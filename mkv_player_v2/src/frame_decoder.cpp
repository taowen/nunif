#include "frame_decoder.h"
#include <iostream>

FrameDecoder::FrameDecoder() 
    : d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , video_codec_context_(nullptr)
    , audio_codec_context_(nullptr)
    , hw_device_ctx_(nullptr)
    , video_codec_(nullptr)
    , audio_codec_(nullptr)
    , audio_resampler_(nullptr)
    , is_initialized_(false)
    , current_audio_frame_index_(0)
    , current_video_frame_index_(0) {
    
    // 初始化AVFrame池
    for (int i = 0; i < AVFRAME_POOL_SIZE; i++) {
        audio_frame_pool_[i] = nullptr;
        video_frame_pool_[i] = nullptr;
    }
}

FrameDecoder::~FrameDecoder() {
    close();
}

bool FrameDecoder::open(const std::string& filepath) {
    // 清理已有资源
    close();
    
    // 1. 打开MKV文件
    if (!demuxer_.open(filepath)) {
        std::cerr << "Failed to open MKV file: " << filepath << std::endl;
        return false;
    }
    
    // 2. 获取codec parameters
    AVCodecParameters* video_codec_params = demuxer_.getVideoCodecParameters();
    AVCodecParameters* audio_codec_params = demuxer_.getAudioCodecParameters();
    if (!video_codec_params || !audio_codec_params) {
        std::cerr << "Failed to get codec parameters" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 3. 查找视频硬件解码器
    if (!findVideoHardwareDecoder(video_codec_params->codec_id)) {
        std::cerr << "Video hardware decoder not found" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 4. 查找音频解码器
    if (!findAudioDecoder(audio_codec_params->codec_id)) {
        std::cerr << "Audio decoder not found" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 5. 创建硬件上下文（FFmpeg会创建自己的D3D11设备）
    if (!createHardwareContext()) {
        std::cerr << "Failed to create hardware context" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 6. 配置视频解码器
    if (!configureVideoDecoder(video_codec_params)) {
        std::cerr << "Failed to configure video decoder" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 7. 配置音频解码器
    if (!configureAudioDecoder(audio_codec_params)) {
        std::cerr << "Failed to configure audio decoder" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 8. 初始化音频重采样器
    if (!initializeAudioResampler()) {
        std::cerr << "Failed to initialize audio resampler" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 9. 初始化AVFrame池
    initializeFramePools();
    
    is_initialized_ = true;
    return true;
}

bool FrameDecoder::readNextFrames(DecodedFrames& decoded_frames) {
    if (!is_initialized_) {
        decoded_frames.audio_frame.is_valid = false;
        decoded_frames.video_frame.is_valid = false;
        return false;
    }
    
    // 从池中获取下一帧的索引
    int audio_idx = current_audio_frame_index_;
    int video_idx = current_video_frame_index_;
    AVFrame* audio_frame = audio_frame_pool_[audio_idx];
    AVFrame* video_frame = video_frame_pool_[video_idx];

    // 清理之前的数据
    av_frame_unref(audio_frame);
    av_frame_unref(video_frame);

    // 设置帧的所有权信息
    decoded_frames.audio_frame.owner = this;
    decoded_frames.audio_frame.pool_index = audio_idx;
    decoded_frames.video_frame.owner = this;
    decoded_frames.video_frame.pool_index = video_idx;
    
    // 重置有效标志
    decoded_frames.audio_frame.is_valid = false;
    decoded_frames.video_frame.is_valid = false;
    
    // 从demuxer获取同步的包
    PacketDemuxer::SyncedPackets synced_packets;
    if (!demuxer_.readNextSyncedPackets(synced_packets)) {
        return false;
    }
    
    // 设置同步时间戳
    decoded_frames.audio_frame.timestamp = synced_packets.timestamp;
    decoded_frames.video_frame.timestamp = synced_packets.timestamp;
    
    // 解码音频包
    if (synced_packets.audio_packet) {
        decoded_frames.audio_frame.is_valid = decodeAudioPacket(
            synced_packets.audio_packet, 
            audio_frame
        );
        av_packet_free(&synced_packets.audio_packet);
    }
    
    // 解码视频包
    if (synced_packets.video_packet) {
        decoded_frames.video_frame.is_valid = decodeVideoPacket(
            synced_packets.video_packet,
            video_frame
        );
        av_packet_free(&synced_packets.video_packet);
    }

    // 如果成功解码，移动到下一个槽位
    if (decoded_frames.audio_frame.is_valid) {
        current_audio_frame_index_ = (current_audio_frame_index_ + 1) % AVFRAME_POOL_SIZE;
    }
    if (decoded_frames.video_frame.is_valid) {
        current_video_frame_index_ = (current_video_frame_index_ + 1) % AVFRAME_POOL_SIZE;
    }
    
    // 至少要有一个有效帧
    return decoded_frames.audio_frame.is_valid || decoded_frames.video_frame.is_valid;
}

bool FrameDecoder::decodeVideoPacket(AVPacket* packet, AVFrame* frame) {
    if (!video_codec_context_ || !packet || !frame) {
        return false;
    }
    
    // 发送包到解码器
    int ret = avcodec_send_packet(video_codec_context_, packet);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        std::cerr << "Error sending video packet to decoder: " << ret << std::endl;
        return false;
    }
    
    // 接收解码后的帧
    ret = avcodec_receive_frame(video_codec_context_, frame);
    if (ret == 0) {
        // 验证是硬件帧
        if (frame->format != AV_PIX_FMT_D3D11) {
            std::cerr << "Expected D3D11 hardware frame, got format: " << frame->format << std::endl;
            av_frame_unref(frame);
            return false;
        }
        return true;
    } else if (ret == AVERROR(EAGAIN)) {
        // 需要更多数据
        return false;
    } else {
        std::cerr << "Error receiving video frame: " << ret << std::endl;
        return false;
    }
}

bool FrameDecoder::decodeAudioPacket(AVPacket* packet, AVFrame* frame) {
    if (!audio_codec_context_ || !packet || !frame) {
        return false;
    }
    
    // 发送包到解码器
    int ret = avcodec_send_packet(audio_codec_context_, packet);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        std::cerr << "Error sending audio packet to decoder: " << ret << std::endl;
        return false;
    }
    
    // 接收解码后的帧
    ret = avcodec_receive_frame(audio_codec_context_, frame);
    if (ret == 0) {
        return true;
    } else if (ret == AVERROR(EAGAIN)) {
        // 需要更多数据
        return false;
    } else {
        std::cerr << "Error receiving audio frame: " << ret << std::endl;
        return false;
    }
}

const char* FrameDecoder::getVideoCodecName() const {
    if (video_codec_) {
        return video_codec_->name;
    }
    return nullptr;
}

const char* FrameDecoder::getAudioCodecName() const {
    if (audio_codec_) {
        return audio_codec_->name;
    }
    return nullptr;
}

AVFrame* FrameDecoder::getFrameFromPool(int index, bool is_audio) const {
    if (index < 0 || index >= AVFRAME_POOL_SIZE) {
        return nullptr;
    }
    return is_audio ? audio_frame_pool_[index] : video_frame_pool_[index];
}


void FrameDecoder::flush() {
    if (is_initialized_) {
        if (video_codec_context_) {
            avcodec_send_packet(video_codec_context_, nullptr);
        }
        if (audio_codec_context_) {
            avcodec_send_packet(audio_codec_context_, nullptr);
        }
    }
}

void FrameDecoder::close() {
    demuxer_.close();
    releaseResources();
    is_initialized_ = false;
}

bool FrameDecoder::createD3D11Device() {
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // 默认适配器
        D3D_DRIVER_TYPE_HARDWARE,   // 硬件驱动
        nullptr,                    // 软件驱动句柄
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT, // 支持视频
        nullptr,                    // 功能级别数组
        0,                          // 功能级别数组大小
        D3D11_SDK_VERSION,          // SDK版本
        &d3d11_device_,             // 输出设备
        nullptr,                    // 输出功能级别
        &d3d11_context_             // 输出设备上下文
    );
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool FrameDecoder::createHardwareContext() {
    // 让FFmpeg自动创建D3D11VA硬件设备上下文
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA hardware device context, error: " << ret << std::endl;
        return false;
    }
    
    // 从硬件设备上下文中提取D3D11设备和设备上下文（参考d11_decode.cpp）
    AVHWDeviceContext* hw_device_ctx = (AVHWDeviceContext*)hw_device_ctx_->data;
    AVD3D11VADeviceContext* d3d11_device_ctx = (AVD3D11VADeviceContext*)hw_device_ctx->hwctx;
    
    // 使用FFmpeg创建的D3D11设备，而不是自己创建的
    if (d3d11_device_) {
        d3d11_device_->Release();
        d3d11_device_ = nullptr;
    }
    if (d3d11_context_) {
        d3d11_context_->Release();
        d3d11_context_ = nullptr;
    }
    
    d3d11_device_ = d3d11_device_ctx->device;
    d3d11_context_ = d3d11_device_ctx->device_context;
    
    // 增加引用计数，防止FFmpeg释放时导致野指针
    if (d3d11_device_) {
        d3d11_device_->AddRef();
    }
    if (d3d11_context_) {
        d3d11_context_->AddRef();
    }
    
    std::cout << "D3D11VA hardware context created successfully" << std::endl;
    std::cout << "Using FFmpeg's D3D11 device for hardware acceleration" << std::endl;
    return true;
}

bool FrameDecoder::findVideoHardwareDecoder(AVCodecID codec_id) {
    // 查找支持D3D11VA的通用解码器
    video_codec_ = avcodec_find_decoder(codec_id);
    if (!video_codec_) {
        std::cerr << "No video decoder found for codec ID: " << codec_id << std::endl;
        return false;
    }
    
    // 检查解码器是否支持D3D11VA硬件加速
    for (int i = 0;; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(video_codec_, i);
        if (!config) {
            break;
        }
        
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
            std::cout << "Found D3D11VA support for video codec: " << video_codec_->name << std::endl;
            return true;
        }
    }
    
    std::cerr << "Video codec does not support D3D11VA hardware acceleration" << std::endl;
    return false;
}

bool FrameDecoder::findAudioDecoder(AVCodecID codec_id) {
    audio_codec_ = avcodec_find_decoder(codec_id);
    if (!audio_codec_) {
        std::cerr << "No audio decoder found for codec ID: " << codec_id << std::endl;
        return false;
    }
    
    std::cout << "Found audio decoder: " << audio_codec_->name << std::endl;
    return true;
}

bool FrameDecoder::configureVideoDecoder(AVCodecParameters* codec_params) {
    // 分配视频解码器上下文
    video_codec_context_ = avcodec_alloc_context3(video_codec_);
    if (!video_codec_context_) {
        std::cerr << "Failed to allocate video codec context" << std::endl;
        return false;
    }
    
    // 复制编解码器参数
    int ret = avcodec_parameters_to_context(video_codec_context_, codec_params);
    if (ret < 0) {
        std::cerr << "Failed to copy video codec parameters to context" << std::endl;
        return false;
    }
    
    // 设置硬件设备上下文
    video_codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    // 强制D3D11硬件解码像素格式
    video_codec_context_->get_format = [](AVCodecContext*, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
        // 只接受D3D11格式，不允许软件解码
        for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
            if (*p == AV_PIX_FMT_D3D11) {
                return *p;
            }
        }
        return AV_PIX_FMT_NONE; // 强制失败，不允许软件解码
    };
    
    // 打开视频解码器
    ret = avcodec_open2(video_codec_context_, video_codec_, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to open video codec, error: " << ret << std::endl;
        return false;
    }
    
    return true;
}

bool FrameDecoder::configureAudioDecoder(AVCodecParameters* codec_params) {
    // 分配音频解码器上下文
    audio_codec_context_ = avcodec_alloc_context3(audio_codec_);
    if (!audio_codec_context_) {
        std::cerr << "Failed to allocate audio codec context" << std::endl;
        return false;
    }
    
    // 复制编解码器参数
    int ret = avcodec_parameters_to_context(audio_codec_context_, codec_params);
    if (ret < 0) {
        std::cerr << "Failed to copy audio codec parameters to context" << std::endl;
        return false;
    }
    
    // 打开音频解码器
    ret = avcodec_open2(audio_codec_context_, audio_codec_, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to open audio codec, error: " << ret << std::endl;
        return false;
    }
    
    return true;
}

bool FrameDecoder::initializeAudioResampler() {
    // 初始化音频重采样器（如果需要）
    // 这里暂时返回true，实际实现可能需要根据输出格式配置重采样
    return true;
}

void FrameDecoder::releaseResources() {
    if (video_codec_context_) {
        avcodec_free_context(&video_codec_context_);
        video_codec_context_ = nullptr;
    }
    
    if (audio_codec_context_) {
        avcodec_free_context(&audio_codec_context_);
        audio_codec_context_ = nullptr;
    }
    
    if (audio_resampler_) {
        swr_free(&audio_resampler_);
        audio_resampler_ = nullptr;
    }
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    
    if (d3d11_context_) {
        d3d11_context_->Release();
        d3d11_context_ = nullptr;
    }
    
    if (d3d11_device_) {
        d3d11_device_->Release();
        d3d11_device_ = nullptr;
    }
    
    // 释放AVFrame池
    releaseFramePools();
    
    video_codec_ = nullptr;
    audio_codec_ = nullptr;
}

void FrameDecoder::initializeFramePools() {
    // 初始化音频帧池
    for (int i = 0; i < AVFRAME_POOL_SIZE; i++) {
        audio_frame_pool_[i] = av_frame_alloc();
    }
    
    // 初始化视频帧池
    for (int i = 0; i < AVFRAME_POOL_SIZE; i++) {
        video_frame_pool_[i] = av_frame_alloc();
    }
    
    current_audio_frame_index_ = 0;
    current_video_frame_index_ = 0;
}


void FrameDecoder::releaseFramePools() {
    // 释放音频帧池
    for (int i = 0; i < AVFRAME_POOL_SIZE; i++) {
        if (audio_frame_pool_[i]) {
            av_frame_free(&audio_frame_pool_[i]);
            audio_frame_pool_[i] = nullptr;
        }
    }
    
    // 释放视频帧池
    for (int i = 0; i < AVFRAME_POOL_SIZE; i++) {
        if (video_frame_pool_[i]) {
            av_frame_free(&video_frame_pool_[i]);
            video_frame_pool_[i] = nullptr;
        }
    }
    
    current_audio_frame_index_ = 0;
    current_video_frame_index_ = 0;
}