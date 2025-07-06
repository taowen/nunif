#include "hw_frame_decoder.h"
#include <iostream>

HwFrameDecoder::HwFrameDecoder() 
    : video_codec_context_(nullptr)
    , audio_codec_context_(nullptr)
    , hw_device_ctx_(nullptr)
    , video_codec_(nullptr)
    , audio_codec_(nullptr)
    , audio_resampler_(nullptr)
    , is_initialized_(false)
    , current_pair_index_(0) {
    
    // 初始化双缓冲HwFramePair
    for (int i = 0; i < 2; i++) {
        borrowed_pairs_[i].audio_frame.frame = nullptr;
        borrowed_pairs_[i].video_frame.frame = nullptr;
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].video_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
    }
}

HwFrameDecoder::~HwFrameDecoder() {
    close();
}

bool HwFrameDecoder::open(const std::string& filepath) {
    // 清理已有资源
    close();
    
    // 保存文件路径
    filepath_ = filepath;
    
    // 1. 打开MKV文件
    if (!demuxer_.open(filepath)) {
        std::cerr << "Failed to open MKV file: " << filepath << std::endl;
        return false;
    }
    
    // 2. 获取codec parameters
    AVCodecParameters* video_codec_params = demuxer_.getReader().getVideoCodecParameters();
    AVCodecParameters* audio_codec_params = demuxer_.getReader().getAudioCodecParameters();
    if (!video_codec_params || !audio_codec_params) {
        std::cerr << "Failed to get codec parameters" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 3. 创建硬件上下文（让FFmpeg自动管理）
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA hardware device context, error: " << ret << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 4. 配置视频解码器
    if (!configureVideoDecoder(video_codec_params)) {
        std::cerr << "Failed to configure video decoder" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 5. 配置音频解码器
    if (!configureAudioDecoder(audio_codec_params)) {
        std::cerr << "Failed to configure audio decoder" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 6. 初始化音频重采样器
    if (!initializeAudioResampler()) {
        std::cerr << "Failed to initialize audio resampler" << std::endl;
        demuxer_.close();
        return false;
    }
    
    // 7. 初始化双缓冲帧
    initializeBorrowedPairs();
    
    is_initialized_ = true;
    return true;
}

bool HwFrameDecoder::readNextHwFramePair(HwFramePair& decoded_frames) {
    if (!is_initialized_) {
        return false;
    }
    
    
    // 从demuxer获取同步的包
    PacketDemuxer::PacketPair synced_packets;
    if (!demuxer_.readNextPacketPair(synced_packets) || !synced_packets.is_valid) {
        return false;
    }
    
    // 获取当前pair的引用
    HwFramePair& current_pair = borrowed_pairs_[current_pair_index_];
    
    // 清理之前的数据，但保持AVFrame结构体不变（池化复用）
    av_frame_unref(current_pair.audio_frame.frame);
    av_frame_unref(current_pair.video_frame.frame);
    
    // 重置有效标志
    current_pair.audio_frame.is_valid = false;
    current_pair.video_frame.is_valid = false;
    
    // 设置同步时间戳
    current_pair.audio_frame.timestamp = synced_packets.timestamp;
    current_pair.video_frame.timestamp = synced_packets.timestamp;
    
    // 解码音频包
    if (synced_packets.audio_packet) {
        current_pair.audio_frame.is_valid = decodeAudioPacket(
            synced_packets.audio_packet, 
            current_pair.audio_frame.frame
        );
        // 注意：不再手动释放包，由PacketDemuxer管理
    }
    
    // 解码视频包
    if (synced_packets.video_packet) {
        current_pair.video_frame.is_valid = decodeVideoPacket(
            synced_packets.video_packet,
            current_pair.video_frame.frame
        );
        // 注意：不再手动释放包，由PacketDemuxer管理
    }

    // 注意：帧池索引将随着current_pair_index_的切换而自动更新
    
    // 至少要有一个有效帧
    bool has_valid_frame = current_pair.audio_frame.is_valid || current_pair.video_frame.is_valid;
    if (has_valid_frame) {
        current_pair.is_valid = true;
        
        // 将当前pair返回给调用者
        decoded_frames = current_pair;
        
        // 切换到下一个pair
        current_pair_index_ = (current_pair_index_ + 1) % 2;
    }
    
    return has_valid_frame;
}

bool HwFrameDecoder::decodeVideoPacket(AVPacket* packet, AVFrame* frame) {
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

bool HwFrameDecoder::decodeAudioPacket(AVPacket* packet, AVFrame* frame) {
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

const char* HwFrameDecoder::getVideoCodecName() const {
    if (video_codec_) {
        return video_codec_->name;
    }
    return nullptr;
}

const char* HwFrameDecoder::getAudioCodecName() const {
    if (audio_codec_) {
        return audio_codec_->name;
    }
    return nullptr;
}




void HwFrameDecoder::close() {
    demuxer_.close();
    releaseResources();
    is_initialized_ = false;
    filepath_.clear();
}


bool HwFrameDecoder::configureVideoDecoder(AVCodecParameters* codec_params) {
    // 查找视频硬件解码器
    video_codec_ = avcodec_find_decoder(codec_params->codec_id);
    if (!video_codec_) {
        std::cerr << "No video decoder found for codec ID: " << codec_params->codec_id << std::endl;
        return false;
    }
    
    // 检查通用解码器是否支持D3D11VA硬件加速
    bool supports_d3d11 = false;
    for (int i = 0;; i++) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(video_codec_, i);
        if (!config) {
            break;
        }
        
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
            std::cout << "Found D3D11VA support for video codec: " << video_codec_->name << std::endl;
            supports_d3d11 = true;
            break;
        }
    }
    
    if (!supports_d3d11) {
        std::cerr << "Video codec does not support D3D11VA hardware acceleration" << std::endl;
        return false;
    }

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

bool HwFrameDecoder::configureAudioDecoder(AVCodecParameters* codec_params) {
    // 查找音频解码器
    audio_codec_ = avcodec_find_decoder(codec_params->codec_id);
    if (!audio_codec_) {
        std::cerr << "No audio decoder found for codec ID: " << codec_params->codec_id << std::endl;
        return false;
    }
    
    std::cout << "Found audio decoder: " << audio_codec_->name << std::endl;

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

bool HwFrameDecoder::initializeAudioResampler() {
    // 初始化音频重采样器（如果需要）
    // 这里暂时返回true，实际实现可能需要根据输出格式配置重采样
    return true;
}

void HwFrameDecoder::releaseResources() {
    // 释放编解码器上下文
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
    
    // 释放池化的AVFrame（只在这里释放！）
    for (int i = 0; i < 2; i++) {
        if (borrowed_pairs_[i].audio_frame.frame) {
            av_frame_free(&borrowed_pairs_[i].audio_frame.frame);
            borrowed_pairs_[i].audio_frame.frame = nullptr;
        }
        if (borrowed_pairs_[i].video_frame.frame) {
            av_frame_free(&borrowed_pairs_[i].video_frame.frame);
            borrowed_pairs_[i].video_frame.frame = nullptr;
        }
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].video_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
    }
    
    video_codec_ = nullptr;
    audio_codec_ = nullptr;
}

void HwFrameDecoder::initializeBorrowedPairs() {
    // 一次性分配所有AVFrame，池化管理避免频繁分配释放
    for (int i = 0; i < 2; i++) {
        // 分配音频和视频帧，生命周期与HwFrameDecoder相同
        if (!borrowed_pairs_[i].audio_frame.frame) {
            borrowed_pairs_[i].audio_frame.frame = av_frame_alloc();
        }
        if (!borrowed_pairs_[i].video_frame.frame) {
            borrowed_pairs_[i].video_frame.frame = av_frame_alloc();
        }
        
        // 初始化状态
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].video_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
        borrowed_pairs_[i].audio_frame.timestamp = 0.0;
        borrowed_pairs_[i].video_frame.timestamp = 0.0;
    }
}




bool HwFrameDecoder::hasValidPair() const {
    return borrowed_pairs_[0].is_valid || borrowed_pairs_[1].is_valid;
}

int HwFrameDecoder::getValidPairCount() const {
    int count = 0;
    if (borrowed_pairs_[0].is_valid) count++;
    if (borrowed_pairs_[1].is_valid) count++;
    return count;
}

bool HwFrameDecoder::tryDecodeFirstVideoFrame(AVFrame* frame) {
    if (!is_initialized_ || !frame) {
        return false;
    }
    
    // 创建独立的临时 demuxer，避免污染主 demuxer 状态
    PacketDemuxer temp_demuxer;
    if (!temp_demuxer.open(filepath_)) {
        return false;
    }
    
    // 清理输出帧
    av_frame_unref(frame);
    
    // 尝试读取视频包直到成功解码出一帧
    PacketDemuxer::PacketPair packet_pair;
    bool success = false;
    
    while (temp_demuxer.readNextPacketPair(packet_pair)) {
        if (packet_pair.is_valid && packet_pair.video_packet) {
            // 解码视频包
            if (decodeVideoPacket(packet_pair.video_packet, frame)) {
                success = true;
                break;
            }
            // 继续尝试下一个包
        }
    }
    
    // 清理临时 demuxer
    temp_demuxer.close();
    
    return success;
}

bool HwFrameDecoder::tryDecodeFirstAudioFrame(AVFrame* frame) {
    if (!is_initialized_ || !frame) {
        return false;
    }
    
    // 创建独立的临时 demuxer，避免污染主 demuxer 状态
    PacketDemuxer temp_demuxer;
    if (!temp_demuxer.open(filepath_)) {
        return false;
    }
    
    // 清理输出帧
    av_frame_unref(frame);
    
    // 尝试读取音频包直到成功解码出一帧
    PacketDemuxer::PacketPair packet_pair;
    bool success = false;
    
    while (temp_demuxer.readNextPacketPair(packet_pair)) {
        if (packet_pair.is_valid && packet_pair.audio_packet) {
            // 解码音频包
            if (decodeAudioPacket(packet_pair.audio_packet, frame)) {
                success = true;
                break;
            }
            // 继续尝试下一个包
        }
    }
    
    // 清理临时 demuxer
    temp_demuxer.close();
    
    return success;
}

ID3D11Device* HwFrameDecoder::getD3D11DeviceFromFrame(AVFrame* frame) {
    if (!frame || frame->format != AV_PIX_FMT_D3D11) {
        return nullptr;
    }
    
    // 从硬件帧获取D3D11纹理
    ID3D11Texture2D* texture = (ID3D11Texture2D*)frame->data[0];
    if (!texture) {
        return nullptr;
    }
    
    // 从纹理获取设备
    ID3D11Device* device = nullptr;
    texture->GetDevice(&device);
    return device;
}

ID3D11DeviceContext* HwFrameDecoder::getD3D11ContextFromFrame(AVFrame* frame) {
    ID3D11Device* device = getD3D11DeviceFromFrame(frame);
    if (!device) {
        return nullptr;
    }
    
    ID3D11DeviceContext* context = nullptr;
    device->GetImmediateContext(&context);
    device->Release(); // 释放从GetDevice获取的引用
    return context;
}