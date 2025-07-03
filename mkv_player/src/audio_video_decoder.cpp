#include "audio_video_decoder.h"
#include <iostream>

AudioVideoDecoder::AudioVideoDecoder() 
    : d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , video_codec_context_(nullptr)
    , audio_codec_context_(nullptr)
    , hw_device_ctx_(nullptr)
    , video_codec_(nullptr)
    , audio_codec_(nullptr)
    , audio_resampler_(nullptr)
    , next_audio_timestamp_(0.0)
    , has_buffered_video_frame_(false)
    , buffered_video_frame_(nullptr)
    , buffered_video_timestamp_(0.0)
    , is_initialized_(false) {
}

AudioVideoDecoder::~AudioVideoDecoder() {
    close();
}

bool AudioVideoDecoder::open(const std::string& filepath) {
    // 清理已有资源
    close();
    
    // 1. 打开MKV文件
    if (!reader_.open(filepath)) {
        std::cerr << "Failed to open MKV file: " << filepath << std::endl;
        return false;
    }
    
    auto stream_info = reader_.getStreamInfo();
    if (stream_info.video_stream_index < 0) {
        std::cerr << "No video stream found in file" << std::endl;
        reader_.close();
        return false;
    }
    
    if (stream_info.audio_stream_index < 0) {
        std::cerr << "No audio stream found in file" << std::endl;
        reader_.close();
        return false;
    }
    
    // 2. 获取codec parameters
    AVCodecParameters* video_codec_params = reader_.getVideoCodecParameters();
    AVCodecParameters* audio_codec_params = reader_.getAudioCodecParameters();
    if (!video_codec_params || !audio_codec_params) {
        std::cerr << "Failed to get codec parameters" << std::endl;
        reader_.close();
        return false;
    }
    
    // 3. 查找视频硬件解码器
    if (!findVideoHardwareDecoder(video_codec_params->codec_id)) {
        std::cerr << "Video hardware decoder not found" << std::endl;
        reader_.close();
        return false;
    }
    
    // 4. 查找音频解码器
    if (!findAudioDecoder(audio_codec_params->codec_id)) {
        std::cerr << "Audio decoder not found" << std::endl;
        reader_.close();
        return false;
    }
    
    // 5. 创建DirectX11设备
    if (!createD3D11Device()) {
        std::cerr << "Failed to create D3D11 device" << std::endl;
        reader_.close();
        return false;
    }
    
    // 6. 创建硬件上下文
    if (!createHardwareContext()) {
        std::cerr << "Failed to create hardware context" << std::endl;
        reader_.close();
        return false;
    }
    
    // 7. 配置视频解码器
    if (!configureVideoDecoder(video_codec_params)) {
        std::cerr << "Failed to configure video decoder" << std::endl;
        reader_.close();
        return false;
    }
    
    // 8. 配置音频解码器
    if (!configureAudioDecoder(audio_codec_params)) {
        std::cerr << "Failed to configure audio decoder" << std::endl;
        reader_.close();
        return false;
    }
    
    // 9. 初始化音频重采样器
    if (!initializeAudioResampler()) {
        std::cerr << "Failed to initialize audio resampler" << std::endl;
        reader_.close();
        return false;
    }
    
    // 10. 分配缓冲视频帧
    buffered_video_frame_ = av_frame_alloc();
    if (!buffered_video_frame_) {
        std::cerr << "Failed to allocate buffered video frame" << std::endl;
        reader_.close();
        return false;
    }
    
    is_initialized_ = true;
    return true;
}

bool AudioVideoDecoder::readNextFrames(DecodedFrames& decoded_frames) {
    if (!is_initialized_) {
        decoded_frames.audio_frame.is_valid = false;
        decoded_frames.video_frame.is_valid = false;
        return false;
    }
    
    // 分配音频帧
    if (!decoded_frames.audio_frame.frame) {
        decoded_frames.audio_frame.frame = av_frame_alloc();
        if (!decoded_frames.audio_frame.frame) {
            decoded_frames.audio_frame.is_valid = false;
            decoded_frames.video_frame.is_valid = false;
            return false;
        }
    }
    
    // 分配视频帧
    if (!decoded_frames.video_frame.frame) {
        decoded_frames.video_frame.frame = av_frame_alloc();
        if (!decoded_frames.video_frame.frame) {
            decoded_frames.audio_frame.is_valid = false;
            decoded_frames.video_frame.is_valid = false;
            return false;
        }
    }
    
    // 1. 先解码下一个音频帧（以音频为主时钟）
    if (!decodeNextAudioFrame(decoded_frames.audio_frame)) {
        decoded_frames.audio_frame.is_valid = false;
        decoded_frames.video_frame.is_valid = false;
        return false;
    }
    
    // 2. 找到匹配的视频帧（基于音频时间戳）
    double audio_timestamp = decoded_frames.audio_frame.timestamp;
    findMatchingVideoFrame(decoded_frames.video_frame, audio_timestamp);
    
    return true;
}

bool AudioVideoDecoder::decodeNextAudioFrame(DecodedFrame& audio_frame) {
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        audio_frame.is_valid = false;
        return false;
    }
    
    while (true) {
        // 尝试从音频解码器获取帧
        int ret = avcodec_receive_frame(audio_codec_context_, audio_frame.frame);
        if (ret == 0) {
            // 成功获取音频帧
            auto stream_info = reader_.getStreamInfo();
            if (audio_frame.frame->pts != AV_NOPTS_VALUE) {
                audio_frame.timestamp = audio_frame.frame->pts * av_q2d(stream_info.audio_time_base);
            } else {
                audio_frame.timestamp = next_audio_timestamp_;
            }
            
            // 更新下一个音频时间戳
            double duration = (double)audio_frame.frame->nb_samples / audio_frame.frame->sample_rate;
            next_audio_timestamp_ = audio_frame.timestamp + duration;
            
            audio_frame.is_valid = true;
            av_packet_free(&packet);
            return true;
        } else if (ret == AVERROR_EOF) {
            audio_frame.is_valid = false;
            av_packet_free(&packet);
            return false;
        } else if (ret == AVERROR(EAGAIN)) {
            // 需要更多输入数据
            while (reader_.readNextPacket(packet)) {
                if (reader_.isAudioPacket(packet)) {
                    // 发送音频包到解码器
                    ret = avcodec_send_packet(audio_codec_context_, packet);
                    av_packet_unref(packet);
                    if (ret < 0 && ret != AVERROR(EAGAIN)) {
                        std::cerr << "Error sending audio packet to decoder: " << ret << std::endl;
                        audio_frame.is_valid = false;
                        av_packet_free(&packet);
                        return false;
                    }
                    break;
                }
                av_packet_unref(packet);
            }
            
            if (reader_.isEOF()) {
                avcodec_send_packet(audio_codec_context_, nullptr);
            }
        } else {
            std::cerr << "Error receiving audio frame: " << ret << std::endl;
            audio_frame.is_valid = false;
            av_packet_free(&packet);
            return false;
        }
    }
}

bool AudioVideoDecoder::findMatchingVideoFrame(DecodedFrame& video_frame, double target_timestamp) {
    // 检查是否有缓冲的视频帧
    if (has_buffered_video_frame_) {
        double time_diff = buffered_video_timestamp_ - target_timestamp;
        
        // 如果缓冲帧太旧，跳过它
        if (time_diff < -0.1) {
            av_frame_unref(buffered_video_frame_);
            has_buffered_video_frame_ = false;
        }
        // 如果缓冲帧在可接受范围内，使用它
        else if (time_diff >= -0.1 && time_diff <= 0.1) {
            av_frame_move_ref(video_frame.frame, buffered_video_frame_);
            video_frame.timestamp = buffered_video_timestamp_;
            video_frame.is_valid = true;
            has_buffered_video_frame_ = false;
            return true;
        }
        // 如果缓冲帧太新，保留它，不返回视频帧
        else {
            video_frame.is_valid = false;
            return true;
        }
    }
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        video_frame.is_valid = false;
        return false;
    }
    
    // 解码视频帧直到找到匹配的或超过目标时间戳
    while (true) {
        // 尝试从视频解码器获取帧
        int ret = avcodec_receive_frame(video_codec_context_, video_frame.frame);
        if (ret == 0) {
            // 成功获取视频帧
            auto stream_info = reader_.getStreamInfo();
            double video_timestamp = 0.0;
            if (video_frame.frame->pts != AV_NOPTS_VALUE) {
                video_timestamp = video_frame.frame->pts * av_q2d(stream_info.video_time_base);
            }
            
            double time_diff = video_timestamp - target_timestamp;
            
            // 如果视频帧太旧，继续解码
            if (time_diff < -0.1) {
                av_frame_unref(video_frame.frame);
                continue;
            }
            // 如果视频帧在可接受范围内，返回它
            else if (time_diff >= -0.1 && time_diff <= 0.1) {
                video_frame.timestamp = video_timestamp;
                video_frame.is_valid = true;
                av_packet_free(&packet);
                return true;
            }
            // 如果视频帧太新，缓冲它
            else {
                av_frame_move_ref(buffered_video_frame_, video_frame.frame);
                buffered_video_timestamp_ = video_timestamp;
                has_buffered_video_frame_ = true;
                video_frame.is_valid = false;
                av_packet_free(&packet);
                return true;
            }
        } else if (ret == AVERROR_EOF) {
            video_frame.is_valid = false;
            av_packet_free(&packet);
            return true;
        } else if (ret == AVERROR(EAGAIN)) {
            // 需要更多输入数据
            bool found_video = false;
            while (reader_.readNextPacket(packet)) {
                if (reader_.isVideoPacket(packet)) {
                    // 发送视频包到解码器
                    ret = avcodec_send_packet(video_codec_context_, packet);
                    av_packet_unref(packet);
                    if (ret < 0 && ret != AVERROR(EAGAIN)) {
                        std::cerr << "Error sending video packet to decoder: " << ret << std::endl;
                        video_frame.is_valid = false;
                        av_packet_free(&packet);
                        return false;
                    }
                    found_video = true;
                    break;
                }
                av_packet_unref(packet);
            }
            
            if (!found_video && reader_.isEOF()) {
                avcodec_send_packet(video_codec_context_, nullptr);
                video_frame.is_valid = false;
                av_packet_free(&packet);
                return true;
            }
        } else {
            std::cerr << "Error receiving video frame: " << ret << std::endl;
            video_frame.is_valid = false;
            av_packet_free(&packet);
            return false;
        }
    }
}

const char* AudioVideoDecoder::getVideoCodecName() const {
    if (video_codec_) {
        return video_codec_->name;
    }
    return nullptr;
}

const char* AudioVideoDecoder::getAudioCodecName() const {
    if (audio_codec_) {
        return audio_codec_->name;
    }
    return nullptr;
}

void AudioVideoDecoder::flush() {
    if (is_initialized_) {
        if (video_codec_context_) {
            avcodec_send_packet(video_codec_context_, nullptr);
        }
        if (audio_codec_context_) {
            avcodec_send_packet(audio_codec_context_, nullptr);
        }
        
        // 清除缓冲的视频帧
        if (has_buffered_video_frame_) {
            av_frame_unref(buffered_video_frame_);
            has_buffered_video_frame_ = false;
        }
    }
}

void AudioVideoDecoder::close() {
    reader_.close();
    releaseResources();
    is_initialized_ = false;
}

bool AudioVideoDecoder::createD3D11Device() {
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

bool AudioVideoDecoder::createHardwareContext() {
    // 让FFmpeg自动创建D3D11VA硬件设备上下文
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA hardware device context, error: " << ret << std::endl;
        return false;
    }
    
    std::cout << "D3D11VA hardware context created successfully" << std::endl;
    return true;
}

bool AudioVideoDecoder::findVideoHardwareDecoder(AVCodecID codec_id) {
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

bool AudioVideoDecoder::findAudioDecoder(AVCodecID codec_id) {
    audio_codec_ = avcodec_find_decoder(codec_id);
    if (!audio_codec_) {
        std::cerr << "No audio decoder found for codec ID: " << codec_id << std::endl;
        return false;
    }
    
    std::cout << "Found audio decoder: " << audio_codec_->name << std::endl;
    return true;
}

bool AudioVideoDecoder::configureVideoDecoder(AVCodecParameters* codec_params) {
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
    video_codec_context_->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
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

bool AudioVideoDecoder::configureAudioDecoder(AVCodecParameters* codec_params) {
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

bool AudioVideoDecoder::initializeAudioResampler() {
    // 初始化音频重采样器（如果需要）
    // 这里暂时返回true，实际实现可能需要根据输出格式配置重采样
    return true;
}

void AudioVideoDecoder::releaseResources() {
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
    
    if (buffered_video_frame_) {
        av_frame_free(&buffered_video_frame_);
        buffered_video_frame_ = nullptr;
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
    
    video_codec_ = nullptr;
    audio_codec_ = nullptr;
    has_buffered_video_frame_ = false;
    next_audio_timestamp_ = 0.0;
    buffered_video_timestamp_ = 0.0;
}