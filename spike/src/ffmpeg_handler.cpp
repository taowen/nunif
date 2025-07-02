#include "ffmpeg_handler.h"
#include "audio_state.h"
#include <iostream>

// 先包含 DirectX 相关头文件
#include <d3d11.h>

extern "C" {
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

// FFmpeg 处理器结构体 - 内部实现
struct FFmpegHandler {
    // FFmpeg 相关
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* videoCodecContext = nullptr;
    AVCodecContext* audioCodecContext = nullptr;
    
    // 硬件解码相关
    AVBufferRef* hwDeviceContext = nullptr;
    enum AVPixelFormat hwPixelFormat = AV_PIX_FMT_NONE;
    
    // 流索引
    int videoStreamIndex = -1;
    int audioStreamIndex = -1;
    
    // 时间基准
    double videoTimeBase = 0.0;
    double audioTimeBase = 0.0;
    
    // 硬件解码状态
    bool isHardwareDecoded = false;
};

// 静态全局变量 - 隐藏在实现文件中
static FFmpegHandler ffmpegState;

// 硬件解码像素格式回调
static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_D3D11) {
            return *p;
        }
    }
    
    std::cerr << "Failed to get HW surface format." << std::endl;
    return AV_PIX_FMT_NONE;
}

bool createFFmpegHandler(const char* filename) {
    // 先清理之前的状态
    destroyFFmpegHandler();
    
    // 打开文件
    if (avformat_open_input(&ffmpegState.formatContext, filename, nullptr, nullptr) < 0) {
        std::cerr << "Error: Cannot open input file" << std::endl;
        return false;
    }
    
    // 获取流信息
    if (avformat_find_stream_info(ffmpegState.formatContext, nullptr) < 0) {
        std::cerr << "Error: Cannot find stream info" << std::endl;
        avformat_close_input(&ffmpegState.formatContext);
        return false;
    }
    
    // 查找视频和音频流
    for (unsigned int i = 0; i < ffmpegState.formatContext->nb_streams; i++) {
        if (ffmpegState.formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && ffmpegState.videoStreamIndex == -1) {
            ffmpegState.videoStreamIndex = i;
        } else if (ffmpegState.formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && ffmpegState.audioStreamIndex == -1) {
            ffmpegState.audioStreamIndex = i;
        }
    }
    
    // 初始化视频解码器（仅硬件解码）
    if (ffmpegState.videoStreamIndex >= 0) {
        AVStream* videoStream = ffmpegState.formatContext->streams[ffmpegState.videoStreamIndex];
        
        // 创建 DirectX 11 硬件设备上下文
        int ret = av_hwdevice_ctx_create(&ffmpegState.hwDeviceContext, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret < 0) {
            char errBuf[256];
            av_strerror(ret, errBuf, sizeof(errBuf));
            std::cerr << "Error: Failed to create D3D11VA hardware device context: " << errBuf << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        // 查找解码器
        const AVCodec* videoCodec = avcodec_find_decoder(videoStream->codecpar->codec_id);
        if (!videoCodec) {
            std::cerr << "Error: No suitable decoder found" << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        // 检查解码器是否支持硬件加速
        bool canUseHardware = false;
        for (int i = 0; ; i++) {
            const AVCodecHWConfig *config = avcodec_get_hw_config(videoCodec, i);
            if (!config) {
                break;
            }
            
            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
                config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
                canUseHardware = true;
                break;
            }
        }
        
        if (!canUseHardware) {
            std::cerr << "Error: Decoder does not support D3D11VA hardware acceleration" << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.videoCodecContext = avcodec_alloc_context3(videoCodec);
        if (!ffmpegState.videoCodecContext) {
            std::cerr << "Error: Failed to allocate video codec context" << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        if (avcodec_parameters_to_context(ffmpegState.videoCodecContext, videoStream->codecpar) < 0) {
            std::cerr << "Error: Failed to copy codec parameters to context" << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        // 设置硬件设备上下文
        ffmpegState.videoCodecContext->hw_device_ctx = av_buffer_ref(ffmpegState.hwDeviceContext);
        ffmpegState.videoCodecContext->get_format = get_hw_format;
        
        // 设置额外的硬件帧数量以避免缓冲池溢出
        ffmpegState.videoCodecContext->extra_hw_frames = 10;
        
        if (avcodec_open2(ffmpegState.videoCodecContext, videoCodec, nullptr) < 0) {
            std::cerr << "Error: Failed to open decoder" << std::endl;
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.videoTimeBase = av_q2d(videoStream->time_base);
        ffmpegState.isHardwareDecoded = true;
        ffmpegState.hwPixelFormat = AV_PIX_FMT_D3D11;
    }
    
    // 初始化音频解码器
    if (ffmpegState.audioStreamIndex >= 0) {
        AVStream* audioStream = ffmpegState.formatContext->streams[ffmpegState.audioStreamIndex];
        const AVCodec* audioCodec = avcodec_find_decoder(audioStream->codecpar->codec_id);
        if (!audioCodec) {
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.audioCodecContext = avcodec_alloc_context3(audioCodec);
        if (!ffmpegState.audioCodecContext) {
            destroyFFmpegHandler();
            return false;
        }
        
        if (avcodec_parameters_to_context(ffmpegState.audioCodecContext, audioStream->codecpar) < 0) {
            destroyFFmpegHandler();
            return false;
        }
        if (avcodec_open2(ffmpegState.audioCodecContext, audioCodec, nullptr) < 0) {
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.audioTimeBase = av_q2d(audioStream->time_base);
    }
    
    return true;
}

void destroyFFmpegHandler() {
    if (ffmpegState.videoCodecContext) { 
        avcodec_free_context(&ffmpegState.videoCodecContext);
    }
    if (ffmpegState.audioCodecContext) { 
        avcodec_free_context(&ffmpegState.audioCodecContext);
    }
    if (ffmpegState.hwDeviceContext) {
        av_buffer_unref(&ffmpegState.hwDeviceContext);
    }
    if (ffmpegState.formatContext) { 
        avformat_close_input(&ffmpegState.formatContext);
    }
    
    // 重置索引和状态
    ffmpegState.videoStreamIndex = -1;
    ffmpegState.audioStreamIndex = -1;
    ffmpegState.videoTimeBase = 0.0;
    ffmpegState.audioTimeBase = 0.0;
    ffmpegState.isHardwareDecoded = false;
    ffmpegState.hwPixelFormat = AV_PIX_FMT_NONE;
}

bool getVideoInfo(VideoInfo* info) {
    if (!info || ffmpegState.videoStreamIndex < 0) return false;
    
    info->width = ffmpegState.videoCodecContext->width;
    info->height = ffmpegState.videoCodecContext->height;
    info->timeBase = ffmpegState.videoTimeBase;
    info->streamIndex = ffmpegState.videoStreamIndex;
    info->isHardwareDecoded = ffmpegState.isHardwareDecoded;
    
    return true;
}

bool getAudioInfo(AudioInfo* info) {
    if (!info || ffmpegState.audioStreamIndex < 0) return false;
    
    info->sampleRate = ffmpegState.audioCodecContext->sample_rate;
    info->channels = ffmpegState.audioCodecContext->ch_layout.nb_channels;
    info->timeBase = ffmpegState.audioTimeBase;
    info->streamIndex = ffmpegState.audioStreamIndex;
    
    return true;
}

bool getAudioConfig(AudioConfig* config) {
    if (!config || ffmpegState.audioStreamIndex < 0) return false;
    
    config->inputSampleRate = ffmpegState.audioCodecContext->sample_rate;
    config->inputChannels = ffmpegState.audioCodecContext->ch_layout.nb_channels;
    config->inputFormat = ffmpegState.audioCodecContext->sample_fmt;
    config->inputChannelLayout = ffmpegState.audioCodecContext->ch_layout;
    
    return true;
}

AVFormatContext* getFormatContext() {
    return ffmpegState.formatContext;
}

AVCodecContext* getVideoCodecContext() {
    return ffmpegState.videoCodecContext;
}

AVCodecContext* getAudioCodecContext() {
    return ffmpegState.audioCodecContext;
}

AVBufferRef* getHwDeviceContext() {
    return ffmpegState.hwDeviceContext;
}

// 新增：获取 D3D11 Device
ID3D11Device* getD3D11Device() {
    if (!ffmpegState.hwDeviceContext) return nullptr;
    
    AVHWDeviceContext* hwDeviceContext = (AVHWDeviceContext*)ffmpegState.hwDeviceContext->data;
    if (hwDeviceContext->type != AV_HWDEVICE_TYPE_D3D11VA) return nullptr;
    
    AVD3D11VADeviceContext* d3d11Context = (AVD3D11VADeviceContext*)hwDeviceContext->hwctx;
    return d3d11Context->device;
}

ID3D11DeviceContext* getD3D11DeviceContext() {
    if (!ffmpegState.hwDeviceContext) return nullptr;
    
    AVHWDeviceContext* hwDeviceContext = (AVHWDeviceContext*)ffmpegState.hwDeviceContext->data;
    if (hwDeviceContext->type != AV_HWDEVICE_TYPE_D3D11VA) return nullptr;
    
    AVD3D11VADeviceContext* d3d11Context = (AVD3D11VADeviceContext*)hwDeviceContext->hwctx;
    return d3d11Context->device_context;
} 