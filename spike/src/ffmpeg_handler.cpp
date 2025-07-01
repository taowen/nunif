#include "ffmpeg_handler.h"
#include "audio_state.h"
#include <iostream>

extern "C" {
#include <libavutil/opt.h>
}

// FFmpeg 处理器结构体 - 内部实现
struct FFmpegHandler {
    // FFmpeg 相关
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* videoCodecContext = nullptr;
    AVCodecContext* audioCodecContext = nullptr;
    SwsContext* swsContext = nullptr;
    
    // 流索引
    int videoStreamIndex = -1;
    int audioStreamIndex = -1;
    
    // 时间基准
    double videoTimeBase = 0.0;
    double audioTimeBase = 0.0;
};

FFmpegHandlerHandle createFFmpegHandler(const char* filename) {
    FFmpegHandler* handler = new FFmpegHandler();
    
    // 打开文件
    if (avformat_open_input(&handler->formatContext, filename, nullptr, nullptr) < 0) {
        delete handler;
        return nullptr;
    }
    
    // 获取流信息
    if (avformat_find_stream_info(handler->formatContext, nullptr) < 0) {
        avformat_close_input(&handler->formatContext);
        delete handler;
        return nullptr;
    }
    
    // 查找视频和音频流
    for (unsigned int i = 0; i < handler->formatContext->nb_streams; i++) {
        if (handler->formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && handler->videoStreamIndex == -1) {
            handler->videoStreamIndex = i;
        } else if (handler->formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && handler->audioStreamIndex == -1) {
            handler->audioStreamIndex = i;
        }
    }
    
    // 初始化视频解码器
    if (handler->videoStreamIndex >= 0) {
        AVStream* videoStream = handler->formatContext->streams[handler->videoStreamIndex];
        const AVCodec* videoCodec = avcodec_find_decoder(videoStream->codecpar->codec_id);
        if (!videoCodec) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        handler->videoCodecContext = avcodec_alloc_context3(videoCodec);
        if (!handler->videoCodecContext) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        if (avcodec_parameters_to_context(handler->videoCodecContext, videoStream->codecpar) < 0) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        if (avcodec_open2(handler->videoCodecContext, videoCodec, nullptr) < 0) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        handler->videoTimeBase = av_q2d(videoStream->time_base);
        
        // 初始化 swscale
        handler->swsContext = sws_getContext(
            handler->videoCodecContext->width, handler->videoCodecContext->height, handler->videoCodecContext->pix_fmt,
            handler->videoCodecContext->width, handler->videoCodecContext->height, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    }
    
    // 初始化音频解码器
    if (handler->audioStreamIndex >= 0) {
        AVStream* audioStream = handler->formatContext->streams[handler->audioStreamIndex];
        const AVCodec* audioCodec = avcodec_find_decoder(audioStream->codecpar->codec_id);
        if (!audioCodec) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        handler->audioCodecContext = avcodec_alloc_context3(audioCodec);
        if (!handler->audioCodecContext) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        if (avcodec_parameters_to_context(handler->audioCodecContext, audioStream->codecpar) < 0) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        if (avcodec_open2(handler->audioCodecContext, audioCodec, nullptr) < 0) {
            destroyFFmpegHandler(handler);
            return nullptr;
        }
        
        handler->audioTimeBase = av_q2d(audioStream->time_base);
    }
    
    return handler;
}

void destroyFFmpegHandler(FFmpegHandlerHandle handle) {
    if (!handle) return;
    
    if (handle->swsContext) sws_freeContext(handle->swsContext);
    if (handle->videoCodecContext) avcodec_free_context(&handle->videoCodecContext);
    if (handle->audioCodecContext) avcodec_free_context(&handle->audioCodecContext);
    if (handle->formatContext) avformat_close_input(&handle->formatContext);
    
    delete handle;
}

bool getVideoInfo(FFmpegHandlerHandle handle, VideoInfo* info) {
    if (!handle || !info || handle->videoStreamIndex < 0) return false;
    
    info->width = handle->videoCodecContext->width;
    info->height = handle->videoCodecContext->height;
    info->timeBase = handle->videoTimeBase;
    info->streamIndex = handle->videoStreamIndex;
    
    return true;
}

bool getAudioInfo(FFmpegHandlerHandle handle, AudioInfo* info) {
    if (!handle || !info || handle->audioStreamIndex < 0) return false;
    
    info->sampleRate = handle->audioCodecContext->sample_rate;
    info->channels = handle->audioCodecContext->ch_layout.nb_channels;
    info->timeBase = handle->audioTimeBase;
    info->streamIndex = handle->audioStreamIndex;
    
    return true;
}

AVFormatContext* getFormatContext(FFmpegHandlerHandle handle) {
    return handle ? handle->formatContext : nullptr;
}

AVCodecContext* getVideoCodecContext(FFmpegHandlerHandle handle) {
    return handle ? handle->videoCodecContext : nullptr;
}

AVCodecContext* getAudioCodecContext(FFmpegHandlerHandle handle) {
    return handle ? handle->audioCodecContext : nullptr;
}

SwsContext* getSwsContext(FFmpegHandlerHandle handle) {
    return handle ? handle->swsContext : nullptr;
} 