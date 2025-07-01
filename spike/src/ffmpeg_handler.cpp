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

// 静态全局变量 - 隐藏在实现文件中
static FFmpegHandler ffmpegState;

bool createFFmpegHandler(const char* filename) {
    // 先清理之前的状态
    destroyFFmpegHandler();
    
    // 打开文件
    if (avformat_open_input(&ffmpegState.formatContext, filename, nullptr, nullptr) < 0) {
        return false;
    }
    
    // 获取流信息
    if (avformat_find_stream_info(ffmpegState.formatContext, nullptr) < 0) {
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
    
    // 初始化视频解码器
    if (ffmpegState.videoStreamIndex >= 0) {
        AVStream* videoStream = ffmpegState.formatContext->streams[ffmpegState.videoStreamIndex];
        const AVCodec* videoCodec = avcodec_find_decoder(videoStream->codecpar->codec_id);
        if (!videoCodec) {
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.videoCodecContext = avcodec_alloc_context3(videoCodec);
        if (!ffmpegState.videoCodecContext) {
            destroyFFmpegHandler();
            return false;
        }
        
        if (avcodec_parameters_to_context(ffmpegState.videoCodecContext, videoStream->codecpar) < 0) {
            destroyFFmpegHandler();
            return false;
        }
        if (avcodec_open2(ffmpegState.videoCodecContext, videoCodec, nullptr) < 0) {
            destroyFFmpegHandler();
            return false;
        }
        
        ffmpegState.videoTimeBase = av_q2d(videoStream->time_base);
        
        // 初始化 swscale
        ffmpegState.swsContext = sws_getContext(
            ffmpegState.videoCodecContext->width, ffmpegState.videoCodecContext->height, ffmpegState.videoCodecContext->pix_fmt,
            ffmpegState.videoCodecContext->width, ffmpegState.videoCodecContext->height, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
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
    if (ffmpegState.swsContext) { 
        sws_freeContext(ffmpegState.swsContext); 
        ffmpegState.swsContext = nullptr;
    }
    if (ffmpegState.videoCodecContext) { 
        avcodec_free_context(&ffmpegState.videoCodecContext);
    }
    if (ffmpegState.audioCodecContext) { 
        avcodec_free_context(&ffmpegState.audioCodecContext);
    }
    if (ffmpegState.formatContext) { 
        avformat_close_input(&ffmpegState.formatContext);
    }
    
    // 重置索引
    ffmpegState.videoStreamIndex = -1;
    ffmpegState.audioStreamIndex = -1;
    ffmpegState.videoTimeBase = 0.0;
    ffmpegState.audioTimeBase = 0.0;
}

bool getVideoInfo(VideoInfo* info) {
    if (!info || ffmpegState.videoStreamIndex < 0) return false;
    
    info->width = ffmpegState.videoCodecContext->width;
    info->height = ffmpegState.videoCodecContext->height;
    info->timeBase = ffmpegState.videoTimeBase;
    info->streamIndex = ffmpegState.videoStreamIndex;
    
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

SwsContext* getSwsContext() {
    return ffmpegState.swsContext;
} 