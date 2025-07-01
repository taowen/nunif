#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

// 不透明句柄类型
typedef struct FFmpegHandler* FFmpegHandlerHandle;

// 视频信息结构
struct VideoInfo {
    int width;
    int height;
    double timeBase;
    int streamIndex;
};

// 音频信息结构
struct AudioInfo {
    int sampleRate;
    int channels;
    double timeBase;
    int streamIndex;
};

// 函数声明
FFmpegHandlerHandle createFFmpegHandler(const char* filename);
void destroyFFmpegHandler(FFmpegHandlerHandle handle);

bool getVideoInfo(FFmpegHandlerHandle handle, VideoInfo* info);
bool getAudioInfo(FFmpegHandlerHandle handle, AudioInfo* info);

AVFormatContext* getFormatContext(FFmpegHandlerHandle handle);
AVCodecContext* getVideoCodecContext(FFmpegHandlerHandle handle);
AVCodecContext* getAudioCodecContext(FFmpegHandlerHandle handle);
SwsContext* getSwsContext(FFmpegHandlerHandle handle);
SwrContext* getSwrContext(FFmpegHandlerHandle handle);

void setSwrContextForHandler(FFmpegHandlerHandle handle, SwrContext* swrContext); 