#pragma once

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

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

// 新增：获取音频配置的封装函数
struct AudioConfig;  // 前向声明

// 函数声明 - 移除句柄参数，内部管理状态
bool createFFmpegHandler(const char* filename);
void destroyFFmpegHandler();

bool getVideoInfo(VideoInfo* info);
bool getAudioInfo(AudioInfo* info);

// 新增：获取音频配置的封装函数
bool getAudioConfig(AudioConfig* config);

AVFormatContext* getFormatContext();
AVCodecContext* getVideoCodecContext();
AVCodecContext* getAudioCodecContext();
SwsContext* getSwsContext(); 