#pragma once

#include <windows.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

// 音频配置结构
struct AudioConfig {
    int inputSampleRate;
    int inputChannels;
    AVSampleFormat inputFormat;
    AVChannelLayout inputChannelLayout;
};

// 函数声明 - 移除setSwrContext，添加音频配置参数
bool initializeAudioState(const AudioConfig* config);
void startAudioPlayback();
void stopAudioPlayback();
void pushAudioFrame(AVFrame* frame);
void cleanupAudioState();