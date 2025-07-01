#pragma once

#include <windows.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

// 函数声明 - 不再需要传递 AudioState 参数
bool initializeAudioState();
void startAudioPlayback();
void stopAudioPlayback();
void pushAudioFrame(AVFrame* frame);
void cleanupAudioState();
void setSwrContext(SwrContext* swrContext);