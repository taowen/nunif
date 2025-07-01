#pragma once

extern "C" {
#include <libavformat/avformat.h>
}

// 视频状态管理接口
bool initializeVideoState();
void cleanupVideoState();

void pushVideoFrame(AVFrame* frame);
AVFrame* getVideoFrameForTime(double currentSeconds, double timeBase);
void clearVideoFrameQueue();

// 获取队列状态
size_t getVideoQueueSize();
bool isVideoQueueFull(); 