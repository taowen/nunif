#include "video_state.h"
#include <queue>
#include <mutex>
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
}

// 视频状态结构体 - 内部实现
struct VideoState {
    std::queue<AVFrame*> videoFrameQueue;
    std::mutex videoQueueMutex;
    static const size_t maxQueueSize = 10;
};

// 静态全局变量 - 隐藏在实现文件中
static VideoState videoState;

bool initializeVideoState() {
    // 清理可能存在的旧状态
    cleanupVideoState();
    return true;
}

void cleanupVideoState() {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    while (!videoState.videoFrameQueue.empty()) {
        videoState.videoFrameQueue.pop();
    }
}

void pushVideoFrame(AVFrame* frame) {
    if (!frame) return;
    
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    if (videoState.videoFrameQueue.size() < videoState.maxQueueSize) {
        videoState.videoFrameQueue.push(frame);
    }
}

AVFrame* getVideoFrameForTime(double currentSeconds, double timeBase) {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    AVFrame* frame = nullptr;
    
    while (!videoState.videoFrameQueue.empty()) {
        AVFrame* candidate = videoState.videoFrameQueue.front();
        
        if (!candidate) {
            std::cerr << "Warning: Found null frame in video queue, removing it" << std::endl;
            videoState.videoFrameQueue.pop();
            continue;
        }
        
        double frameTime = candidate->pts * timeBase;
        
        if (frameTime <= currentSeconds + 0.04) { // 40ms 容差
            videoState.videoFrameQueue.pop();
            frame = candidate;
        } else {
            break;
        }
    }
    
    return frame;
}

void clearVideoFrameQueue() {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    while (!videoState.videoFrameQueue.empty()) {
        videoState.videoFrameQueue.pop();
    }
}

size_t getVideoQueueSize() {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    return videoState.videoFrameQueue.size();
}

bool isVideoQueueFull() {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    return videoState.videoFrameQueue.size() >= videoState.maxQueueSize;
} 