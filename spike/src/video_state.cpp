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
    if (!frame) {
        std::cerr << "[VideoState] ERROR: Attempted to push null frame" << std::endl;
        return;
    }
    
    // Validate frame before adding to queue
    if (frame->format < 0 || frame->width <= 0 || frame->height <= 0) {
        std::cerr << "[VideoState] ERROR: Invalid frame - format: " << frame->format 
                  << ", dimensions: " << frame->width << "x" << frame->height << std::endl;
        return;
    }
    
    if (frame->pts == AV_NOPTS_VALUE) {
        std::cerr << "[VideoState] WARNING: Frame has no PTS, skipping" << std::endl;
        return;
    }
    
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    if (videoState.videoFrameQueue.size() < videoState.maxQueueSize) {
        videoState.videoFrameQueue.push(frame);
        std::cout << "[VideoState] Pushed valid frame to queue, queue size: " << videoState.videoFrameQueue.size() << std::endl;
    } else {
        std::cerr << "[VideoState] WARNING: Video queue full, dropping frame" << std::endl;
    }
}

AVFrame* getVideoFrameForTime(double currentSeconds, double timeBase) {
    std::lock_guard<std::mutex> lock(videoState.videoQueueMutex);
    AVFrame* frame = nullptr;
    
    while (!videoState.videoFrameQueue.empty()) {
        AVFrame* candidate = videoState.videoFrameQueue.front();
        
        if (!candidate) {
            std::cerr << "[VideoState] WARNING: Found null frame in video queue, removing it" << std::endl;
            videoState.videoFrameQueue.pop();
            continue;
        }
        
        // Additional validation
        if (candidate->format < 0 || candidate->width <= 0 || candidate->height <= 0) {
            std::cerr << "[VideoState] WARNING: Found invalid frame in queue, removing it" << std::endl;
            videoState.videoFrameQueue.pop();
            av_frame_free(&candidate);
            continue;
        }
        
        if (candidate->pts == AV_NOPTS_VALUE) {
            std::cerr << "[VideoState] WARNING: Frame has no PTS, removing it" << std::endl;
            videoState.videoFrameQueue.pop();
            av_frame_free(&candidate);
            continue;
        }
        
        double frameTime = candidate->pts * timeBase;
        
        if (frameTime <= currentSeconds + 0.04) { // 40ms tolerance
            videoState.videoFrameQueue.pop();
            frame = candidate;
            std::cout << "[VideoState] Retrieved frame with PTS: " << candidate->pts << ", time: " << frameTime << std::endl;
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