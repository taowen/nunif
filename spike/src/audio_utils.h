#pragma once

#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <queue>
#include <mutex>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

// 音频上下文结构体
struct AudioContext {
    IMMDeviceEnumerator* deviceEnumerator = nullptr;
    IMMDevice* audioDevice = nullptr;
    IAudioClient* audioClient = nullptr;
    IAudioRenderClient* renderClient = nullptr;
    UINT32 bufferFrameCount = 0;
    WAVEFORMATEX* audioFormat = nullptr;
    SwrContext* swrContext = nullptr;
    
    // 队列和同步
    std::queue<AVFrame*>* audioFrameQueue = nullptr;
    std::mutex* audioQueueMutex = nullptr;
    std::atomic<bool>* shouldStop = nullptr;
    size_t maxQueueSize = 10;
};

// 函数声明
bool initializeAudio(AudioContext& audioCtx);
void audioLoop(AudioContext& audioCtx);
void cleanupAudio(AudioContext& audioCtx); 