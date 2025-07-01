#include "audio_utils.h"
#include <iostream>
#include <thread>
#include <chrono>

extern "C" {
#include <libavutil/opt.h>
}

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

bool initializeAudio(AudioContext& audioCtx) {
    CoInitialize(nullptr);
    
    // 创建设备枚举器
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&audioCtx.deviceEnumerator);
    if (FAILED(hr)) return false;
    
    // 获取默认音频设备
    hr = audioCtx.deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &audioCtx.audioDevice);
    if (FAILED(hr)) return false;
    
    // 激活音频客户端
    hr = audioCtx.audioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioCtx.audioClient);
    if (FAILED(hr)) return false;
    
    // 设置音频格式
    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = 48000;
    format.wBitsPerSample = 16;
    format.nBlockAlign = (format.nChannels * format.wBitsPerSample) / 8;
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
    
    // 初始化音频客户端
    hr = audioCtx.audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, &format, nullptr);
    if (FAILED(hr)) return false;
    
    // 获取缓冲区大小
    audioCtx.audioClient->GetBufferSize(&audioCtx.bufferFrameCount);
    
    // 获取渲染客户端
    hr = audioCtx.audioClient->GetService(__uuidof(IAudioRenderClient), (void**)&audioCtx.renderClient);
    if (FAILED(hr)) return false;
    
    return true;
}

void audioLoop(AudioContext& audioCtx) {
    BYTE* audioBuffer;
    UINT32 numFramesPadding;
    
    while (!*(audioCtx.shouldStop)) {
        audioCtx.audioClient->GetCurrentPadding(&numFramesPadding);
        UINT32 numFramesAvailable = audioCtx.bufferFrameCount - numFramesPadding;
        
        if (numFramesAvailable > 0) {
            std::lock_guard<std::mutex> lock(*(audioCtx.audioQueueMutex));
            if (!audioCtx.audioFrameQueue->empty()) {
                AVFrame* frame = audioCtx.audioFrameQueue->front();
                audioCtx.audioFrameQueue->pop();
                
                if (SUCCEEDED(audioCtx.renderClient->GetBuffer(numFramesAvailable, &audioBuffer))) {
                    // 重采样音频数据
                    int outputSamples = swr_convert(audioCtx.swrContext,
                        &audioBuffer, numFramesAvailable,
                        (const uint8_t**)frame->data, frame->nb_samples);
                    
                    audioCtx.renderClient->ReleaseBuffer(outputSamples, 0);
                }
                
                av_frame_free(&frame);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void cleanupAudio(AudioContext& audioCtx) {
    // 清理音频队列
    if (audioCtx.audioFrameQueue && audioCtx.audioQueueMutex) {
        std::lock_guard<std::mutex> lock(*(audioCtx.audioQueueMutex));
        while (!audioCtx.audioFrameQueue->empty()) {
            AVFrame* frame = audioCtx.audioFrameQueue->front();
            audioCtx.audioFrameQueue->pop();
            av_frame_free(&frame);
        }
    }
    
    // 清理音频资源
    if (audioCtx.renderClient) audioCtx.renderClient->Release();
    if (audioCtx.audioClient) audioCtx.audioClient->Release();
    if (audioCtx.audioDevice) audioCtx.audioDevice->Release();
    if (audioCtx.deviceEnumerator) audioCtx.deviceEnumerator->Release();
    
    CoUninitialize();
} 