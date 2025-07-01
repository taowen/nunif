#include "audio_state.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <queue>
#include <mutex>
#include <atomic>
#include <thread>

extern "C" {
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

// 音频状态结构体 - 现在是内部实现
struct AudioState {
    // WASAPI 相关
    IMMDeviceEnumerator* deviceEnumerator = nullptr;
    IMMDevice* audioDevice = nullptr;
    IAudioClient* audioClient = nullptr;
    IAudioRenderClient* renderClient = nullptr;
    UINT32 bufferFrameCount = 0;
    WAVEFORMATEX* audioFormat = nullptr;
    SwrContext* swrContext = nullptr;
    
    // 内部队列和同步
    std::queue<AVFrame*> audioFrameQueue;
    std::mutex audioQueueMutex;
    std::atomic<bool> shouldStop{false};
    std::atomic<bool> isPlaying{false};
    std::thread audioThread;
    
    // 配置
    size_t maxQueueSize = 10;
};

// 静态全局变量 - 隐藏在实现文件中
static AudioState audioState;

static void audioLoop() {
    BYTE* audioBuffer;
    UINT32 numFramesPadding;
    
    while (audioState.isPlaying && !audioState.shouldStop) {
        audioState.audioClient->GetCurrentPadding(&numFramesPadding);
        UINT32 numFramesAvailable = audioState.bufferFrameCount - numFramesPadding;
        
        if (numFramesAvailable > 0) {
            std::lock_guard<std::mutex> lock(audioState.audioQueueMutex);
            if (!audioState.audioFrameQueue.empty()) {
                AVFrame* frame = audioState.audioFrameQueue.front();
                audioState.audioFrameQueue.pop();
                
                if (SUCCEEDED(audioState.renderClient->GetBuffer(numFramesAvailable, &audioBuffer))) {
                    // 重采样音频数据
                    int outputSamples = swr_convert(audioState.swrContext,
                        &audioBuffer, numFramesAvailable,
                        (const uint8_t**)frame->data, frame->nb_samples);
                    
                    audioState.renderClient->ReleaseBuffer(outputSamples, 0);
                }
                
                av_frame_free(&frame);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

bool initializeAudioState(const AudioConfig* config) {
    CoInitialize(nullptr);
    
    // 创建设备枚举器
    HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&audioState.deviceEnumerator);
    if (FAILED(hr)) return false;
    
    // 获取默认音频设备
    hr = audioState.deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &audioState.audioDevice);
    if (FAILED(hr)) return false;
    
    // 激活音频客户端
    hr = audioState.audioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioState.audioClient);
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
    hr = audioState.audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, &format, nullptr);
    if (FAILED(hr)) return false;
    
    // 获取缓冲区大小
    audioState.audioClient->GetBufferSize(&audioState.bufferFrameCount);
    
    // 获取渲染客户端
    hr = audioState.audioClient->GetService(__uuidof(IAudioRenderClient), (void**)&audioState.renderClient);
    if (FAILED(hr)) return false;
    
    // 内部创建和管理SwrContext - 提高内聚性
    if (config) {
        audioState.swrContext = swr_alloc();
        if (!audioState.swrContext) {
            return false;
        }
        
        // 创建输出通道布局
        AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_STEREO;
        
        // 设置重采样参数
        av_opt_set_chlayout(audioState.swrContext, "out_chlayout", &out_ch_layout, 0);
        av_opt_set_int(audioState.swrContext, "out_sample_rate", 48000, 0);
        av_opt_set_sample_fmt(audioState.swrContext, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
        
        av_opt_set_chlayout(audioState.swrContext, "in_chlayout", &config->inputChannelLayout, 0);
        av_opt_set_int(audioState.swrContext, "in_sample_rate", config->inputSampleRate, 0);
        av_opt_set_sample_fmt(audioState.swrContext, "in_sample_fmt", config->inputFormat, 0);
        
        if (swr_init(audioState.swrContext) < 0) {
            swr_free(&audioState.swrContext);
            return false;
        }
    }
    
    return true;
}

void startAudioPlayback() {
    if (audioState.isPlaying) return;
    
    audioState.shouldStop = false;
    audioState.isPlaying = true;
    
    // 启动音频客户端
    if (audioState.audioClient) {
        audioState.audioClient->Start();
    }
    
    // 启动音频线程
    audioState.audioThread = std::thread(audioLoop);
}

void stopAudioPlayback() {
    audioState.shouldStop = true;
    audioState.isPlaying = false;
    
    if (audioState.audioThread.joinable()) {
        audioState.audioThread.join();
    }
    
    if (audioState.audioClient) {
        audioState.audioClient->Stop();
    }
}

void pushAudioFrame(AVFrame* frame) {
    if (!frame) return;
    
    std::lock_guard<std::mutex> lock(audioState.audioQueueMutex);
    if (audioState.audioFrameQueue.size() < audioState.maxQueueSize) {
        AVFrame* clonedFrame = av_frame_clone(frame);
        audioState.audioFrameQueue.push(clonedFrame);
    }
}

void cleanupAudioState() {
    // 停止播放
    stopAudioPlayback();
    
    // 清理音频队列
    {
        std::lock_guard<std::mutex> lock(audioState.audioQueueMutex);
        while (!audioState.audioFrameQueue.empty()) {
            AVFrame* frame = audioState.audioFrameQueue.front();
            audioState.audioFrameQueue.pop();
            av_frame_free(&frame);
        }
    }
    
    // 内部清理SwrContext - 提高内聚性
    if (audioState.swrContext) {
        swr_free(&audioState.swrContext);
        audioState.swrContext = nullptr;
    }
    
    // 清理音频资源
    if (audioState.renderClient) audioState.renderClient->Release();
    if (audioState.audioClient) audioState.audioClient->Release();
    if (audioState.audioDevice) audioState.audioDevice->Release();
    if (audioState.deviceEnumerator) audioState.deviceEnumerator->Release();
    
    CoUninitialize();
} 