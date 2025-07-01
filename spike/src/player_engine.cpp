#include "player_engine.h"
#include "ffmpeg_handler.h"
#include "dx11_renderer.h"
#include "audio_state.h"
#include "video_state.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// 播放引擎状态结构
struct PlayerEngineState {
    // 播放控制
    std::atomic<bool> playing{false};
    std::atomic<bool> shouldStop{false};
    std::thread decodingThread;
    std::thread renderThread;
    
    // 同步相关
    std::chrono::high_resolution_clock::time_point startTime;
    
    // 媒体信息缓存
    bool hasVideo = false;
    bool hasAudio = false;
    VideoInfo videoInfo = {};
    AudioInfo audioInfo = {};
};

// 内部函数声明
static void decodingLoop(PlayerEngineState* state);
static void renderLoop(PlayerEngineState* state);

PlayerEngineHandle createPlayerEngine(const char* filename, HWND hwnd) {
    PlayerEngineState* state = new PlayerEngineState();
    
    // 初始化 FFmpeg 处理器
    if (!createFFmpegHandler(filename)) {
        delete state;
        return nullptr;
    }
    
    // 初始化 DirectX11 渲染器
    if (!createDX11Renderer(hwnd)) {
        destroyFFmpegHandler();
        delete state;
        return nullptr;
    }
    
    // 初始化视频状态管理
    if (!initializeVideoState()) {
        destroyDX11Renderer();
        destroyFFmpegHandler();
        delete state;
        return nullptr;
    }
    
    // 获取媒体信息
    state->hasVideo = getVideoInfo(&state->videoInfo);
    state->hasAudio = getAudioInfo(&state->audioInfo);
    
    // 初始化音频系统 - 使用封装的配置获取函数
    if (state->hasAudio) {
        AudioConfig audioConfig;
        if (getAudioConfig(&audioConfig)) {
            if (!initializeAudioState(&audioConfig)) {
                std::cerr << "Warning: Failed to initialize audio" << std::endl;
                state->hasAudio = false;
            }
        } else {
            std::cerr << "Warning: Failed to get audio configuration" << std::endl;
            state->hasAudio = false;
        }
    }
    
    // 创建视频纹理
    if (state->hasVideo) {
        createVideoTexture(state->videoInfo.width, state->videoInfo.height);
    }
    
    return static_cast<PlayerEngineHandle>(state);
}

void destroyPlayerEngine(PlayerEngineHandle handle) {
    if (!handle) return;
    
    PlayerEngineState* state = static_cast<PlayerEngineState*>(handle);
    
    // 停止播放
    stopPlayback(handle);
    
    // 清理视频状态 - 使用新接口
    cleanupVideoState();
    
    // 清理音频
    if (state->hasAudio) {
        cleanupAudioState();
    }
    
    // 清理组件
    destroyDX11Renderer();
    destroyFFmpegHandler();
    
    delete state;
}

bool startPlayback(PlayerEngineHandle handle) {
    if (!handle) return false;
    
    PlayerEngineState* state = static_cast<PlayerEngineState*>(handle);
    
    if (state->playing) return true;
    
    state->playing = true;
    state->shouldStop = false;
    state->startTime = std::chrono::high_resolution_clock::now();
    
    // 启动解码线程
    state->decodingThread = std::thread([state]() {
        decodingLoop(state);
    });
    
    // 启动音频播放
    if (state->hasAudio) {
        startAudioPlayback();
    }
    
    // 启动渲染线程
    state->renderThread = std::thread([state]() {
        renderLoop(state);
    });
    
    return true;
}

void stopPlayback(PlayerEngineHandle handle) {
    if (!handle) return;
    
    PlayerEngineState* state = static_cast<PlayerEngineState*>(handle);
    
    state->shouldStop = true;
    state->playing = false;
    
    // 等待线程结束
    if (state->decodingThread.joinable()) {
        state->decodingThread.join();
    }
    if (state->renderThread.joinable()) {
        state->renderThread.join();
    }
    
    // 停止音频播放
    if (state->hasAudio) {
        stopAudioPlayback();
    }
}

bool isPlaying(PlayerEngineHandle handle) {
    if (!handle) return false;
    
    PlayerEngineState* state = static_cast<PlayerEngineState*>(handle);
    return state->playing;
}

bool getMediaDimensions(PlayerEngineHandle handle, int* width, int* height) {
    if (!handle || !width || !height) return false;
    
    PlayerEngineState* state = static_cast<PlayerEngineState*>(handle);
    
    if (!state->hasVideo) return false;
    
    *width = state->videoInfo.width;
    *height = state->videoInfo.height;
    return true;
}

// 内部函数实现
static void decodingLoop(PlayerEngineState* state) {
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    AVFormatContext* formatContext = getFormatContext();
    AVCodecContext* videoCodecContext = getVideoCodecContext();
    AVCodecContext* audioCodecContext = getAudioCodecContext();
    
    while (!state->shouldStop && av_read_frame(formatContext, packet) >= 0) {
        if (state->hasVideo && packet->stream_index == state->videoInfo.streamIndex) {
            if (avcodec_send_packet(videoCodecContext, packet) == 0) {
                while (avcodec_receive_frame(videoCodecContext, frame) == 0) {
                    // 使用新的视频状态接口
                    pushVideoFrame(frame);
                }
            }
        } else if (state->hasAudio && packet->stream_index == state->audioInfo.streamIndex) {
            if (avcodec_send_packet(audioCodecContext, packet) == 0) {
                while (avcodec_receive_frame(audioCodecContext, frame) == 0) {
                    pushAudioFrame(frame);
                }
            }
        }
        
        av_packet_unref(packet);
        
        // 队列大小控制 - 使用新接口
        if (isVideoQueueFull()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    av_packet_free(&packet);
    av_frame_free(&frame);
}

static void renderLoop(PlayerEngineState* state) {
    while (state->playing && !state->shouldStop) {
        // 获取当前时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - state->startTime);
        double currentSeconds = elapsed.count() / 1000000.0;
        
        // 从队列中获取视频帧 - 使用新接口
        AVFrame* frame = getVideoFrameForTime(currentSeconds, state->videoInfo.timeBase);
        
        if (frame && state->hasVideo) {
            updateVideoTexture(frame);
            av_frame_free(&frame);
        }
        
        // 渲染
        renderFrame();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
} 