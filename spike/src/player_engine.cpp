#include "player_engine.h"
#include "ffmpeg_handler.h"
#include "dx11_renderer.h"
#include "audio_state.h"
#include "video_state.h"
#include "decode_loop.h"
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

// 全局当前播放引擎
static PlayerEngineHandle g_currentPlayerEngine = nullptr;

// 播放引擎状态结构
struct PlayerEngineState {
    // 播放控制
    std::atomic<bool> playing{false};
    std::atomic<bool> shouldStop{false};
    std::thread decodingThread;
    std::thread renderThread;
    
    // 解码循环实例
    std::unique_ptr<DecodeLoop> decodeLoop;
    
    // 同步相关
    std::chrono::high_resolution_clock::time_point startTime;
    
    // 媒体信息缓存
    bool hasVideo = false;
    bool hasAudio = false;
    VideoInfo videoInfo = {};
    AudioInfo audioInfo = {};
};

// 前向声明 - 在结构体定义之后
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
    
    // 创建解码循环实例
    DecodeState decodeState;
    decodeState.shouldStop = &state->shouldStop;
    decodeState.hasVideo = state->hasVideo;
    decodeState.hasAudio = state->hasAudio;
    decodeState.videoStreamIndex = state->videoInfo.streamIndex;
    decodeState.audioStreamIndex = state->audioInfo.streamIndex;
    
    state->decodeLoop = std::make_unique<DecodeLoop>(
        decodeState,
        [](AVFrame* frame) { pushVideoFrame(frame); },  // 视频帧回调
        [](AVFrame* frame) { pushAudioFrame(frame); },  // 音频帧回调
        []() { return isVideoQueueFull(); }             // 队列满检查回调
    );
    
    // 设置 FFmpeg 上下文
    state->decodeLoop->setContexts(
        getFormatContext(),
        getVideoCodecContext(),
        getAudioCodecContext()
    );
    
    PlayerEngineHandle handle = static_cast<PlayerEngineHandle>(state);
    
    // 自动设置为当前播放引擎
    g_currentPlayerEngine = handle;
    
    return handle;
}
void stopPlayback(PlayerEngineHandle handle);
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
        state->decodeLoop->run();
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

void destroyCurrentPlayerEngine() {
    if (g_currentPlayerEngine) {
        destroyPlayerEngine(g_currentPlayerEngine);
        g_currentPlayerEngine = nullptr;
    }
}

// 便利接口实现
bool startCurrentPlayback() {
    return g_currentPlayerEngine ? startPlayback(g_currentPlayerEngine) : false;
}

void stopCurrentPlayback() {
    if (g_currentPlayerEngine) {
        stopPlayback(g_currentPlayerEngine);
    }
}

bool isCurrentPlaying() {
    return g_currentPlayerEngine ? isPlaying(g_currentPlayerEngine) : false;
} 