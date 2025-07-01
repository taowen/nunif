#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <avrt.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <iostream>
#include "shader_utils.h"
#include "vertex_buffer_utils.h"
#include "audio_state.h"
#include "dx11_renderer.h"
#include "ffmpeg_handler.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

// 全局变量
// FFmpeg 相关 - 现在使用句柄
FFmpegHandlerHandle ffmpegHandler = nullptr;

// DirectX11 相关 - 使用不透明句柄
DX11RendererHandle dx11Renderer = nullptr;

// 窗口相关
HWND hwnd = nullptr;

// 播放控制
std::atomic<bool> playing(false);
std::atomic<bool> shouldStop(false);
std::thread decodingThread;
std::thread audioThread;

// 同步相关
std::chrono::high_resolution_clock::time_point startTime;

// 帧队列
std::queue<AVFrame*> videoFrameQueue;
std::mutex videoQueueMutex;
const size_t maxQueueSize = 10;

// 全局函数声明
bool initializeDX11(HWND hwnd);
bool initializeAudioState();
void play();
void stop();
void decodingLoop();
void renderLoop();
void cleanup();

bool initializeDX11(HWND hwnd) {
    dx11Renderer = createDX11Renderer(hwnd);
    return dx11Renderer != nullptr;
}

void play() {
    if (playing) return;
    
    playing = true;
    shouldStop = false;
    startTime = std::chrono::high_resolution_clock::now();
    
    // 启动解码线程
    decodingThread = std::thread(&decodingLoop);
    
    // 检查是否有音频流
    AudioInfo audioInfo;
    if (getAudioInfo(ffmpegHandler, &audioInfo)) {
        startAudioPlayback();
    }
    
    // 启动渲染循环
    renderLoop();
}

void stop() {
    shouldStop = true;
    playing = false;
    
    if (decodingThread.joinable()) {
        decodingThread.join();
    }
    
    // 停止音频播放
    stopAudioPlayback();
}

void decodingLoop() {
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    AVFormatContext* formatContext = getFormatContext(ffmpegHandler);
    AVCodecContext* videoCodecContext = getVideoCodecContext(ffmpegHandler);
    AVCodecContext* audioCodecContext = getAudioCodecContext(ffmpegHandler);
    
    VideoInfo videoInfo;
    AudioInfo audioInfo;
    bool hasVideo = getVideoInfo(ffmpegHandler, &videoInfo);
    bool hasAudio = getAudioInfo(ffmpegHandler, &audioInfo);
    
    while (!shouldStop && av_read_frame(formatContext, packet) >= 0) {
        if (hasVideo && packet->stream_index == videoInfo.streamIndex) {
            if (avcodec_send_packet(videoCodecContext, packet) == 0) {
                while (avcodec_receive_frame(videoCodecContext, frame) == 0) {
                    std::lock_guard<std::mutex> lock(videoQueueMutex);
                    if (videoFrameQueue.size() < maxQueueSize) {
                        AVFrame* clonedFrame = av_frame_clone(frame);
                        videoFrameQueue.push(clonedFrame);
                    }
                }
            }
        } else if (hasAudio && packet->stream_index == audioInfo.streamIndex) {
            if (avcodec_send_packet(audioCodecContext, packet) == 0) {
                while (avcodec_receive_frame(audioCodecContext, frame) == 0) {
                    pushAudioFrame(frame);
                }
            }
        }
        
        av_packet_unref(packet);
        
        // 简单的队列大小控制
        if (videoFrameQueue.size() >= maxQueueSize) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    av_packet_free(&packet);
    av_frame_free(&frame);
}

void renderLoop() {
    VideoInfo videoInfo;
    bool hasVideo = getVideoInfo(ffmpegHandler, &videoInfo);
    
    while (playing && !shouldStop) {
        // 获取当前时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime);
        double currentSeconds = elapsed.count() / 1000000.0;
        
        // 从队列中获取视频帧
        AVFrame* frame = nullptr;
        {
            std::lock_guard<std::mutex> lock(videoQueueMutex);
            while (!videoFrameQueue.empty()) {
                AVFrame* candidate = videoFrameQueue.front();
                double frameTime = candidate->pts * videoInfo.timeBase;
                
                if (frameTime <= currentSeconds + 0.04) { // 40ms 容差
                    videoFrameQueue.pop();
                    if (frame) av_frame_free(&frame);
                    frame = candidate;
                } else {
                    break;
                }
            }
        }
        
        if (frame) {
            updateVideoTexture(dx11Renderer, frame, getVideoCodecContext(ffmpegHandler), getSwsContext(ffmpegHandler));
            av_frame_free(&frame);
        }
        
        // 渲染
        renderFrame(dx11Renderer);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
}

void cleanup() {
    stop();
    
    // 清理队列
    {
        std::lock_guard<std::mutex> lock(videoQueueMutex);
        while (!videoFrameQueue.empty()) {
            AVFrame* frame = videoFrameQueue.front();
            videoFrameQueue.pop();
            av_frame_free(&frame);
        }
    }
    
    // 清理音频
    cleanupAudioState();
    
    // 清理 FFmpeg
    if (ffmpegHandler) {
        destroyFFmpegHandler(ffmpegHandler);
        ffmpegHandler = nullptr;
    }
    
    // 清理 DirectX11
    if (dx11Renderer) {
        destroyDX11Renderer(dx11Renderer);
        dx11Renderer = nullptr;
    }
}

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd_param, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        break;
        
    case WM_DESTROY:
        cleanup();
        PostQuitMessage(0);
        break;
        
    case WM_KEYDOWN:
        if (wParam == VK_SPACE) {
            // 空格键暂停/播放（简单实现）
            if (playing) {
                stop();
            }
        }
        break;
        
    default:
        return DefWindowProc(hwnd_param, uMsg, wParam, lParam);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mkv_file>" << std::endl;
        return -1;
    }
    
    // Get screen dimensions
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // 注册窗口类 - 使用 ANSI 字符串
    WNDCLASSA wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "MKVPlayer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    
    RegisterClassA(&wc);
    
    // 创建全屏窗口 - 使用 ANSI 版本
    hwnd = CreateWindowExA(
        WS_EX_TOPMOST, "MKVPlayer", "MKV Player",
        WS_POPUP | WS_VISIBLE,
        0, 0, screenWidth, screenHeight,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );
    
    if (!hwnd) {
        std::cerr << "Failed to create window" << std::endl;
        return -1;
    }
    
    ShowWindow(hwnd, SW_MAXIMIZE);
    UpdateWindow(hwnd);
    
    // 初始化FFmpeg处理器
    ffmpegHandler = createFFmpegHandler(argv[1]);
    
    // 初始化播放器
    if (ffmpegHandler && initializeDX11(hwnd) && initializeAudioState()) {
        // 创建视频纹理
        VideoInfo videoInfo;
        if (getVideoInfo(ffmpegHandler, &videoInfo)) {
            createVideoTexture(dx11Renderer, videoInfo.width, videoInfo.height);
        }
        
        // 开始播放
        std::thread playThread([]() {
            play();
        });
        
        // 消息循环
        MSG msg = {};
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        
        playThread.join();
    } else {
        std::cerr << "Failed to initialize player" << std::endl;
        MessageBoxA(hwnd, "Failed to initialize player", "Error", MB_OK | MB_ICONERROR);
    }
    
    return 0;
}
