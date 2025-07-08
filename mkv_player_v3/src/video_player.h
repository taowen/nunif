#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <memory>
#include <chrono>

// 前向声明
class FakeVideoSignal;
class FrameRateConverter;

class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    
    // 初始化DirectX11渲染环境
    bool initialize(HWND hwnd);
    
    // 定时器回调 - 由外部定时器调用
    // 这个方法会：
    // 1. 检查是否需要获取新的视频帧
    // 2. 渲染当前帧
    // 3. Present到屏幕
    // 4. 统计帧率信息
    void onTimer();
    
private:
    // DirectX11资源
    HWND hwnd_;
    ID3D11Device* device_;
    ID3D11DeviceContext* context_;
    IDXGISwapChain* swap_chain_;
    ID3D11RenderTargetView* render_target_view_;
    
    // 视频信号和帧率转换
    std::unique_ptr<FakeVideoSignal> video_signal_;
    std::unique_ptr<FrameRateConverter> frame_converter_;
    
    // 当前帧状态
    struct Frame {
        float color[3];
        double timestamp;
    };
    Frame current_frame_;
    bool has_frame_;
    
    // 统计信息
    int render_count_;
    std::chrono::high_resolution_clock::time_point last_stats_time_;
    
    void renderFrame(const Frame& frame);
    void cleanup();
};