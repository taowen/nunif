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
    
    // 初始化 - 外部传入D3D11资源
    bool initialize(ID3D11Device* device, 
                   ID3D11DeviceContext* context, 
                   ID3D11RenderTargetView* render_target_view,
                   IDXGISwapChain* swap_chain = nullptr);
    
    // 定时器回调 - 由外部定时器调用
    void onTimer();
    
private:
    // DirectX11资源 - 外部传入，不负责释放
    ID3D11Device* device_;
    ID3D11DeviceContext* context_;
    IDXGISwapChain* swap_chain_;  // 可选，测试时为nullptr
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
};