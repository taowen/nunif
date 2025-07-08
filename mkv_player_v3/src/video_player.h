#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <memory>
#include <chrono>
#include <string>

// 前向声明
class AsyncRgbVideoDecoder;

class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();
    
    // 初始化 - 外部传入D3D11资源
    bool initialize(ID3D11Device* device, 
                   ID3D11DeviceContext* context, 
                   ID3D11RenderTargetView* render_target_view,
                   IDXGISwapChain* swap_chain = nullptr);
    
    // 打开MKV文件
    bool open(const std::string& filepath);
    
    // 定时器回调 - 由外部定时器调用
    void onTimer();
    
    // 关闭
    void close();
    
private:
    // DirectX11资源 - 外部传入，不负责释放
    ID3D11Device* device_;
    ID3D11DeviceContext* context_;
    IDXGISwapChain* swap_chain_;  // 可选，测试时为nullptr
    ID3D11RenderTargetView* render_target_view_;
    
    // 视频解码器 - 内部持有
    std::unique_ptr<AsyncRgbVideoDecoder> video_decoder_;
    
    // 统计信息
    int render_count_;
    std::chrono::high_resolution_clock::time_point last_stats_time_;
    
    void renderVideoTexture(ID3D11Texture2D* texture, ID3D11ShaderResourceView* srv);
};