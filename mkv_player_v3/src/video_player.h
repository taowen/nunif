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
    bool initialize(ID3D11RenderTargetView* render_target_view);
    
    // 打开MKV文件
    bool open(const std::string& filepath);
    
    // 定时器回调 - 由外部定时器调用
    void onTimer();
    
    // 关闭
    void close();
    
    // 获取内部D3D11设备资源
    ID3D11Device* getD3D11Device() const;
    ID3D11DeviceContext* getD3D11Context() const;
    
private:
    // DirectX11资源 - 外部传入，不负责释放
    ID3D11RenderTargetView* render_target_view_;
    
    // DirectX11资源 - 内部创建和管理
    ID3D11DeviceContext* device_context_;  // VideoPlayer专用的设备上下文
    
    // Shader渲染资源 - 内部管理
    ID3D11VertexShader* vertex_shader_;
    ID3D11PixelShader* pixel_shader_;
    ID3D11InputLayout* input_layout_;
    ID3D11SamplerState* texture_sampler_;
    ID3D11Buffer* vertex_buffer_;
    
    // 视频解码器 - 内部持有
    std::unique_ptr<AsyncRgbVideoDecoder> video_decoder_;
    
    // 统计信息
    int render_count_;
    std::chrono::high_resolution_clock::time_point last_stats_time_;
    
    bool initializeShaders();
    void cleanupShaders();
    void renderVideoTexture(ID3D11Texture2D* texture, ID3D11ShaderResourceView* srv);
};