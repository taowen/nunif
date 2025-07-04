#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <Windows.h>
#include <string>

class VideoRenderer {
public:
    VideoRenderer();
    ~VideoRenderer();

    // 初始化DirectX11渲染系统
    bool initialize(HWND window_handle, int width, int height);
    
    // 渲染RGB纹理到屏幕
    bool renderFrame(ID3D11Texture2D* rgb_texture, ID3D11ShaderResourceView* rgb_srv);
    
    // 呈现到屏幕
    void present();
    
    // 资源管理
    void close();
    
    // 获取D3D11设备（用于纹理兼容性）
    ID3D11Device* getDevice() const { return d3d11_device_; }
    ID3D11DeviceContext* getContext() const { return d3d11_context_; }

private:
    // DirectX11 核心组件
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    IDXGISwapChain* swap_chain_;
    ID3D11RenderTargetView* render_target_view_;
    
    // 渲染管线组件
    ID3D11VertexShader* vertex_shader_;
    ID3D11PixelShader* pixel_shader_;
    ID3D11InputLayout* input_layout_;
    ID3D11Buffer* vertex_buffer_;
    ID3D11SamplerState* sampler_state_;
    ID3D11RasterizerState* rasterizer_state_;
    
    // 视口信息
    D3D11_VIEWPORT viewport_;
    
    // 内部方法
    bool createDeviceAndSwapChain(HWND window_handle, int width, int height);
    bool createRenderTargetView();
    bool createShaders();
    bool createGeometry();
    bool createSamplerState();
    void releaseResources();
    
    // 着色器编译辅助
    bool compileShader(const std::string& shader_source, const std::string& entry_point, 
                      const std::string& target, ID3DBlob** blob);
};