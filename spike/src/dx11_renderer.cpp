#include "dx11_renderer.h"
#include "shader_utils.h"
#include "vertex_buffer_utils.h"
#include <d3dcompiler.h>
#include <iostream>
#include <vector>
#include <d3d11.h>
#include <dxgi.h>

extern "C" {
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// DirectX11 渲染器状态结构 - 内部实现
struct DX11RendererState {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* deviceContext = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11RenderTargetView* renderTargetView = nullptr;
    ID3D11VertexShader* vertexShader = nullptr;
    ID3D11PixelShader* pixelShader = nullptr;
    ID3D11Buffer* vertexBuffer = nullptr;
    ID3D11InputLayout* inputLayout = nullptr;
    ID3D11Texture2D* videoTexture = nullptr;
    ID3D11ShaderResourceView* videoSRV = nullptr;
    ID3D11SamplerState* samplerState = nullptr;
    int windowWidth = 800;
    int windowHeight = 600;
    
    // 内部管理像素格式转换 - 提高内聚性
    SwsContext* internalSwsContext = nullptr;
    int videoWidth = 0;
    int videoHeight = 0;
    AVPixelFormat inputPixelFormat = AV_PIX_FMT_NONE;
};

// 静态全局变量 - 隐藏在实现文件中
static DX11RendererState dx11State;

bool createDX11Renderer(HWND hwnd) {
    // Get screen dimensions for fullscreen
    dx11State.windowWidth = GetSystemMetrics(SM_CXSCREEN);
    dx11State.windowHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // 创建设备和交换链
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = dx11State.windowWidth;
    swapChainDesc.BufferDesc.Height = dx11State.windowHeight;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Windowed = TRUE;
    
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &swapChainDesc, &dx11State.swapChain, &dx11State.device, &featureLevel, &dx11State.deviceContext
    );
    
    if (FAILED(hr)) {
        return false;
    }
    
    // 创建渲染目标视图
    ID3D11Texture2D* backBuffer;
    dx11State.swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    dx11State.device->CreateRenderTargetView(backBuffer, nullptr, &dx11State.renderTargetView);
    backBuffer->Release();
    
    // 创建着色器
    if (!createShaders(dx11State.device, &dx11State.vertexShader, &dx11State.pixelShader, &dx11State.inputLayout)) {
        return false;
    }
    
    // 创建顶点缓冲区
    if (!createVertexBuffer(dx11State.device, &dx11State.vertexBuffer)) {
        return false;
    }
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    dx11State.device->CreateSamplerState(&samplerDesc, &dx11State.samplerState);
    
    return true;
}

void destroyDX11Renderer() {
    // 清理内部SwsContext
    if (dx11State.internalSwsContext) {
        sws_freeContext(dx11State.internalSwsContext);
        dx11State.internalSwsContext = nullptr;
    }
    
    if (dx11State.samplerState) { dx11State.samplerState->Release(); dx11State.samplerState = nullptr; }
    if (dx11State.videoSRV) { dx11State.videoSRV->Release(); dx11State.videoSRV = nullptr; }
    if (dx11State.videoTexture) { dx11State.videoTexture->Release(); dx11State.videoTexture = nullptr; }
    if (dx11State.inputLayout) { dx11State.inputLayout->Release(); dx11State.inputLayout = nullptr; }
    if (dx11State.vertexBuffer) { dx11State.vertexBuffer->Release(); dx11State.vertexBuffer = nullptr; }
    if (dx11State.pixelShader) { dx11State.pixelShader->Release(); dx11State.pixelShader = nullptr; }
    if (dx11State.vertexShader) { dx11State.vertexShader->Release(); dx11State.vertexShader = nullptr; }
    if (dx11State.renderTargetView) { dx11State.renderTargetView->Release(); dx11State.renderTargetView = nullptr; }
    if (dx11State.swapChain) { dx11State.swapChain->Release(); dx11State.swapChain = nullptr; }
    if (dx11State.deviceContext) { dx11State.deviceContext->Release(); dx11State.deviceContext = nullptr; }
    if (dx11State.device) { dx11State.device->Release(); dx11State.device = nullptr; }
}

void createVideoTexture(int width, int height) {
    // 记录视频尺寸
    dx11State.videoWidth = width;
    dx11State.videoHeight = height;
    
    if (dx11State.videoTexture) {
        dx11State.videoTexture->Release();
        dx11State.videoTexture = nullptr;
    }
    if (dx11State.videoSRV) {
        dx11State.videoSRV->Release();
        dx11State.videoSRV = nullptr;
    }
    
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.Usage = D3D11_USAGE_DYNAMIC;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    dx11State.device->CreateTexture2D(&textureDesc, nullptr, &dx11State.videoTexture);
    dx11State.device->CreateShaderResourceView(dx11State.videoTexture, nullptr, &dx11State.videoSRV);
}

void updateVideoTexture(AVFrame* frame) {
    if (!frame || !dx11State.videoTexture) return;
    
    // 内部管理SwsContext - 提高内聚性，减少外部依赖
    if (!dx11State.internalSwsContext || dx11State.inputPixelFormat != frame->format) {
        if (dx11State.internalSwsContext) {
            sws_freeContext(dx11State.internalSwsContext);
        }
        
        dx11State.inputPixelFormat = static_cast<AVPixelFormat>(frame->format);
        dx11State.internalSwsContext = sws_getContext(
            frame->width, frame->height, dx11State.inputPixelFormat,
            dx11State.videoWidth, dx11State.videoHeight, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    }
    
    if (!dx11State.internalSwsContext) return;
    
    // 创建临时 RGBA 缓冲区
    std::vector<uint8_t> rgbaBuffer(dx11State.videoWidth * dx11State.videoHeight * 4);
    
    uint8_t* rgbaData[1] = { rgbaBuffer.data() };
    int rgbaLinesize[1] = { dx11State.videoWidth * 4 };
    
    // 转换为 RGBA
    sws_scale(dx11State.internalSwsContext, frame->data, frame->linesize, 
              0, frame->height, rgbaData, rgbaLinesize);
    
    // 更新纹理
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (SUCCEEDED(dx11State.deviceContext->Map(dx11State.videoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))) {
        uint8_t* dest = static_cast<uint8_t*>(mappedResource.pData);
        uint8_t* src = rgbaBuffer.data();
        
        for (int y = 0; y < dx11State.videoHeight; y++) {
            memcpy(dest + y * mappedResource.RowPitch, src + y * dx11State.videoWidth * 4, dx11State.videoWidth * 4);
        }
        
        dx11State.deviceContext->Unmap(dx11State.videoTexture, 0);
    }
}

void renderFrame() {
    if (!dx11State.device) return;
    
    // 渲染
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    dx11State.deviceContext->ClearRenderTargetView(dx11State.renderTargetView, clearColor);
    
    dx11State.deviceContext->OMSetRenderTargets(1, &dx11State.renderTargetView, nullptr);
    
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(dx11State.windowWidth);
    viewport.Height = static_cast<float>(dx11State.windowHeight);
    viewport.MaxDepth = 1.0f;
    dx11State.deviceContext->RSSetViewports(1, &viewport);
    
    dx11State.deviceContext->IASetInputLayout(dx11State.inputLayout);
    dx11State.deviceContext->VSSetShader(dx11State.vertexShader, nullptr, 0);
    dx11State.deviceContext->PSSetShader(dx11State.pixelShader, nullptr, 0);
    
    if (dx11State.videoSRV) {
        dx11State.deviceContext->PSSetShaderResources(0, 1, &dx11State.videoSRV);
        dx11State.deviceContext->PSSetSamplers(0, 1, &dx11State.samplerState);
    }
    
    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    dx11State.deviceContext->IASetVertexBuffers(0, 1, &dx11State.vertexBuffer, &stride, &offset);
    dx11State.deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    
    dx11State.deviceContext->Draw(4, 0);
    
    dx11State.swapChain->Present(1, 0);
} 