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

// DirectX11 渲染器状态结构 - 现在是内部实现
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

DX11RendererHandle createDX11Renderer(HWND hwnd) {
    DX11RendererState* state = new DX11RendererState();
    
    // Get screen dimensions for fullscreen
    state->windowWidth = GetSystemMetrics(SM_CXSCREEN);
    state->windowHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // 创建设备和交换链
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = state->windowWidth;
    swapChainDesc.BufferDesc.Height = state->windowHeight;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Windowed = TRUE;
    
    D3D_FEATURE_LEVEL featureLevel;
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &swapChainDesc, &state->swapChain, &state->device, &featureLevel, &state->deviceContext
    );
    
    if (FAILED(hr)) {
        delete state;
        return nullptr;
    }
    
    // 创建渲染目标视图
    ID3D11Texture2D* backBuffer;
    state->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    state->device->CreateRenderTargetView(backBuffer, nullptr, &state->renderTargetView);
    backBuffer->Release();
    
    // 创建着色器
    if (!createShaders(state->device, &state->vertexShader, &state->pixelShader, &state->inputLayout)) {
        delete state;
        return nullptr;
    }
    
    // 创建顶点缓冲区
    if (!createVertexBuffer(state->device, &state->vertexBuffer)) {
        delete state;
        return nullptr;
    }
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    state->device->CreateSamplerState(&samplerDesc, &state->samplerState);
    
    return static_cast<DX11RendererHandle>(state);
}

void destroyDX11Renderer(DX11RendererHandle handle) {
    if (!handle) return;
    
    DX11RendererState* state = static_cast<DX11RendererState*>(handle);
    
    // 清理内部SwsContext
    if (state->internalSwsContext) {
        sws_freeContext(state->internalSwsContext);
    }
    
    if (state->samplerState) state->samplerState->Release();
    if (state->videoSRV) state->videoSRV->Release();
    if (state->videoTexture) state->videoTexture->Release();
    if (state->inputLayout) state->inputLayout->Release();
    if (state->vertexBuffer) state->vertexBuffer->Release();
    if (state->pixelShader) state->pixelShader->Release();
    if (state->vertexShader) state->vertexShader->Release();
    if (state->renderTargetView) state->renderTargetView->Release();
    if (state->swapChain) state->swapChain->Release();
    if (state->deviceContext) state->deviceContext->Release();
    if (state->device) state->device->Release();
    
    delete state;
}

void createVideoTexture(DX11RendererHandle handle, int width, int height) {
    if (!handle) return;
    
    DX11RendererState* state = static_cast<DX11RendererState*>(handle);
    
    // 记录视频尺寸
    state->videoWidth = width;
    state->videoHeight = height;
    
    if (state->videoTexture) {
        state->videoTexture->Release();
        state->videoTexture = nullptr;
    }
    if (state->videoSRV) {
        state->videoSRV->Release();
        state->videoSRV = nullptr;
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
    
    state->device->CreateTexture2D(&textureDesc, nullptr, &state->videoTexture);
    state->device->CreateShaderResourceView(state->videoTexture, nullptr, &state->videoSRV);
}

void updateVideoTexture(DX11RendererHandle handle, AVFrame* frame) {
    if (!handle || !frame) return;
    
    DX11RendererState* state = static_cast<DX11RendererState*>(handle);
    
    if (!state->videoTexture) return;
    
    // 内部管理SwsContext - 提高内聚性，减少外部依赖
    if (!state->internalSwsContext || state->inputPixelFormat != frame->format) {
        if (state->internalSwsContext) {
            sws_freeContext(state->internalSwsContext);
        }
        
        state->inputPixelFormat = static_cast<AVPixelFormat>(frame->format);
        state->internalSwsContext = sws_getContext(
            frame->width, frame->height, state->inputPixelFormat,
            state->videoWidth, state->videoHeight, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
    }
    
    if (!state->internalSwsContext) return;
    
    // 创建临时 RGBA 缓冲区
    std::vector<uint8_t> rgbaBuffer(state->videoWidth * state->videoHeight * 4);
    
    uint8_t* rgbaData[1] = { rgbaBuffer.data() };
    int rgbaLinesize[1] = { state->videoWidth * 4 };
    
    // 转换为 RGBA
    sws_scale(state->internalSwsContext, frame->data, frame->linesize, 
              0, frame->height, rgbaData, rgbaLinesize);
    
    // 更新纹理
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (SUCCEEDED(state->deviceContext->Map(state->videoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))) {
        uint8_t* dest = static_cast<uint8_t*>(mappedResource.pData);
        uint8_t* src = rgbaBuffer.data();
        
        for (int y = 0; y < state->videoHeight; y++) {
            memcpy(dest + y * mappedResource.RowPitch, src + y * state->videoWidth * 4, state->videoWidth * 4);
        }
        
        state->deviceContext->Unmap(state->videoTexture, 0);
    }
}

void renderFrame(DX11RendererHandle handle) {
    if (!handle) return;
    
    DX11RendererState* state = static_cast<DX11RendererState*>(handle);
    
    // 渲染
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    state->deviceContext->ClearRenderTargetView(state->renderTargetView, clearColor);
    
    state->deviceContext->OMSetRenderTargets(1, &state->renderTargetView, nullptr);
    
    D3D11_VIEWPORT viewport = {};
    viewport.Width = static_cast<float>(state->windowWidth);
    viewport.Height = static_cast<float>(state->windowHeight);
    viewport.MaxDepth = 1.0f;
    state->deviceContext->RSSetViewports(1, &viewport);
    
    state->deviceContext->IASetInputLayout(state->inputLayout);
    state->deviceContext->VSSetShader(state->vertexShader, nullptr, 0);
    state->deviceContext->PSSetShader(state->pixelShader, nullptr, 0);
    
    if (state->videoSRV) {
        state->deviceContext->PSSetShaderResources(0, 1, &state->videoSRV);
        state->deviceContext->PSSetSamplers(0, 1, &state->samplerState);
    }
    
    UINT stride = sizeof(Vertex);
    UINT offset = 0;
    state->deviceContext->IASetVertexBuffers(0, 1, &state->vertexBuffer, &stride, &offset);
    state->deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    
    state->deviceContext->Draw(4, 0);
    
    state->swapChain->Present(1, 0);
} 