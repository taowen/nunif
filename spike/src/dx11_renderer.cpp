#include "dx11_renderer.h"
#include "shader_utils.h"
#include "vertex_buffer_utils.h"
#include <d3dcompiler.h>
#include <iostream>
#include <vector>

extern "C" {
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

bool initializeDX11Renderer(HWND hwnd, DX11RendererState* state) {
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
    
    if (FAILED(hr)) return false;
    
    // 创建渲染目标视图
    ID3D11Texture2D* backBuffer;
    state->swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    state->device->CreateRenderTargetView(backBuffer, nullptr, &state->renderTargetView);
    backBuffer->Release();
    
    // 创建着色器
    if (!createShaders(state->device, &state->vertexShader, &state->pixelShader, &state->inputLayout)) return false;
    
    // 创建顶点缓冲区
    if (!createVertexBuffer(state->device, &state->vertexBuffer)) return false;
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    state->device->CreateSamplerState(&samplerDesc, &state->samplerState);
    
    return true;
}

void createVideoTexture(DX11RendererState* state, int width, int height) {
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

void updateVideoTexture(DX11RendererState* state, AVFrame* frame, AVCodecContext* videoCodecContext, SwsContext* swsContext) {
    if (!state->videoTexture || !swsContext) return;
    
    // 创建临时 RGBA 缓冲区
    int width = videoCodecContext->width;
    int height = videoCodecContext->height;
    std::vector<uint8_t> rgbaBuffer(width * height * 4);
    
    uint8_t* rgbaData[1] = { rgbaBuffer.data() };
    int rgbaLinesize[1] = { width * 4 };
    
    // 转换为 RGBA
    sws_scale(swsContext, frame->data, frame->linesize, 0, height, rgbaData, rgbaLinesize);
    
    // 更新纹理
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (SUCCEEDED(state->deviceContext->Map(state->videoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))) {
        uint8_t* dest = static_cast<uint8_t*>(mappedResource.pData);
        uint8_t* src = rgbaBuffer.data();
        
        for (int y = 0; y < height; y++) {
            memcpy(dest + y * mappedResource.RowPitch, src + y * width * 4, width * 4);
        }
        
        state->deviceContext->Unmap(state->videoTexture, 0);
    }
}

void renderFrame(DX11RendererState* state) {
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

void cleanupDX11Renderer(DX11RendererState* state) {
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
    
    // 重置所有指针
    *state = {};
} 