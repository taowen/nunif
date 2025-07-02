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

bool createDX11Renderer(HWND hwnd, ID3D11Device* externalDevice, ID3D11DeviceContext* externalContext) {
    std::cout << "[DX11Renderer] Creating DX11 renderer..." << std::endl;
    
    // Get screen dimensions for fullscreen
    dx11State.windowWidth = GetSystemMetrics(SM_CXSCREEN);
    dx11State.windowHeight = GetSystemMetrics(SM_CYSCREEN);
    
    std::cout << "[DX11Renderer] Screen dimensions: " << dx11State.windowWidth << "x" << dx11State.windowHeight << std::endl;
    
    dx11State.device = externalDevice;
    dx11State.deviceContext = externalContext;
    dx11State.device->AddRef();
    dx11State.deviceContext->AddRef();
    
    std::cout << "[DX11Renderer] Using external D3D11 device and context" << std::endl;
    
    // 创建交换链（需要从 device 获取 DXGI factory）
    IDXGIDevice* dxgiDevice = nullptr;
    IDXGIAdapter* dxgiAdapter = nullptr;
    IDXGIFactory* dxgiFactory = nullptr;
    
    dx11State.device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
    dxgiDevice->GetAdapter(&dxgiAdapter);
    dxgiAdapter->GetParent(__uuidof(IDXGIFactory), (void**)&dxgiFactory);
    
    DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
    swapChainDesc.BufferCount = 1;
    swapChainDesc.BufferDesc.Width = dx11State.windowWidth;
    swapChainDesc.BufferDesc.Height = dx11State.windowHeight;
    swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.OutputWindow = hwnd;
    swapChainDesc.SampleDesc.Count = 1;
    swapChainDesc.Windowed = TRUE;
    
    HRESULT hr = dxgiFactory->CreateSwapChain(dx11State.device, &swapChainDesc, &dx11State.swapChain);
    
    dxgiFactory->Release();
    dxgiAdapter->Release();
    dxgiDevice->Release();
    
    if (FAILED(hr)) {
        std::cerr << "[DX11Renderer] Failed to create swap chain, HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    std::cout << "[DX11Renderer] Swap chain created successfully" << std::endl;
    
    // 创建渲染目标视图
    ID3D11Texture2D* backBuffer;
    dx11State.swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
    dx11State.device->CreateRenderTargetView(backBuffer, nullptr, &dx11State.renderTargetView);
    backBuffer->Release();
    
    // 创建着色器
    if (!createShaders(dx11State.device, &dx11State.vertexShader, &dx11State.pixelShader, &dx11State.inputLayout)) {
        std::cerr << "[DX11Renderer] Failed to create shaders" << std::endl;
        return false;
    }
    
    std::cout << "[DX11Renderer] Shaders created successfully" << std::endl;
    
    // 创建顶点缓冲区
    if (!createVertexBuffer(dx11State.device, &dx11State.vertexBuffer)) {
        std::cerr << "[DX11Renderer] Failed to create vertex buffer" << std::endl;
        return false;
    }
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    dx11State.device->CreateSamplerState(&samplerDesc, &dx11State.samplerState);
    
    std::cout << "[DX11Renderer] DX11 renderer created successfully" << std::endl;
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
    std::cout << "[DX11Renderer] Creating video texture: " << width << "x" << height << std::endl;
    
    // 验证输入参数
    if (width <= 0 || height <= 0) {
        std::cerr << "[DX11Renderer] ERROR: Invalid texture dimensions: " << width << "x" << height << std::endl;
        return;
    }
    
    // 记录视频尺寸
    dx11State.videoWidth = width;
    dx11State.videoHeight = height;
    
    if (dx11State.videoTexture) {
        std::cout << "[DX11Renderer] Releasing existing video texture" << std::endl;
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
    
    HRESULT hr = dx11State.device->CreateTexture2D(&textureDesc, nullptr, &dx11State.videoTexture);
    if (FAILED(hr)) {
        std::cerr << "[DX11Renderer] ERROR: Failed to create video texture, HRESULT: 0x" << std::hex << hr << std::endl;
        return;
    }
    
    hr = dx11State.device->CreateShaderResourceView(dx11State.videoTexture, nullptr, &dx11State.videoSRV);
    if (FAILED(hr)) {
        std::cerr << "[DX11Renderer] ERROR: Failed to create shader resource view, HRESULT: 0x" << std::hex << hr << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] Video texture created successfully" << std::endl;
}

void updateVideoTextureFromHardwareFrame(AVFrame* frame) {
    if (!frame || !dx11State.videoTexture) {
        std::cerr << "[DX11Renderer] ERROR: Invalid frame or texture in updateVideoTextureFromHardwareFrame" << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] Updating video texture from hardware frame" << std::endl;
    std::cout << "[DX11Renderer] Frame format: " << frame->format << " (expected AV_PIX_FMT_D3D11=" << AV_PIX_FMT_D3D11 << ")" << std::endl;
    std::cout << "[DX11Renderer] Frame dimensions: " << frame->width << "x" << frame->height << std::endl;
    
    // 检查是否是硬件解码帧
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cout << "[DX11Renderer] Not a D3D11 frame, falling back to software processing" << std::endl;
        // 回退到软件解码处理
        updateVideoTexture(frame);
        return;
    }
    
    // 获取 D3D11 纹理
    ID3D11Texture2D* srcTexture = (ID3D11Texture2D*)frame->data[0];
    int arrayIndex = (int)(intptr_t)frame->data[1];
    
    if (!srcTexture) {
        std::cerr << "[DX11Renderer] ERROR: No D3D11 texture in frame data" << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] Source texture: " << srcTexture << ", array index: " << arrayIndex << std::endl;
    
    // 获取源纹理描述
    D3D11_TEXTURE2D_DESC srcDesc;
    srcTexture->GetDesc(&srcDesc);
    
    std::cout << "[DX11Renderer] Source texture desc - Width: " << srcDesc.Width << ", Height: " << srcDesc.Height 
              << ", Format: " << srcDesc.Format << ", MipLevels: " << srcDesc.MipLevels << std::endl;
    
    // 创建用于复制的子资源索引
    UINT srcSubresource = D3D11CalcSubresource(0, arrayIndex, srcDesc.MipLevels);
    
    std::cout << "[DX11Renderer] Copying from subresource: " << srcSubresource << std::endl;
    
    // 直接复制纹理内容
    dx11State.deviceContext->CopySubresourceRegion(
        dx11State.videoTexture, 0,
        0, 0, 0,
        srcTexture, srcSubresource,
        nullptr
    );
    
    std::cout << "[DX11Renderer] Hardware frame texture copy completed" << std::endl;
}

// 修改原有的 updateVideoTexture 函数，支持硬件解码帧
void updateVideoTexture(AVFrame* frame) {
    if (!frame || !dx11State.videoTexture) {
        std::cerr << "[DX11Renderer] ERROR: Invalid frame or texture in updateVideoTexture" << std::endl;
        return;
    }
    
    // Add comprehensive frame validation before processing
    if (frame->format < 0) {
        std::cerr << "[DX11Renderer] ERROR: Invalid frame format: " << frame->format << std::endl;
        return;
    }
    
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "[DX11Renderer] ERROR: Invalid frame dimensions: " << frame->width << "x" << frame->height << std::endl;
        return;
    }
    
    if (frame->pts == AV_NOPTS_VALUE) {
        std::cerr << "[DX11Renderer] WARNING: Frame has no PTS, skipping" << std::endl;
        return;
    }
    
    // Additional validation for frame data
    if (!frame->data[0]) {
        std::cerr << "[DX11Renderer] ERROR: Frame data[0] is null" << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] Updating video texture" << std::endl;
    std::cout << "[DX11Renderer] Frame info - format: " << frame->format << ", dimensions: " << frame->width << "x" << frame->height << std::endl;
    std::cout << "[DX11Renderer] Frame PTS: " << frame->pts << std::endl;
    
    // 如果是硬件解码帧，使用专门的处理函数
    if (frame->format == AV_PIX_FMT_D3D11) {
        std::cout << "[DX11Renderer] Hardware frame detected, using hardware path" << std::endl;
        updateVideoTextureFromHardwareFrame(frame);
        return;
    }
    
    std::cout << "[DX11Renderer] Software frame detected, using software path" << std::endl;
    
    // 验证目标纹理尺寸
    if (dx11State.videoWidth <= 0 || dx11State.videoHeight <= 0) {
        std::cerr << "[DX11Renderer] ERROR: Invalid target texture dimensions: " << dx11State.videoWidth << "x" << dx11State.videoHeight << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] Target texture dimensions: " << dx11State.videoWidth << "x" << dx11State.videoHeight << std::endl;
    
    // 内部管理SwsContext - 提高内聚性，减少外部依赖
    if (!dx11State.internalSwsContext || dx11State.inputPixelFormat != frame->format) {
        std::cout << "[DX11Renderer] Creating/recreating SwsContext" << std::endl;
        std::cout << "[DX11Renderer] Input format changed from " << dx11State.inputPixelFormat << " to " << frame->format << std::endl;
        
        if (dx11State.internalSwsContext) {
            std::cout << "[DX11Renderer] Freeing existing SwsContext" << std::endl;
            sws_freeContext(dx11State.internalSwsContext);
        }
        
        dx11State.inputPixelFormat = static_cast<AVPixelFormat>(frame->format);
        
        std::cout << "[DX11Renderer] Creating SwsContext with parameters:" << std::endl;
        std::cout << "[DX11Renderer]   Input: " << frame->width << "x" << frame->height << ", format: " << dx11State.inputPixelFormat << std::endl;
        std::cout << "[DX11Renderer]   Output: " << dx11State.videoWidth << "x" << dx11State.videoHeight << ", format: AV_PIX_FMT_RGBA" << std::endl;
        
        dx11State.internalSwsContext = sws_getContext(
            frame->width, frame->height, dx11State.inputPixelFormat,
            dx11State.videoWidth, dx11State.videoHeight, AV_PIX_FMT_RGBA,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!dx11State.internalSwsContext) {
            std::cerr << "[DX11Renderer] ERROR: Failed to create SwsContext!" << std::endl;
            std::cerr << "[DX11Renderer] Input params: " << frame->width << "x" << frame->height << ", format: " << dx11State.inputPixelFormat << std::endl;
            std::cerr << "[DX11Renderer] Output params: " << dx11State.videoWidth << "x" << dx11State.videoHeight << ", format: AV_PIX_FMT_RGBA" << std::endl;
            return;
        }
        
        std::cout << "[DX11Renderer] SwsContext created successfully" << std::endl;
    }
    
    if (!dx11State.internalSwsContext) {
        std::cerr << "[DX11Renderer] ERROR: No SwsContext available" << std::endl;
        return;
    }
    
    // 创建临时 RGBA 缓冲区
    size_t bufferSize = dx11State.videoWidth * dx11State.videoHeight * 4;
    std::cout << "[DX11Renderer] Creating RGBA buffer of size: " << bufferSize << " bytes" << std::endl;
    std::vector<uint8_t> rgbaBuffer(bufferSize);
    
    uint8_t* rgbaData[1] = { rgbaBuffer.data() };
    int rgbaLinesize[1] = { dx11State.videoWidth * 4 };
    
    std::cout << "[DX11Renderer] Starting sws_scale conversion" << std::endl;
    std::cout << "[DX11Renderer] Input data pointers: " << (void*)frame->data[0] << ", " << (void*)frame->data[1] << ", " << (void*)frame->data[2] << std::endl;
    std::cout << "[DX11Renderer] Input linesize: " << frame->linesize[0] << ", " << frame->linesize[1] << ", " << frame->linesize[2] << std::endl;
    
    // 转换为 RGBA
    int result = sws_scale(dx11State.internalSwsContext, frame->data, frame->linesize, 
                          0, frame->height, rgbaData, rgbaLinesize);
    
    if (result != dx11State.videoHeight) {
        std::cerr << "[DX11Renderer] ERROR: sws_scale failed or returned unexpected result: " << result 
                  << " (expected: " << dx11State.videoHeight << ")" << std::endl;
        return;
    }
    
    std::cout << "[DX11Renderer] sws_scale completed successfully, processed " << result << " lines" << std::endl;
    
    // 更新纹理
    std::cout << "[DX11Renderer] Mapping texture for update" << std::endl;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    HRESULT hr = dx11State.deviceContext->Map(dx11State.videoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource);
    if (SUCCEEDED(hr)) {
        uint8_t* dest = static_cast<uint8_t*>(mappedResource.pData);
        uint8_t* src = rgbaBuffer.data();
        
        std::cout << "[DX11Renderer] Copying data to texture, row pitch: " << mappedResource.RowPitch << std::endl;
        
        for (int y = 0; y < dx11State.videoHeight; y++) {
            memcpy(dest + y * mappedResource.RowPitch, src + y * dx11State.videoWidth * 4, dx11State.videoWidth * 4);
        }
        
        dx11State.deviceContext->Unmap(dx11State.videoTexture, 0);
        std::cout << "[DX11Renderer] Texture update completed successfully" << std::endl;
    } else {
        std::cerr << "[DX11Renderer] ERROR: Failed to map texture for update, HRESULT: 0x" << std::hex << hr << std::endl;
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