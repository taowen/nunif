#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
}

// DirectX11 渲染器状态结构
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
};

// 函数声明
bool initializeDX11Renderer(HWND hwnd, DX11RendererState* state);
void createVideoTexture(DX11RendererState* state, int width, int height);
void updateVideoTexture(DX11RendererState* state, AVFrame* frame, AVCodecContext* videoCodecContext, SwsContext* swsContext);
void renderFrame(DX11RendererState* state);
void cleanupDX11Renderer(DX11RendererState* state); 