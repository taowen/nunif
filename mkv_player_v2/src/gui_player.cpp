#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <io.h>
#include <fcntl.h>
#include "async_rgb_frame_decoder.h"

using Microsoft::WRL::ComPtr;

class GUIPlayer {
public:
    GUIPlayer() : m_hwnd(nullptr), m_startTime(std::chrono::steady_clock::now()), m_autoExitTimeMs(0), m_videoPath("") {}
    
    void SetAutoExitTime(int timeMs) {
        m_autoExitTimeMs = timeMs;
    }
    
    void SetVideoPath(const std::string& path) {
        m_videoPath = path;
    }
    
    bool Initialize(HINSTANCE hInstance, int nCmdShow) {
        std::cout << "GUI Player starting..." << std::endl;
        
        if (!CreateAppWindow(hInstance, nCmdShow)) {
            std::cout << "Failed to create app window" << std::endl;
            return false;
        }
        
        std::cout << "App window created successfully" << std::endl;
        
        // 先初始化解码器（如果有视频文件）
        if (!m_videoPath.empty()) {
            if (!m_decoder.open(m_videoPath)) {
                std::cout << "Failed to open video file: " << m_videoPath << std::endl;
                return false;
            }
            std::cout << "Video decoder initialized successfully" << std::endl;
        }
        
        if (!InitializeD3D()) {
            std::cout << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        std::cout << "DirectX11 initialized successfully" << std::endl;
        
        return true;
    }
    
    void Run() {
        std::cout << "Starting render loop..." << std::endl;
        
        MSG msg = {};
        int frameCount = 0;
        while (WM_QUIT != msg.message) {
            if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            } else {
                if (m_autoExitTimeMs > 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime);
                    if (elapsed.count() >= m_autoExitTimeMs) {
                        std::cout << "Auto-exit triggered after " << elapsed.count() << "ms" << std::endl;
                        PostQuitMessage(0);
                        break;
                    }
                }
                Render();
                frameCount++;
                if (frameCount % 60 == 0) {
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_startTime);
                    std::cout << "Frame " << frameCount << " at " << elapsed.count() << "ms" << std::endl;
                }
            }
        }
        
        std::cout << "Render loop ended, total frames: " << frameCount << std::endl;
    }
    
    void Cleanup() {
        std::cout << "Cleaning up resources..." << std::endl;
        if (m_deviceContext) {
            m_deviceContext->ClearState();
        }
        std::cout << "GUI Player shutdown complete" << std::endl;
    }

private:
    HWND m_hwnd;
    ComPtr<ID3D11Device> m_device;
    ComPtr<ID3D11DeviceContext> m_deviceContext;
    ComPtr<IDXGISwapChain> m_swapChain;
    ComPtr<ID3D11RenderTargetView> m_renderTargetView;
    
    // 渲染资源
    ComPtr<ID3D11VertexShader> m_vertexShader;
    ComPtr<ID3D11PixelShader> m_pixelShader;
    ComPtr<ID3D11InputLayout> m_inputLayout;
    ComPtr<ID3D11Buffer> m_vertexBuffer;
    ComPtr<ID3D11SamplerState> m_samplerState;
    
    std::chrono::steady_clock::time_point m_startTime;
    int m_autoExitTimeMs;
    
    std::string m_videoPath;
    AsyncRGBFrameDecoder m_decoder;
    
    bool CreateAppWindow(HINSTANCE hInstance, int nCmdShow) {
        WNDCLASSEXW wcex = {};
        wcex.cbSize = sizeof(WNDCLASSEXW);
        wcex.style = CS_HREDRAW | CS_VREDRAW;
        wcex.lpfnWndProc = WindowProc;
        wcex.hInstance = hInstance;
        wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        wcex.lpszClassName = L"GUIPlayerClass";
        
        if (!RegisterClassExW(&wcex)) {
            return false;
        }
        
        m_hwnd = CreateWindowW(
            L"GUIPlayerClass",
            L"GUI Player - DirectX11 Color Test",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            800, 600,
            nullptr, nullptr, hInstance, this
        );
        
        if (!m_hwnd) {
            return false;
        }
        
        ShowWindow(m_hwnd, nCmdShow);
        UpdateWindow(m_hwnd);
        
        return true;
    }
    
    bool InitializeD3D() {
        // 必须从解码器获取D3D11设备以确保兼容性
        if (m_videoPath.empty()) {
            std::cout << "No video file provided, cannot initialize D3D11" << std::endl;
            return false;
        }
        
        // 从解码器获取D3D11设备
        m_device = m_decoder.getD3D11Device();
        if (!m_device) {
            std::cout << "Failed to get D3D11 device from decoder" << std::endl;
            return false;
        }
        
        // 从设备获取device context
        m_device->GetImmediateContext(&m_deviceContext);
        std::cout << "Using D3D11 device from decoder" << std::endl;
        
        // 创建DXGI Factory和SwapChain
        ComPtr<IDXGIDevice> dxgiDevice;
        HRESULT hr = m_device.As(&dxgiDevice);
        if (FAILED(hr)) {
            return false;
        }
        
        ComPtr<IDXGIAdapter> dxgiAdapter;
        hr = dxgiDevice->GetAdapter(&dxgiAdapter);
        if (FAILED(hr)) {
            return false;
        }
        
        ComPtr<IDXGIFactory> dxgiFactory;
        hr = dxgiAdapter->GetParent(__uuidof(IDXGIFactory), (void**)&dxgiFactory);
        if (FAILED(hr)) {
            return false;
        }
        
        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        swapChainDesc.BufferCount = 1;
        swapChainDesc.BufferDesc.Width = 800;
        swapChainDesc.BufferDesc.Height = 600;
        swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
        swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.OutputWindow = m_hwnd;
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.SampleDesc.Quality = 0;
        swapChainDesc.Windowed = TRUE;
        
        hr = dxgiFactory->CreateSwapChain(m_device.Get(), &swapChainDesc, &m_swapChain);
        if (FAILED(hr)) {
            return false;
        }
        
        ComPtr<ID3D11Texture2D> backBuffer;
        hr = m_swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
        if (FAILED(hr)) {
            return false;
        }
        
        hr = m_device->CreateRenderTargetView(backBuffer.Get(), nullptr, &m_renderTargetView);
        if (FAILED(hr)) {
            return false;
        }
        
        m_deviceContext->OMSetRenderTargets(1, m_renderTargetView.GetAddressOf(), nullptr);
        
        D3D11_VIEWPORT viewport = {};
        viewport.Width = 800.0f;
        viewport.Height = 600.0f;
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        viewport.TopLeftX = 0;
        viewport.TopLeftY = 0;
        
        m_deviceContext->RSSetViewports(1, &viewport);
        
        // 初始化渲染资源
        if (!InitializeRenderResources()) {
            return false;
        }
        
        return true;
    }
    
    void Render() {
        if (!m_videoPath.empty() && m_decoder.isInitialized()) {
            RGBFrameDecoder::RGBFramePair rgbPair;
            if (m_decoder.readNextRGBFramePair(rgbPair)) {
                if (rgbPair.is_valid && rgbPair.rgb_frame.hasValidResources()) {
                    RenderVideoFrame(rgbPair.rgb_frame.rgb_texture.Get());
                } else {
                    std::cerr << "ERROR: RGB frame pair is invalid or has no valid resources" << std::endl;
                    PostQuitMessage(-1);
                    return;
                }
            } else {
                std::cerr << "ERROR: Failed to read next RGB frame pair" << std::endl;
                PostQuitMessage(-1);
                return;
            }
        } else {
            // 无视频文件时，显示黑屏
            float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            m_deviceContext->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
        }
        
        m_swapChain->Present(1, 0);  // 启用垂直同步
    }
    
    bool InitializeRenderResources() {
        // 创建vertex shader
        const char* vertexShaderSource = R"(
            struct VSInput {
                float2 position : POSITION;
                float2 texCoord : TEXCOORD;
            };
            
            struct VSOutput {
                float4 position : SV_POSITION;
                float2 texCoord : TEXCOORD;
            };
            
            VSOutput main(VSInput input) {
                VSOutput output;
                output.position = float4(input.position, 0.0, 1.0);
                output.texCoord = input.texCoord;
                return output;
            }
        )";
        
        const char* pixelShaderSource = R"(
            Texture2D videoTexture : register(t0);
            SamplerState videoSampler : register(s0);
            
            struct PSInput {
                float4 position : SV_POSITION;
                float2 texCoord : TEXCOORD;
            };
            
            float4 main(PSInput input) : SV_TARGET {
                return videoTexture.Sample(videoSampler, input.texCoord);
            }
        )";
        
        // 编译vertex shader
        ComPtr<ID3DBlob> vsBlob;
        ComPtr<ID3DBlob> errorBlob;
        HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "Vertex shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            }
            return false;
        }
        
        hr = m_device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vertexShader);
        if (FAILED(hr)) {
            return false;
        }
        
        // 编译pixel shader
        ComPtr<ID3DBlob> psBlob;
        hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "Pixel shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            }
            return false;
        }
        
        hr = m_device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_pixelShader);
        if (FAILED(hr)) {
            return false;
        }
        
        // 创建input layout
        D3D11_INPUT_ELEMENT_DESC inputElements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
        };
        
        hr = m_device->CreateInputLayout(inputElements, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_inputLayout);
        if (FAILED(hr)) {
            return false;
        }
        
        // 创建全屏四边形顶点缓冲区
        struct Vertex {
            float x, y;      // position
            float u, v;      // texcoord
        };
        
        Vertex vertices[] = {
            {-1.0f, -1.0f, 0.0f, 1.0f},  // 左下
            {-1.0f,  1.0f, 0.0f, 0.0f},  // 左上
            { 1.0f, -1.0f, 1.0f, 1.0f},  // 右下
            { 1.0f,  1.0f, 1.0f, 0.0f}   // 右上
        };
        
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.ByteWidth = sizeof(vertices);
        bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        
        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = vertices;
        
        hr = m_device->CreateBuffer(&bufferDesc, &initData, &m_vertexBuffer);
        if (FAILED(hr)) {
            return false;
        }
        
        // 创建sampler state
        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        
        hr = m_device->CreateSamplerState(&samplerDesc, &m_samplerState);
        if (FAILED(hr)) {
            return false;
        }
        
        return true;
    }
    
    void RenderVideoFrame(ID3D11Texture2D* videoTexture) {
        // 清空渲染目标
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        m_deviceContext->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
        
        // 创建shader resource view
        ComPtr<ID3D11ShaderResourceView> srv;
        HRESULT hr = m_device->CreateShaderResourceView(videoTexture, nullptr, &srv);
        if (FAILED(hr)) {
            std::cerr << "ERROR: Failed to create shader resource view, HRESULT: 0x" << std::hex << hr << std::endl;
            return;
        }
        
        // 设置渲染状态
        m_deviceContext->IASetInputLayout(m_inputLayout.Get());
        m_deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        
        // 设置顶点缓冲区
        UINT stride = sizeof(float) * 4;  // position(2) + texcoord(2)
        UINT offset = 0;
        m_deviceContext->IASetVertexBuffers(0, 1, m_vertexBuffer.GetAddressOf(), &stride, &offset);
        
        // 设置shaders
        m_deviceContext->VSSetShader(m_vertexShader.Get(), nullptr, 0);
        m_deviceContext->PSSetShader(m_pixelShader.Get(), nullptr, 0);
        
        // 设置纹理和采样器
        m_deviceContext->PSSetShaderResources(0, 1, srv.GetAddressOf());
        m_deviceContext->PSSetSamplers(0, 1, m_samplerState.GetAddressOf());
        
        // 渲染全屏四边形
        m_deviceContext->Draw(4, 0);
    }
    
    
    
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        GUIPlayer* player = nullptr;
        
        if (msg == WM_NCCREATE) {
            CREATESTRUCT* createStruct = (CREATESTRUCT*)lParam;
            player = (GUIPlayer*)createStruct->lpCreateParams;
            SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)player);
        } else {
            player = (GUIPlayer*)GetWindowLongPtr(hwnd, GWLP_USERDATA);
        }
        
        switch (msg) {
            case WM_DESTROY:
                PostQuitMessage(0);
                break;
            case WM_KEYDOWN:
                if (wParam == VK_ESCAPE) {
                    PostQuitMessage(0);
                }
                break;
            default:
                return DefWindowProc(hwnd, msg, wParam, lParam);
        }
        
        return 0;
    }
};

int main() {
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    int nCmdShow = SW_SHOW;
    LPWSTR lpCmdLine = GetCommandLineW();
    
    int argc;
    LPWSTR* argv = CommandLineToArgvW(lpCmdLine, &argc);
    GUIPlayer player;
    
    for (int i = 1; i < argc; i++) {
        std::wstring arg = argv[i];
        if (arg.find(L".mkv") != std::wstring::npos || arg.find(L".mp4") != std::wstring::npos) {
            int size = WideCharToMultiByte(CP_UTF8, 0, arg.c_str(), -1, nullptr, 0, nullptr, nullptr);
            std::string videoPath(size, 0);
            WideCharToMultiByte(CP_UTF8, 0, arg.c_str(), -1, &videoPath[0], size, nullptr, nullptr);
            videoPath.resize(size - 1);
            player.SetVideoPath(videoPath);
            std::cout << "Video path set to: " << videoPath << std::endl;
        } else {
            int autoExitTime = _wtoi(argv[i]);
            if (autoExitTime > 0) {
                player.SetAutoExitTime(autoExitTime);
            }
        }
    }
    
    if (!player.Initialize(hInstance, nCmdShow)) {
        return -1;
    }
    
    player.Run();
    player.Cleanup();
    
    return 0;
}