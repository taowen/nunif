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
        
        if (!InitializeD3D()) {
            std::cout << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        std::cout << "DirectX11 initialized successfully" << std::endl;
        
        if (!m_videoPath.empty()) {
            if (!m_decoder.open(m_videoPath)) {
                std::cout << "Failed to open video file: " << m_videoPath << std::endl;
                return false;
            }
            std::cout << "Video decoder initialized successfully" << std::endl;
        }
        
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
        
        D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
        
        HRESULT hr = D3D11CreateDeviceAndSwapChain(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            &featureLevel, 1,
            D3D11_SDK_VERSION,
            &swapChainDesc,
            &m_swapChain,
            &m_device,
            nullptr,
            &m_deviceContext
        );
        
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
            std::cerr << "ERROR: Video path is empty or decoder is not initialized" << std::endl;
            PostQuitMessage(-1);
            return;
        }
        
        m_swapChain->Present(0, 0);
    }
    
    void RenderVideoFrame(ID3D11Texture2D* videoTexture) {
        ComPtr<ID3D11ShaderResourceView> srv;
        HRESULT hr = m_device->CreateShaderResourceView(videoTexture, nullptr, &srv);
        if (FAILED(hr)) {
            std::cerr << "ERROR: Failed to create shader resource view, HRESULT: 0x" << std::hex << hr << std::endl;
            PostQuitMessage(-1);
            return;
        }
        
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        m_deviceContext->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
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