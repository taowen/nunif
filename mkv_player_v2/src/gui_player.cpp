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

using Microsoft::WRL::ComPtr;

class GUIPlayer {
public:
    GUIPlayer() : m_hwnd(nullptr), m_colorIndex(0), m_lastColorChange(std::chrono::steady_clock::now()), m_startTime(std::chrono::steady_clock::now()), m_autoExitTimeMs(0) {}
    
    void SetAutoExitTime(int timeMs) {
        m_autoExitTimeMs = timeMs;
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
                    std::cout << "Frame " << frameCount << " at " << elapsed.count() << "ms, color: " << GetCurrentColorName() << std::endl;
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
    
    int m_colorIndex;
    std::chrono::steady_clock::time_point m_lastColorChange;
    std::chrono::steady_clock::time_point m_startTime;
    int m_autoExitTimeMs;
    
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
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastColorChange);
        
        if (elapsed.count() >= 1000) {
            int oldColorIndex = m_colorIndex;
            m_colorIndex = (m_colorIndex + 1) % 3;
            m_lastColorChange = now;
            std::cout << "Color changed from " << GetColorName(oldColorIndex) << " to " << GetColorName(m_colorIndex) << std::endl;
        }
        
        float clearColor[4];
        switch (m_colorIndex) {
            case 0:
                clearColor[0] = 1.0f; clearColor[1] = 0.0f; clearColor[2] = 0.0f; clearColor[3] = 1.0f;
                break;
            case 1:
                clearColor[0] = 0.0f; clearColor[1] = 1.0f; clearColor[2] = 0.0f; clearColor[3] = 1.0f;
                break;
            case 2:
                clearColor[0] = 0.0f; clearColor[1] = 0.0f; clearColor[2] = 1.0f; clearColor[3] = 1.0f;
                break;
        }
        
        m_deviceContext->ClearRenderTargetView(m_renderTargetView.Get(), clearColor);
        m_swapChain->Present(0, 0);
    }
    
    const char* GetColorName(int colorIndex) {
        switch (colorIndex) {
            case 0: return "Red";
            case 1: return "Green";
            case 2: return "Blue";
            default: return "Unknown";
        }
    }
    
    const char* GetCurrentColorName() {
        return GetColorName(m_colorIndex);
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
    LPWSTR cmdArg = nullptr;
    if (argc > 1) {
        cmdArg = argv[1];
    }
    GUIPlayer player;
    
    if (cmdArg && wcslen(cmdArg) > 0) {
        int autoExitTime = _wtoi(cmdArg);
        if (autoExitTime > 0) {
            player.SetAutoExitTime(autoExitTime);
        }
    }
    
    if (!player.Initialize(hInstance, nCmdShow)) {
        return -1;
    }
    
    player.Run();
    player.Cleanup();
    
    return 0;
}