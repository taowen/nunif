#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <memory>
#include <iostream>
#include "video_player.h"
#include "async_rgb_video_decoder.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

// 全局变量
std::unique_ptr<VideoPlayer> g_video_player;
std::unique_ptr<AsyncRgbVideoDecoder> g_decoder;
ID3D11Device* g_device = nullptr;
ID3D11DeviceContext* g_context = nullptr;
IDXGISwapChain* g_swap_chain = nullptr;
ID3D11RenderTargetView* g_render_target_view = nullptr;

constexpr UINT_PTR TIMER_ID = 1;
constexpr UINT TIMER_INTERVAL = 16;  // 16ms ≈ 60fps

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        {
            // 创建D3D11设备和交换链
            DXGI_SWAP_CHAIN_DESC scd = {};
            scd.BufferCount = 2;
            scd.BufferDesc.Width = 800;
            scd.BufferDesc.Height = 600;
            scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            scd.BufferDesc.RefreshRate.Numerator = 60;
            scd.BufferDesc.RefreshRate.Denominator = 1;
            scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            scd.OutputWindow = hwnd;
            scd.SampleDesc.Count = 1;
            scd.SampleDesc.Quality = 0;
            scd.Windowed = TRUE;
            scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
            
            HRESULT hr = D3D11CreateDeviceAndSwapChain(
                nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 
                D3D11_CREATE_DEVICE_DEBUG,
                nullptr, 0, D3D11_SDK_VERSION,
                &scd, &g_swap_chain, &g_device, nullptr, &g_context);
                
            if (SUCCEEDED(hr)) {
                // 创建渲染目标
                ID3D11Texture2D* back_buffer = nullptr;
                hr = g_swap_chain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
                if (SUCCEEDED(hr)) {
                    hr = g_device->CreateRenderTargetView(back_buffer, nullptr, &g_render_target_view);
                    back_buffer->Release();
                }
            }
            
            if (SUCCEEDED(hr)) {
                // 初始化VideoPlayer
                g_video_player = std::make_unique<VideoPlayer>();
                if (g_video_player->initialize(g_device, g_context, g_render_target_view, g_swap_chain)) {
                    SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL, nullptr);
                    std::cout << "GUI播放器启动 - 依赖注入架构 (16ms间隔)" << std::endl;
                } else {
                    std::cerr << "Failed to initialize video player" << std::endl;
                    PostQuitMessage(1);
                }
            } else {
                std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
                PostQuitMessage(1);
            }
        }
        break;
        
    case WM_TIMER:
        if (wParam == TIMER_ID && g_video_player) {
            g_video_player->onTimer();
        }
        break;
        
    case WM_DESTROY:
        KillTimer(hwnd, TIMER_ID);
        if (g_video_player) {
            g_video_player.reset();
        }
        // 释放D3D11资源
        if (g_render_target_view) {
            g_render_target_view->Release();
            g_render_target_view = nullptr;
        }
        if (g_swap_chain) {
            g_swap_chain->Release();
            g_swap_chain = nullptr;
        }
        if (g_context) {
            g_context->Release();
            g_context = nullptr;
        }
        if (g_device) {
            g_device->Release();
            g_device = nullptr;
        }
        PostQuitMessage(0);
        break;
        
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            PostMessage(hwnd, WM_CLOSE, 0, 0);
        }
        break;
        
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    
    return 0;
}

// 主函数
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // 注册窗口类
    WNDCLASSA wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "GUIPlayer";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    
    if (!RegisterClassA(&wc)) {
        std::cerr << "Failed to register window class" << std::endl;
        return 1;
    }
    
    // 创建窗口
    HWND hwnd = CreateWindowA(
        "GUIPlayer", "GUI播放器 - onTimer()回调架构 - 24fps->60Hz",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        800, 600, nullptr, nullptr, hInstance, nullptr);
        
    if (!hwnd) {
        std::cerr << "Failed to create window" << std::endl;
        return 1;
    }
    
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    
    std::cout << "窗口已创建 - 定时器回调架构，按ESC键退出" << std::endl;
    
    // 消息循环
    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return static_cast<int>(msg.wParam);
}