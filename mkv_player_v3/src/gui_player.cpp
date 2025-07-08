#include <windows.h>
#include <memory>
#include <iostream>
#include "video_player.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

// 全局变量
std::unique_ptr<VideoPlayer> g_video_player;
constexpr UINT_PTR TIMER_ID = 1;
constexpr UINT TIMER_INTERVAL = 16;  // 16ms ≈ 60fps

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        g_video_player = std::make_unique<VideoPlayer>();
        if (!g_video_player->initialize(hwnd)) {
            std::cerr << "Failed to initialize video player" << std::endl;
            PostQuitMessage(1);
        } else {
            // 启动定时器 - 每16ms调用一次onTimer()
            SetTimer(hwnd, TIMER_ID, TIMER_INTERVAL, nullptr);
            std::cout << "GUI播放器启动 - 定时器驱动架构 (16ms间隔)" << std::endl;
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