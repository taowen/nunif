#include <windows.h>
#include <iostream>
#include "player_engine.h"

// 全局变量
HWND hwnd = nullptr;
PlayerEngineHandle playerEngine = nullptr;

// 函数声明
LRESULT CALLBACK WindowProc(HWND hwnd_param, UINT uMsg, WPARAM wParam, LPARAM lParam);
bool initializeWindow(HINSTANCE hInstance);
void cleanup();

bool initializeWindow(HINSTANCE hInstance) {
    // 获取屏幕尺寸
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // 注册窗口类
    WNDCLASSA wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "MKVPlayer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    
    if (!RegisterClassA(&wc)) {
        return false;
    }
    
    // 创建全屏窗口
    hwnd = CreateWindowExA(
        WS_EX_TOPMOST, "MKVPlayer", "MKV Player",
        WS_POPUP | WS_VISIBLE,
        0, 0, screenWidth, screenHeight,
        nullptr, nullptr, hInstance, nullptr
    );
    
    if (!hwnd) {
        return false;
    }
    
    ShowWindow(hwnd, SW_MAXIMIZE);
    UpdateWindow(hwnd);
    
    return true;
}

void cleanup() {
    if (playerEngine) {
        destroyPlayerEngine(playerEngine);
        playerEngine = nullptr;
    }
}

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd_param, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        break;
        
    case WM_DESTROY:
        cleanup();
        PostQuitMessage(0);
        break;
        
    case WM_KEYDOWN:
        if (wParam == VK_SPACE && playerEngine) {
            // 空格键暂停/播放
            if (isPlaying(playerEngine)) {
                stopPlayback(playerEngine);
            } else {
                startPlayback(playerEngine);
            }
        } else if (wParam == VK_ESCAPE) {
            // ESC键退出
            PostMessage(hwnd_param, WM_CLOSE, 0, 0);
        }
        break;
        
    default:
        return DefWindowProc(hwnd_param, uMsg, wParam, lParam);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mkv_file>" << std::endl;
        return -1;
    }
    
    HINSTANCE hInstance = GetModuleHandle(nullptr);
    
    // 初始化窗口
    if (!initializeWindow(hInstance)) {
        std::cerr << "Failed to create window" << std::endl;
        return -1;
    }
    
    // 创建播放引擎
    playerEngine = createPlayerEngine(argv[1], hwnd);
    if (!playerEngine) {
        std::cerr << "Failed to initialize player engine" << std::endl;
        MessageBoxA(hwnd, "Failed to initialize player engine", "Error", MB_OK | MB_ICONERROR);
        cleanup();
        return -1;
    }
    
    // 开始播放
    if (!startPlayback(playerEngine)) {
        std::cerr << "Failed to start playback" << std::endl;
        MessageBoxA(hwnd, "Failed to start playback", "Error", MB_OK | MB_ICONERROR);
        cleanup();
        return -1;
    }
    
    // 消息循环
    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return 0;
}
