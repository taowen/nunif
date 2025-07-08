#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <memory>
#include <iostream>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

// 假的视频信号源 - 三色轮替24fps
class FakeVideoSignal {
public:
    struct Frame {
        float color[3];  // RGB
        double timestamp;
    };
    
    FakeVideoSignal() : frame_count_(0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    bool getNextFrame(Frame& frame) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        
        // 24fps = 1/24 = 0.041667秒每帧
        double frame_duration = 1.0 / 24.0;
        int target_frame = static_cast<int>(elapsed / frame_duration);
        
        if (target_frame <= frame_count_) {
            return false;  // 还没到下一帧时间
        }
        
        frame_count_ = target_frame;
        frame.timestamp = frame_count_ * frame_duration;
        
        // 三色轮替：红->绿->蓝->红...
        int color_index = frame_count_ % 3;
        frame.color[0] = (color_index == 0) ? 1.0f : 0.0f;  // R
        frame.color[1] = (color_index == 1) ? 1.0f : 0.0f;  // G
        frame.color[2] = (color_index == 2) ? 1.0f : 0.0f;  // B
        
        return true;
    }
    
    int getFrameCount() const { return frame_count_; }
    
private:
    int frame_count_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// 帧率转换器 - 24fps到60Hz
class FrameRateConverter {
public:
    FrameRateConverter() : frame_time_accumulator_(0.0), current_display_count_(0) {}
    
    bool shouldDisplayFrame(const FakeVideoSignal::Frame&) {
        // 24fps → 60Hz: 每个视频帧需要显示 60/24 = 2.5 次
        double source_fps = 24.0;
        double target_fps = 60.0;
        double frame_duration = 1.0 / source_fps;
        double display_interval = 1.0 / target_fps;
        
        if (current_display_count_ == 0) {
            // 新的视频帧，计算需要显示多少次
            frame_time_accumulator_ += frame_duration;
            required_display_count_ = static_cast<int>(frame_time_accumulator_ / display_interval + 0.5);
            frame_time_accumulator_ -= required_display_count_ * display_interval;
        }
        
        current_display_count_++;
        
        if (current_display_count_ >= required_display_count_) {
            current_display_count_ = 0;
            return false;  // 这个视频帧显示完了，需要新帧
        }
        
        return true;  // 继续显示当前帧
    }
    
private:
    double frame_time_accumulator_;
    int current_display_count_;
    int required_display_count_;
};

// DirectX11渲染器
class VideoRenderer {
public:
    VideoRenderer() : hwnd_(nullptr), should_stop_(false) {}
    
    ~VideoRenderer() {
        cleanup();
    }
    
    bool initialize(HWND hwnd) {
        hwnd_ = hwnd;
        
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
            &scd, &swap_chain_, &device_, nullptr, &context_);
            
        if (FAILED(hr)) {
            std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建渲染目标
        ID3D11Texture2D* back_buffer = nullptr;
        hr = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to get back buffer: " << std::hex << hr << std::endl;
            return false;
        }
        
        hr = device_->CreateRenderTargetView(back_buffer, nullptr, &render_target_view_);
        back_buffer->Release();
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create render target view: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 设置视口
        D3D11_VIEWPORT vp = {};
        vp.Width = 800.0f;
        vp.Height = 600.0f;
        vp.MinDepth = 0.0f;
        vp.MaxDepth = 1.0f;
        vp.TopLeftX = 0;
        vp.TopLeftY = 0;
        context_->RSSetViewports(1, &vp);
        
        return true;
    }
    
    void startRenderLoop() {
        should_stop_ = false;
        render_thread_ = std::thread(&VideoRenderer::renderLoop, this);
    }
    
    void stopRenderLoop() {
        should_stop_ = true;
        if (render_thread_.joinable()) {
            render_thread_.join();
        }
    }
    
private:
    void renderLoop() {
        FakeVideoSignal video_signal;
        FrameRateConverter frame_converter;
        FakeVideoSignal::Frame current_frame = {};
        bool has_frame = false;
        
        std::cout << "开始渲染循环 - 24fps视频信号 -> 60Hz显示" << std::endl;
        
        auto last_stats_time = std::chrono::high_resolution_clock::now();
        int render_count = 0;
        
        while (!should_stop_) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 检查是否需要新的视频帧
            if (!has_frame || !frame_converter.shouldDisplayFrame(current_frame)) {
                if (video_signal.getNextFrame(current_frame)) {
                    has_frame = true;
                    std::cout << "新视频帧 #" << video_signal.getFrameCount() 
                              << " - 颜色: RGB(" << current_frame.color[0] << "," 
                              << current_frame.color[1] << "," << current_frame.color[2] << ")" << std::endl;
                }
            }
            
            // 渲染当前帧
            if (has_frame) {
                renderFrame(current_frame);
            }
            
            // VSync - 60Hz同步
            HRESULT hr = swap_chain_->Present(1, 0);
            if (FAILED(hr)) {
                std::cerr << "Present failed: " << std::hex << hr << std::endl;
                break;
            }
            
            render_count++;
            
            // 每秒统计一次
            auto now = std::chrono::high_resolution_clock::now();
            auto stats_elapsed = std::chrono::duration<double>(now - last_stats_time).count();
            if (stats_elapsed >= 1.0) {
                std::cout << "渲染统计: " << render_count << " 帧/秒, 视频帧: " 
                          << video_signal.getFrameCount() << std::endl;
                render_count = 0;
                last_stats_time = now;
            }
        }
        
        std::cout << "渲染循环结束" << std::endl;
    }
    
    void renderFrame(const FakeVideoSignal::Frame& frame) {
        // 清屏为指定颜色
        float clear_color[4] = { frame.color[0], frame.color[1], frame.color[2], 1.0f };
        context_->ClearRenderTargetView(render_target_view_, clear_color);
        context_->OMSetRenderTargets(1, &render_target_view_, nullptr);
    }
    
    void cleanup() {
        if (render_target_view_) {
            render_target_view_->Release();
            render_target_view_ = nullptr;
        }
        if (swap_chain_) {
            swap_chain_->Release();
            swap_chain_ = nullptr;
        }
        if (context_) {
            context_->Release();
            context_ = nullptr;
        }
        if (device_) {
            device_->Release();
            device_ = nullptr;
        }
    }
    
    HWND hwnd_;
    ID3D11Device* device_ = nullptr;
    ID3D11DeviceContext* context_ = nullptr;
    IDXGISwapChain* swap_chain_ = nullptr;
    ID3D11RenderTargetView* render_target_view_ = nullptr;
    
    std::thread render_thread_;
    std::atomic<bool> should_stop_;
};

// 全局变量
std::unique_ptr<VideoRenderer> g_renderer;

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
        g_renderer = std::make_unique<VideoRenderer>();
        if (!g_renderer->initialize(hwnd)) {
            std::cerr << "Failed to initialize renderer" << std::endl;
            PostQuitMessage(1);
        } else {
            g_renderer->startRenderLoop();
            std::cout << "GUI播放器启动 - 独立渲染线程已开始" << std::endl;
        }
        break;
        
    case WM_DESTROY:
        if (g_renderer) {
            g_renderer->stopRenderLoop();
            g_renderer.reset();
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
        "GUIPlayer", "GUI播放器 - 24fps视频 -> 60Hz显示",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        800, 600, nullptr, nullptr, hInstance, nullptr);
        
    if (!hwnd) {
        std::cerr << "Failed to create window" << std::endl;
        return 1;
    }
    
    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);
    
    std::cout << "窗口已创建，按ESC键退出" << std::endl;
    
    // 消息循环
    MSG msg = {};
    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return static_cast<int>(msg.wParam);
}