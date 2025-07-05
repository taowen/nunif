#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <iostream>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

class SimpleRedWindow {
public:
    SimpleRedWindow() : window_handle_(nullptr), should_close_(false) {}
    
    ~SimpleRedWindow() {
        cleanup();
    }
    
    bool initialize() {
        // 创建窗口
        if (!createWindow()) {
            std::cerr << "Failed to create window" << std::endl;
            return false;
        }
        
        // 初始化DirectX11
        if (!initializeDirectX11()) {
            std::cerr << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        return true;
    }
    
    void run() {
        ShowWindow(window_handle_, SW_SHOWDEFAULT);
        UpdateWindow(window_handle_);
        
        MSG msg = {};
        while (!should_close_) {
            // 处理消息
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
                
                if (msg.message == WM_QUIT) {
                    should_close_ = true;
                }
            }
            
            // 渲染红色
            render();
        }
    }
    
private:
    HWND window_handle_;
    bool should_close_;
    
    ComPtr<ID3D11Device> d3d11_device_;
    ComPtr<ID3D11DeviceContext> d3d11_context_;
    ComPtr<IDXGISwapChain> swap_chain_;
    ComPtr<ID3D11RenderTargetView> render_target_view_;
    
    bool createWindow() {
        // 注册窗口类
        WNDCLASSW wc = {};
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.lpszClassName = L"SimpleRedWindow";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        
        RegisterClassW(&wc);
        
        // 创建窗口
        window_handle_ = CreateWindowExW(
            0,
            L"SimpleRedWindow",
            L"Simple Red Window - DirectX11",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT,
            800, 600,
            nullptr, nullptr,
            GetModuleHandle(nullptr),
            this
        );
        
        return window_handle_ != nullptr;
    }
    
    bool initializeDirectX11() {
        // 创建设备和设备上下文
        D3D_FEATURE_LEVEL feature_levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        
        UINT create_device_flags = 0;
#ifdef _DEBUG
        create_device_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        
        D3D_FEATURE_LEVEL feature_level;
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            create_device_flags,
            feature_levels,
            ARRAYSIZE(feature_levels),
            D3D11_SDK_VERSION,
            &d3d11_device_,
            &feature_level,
            &d3d11_context_
        );
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create D3D11 device, error: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 获取窗口客户区尺寸
        RECT client_rect;
        GetClientRect(window_handle_, &client_rect);
        UINT width = client_rect.right - client_rect.left;
        UINT height = client_rect.bottom - client_rect.top;
        
        // 创建交换链
        ComPtr<IDXGIDevice> dxgi_device;
        hr = d3d11_device_.As(&dxgi_device);
        if (FAILED(hr)) return false;
        
        ComPtr<IDXGIAdapter> dxgi_adapter;
        hr = dxgi_device->GetAdapter(&dxgi_adapter);
        if (FAILED(hr)) return false;
        
        ComPtr<IDXGIFactory> dxgi_factory;
        hr = dxgi_adapter->GetParent(IID_PPV_ARGS(&dxgi_factory));
        if (FAILED(hr)) return false;
        
        DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
        swap_chain_desc.BufferCount = 1;
        swap_chain_desc.BufferDesc.Width = width;
        swap_chain_desc.BufferDesc.Height = height;
        swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swap_chain_desc.BufferDesc.RefreshRate.Numerator = 60;
        swap_chain_desc.BufferDesc.RefreshRate.Denominator = 1;
        swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.OutputWindow = window_handle_;
        swap_chain_desc.SampleDesc.Count = 1;
        swap_chain_desc.SampleDesc.Quality = 0;
        swap_chain_desc.Windowed = TRUE;
        
        hr = dxgi_factory->CreateSwapChain(d3d11_device_.Get(), &swap_chain_desc, &swap_chain_);
        if (FAILED(hr)) {
            std::cerr << "Failed to create swap chain, error: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建渲染目标视图
        ComPtr<ID3D11Texture2D> back_buffer;
        hr = swap_chain_->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
        if (FAILED(hr)) return false;
        
        hr = d3d11_device_->CreateRenderTargetView(back_buffer.Get(), nullptr, &render_target_view_);
        if (FAILED(hr)) {
            std::cerr << "Failed to create render target view, error: " << std::hex << hr << std::endl;
            return false;
        }
        
        // 设置渲染目标
        d3d11_context_->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr);
        
        // 设置视口
        D3D11_VIEWPORT viewport = {};
        viewport.TopLeftX = 0;
        viewport.TopLeftY = 0;
        viewport.Width = static_cast<float>(width);
        viewport.Height = static_cast<float>(height);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        d3d11_context_->RSSetViewports(1, &viewport);
        
        std::cout << "DirectX11 initialized successfully" << std::endl;
        std::cout << "Window size: " << width << "x" << height << std::endl;
        
        return true;
    }
    
    void render() {
        // 清除屏幕为红色
        float red_color[4] = { 1.0f, 0.0f, 0.0f, 1.0f }; // RGBA
        d3d11_context_->ClearRenderTargetView(render_target_view_.Get(), red_color);
        
        // 呈现
        HRESULT hr = swap_chain_->Present(1, 0);
        if (FAILED(hr)) {
            std::cerr << "Present failed with error: " << std::hex << hr << std::endl;
            if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) {
                std::cerr << "DirectX11 device lost!" << std::endl;
                should_close_ = true;
            }
        }
    }
    
    void cleanup() {
        if (render_target_view_) render_target_view_.Reset();
        if (swap_chain_) swap_chain_.Reset();
        if (d3d11_context_) d3d11_context_.Reset();
        if (d3d11_device_) d3d11_device_.Reset();
        
        if (window_handle_) {
            DestroyWindow(window_handle_);
            window_handle_ = nullptr;
        }
    }
    
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
        SimpleRedWindow* window = nullptr;
        
        if (uMsg == WM_NCCREATE) {
            CREATESTRUCT* create_struct = reinterpret_cast<CREATESTRUCT*>(lParam);
            window = reinterpret_cast<SimpleRedWindow*>(create_struct->lpCreateParams);
            SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(window));
        } else {
            window = reinterpret_cast<SimpleRedWindow*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
        }
        
        switch (uMsg) {
            case WM_DESTROY:
                PostQuitMessage(0);
                if (window) {
                    window->should_close_ = true;
                }
                return 0;
                
            case WM_KEYDOWN:
                if (wParam == VK_ESCAPE) {
                    PostQuitMessage(0);
                    if (window) {
                        window->should_close_ = true;
                    }
                }
                return 0;
        }
        
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
};

int main() {
    std::cout << "=== Simple Red Window - DirectX11 ===" << std::endl;
    std::cout << "Press ESC to exit" << std::endl;
    
    SimpleRedWindow window;
    if (!window.initialize()) {
        std::cerr << "Failed to initialize window" << std::endl;
        return 1;
    }
    
    window.run();
    
    std::cout << "Window closed successfully" << std::endl;
    return 0;
}