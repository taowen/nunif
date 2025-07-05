#include "gui_media_sink.h"
#include "media_player.h"
#include <iostream>
#include <d3dcompiler.h>
#include <vector>
#include <algorithm>
#include <cmath>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

// 顶点着色器源码
const char* vertex_shader_source = R"(
struct VS_INPUT {
    float3 pos : POSITION;
    float2 tex : TEXCOORD0;
};

struct VS_OUTPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

VS_OUTPUT main(VS_INPUT input) {
    VS_OUTPUT output;
    output.pos = float4(input.pos, 1.0f);
    output.tex = input.tex;
    return output;
}
)";

// 像素着色器源码
const char* pixel_shader_source = R"(
Texture2D videoTexture : register(t0);
SamplerState textureSampler : register(s0);

struct PS_INPUT {
    float4 pos : SV_POSITION;
    float2 tex : TEXCOORD0;
};

float4 main(PS_INPUT input) : SV_TARGET {
    return videoTexture.Sample(textureSampler, input.tex);
}
)";

// 顶点结构
struct Vertex {
    float x, y, z;
    float u, v;
};

GUIMediaSink::GUIMediaSink()
    : window_handle_(nullptr)
    , should_close_(false)
    , video_width_(0)
    , video_height_(0)
    , audio_sample_rate_(0)
    , audio_channels_(0)
    , is_paused_(false)
    , paused_duration_(0.0)
    , last_video_timestamp_(0.0)
    , has_new_frame_(false)
    , test_mode_(false)
    , auto_close_ms_(1000) 
    , test_start_time_(std::chrono::steady_clock::now())
    , frame_queue_(std::make_unique<FrameQueue>())
    , should_stop_decoder_(false)
    , media_player_(nullptr) {
}

GUIMediaSink::~GUIMediaSink() {
    stopPlayback();
    close();
}

bool GUIMediaSink::initialize(int video_width, int video_height, 
                             int audio_sample_rate, int audio_channels) {
    video_width_ = video_width;
    video_height_ = video_height;
    audio_sample_rate_ = audio_sample_rate;
    audio_channels_ = audio_channels;
    
    
    // 创建窗口（如果失败也继续，只是无法显示）
    if (!createWindow("Video Player - " + std::to_string(video_width_) + "x" + std::to_string(video_height_))) {
        std::cerr << "Failed to create window, continuing without display" << std::endl;
        // 不返回false，继续初始化DirectX11用于解码
    }
    
    // 只有有窗口时才初始化DirectX11渲染
    if (window_handle_) {
        // 初始化DirectX11
        if (!initializeDirectX11()) {
            std::cerr << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        // 创建渲染目标
        if (!createRenderTargets()) {
            std::cerr << "Failed to create render targets" << std::endl;
            return false;
        }
        
        // 创建着色器
        if (!createShaders()) {
            std::cerr << "Failed to create shaders" << std::endl;
            return false;
        }
        
        // 创建几何体
        if (!createGeometry()) {
            std::cerr << "Failed to create geometry" << std::endl;
            return false;
        }
    } else {
        std::cerr << "No window handle, skipping DirectX11 initialization" << std::endl;
    }
    
    // 记录开始时间
    start_time_ = std::chrono::steady_clock::now();
    
    return true;
}

void GUIMediaSink::onVideoFrame(ID3D11Texture2D* rgb_texture, 
                               ID3D11ShaderResourceView* rgb_srv,
                               double timestamp, int width, int height) {
    if (!rgb_texture) {
        std::cerr << "onVideoFrame: Invalid texture" << std::endl;
        return;
    }
    
    // 创建帧对象并推送到队列
    VideoFrame frame(rgb_texture, rgb_srv, timestamp, width, height);
    if (!frame_queue_->push(frame, 10)) { // 10ms 超时
    }
}

void GUIMediaSink::onAudioFrame(const int16_t* samples, int sample_count,
                               double timestamp, int sample_rate, int channels) {
    // TODO: 实现音频播放
    // 这里暂时跳过音频处理
}

void GUIMediaSink::pause() {
    if (!is_paused_) {
        is_paused_ = true;
        pause_time_ = std::chrono::steady_clock::now();
    }
}

void GUIMediaSink::resume() {
    if (is_paused_) {
        is_paused_ = false;
        auto now = std::chrono::steady_clock::now();
        paused_duration_ += std::chrono::duration<double>(now - pause_time_).count();
    }
}

bool GUIMediaSink::isPaused() const {
    return is_paused_;
}

void GUIMediaSink::close() {
    cleanup();
}

bool GUIMediaSink::createWindow(const std::string& title) {
    window_title_ = title;
    
    if (!createWindowClass()) {
        std::cerr << "Failed to create window class" << std::endl;
        return false;
    }
    
    // 限制窗口大小（4K视频太大了）
    int display_width = (video_width_ < 1920) ? video_width_ : 1920;
    int display_height = (video_height_ < 1080) ? video_height_ : 1080;
    
    // 计算窗口大小（包含标题栏和边框）
    RECT window_rect = { 0, 0, display_width, display_height };
    AdjustWindowRect(&window_rect, WS_OVERLAPPEDWINDOW, FALSE);
    
    int window_width = window_rect.right - window_rect.left;
    int window_height = window_rect.bottom - window_rect.top;
    
    // 创建窗口（先不传this指针）
    window_handle_ = CreateWindowExW(
        0,
        L"GUIMediaSink",
        std::wstring(title.begin(), title.end()).c_str(),
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        window_width, window_height,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        nullptr
    );
    
    // 如果窗口创建成功，设置用户数据
    if (window_handle_) {
        SetWindowLongPtr(window_handle_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
    }
    
    if (!window_handle_) {
        DWORD error = GetLastError();
        std::cerr << "Failed to create window, error: " << std::hex << error << " (decimal: " << std::dec << error << ")" << std::endl;
        std::cerr << "Window class: GUIMediaSink" << std::endl;
        std::cerr << "Window size: " << window_width << "x" << window_height << std::endl;
        return false;
    }
    
    return true;
}

void GUIMediaSink::showWindow() {
    if (window_handle_) {
        ShowWindow(window_handle_, SW_SHOWDEFAULT);
        UpdateWindow(window_handle_);
        
        // 清理消息队列中的残留消息
        MSG msg;
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) {
                // 忽略残留的WM_QUIT消息
                continue;
            }
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }
}

bool GUIMediaSink::processMessages() {
    // 检查测试模式自动关闭
    if (test_mode_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - test_start_time_).count();
        
        if (elapsed_ms >= auto_close_ms_) {
            // 发送WM_CLOSE消息来正确关闭窗口
            if (window_handle_) {
                PostMessage(window_handle_, WM_CLOSE, 0, 0);
            }
            should_close_ = true;
            return false;
        }
    }
    
    MSG msg = {};
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
        
        if (msg.message == WM_QUIT) {
            should_close_ = true;
        }
    }
    
    return !should_close_;
}

void GUIMediaSink::present() {
    if (swap_chain_) {
        HRESULT hr = swap_chain_->Present(1, 0);
        if (FAILED(hr)) {
            std::cerr << "Present failed: " << std::hex << hr << std::endl;
        }
    }
}

// 主动渲染循环 - 最佳实践
bool GUIMediaSink::renderLoop() {
    if (!window_handle_) {
        return false;
    }
    if (!d3d11_context_) {
        return false;
    }
    
    static auto last_render_time = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_render_time).count();
    
    // 目标帧率: 60fps = 16.67ms per frame
    const int target_frame_time_ms = 16;
    
    // 处理帧队列（消费解码线程产生的帧）
    processFrameQueue();
    
    // 主动渲染 - 不依赖WM_PAINT
    if (elapsed_ms >= target_frame_time_ms) {
        renderFrame();
        present();
        last_render_time = now;
    }
    
    // 限制CPU使用率
    if (elapsed_ms < target_frame_time_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    return true;
}

bool GUIMediaSink::createWindowClass() {
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"GUIMediaSink";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    
    ATOM result = RegisterClassW(&wc);
    if (result == 0) {
        DWORD error = GetLastError();
        if (error == ERROR_CLASS_ALREADY_EXISTS) {
            // 窗口类已存在，这是正常的
            return true;
        } else {
            std::cerr << "Failed to register window class, error: " << std::hex << error << " (decimal: " << std::dec << error << ")" << std::endl;
            return false;
        }
    }
    
    return true;
}

bool GUIMediaSink::initializeDirectX11() {
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
        std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
        return false;
    }
    
    
    return true;
}

bool GUIMediaSink::createRenderTargets() {
    // 获取窗口客户区尺寸
    RECT client_rect;
    GetClientRect(window_handle_, &client_rect);
    UINT width = client_rect.right - client_rect.left;
    UINT height = client_rect.bottom - client_rect.top;
    
    // 创建交换链
    ComPtr<IDXGIDevice> dxgi_device;
    HRESULT hr = d3d11_device_.As(&dxgi_device);
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
        std::cerr << "Failed to create swap chain: " << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建渲染目标视图
    ComPtr<ID3D11Texture2D> back_buffer;
    hr = swap_chain_->GetBuffer(0, IID_PPV_ARGS(&back_buffer));
    if (FAILED(hr)) return false;
    
    hr = d3d11_device_->CreateRenderTargetView(back_buffer.Get(), nullptr, &render_target_view_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view: " << std::hex << hr << std::endl;
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
    
    return true;
}

bool GUIMediaSink::createShaders() {
    ComPtr<ID3DBlob> vs_blob;
    ComPtr<ID3DBlob> ps_blob;
    ComPtr<ID3DBlob> error_blob;
    
    // 编译顶点着色器
    HRESULT hr = D3DCompile(
        vertex_shader_source,
        strlen(vertex_shader_source),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "vs_5_0",
        0,
        0,
        &vs_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Vertex shader compile error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    // 编译像素着色器
    hr = D3DCompile(
        pixel_shader_source,
        strlen(pixel_shader_source),
        nullptr,
        nullptr,
        nullptr,
        "main",
        "ps_5_0",
        0,
        0,
        &ps_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Pixel shader compile error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    // 创建着色器
    hr = d3d11_device_->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vertex_shader_);
    if (FAILED(hr)) return false;
    
    hr = d3d11_device_->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &pixel_shader_);
    if (FAILED(hr)) return false;
    
    // 创建输入布局
    D3D11_INPUT_ELEMENT_DESC input_elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };
    
    hr = d3d11_device_->CreateInputLayout(
        input_elements,
        ARRAYSIZE(input_elements),
        vs_blob->GetBufferPointer(),
        vs_blob->GetBufferSize(),
        &input_layout_
    );
    
    if (FAILED(hr)) return false;
    
    // 创建采样器状态
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    
    hr = d3d11_device_->CreateSamplerState(&sampler_desc, &sampler_state_);
    if (FAILED(hr)) return false;
    
    return true;
}

bool GUIMediaSink::createGeometry() {
    // 创建全屏四边形
    Vertex vertices[] = {
        {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f}, // 左下
        {-1.0f,  1.0f, 0.0f, 0.0f, 0.0f}, // 左上
        { 1.0f,  1.0f, 0.0f, 1.0f, 0.0f}, // 右上
        { 1.0f, -1.0f, 0.0f, 1.0f, 1.0f}  // 右下
    };
    
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = sizeof(vertices);
    buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    buffer_desc.CPUAccessFlags = 0;
    
    D3D11_SUBRESOURCE_DATA init_data = {};
    init_data.pSysMem = vertices;
    
    HRESULT hr = d3d11_device_->CreateBuffer(&buffer_desc, &init_data, &vertex_buffer_);
    if (FAILED(hr)) return false;
    
    // 创建索引缓冲区
    UINT indices[] = {
        0, 1, 2,
        0, 2, 3
    };
    
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = sizeof(indices);
    buffer_desc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    buffer_desc.CPUAccessFlags = 0;
    
    init_data.pSysMem = indices;
    
    hr = d3d11_device_->CreateBuffer(&buffer_desc, &init_data, &index_buffer_);
    if (FAILED(hr)) return false;
    
    return true;
}

void GUIMediaSink::updateVideoTexture(ID3D11Texture2D* source_texture) {
    if (!source_texture || !d3d11_context_) {
        std::cerr << "updateVideoTexture: Invalid parameters" << std::endl;
        return;
    }
    
    // 第一次创建视频纹理
    if (!video_texture_) {
        D3D11_TEXTURE2D_DESC desc;
        source_texture->GetDesc(&desc);
        
        
        // 创建可绑定到着色器的纹理
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = 0;
        
        HRESULT hr = d3d11_device_->CreateTexture2D(&desc, nullptr, &video_texture_);
        if (FAILED(hr)) {
            std::cerr << "Failed to create video texture: " << std::hex << hr << std::endl;
            return;
        }
        
        // 创建着色器资源视图
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = desc.Format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        
        hr = d3d11_device_->CreateShaderResourceView(video_texture_.Get(), &srv_desc, &video_srv_);
        if (FAILED(hr)) {
            std::cerr << "Failed to create video SRV: " << std::hex << hr << std::endl;
            return;
        }
        
    }
    
    // 复制纹理数据
    d3d11_context_->CopyResource(video_texture_.Get(), source_texture);
}

void GUIMediaSink::renderFrame() {
    if (!d3d11_context_ || !render_target_view_) {
        return;
    }
    
    // 清除背景 - 使用黑色背景而不是动态颜色
    float clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    d3d11_context_->ClearRenderTargetView(render_target_view_.Get(), clear_color);
    
    // 如果有视频纹理，渲染它
    if (video_srv_) {
        // 设置着色器
        d3d11_context_->VSSetShader(vertex_shader_.Get(), nullptr, 0);
        d3d11_context_->PSSetShader(pixel_shader_.Get(), nullptr, 0);
        
        // 设置输入布局
        d3d11_context_->IASetInputLayout(input_layout_.Get());
        
        // 设置顶点缓冲区
        UINT stride = sizeof(Vertex);
        UINT offset = 0;
        d3d11_context_->IASetVertexBuffers(0, 1, vertex_buffer_.GetAddressOf(), &stride, &offset);
        d3d11_context_->IASetIndexBuffer(index_buffer_.Get(), DXGI_FORMAT_R32_UINT, 0);
        d3d11_context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        
        // 设置纹理和采样器
        d3d11_context_->PSSetShaderResources(0, 1, video_srv_.GetAddressOf());
        d3d11_context_->PSSetSamplers(0, 1, sampler_state_.GetAddressOf());
        
        // 绘制
        d3d11_context_->DrawIndexed(6, 0, 0);
    }
}

void GUIMediaSink::cleanup() {
    // 重置所有COM对象
    video_srv_.Reset();
    video_texture_.Reset();
    sampler_state_.Reset();
    input_layout_.Reset();
    pixel_shader_.Reset();
    vertex_shader_.Reset();
    index_buffer_.Reset();
    vertex_buffer_.Reset();
    render_target_view_.Reset();
    swap_chain_.Reset();
    d3d11_context_.Reset();
    d3d11_device_.Reset();
    
    // 销毁窗口
    if (window_handle_) {
        DestroyWindow(window_handle_);
        window_handle_ = nullptr;
    }
}

LRESULT CALLBACK GUIMediaSink::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    GUIMediaSink* sink = reinterpret_cast<GUIMediaSink*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    
    if (sink) {
        return sink->handleMessage(uMsg, wParam, lParam);
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

LRESULT GUIMediaSink::handleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            DestroyWindow(window_handle_);
            return 0;
            
        case WM_DESTROY:
            PostQuitMessage(0);
            should_close_ = true;
            return 0;
            
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
                should_close_ = true;
            } else if (wParam == VK_SPACE) {
                // 空格键切换暂停/恢复
                if (is_paused_) {
                    resume();
                } else {
                    pause();
                }
            }
            return 0;
            
        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                BeginPaint(window_handle_, &ps);
                // 在WM_PAINT中不执行实际渲染，只处理Windows绘制消息
                EndPaint(window_handle_, &ps);
                return 0;
            }
    }
    
    return DefWindowProc(window_handle_, uMsg, wParam, lParam);
}

void GUIMediaSink::setTestMode(bool enabled, int auto_close_ms) {
    test_mode_ = enabled;
    auto_close_ms_ = auto_close_ms;
    if (enabled) {
        // 重置测试开始时间为当前时间
        test_start_time_ = std::chrono::steady_clock::now();
    }
}

void GUIMediaSink::setVideoDimensions(int width, int height) {
    video_width_ = width;
    video_height_ = height;
}

// 多线程播放控制
bool GUIMediaSink::startPlayback(MediaPlayer* player, const std::string& filepath) {
    if (!player) {
        std::cerr << "startPlayback: Invalid MediaPlayer" << std::endl;
        return false;
    }
    
    // 停止现有播放
    stopPlayback();
    
    media_player_ = player;
    current_filepath_ = filepath;
    should_stop_decoder_ = false;
    
    // 启动解码线程
    decoder_thread_ = std::make_unique<std::thread>(&GUIMediaSink::decoderThreadLoop, this);
    
    return true;
}

void GUIMediaSink::stopPlayback() {
    // 停止解码线程
    should_stop_decoder_ = true;
    
    if (decoder_thread_ && decoder_thread_->joinable()) {
        decoder_thread_->join();
        decoder_thread_.reset();
    }
    
    // 清空帧队列
    if (frame_queue_) {
        frame_queue_->clear();
    }
    
    media_player_ = nullptr;
    current_filepath_.clear();
    
}

bool GUIMediaSink::isPlaybackRunning() const {
    return decoder_thread_ && decoder_thread_->joinable() && !should_stop_decoder_;
}

// 解码线程循环
void GUIMediaSink::decoderThreadLoop() {
    
    if (!media_player_) {
        std::cerr << "decoderThreadLoop: No MediaPlayer available" << std::endl;
        return;
    }
    
    // 打开文件
    if (!media_player_->openFile(current_filepath_)) {
        std::cerr << "decoderThreadLoop: Failed to open file: " << current_filepath_ << std::endl;
        return;
    }
    
    
    // 解码循环
    while (!should_stop_decoder_) {
        if (!media_player_->playOneFrame()) {
            break;
        }
        
        // 检查队列是否过满，如果是则稍微等待
        if (frame_queue_->size() > FrameQueue::MAX_QUEUE_SIZE * 0.8) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    // 停止队列
    frame_queue_->stop();
    
}

// 处理帧队列（在主线程调用）
void GUIMediaSink::processFrameQueue() {
    VideoFrame frame;
    
    // 消费所有可用帧（但不阻塞）
    while (frame_queue_->pop(frame, 1)) { // 1ms 超时，不阻塞
        if (frame.is_valid && frame.texture) {
            // 更新视频纹理
            updateVideoTexture(frame.texture);
            
            // 更新时间戳
            last_video_timestamp_ = frame.timestamp;
            has_new_frame_ = true;
        }
    }
}