#include "gui_media_sink.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cstring>

// 防止Windows宏冲突
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// 静态窗口过程函数
LRESULT CALLBACK GUIMediaSink::WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    GUIMediaSink* sink = nullptr;
    
    if (uMsg == WM_NCCREATE) {
        // 在窗口创建时存储this指针
        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(lParam);
        sink = reinterpret_cast<GUIMediaSink*>(pCreate->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(sink));
    } else {
        // 获取存储的this指针
        sink = reinterpret_cast<GUIMediaSink*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    }
    
    if (sink) {
        switch (uMsg) {
        case WM_CLOSE:
            sink->window_should_close_ = true;
            return 0;
            
        case WM_KEYDOWN:
            switch (wParam) {
            case VK_SPACE:
                // 空格键：暂停/恢复
                if (sink->is_paused_) {
                    sink->resume();
                } else {
                    sink->pause();
                }
                return 0;
            case VK_ESCAPE:
                // ESC键：关闭窗口
                sink->window_should_close_ = true;
                return 0;
            }
            break;
            
        case WM_PAINT:
            {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(hwnd, &ps);
                // DirectX渲染会覆盖这里，但仍需要响应WM_PAINT
                EndPaint(hwnd, &ps);
            }
            return 0;
        }
    }
    
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

GUIMediaSink::GUIMediaSink()
    : window_handle_(nullptr)
    , window_should_close_(false)
    , sound_buffer_size_(0)
    , write_position_(0)
    , audio_clock_(0.0)
    , is_paused_(false)
    , video_width_(0)
    , video_height_(0)
    , audio_sample_rate_(0)
    , audio_channels_(0) {
}

GUIMediaSink::~GUIMediaSink() {
    close();
}

bool GUIMediaSink::initialize(int video_width, int video_height, 
                             int audio_sample_rate, int audio_channels) {
    video_width_ = video_width;
    video_height_ = video_height;
    audio_sample_rate_ = audio_sample_rate;
    audio_channels_ = audio_channels;
    
    std::cout << "=== GUI Media Sink Initializing ===" << std::endl;
    std::cout << "Video: " << video_width_ << "x" << video_height_ << std::endl;
    std::cout << "Audio: " << audio_sample_rate_ << "Hz, " << audio_channels_ << " channels" << std::endl;
    
    // 1. 创建窗口
    if (!createWindow()) {
        std::cerr << "Failed to create window" << std::endl;
        return false;
    }
    
    // 2. 初始化DirectX11
    if (!initializeDirectX11()) {
        std::cerr << "Failed to initialize DirectX11" << std::endl;
        return false;
    }
    
    // 3. 创建渲染管线
    if (!createRenderPipeline()) {
        std::cerr << "Failed to create render pipeline" << std::endl;
        return false;
    }
    
    // 4. 初始化DirectSound
    if (!initializeDirectSound()) {
        std::cerr << "Failed to initialize DirectSound" << std::endl;
        // 音频失败不是致命错误，继续运行
    }
    
    start_time_ = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== GUI Media Sink Initialized Successfully ===" << std::endl;
    std::cout << "Controls: SPACE=Pause/Resume, ESC=Exit" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    return true;
}

void GUIMediaSink::onVideoFrame(ID3D11Texture2D* rgb_texture, 
                               ID3D11ShaderResourceView* rgb_srv,
                               double timestamp, int width, int height) {
    if (!rgb_texture || !rgb_srv || window_should_close_) {
        return;
    }
    
    // 渲染纹理到屏幕
    renderTextureToScreen(rgb_texture, rgb_srv);
    
    // 简单的帧率控制（等待垂直同步）
    swap_chain_->Present(1, 0);
    
    // 打印简单的进度信息（每秒更新一次）
    static double last_print_time = 0.0;
    if (timestamp - last_print_time >= 1.0) {
        std::cout << "\\rVideo: " << std::fixed << std::setprecision(1) 
                  << timestamp << "s (" << width << "x" << height << ")" << std::flush;
        last_print_time = timestamp;
    }
}

void GUIMediaSink::onAudioFrame(const int16_t* samples, int sample_count,
                               double timestamp, int sample_rate, int channels) {
    if (!sound_buffer_ || !samples || sample_count <= 0) {
        return;
    }
    
    // 计算需要写入的字节数
    DWORD bytes_to_write = sample_count * channels * sizeof(int16_t);
    
    // 锁定音频缓冲区
    void* audio_ptr1 = nullptr;
    void* audio_ptr2 = nullptr;
    DWORD audio_bytes1 = 0;
    DWORD audio_bytes2 = 0;
    
    HRESULT hr = sound_buffer_->Lock(write_position_, bytes_to_write,
                                    &audio_ptr1, &audio_bytes1,
                                    &audio_ptr2, &audio_bytes2, 0);
    
    if (SUCCEEDED(hr)) {
        // 写入第一段数据
        if (audio_ptr1 && audio_bytes1 > 0) {
            DWORD copy_size = (bytes_to_write < audio_bytes1) ? bytes_to_write : audio_bytes1;
            memcpy(audio_ptr1, samples, copy_size);
        }
        
        // 写入第二段数据（如果缓冲区环绕）
        if (audio_ptr2 && audio_bytes2 > 0 && bytes_to_write > audio_bytes1) {
            DWORD remaining = bytes_to_write - audio_bytes1;
            DWORD copy_size = (remaining < audio_bytes2) ? remaining : audio_bytes2;
            memcpy(audio_ptr2, 
                   reinterpret_cast<const uint8_t*>(samples) + audio_bytes1,
                   copy_size);
        }
        
        sound_buffer_->Unlock(audio_ptr1, audio_bytes1, audio_ptr2, audio_bytes2);
        
        // 更新写入位置
        write_position_ = (write_position_ + bytes_to_write) % sound_buffer_size_;
    }
    
    // 更新音频时钟
    audio_clock_ = timestamp;
}

double GUIMediaSink::getCurrentTime() const {
    if (is_paused_) {
        return audio_clock_;
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    return elapsed.count() / 1000.0;
}

bool GUIMediaSink::shouldSkipFrame(double timestamp) const {
    if (is_paused_) {
        return true;
    }
    
    // 基于音频时钟的简单同步
    double diff = timestamp - audio_clock_;
    
    // 如果视频比音频落后超过100ms，跳过帧
    return diff < -0.1;
}

void GUIMediaSink::pause() {
    is_paused_ = true;
    
    if (sound_buffer_) {
        sound_buffer_->Stop();
    }
    
    std::cout << "\\n[PAUSED]" << std::endl;
}

void GUIMediaSink::resume() {
    is_paused_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();
    
    if (sound_buffer_) {
        sound_buffer_->Play(0, 0, DSBPLAY_LOOPING);
    }
    
    std::cout << "\\n[RESUMED]" << std::endl;
}

bool GUIMediaSink::isPaused() const {
    return is_paused_;
}

void GUIMediaSink::close() {
    releaseResources();
    
    if (window_handle_) {
        DestroyWindow(window_handle_);
        window_handle_ = nullptr;
    }
    
    std::cout << "\\n=== GUI Media Sink Closed ===" << std::endl;
}

bool GUIMediaSink::processWindowMessages() {
    MSG msg;
    while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            window_should_close_ = true;
        }
        
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    
    return !window_should_close_;
}

bool GUIMediaSink::createWindow() {
    // 注册窗口类
    const wchar_t* CLASS_NAME = L"MKVPlayerWindow";
    
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = CLASS_NAME;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    
    RegisterClassW(&wc);
    
    // 计算窗口大小
    int window_width = std::max(video_width_, 640);
    int window_height = std::max(video_height_, 480);
    
    // 创建窗口
    window_handle_ = CreateWindowExW(
        0,
        CLASS_NAME,
        L"MKV Player",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        window_width, window_height,
        nullptr, nullptr,
        GetModuleHandle(nullptr),
        this  // 传递this指针
    );
    
    if (!window_handle_) {
        std::cerr << "Failed to create window" << std::endl;
        return false;
    }
    
    ShowWindow(window_handle_, SW_SHOW);
    UpdateWindow(window_handle_);
    
    return true;
}

bool GUIMediaSink::initializeDirectX11() {
    // 创建DXGI交换链描述
    DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
    swap_chain_desc.BufferCount = 1;
    swap_chain_desc.BufferDesc.Width = video_width_;
    swap_chain_desc.BufferDesc.Height = video_height_;
    swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swap_chain_desc.BufferDesc.RefreshRate.Numerator = 60;
    swap_chain_desc.BufferDesc.RefreshRate.Denominator = 1;
    swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_chain_desc.OutputWindow = window_handle_;
    swap_chain_desc.SampleDesc.Count = 1;
    swap_chain_desc.SampleDesc.Quality = 0;
    swap_chain_desc.Windowed = TRUE;
    
    // 创建设备和交换链
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
        nullptr, 0,
        D3D11_SDK_VERSION,
        &swap_chain_desc,
        &swap_chain_,
        &d3d11_device_,
        nullptr,
        &d3d11_context_
    );
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device and swap chain: 0x" 
                  << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建渲染目标视图
    ComPtr<ID3D11Texture2D> back_buffer;
    hr = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)back_buffer.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer" << std::endl;
        return false;
    }
    
    hr = d3d11_device_->CreateRenderTargetView(back_buffer.Get(), nullptr, &render_target_view_);
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view" << std::endl;
        return false;
    }
    
    // 设置视口
    viewport_.TopLeftX = 0;
    viewport_.TopLeftY = 0;
    viewport_.Width = static_cast<FLOAT>(video_width_);
    viewport_.Height = static_cast<FLOAT>(video_height_);
    viewport_.MinDepth = 0.0f;
    viewport_.MaxDepth = 1.0f;
    
    d3d11_context_->RSSetViewports(1, &viewport_);
    d3d11_context_->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr);
    
    return true;
}

bool GUIMediaSink::initializeDirectSound() {
    // 创建DirectSound对象
    HRESULT hr = DirectSoundCreate8(nullptr, &dsound_, nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to create DirectSound: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 设置协作级别
    hr = dsound_->SetCooperativeLevel(window_handle_, DSSCL_PRIORITY);
    if (FAILED(hr)) {
        std::cerr << "Failed to set DirectSound cooperative level" << std::endl;
        return false;
    }
    
    // 创建音频缓冲区描述
    WAVEFORMATEX wave_format = {};
    wave_format.wFormatTag = WAVE_FORMAT_PCM;
    wave_format.nChannels = audio_channels_;
    wave_format.nSamplesPerSec = audio_sample_rate_;
    wave_format.wBitsPerSample = 16;
    wave_format.nBlockAlign = wave_format.nChannels * wave_format.wBitsPerSample / 8;
    wave_format.nAvgBytesPerSec = wave_format.nSamplesPerSec * wave_format.nBlockAlign;
    
    // 计算缓冲区大小（1秒的音频数据）
    sound_buffer_size_ = wave_format.nAvgBytesPerSec;
    
    DSBUFFERDESC buffer_desc = {};
    buffer_desc.dwSize = sizeof(DSBUFFERDESC);
    buffer_desc.dwFlags = DSBCAPS_GETCURRENTPOSITION2 | DSBCAPS_GLOBALFOCUS;
    buffer_desc.dwBufferBytes = sound_buffer_size_;
    buffer_desc.lpwfxFormat = &wave_format;
    
    // 创建音频缓冲区
    ComPtr<IDirectSoundBuffer> temp_buffer;
    hr = dsound_->CreateSoundBuffer(&buffer_desc, temp_buffer.GetAddressOf(), nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sound buffer" << std::endl;
        return false;
    }
    
    // 查询IDirectSoundBuffer8接口
    hr = temp_buffer->QueryInterface(IID_IDirectSoundBuffer8, (void**)sound_buffer_.GetAddressOf());
    
    if (FAILED(hr)) {
        std::cerr << "Failed to query IDirectSoundBuffer8 interface" << std::endl;
        return false;
    }
    
    // 开始播放音频缓冲区
    sound_buffer_->Play(0, 0, DSBPLAY_LOOPING);
    
    return true;
}

bool GUIMediaSink::createRenderPipeline() {
    // 编译顶点着色器
    ComPtr<ID3DBlob> vs_blob;
    ComPtr<ID3DBlob> error_blob;
    
    const char* vs_source = getVertexShaderSource();
    HRESULT hr = D3DCompile(vs_source, strlen(vs_source), nullptr, nullptr, nullptr,
                           "main", "vs_5_0", 0, 0, &vs_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Vertex shader compilation error: " 
                      << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    hr = d3d11_device_->CreateVertexShader(vs_blob->GetBufferPointer(), 
                                          vs_blob->GetBufferSize(), 
                                          nullptr, &vertex_shader_);
    if (FAILED(hr)) {
        return false;
    }
    
    // 创建输入布局
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };
    
    hr = d3d11_device_->CreateInputLayout(layout, 2, 
                                         vs_blob->GetBufferPointer(),
                                         vs_blob->GetBufferSize(),
                                         &input_layout_);
    
    if (FAILED(hr)) {
        return false;
    }
    
    // 编译像素着色器
    ComPtr<ID3DBlob> ps_blob;
    error_blob.Reset(); // 重置错误blob
    const char* ps_source = getPixelShaderSource();
    hr = D3DCompile(ps_source, strlen(ps_source), nullptr, nullptr, nullptr,
                   "main", "ps_5_0", 0, 0, &ps_blob, &error_blob);
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Pixel shader compilation error: " 
                      << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    
    hr = d3d11_device_->CreatePixelShader(ps_blob->GetBufferPointer(),
                                         ps_blob->GetBufferSize(),
                                         nullptr, &pixel_shader_);
    
    if (FAILED(hr)) {
        return false;
    }
    
    // 创建全屏四边形顶点缓冲区
    struct Vertex {
        float x, y;    // 位置
        float u, v;    // 纹理坐标
    };
    
    Vertex vertices[] = {
        {-1.0f, -1.0f, 0.0f, 1.0f},  // 左下
        {-1.0f,  1.0f, 0.0f, 0.0f},  // 左上
        { 1.0f, -1.0f, 1.0f, 1.0f},  // 右下
        { 1.0f,  1.0f, 1.0f, 0.0f}   // 右上
    };
    
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.Usage = D3D11_USAGE_DEFAULT;
    buffer_desc.ByteWidth = sizeof(vertices);
    buffer_desc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA init_data = {};
    init_data.pSysMem = vertices;
    
    hr = d3d11_device_->CreateBuffer(&buffer_desc, &init_data, &vertex_buffer_);
    if (FAILED(hr)) {
        return false;
    }
    
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
    return SUCCEEDED(hr);
}

void GUIMediaSink::renderTextureToScreen(ID3D11Texture2D* texture, ID3D11ShaderResourceView* srv) {
    // 清空渲染目标
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    d3d11_context_->ClearRenderTargetView(render_target_view_.Get(), clear_color);
    
    // 设置渲染管线
    d3d11_context_->IASetInputLayout(input_layout_.Get());
    d3d11_context_->VSSetShader(vertex_shader_.Get(), nullptr, 0);
    d3d11_context_->PSSetShader(pixel_shader_.Get(), nullptr, 0);
    
    // 设置顶点缓冲区
    UINT stride = sizeof(float) * 4; // x, y, u, v
    UINT offset = 0;
    d3d11_context_->IASetVertexBuffers(0, 1, vertex_buffer_.GetAddressOf(), &stride, &offset);
    d3d11_context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    
    // 设置纹理和采样器
    d3d11_context_->PSSetShaderResources(0, 1, &srv);
    d3d11_context_->PSSetSamplers(0, 1, sampler_state_.GetAddressOf());
    
    // 绘制全屏四边形
    d3d11_context_->Draw(4, 0);
}

void GUIMediaSink::releaseResources() {
    if (sound_buffer_) {
        sound_buffer_->Stop();
    }
    
    // ComPtr会处理所有COM对象的释放
    d3d11_device_.Reset();
    d3d11_context_.Reset();
    swap_chain_.Reset();
    render_target_view_.Reset();
    vertex_shader_.Reset();
    pixel_shader_.Reset();
    input_layout_.Reset();
    vertex_buffer_.Reset();
    sampler_state_.Reset();
    dsound_.Reset();
    sound_buffer_.Reset();
}

const char* GUIMediaSink::getVertexShaderSource() {
    return R"(
struct VSInput {
    float2 position : POSITION;
    float2 texcoord : TEXCOORD;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.texcoord = input.texcoord;
    return output;
}
)";
}

const char* GUIMediaSink::getPixelShaderSource() {
    return R"(
Texture2D tex : register(t0);
SamplerState samp : register(s0);

struct PSInput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

float4 main(PSInput input) : SV_TARGET {
    return tex.Sample(samp, input.texcoord);
}
)";
}