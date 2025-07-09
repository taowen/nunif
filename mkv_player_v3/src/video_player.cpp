#include "video_player.h"
#include "async_rgb_video_decoder.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// VideoPlayer实现
VideoPlayer::VideoPlayer() 
    : render_target_view_(nullptr)
    , device_context_(nullptr)
    , vertex_shader_(nullptr)
    , pixel_shader_(nullptr)
    , input_layout_(nullptr)
    , texture_sampler_(nullptr)
    , vertex_buffer_(nullptr)
    , video_decoder_(std::make_unique<AsyncRgbVideoDecoder>())
    , render_count_(0)
    , last_stats_time_(std::chrono::high_resolution_clock::now())
{
}

VideoPlayer::~VideoPlayer() {
    cleanupShaders();
    
    // 释放内部创建的D3D11资源
    if (device_context_) {
        device_context_->Release();
        device_context_ = nullptr;
    }
    
    // 不负责释放外部传入的D3D11资源
}

bool VideoPlayer::initialize(ID3D11RenderTargetView* render_target_view) {
    if (!render_target_view) {
        return false;
    }
    
    render_target_view_ = render_target_view;
    
    // 等待解码器初始化，获取共享的D3D11设备
    // 这里可能需要等待解码器准备好
    int retry_count = 0;
    while (retry_count < 10 && !getD3D11Device()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        retry_count++;
    }
    
    ID3D11Device* shared_device = getD3D11Device();
    if (!shared_device) {
        std::cerr << "Failed to get D3D11 device from video decoder" << std::endl;
        return false;
    }
    
    // 为VideoPlayer创建专用的即时设备上下文
    shared_device->GetImmediateContext(&device_context_);
    if (!device_context_) {
        std::cerr << "Failed to get immediate context from device" << std::endl;
        return false;
    }
    
    // 初始化Shader渲染管线
    if (!initializeShaders()) {
        std::cerr << "Failed to initialize shaders" << std::endl;
        return false;
    }
    
    std::cout << "VideoPlayer initialized successfully with render target: " << render_target_view << std::endl;
    std::cout << "VideoPlayer created own device context: " << device_context_ << std::endl;
    return true;
}

bool VideoPlayer::open(const std::string& filepath) {
    if (!video_decoder_) {
        return false;
    }
    
    return video_decoder_->open(filepath);
}

void VideoPlayer::close() {
    if (video_decoder_) {
        video_decoder_->close();
    }
}

ID3D11Device* VideoPlayer::getD3D11Device() const {
    return video_decoder_->getD3D11Device();
}

ID3D11DeviceContext* VideoPlayer::getD3D11Context() const {
    return device_context_;
}

void VideoPlayer::onTimer() {
    // 如果未初始化，直接返回
    if (!render_target_view_ || !video_decoder_) {
        return;
    }
    
    // 获取视频帧（阻塞调用）
    AsyncRgbVideoDecoder::DecodedFrame frame;
    if (video_decoder_->readNextFrame(frame) && frame.is_valid) {
        // 调试输出视频帧信息
        std::cout << "获取视频帧 #" << render_count_ 
                  << " - RGB纹理: " << frame.rgb_frame.rgb_texture.Get()
                  << " - SRV: " << frame.rgb_frame.rgb_srv.Get() << std::endl;
        
        // 渲染真实视频纹理
        renderVideoTexture(frame.rgb_frame.rgb_texture.Get(), frame.rgb_frame.rgb_srv.Get());
        
        std::cout << "渲染视频帧 #" << render_count_ << std::endl;
    } else {
        std::cout << "无法获取视频帧 #" << render_count_ << " (readNextFrame失败或frame无效)" << std::endl;
    }
    
    render_count_++;
    
    // 每秒统计一次
    auto now = std::chrono::high_resolution_clock::now();
    auto stats_elapsed = std::chrono::duration<double>(now - last_stats_time_).count();
    if (stats_elapsed >= 1.0) {
        std::cout << "渲染统计: " << render_count_ << " 帧/秒" << std::endl;
        render_count_ = 0;
        last_stats_time_ = now;
    }
}

void VideoPlayer::renderVideoTexture(ID3D11Texture2D* texture, ID3D11ShaderResourceView* srv) {
    if (!texture || !srv || !vertex_shader_ || !pixel_shader_) {
        return;
    }
    
    // 获取渲染目标描述符来设置视口
    ID3D11Resource* render_target_resource = nullptr;
    render_target_view_->GetResource(&render_target_resource);
    
    ID3D11Texture2D* render_target_texture = nullptr;
    HRESULT hr = render_target_resource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&render_target_texture);
    render_target_resource->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to get render target texture" << std::endl;
        return;
    }
    
    D3D11_TEXTURE2D_DESC rt_desc;
    render_target_texture->GetDesc(&rt_desc);
    render_target_texture->Release();
    
    // 设置视口
    D3D11_VIEWPORT viewport = {};
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
    viewport.Width = (float)rt_desc.Width;
    viewport.Height = (float)rt_desc.Height;
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    
    // 清屏
    float clear_color[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    getD3D11Context()->ClearRenderTargetView(render_target_view_, clear_color);
    
    // 设置渲染管线状态
    getD3D11Context()->RSSetViewports(1, &viewport);
    getD3D11Context()->OMSetRenderTargets(1, &render_target_view_, nullptr);
    
    // 设置Shader
    getD3D11Context()->VSSetShader(vertex_shader_, nullptr, 0);
    getD3D11Context()->PSSetShader(pixel_shader_, nullptr, 0);
    getD3D11Context()->IASetInputLayout(nullptr);  // 全屏三角形不需要input layout
    getD3D11Context()->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    
    // 绑定视频纹理到Pixel Shader
    getD3D11Context()->PSSetShaderResources(0, 1, &srv);
    getD3D11Context()->PSSetSamplers(0, 1, &texture_sampler_);
    
    // 渲染全屏三角形（无需vertex buffer）
    getD3D11Context()->Draw(3, 0);  // 3个顶点组成一个大三角形
    
    // 解绑资源
    ID3D11ShaderResourceView* nullSRV = nullptr;
    getD3D11Context()->PSSetShaderResources(0, 1, &nullSRV);
    
    std::cout << "Shader渲染视频纹理到渲染目标: " << rt_desc.Width << "x" << rt_desc.Height << std::endl;
}

bool VideoPlayer::initializeShaders() {
    // 简单的Vertex Shader - 全屏四边形
    const char* vertexShaderSource = R"(
        struct VS_OUTPUT {
            float4 pos : SV_POSITION;
            float2 tex : TEXCOORD0;
        };
        
        VS_OUTPUT main(uint vertexId : SV_VertexID) {
            VS_OUTPUT output;
            
            // 使用vertex ID生成全屏三角形
            float2 texcoord = float2((vertexId << 1) & 2, vertexId & 2);
            output.pos = float4(texcoord * float2(2, -2) + float2(-1, 1), 0, 1);
            output.tex = texcoord;
            
            return output;
        }
    )";
    
    // 简单的Pixel Shader - 采样纹理
    const char* pixelShaderSource = R"(
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
    
    // 编译Vertex Shader
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr, nullptr, nullptr,
                            "main", "vs_4_0", 0, 0, &vsBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Vertex Shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return false;
    }
    
    hr = getD3D11Device()->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vertex_shader_);
    if (FAILED(hr)) {
        vsBlob->Release();
        std::cerr << "Failed to create vertex shader" << std::endl;
        return false;
    }
    
    // 对于全屏三角形，我们不需要输入布局，设置为nullptr
    input_layout_ = nullptr;
    vsBlob->Release();
    
    // 编译Pixel Shader
    ID3DBlob* psBlob = nullptr;
    hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr, nullptr, nullptr,
                    "main", "ps_4_0", 0, 0, &psBlob, &errorBlob);
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Pixel Shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return false;
    }
    
    hr = getD3D11Device()->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pixel_shader_);
    psBlob->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create pixel shader" << std::endl;
        return false;
    }
    
    // 创建纹理采样器
    D3D11_SAMPLER_DESC samplerDesc = {};
    samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    samplerDesc.MinLOD = 0;
    samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
    
    hr = getD3D11Device()->CreateSamplerState(&samplerDesc, &texture_sampler_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create sampler state" << std::endl;
        return false;
    }
    
    std::cout << "Shader渲染管线初始化成功" << std::endl;
    return true;
}

void VideoPlayer::cleanupShaders() {
    if (vertex_shader_) { vertex_shader_->Release(); vertex_shader_ = nullptr; }
    if (pixel_shader_) { pixel_shader_->Release(); pixel_shader_ = nullptr; }
    if (input_layout_) { input_layout_->Release(); input_layout_ = nullptr; }
    if (texture_sampler_) { texture_sampler_->Release(); texture_sampler_ = nullptr; }
    if (vertex_buffer_) { vertex_buffer_->Release(); vertex_buffer_ = nullptr; }
}

