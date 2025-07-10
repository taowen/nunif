#include "video_player.h"
#include <d3dcompiler.h>
#include <iostream>
#include <array>

VideoPlayer::VideoPlayer() 
    : device_(nullptr), context_(nullptr), video_width_(0), video_height_(0), has_new_frame_(false) {}

VideoPlayer::~VideoPlayer() {
    close();
}

bool VideoPlayer::open(const std::string& filepath) {
    close();
    
    decoder_ = std::make_unique<RgbVideoDecoder>();
    if (!decoder_->open(filepath)) {
        return false;
    }
    
    device_ = decoder_->getD3D11Device();
    context_ = decoder_->getD3D11Context();
    if (!device_ || !context_) {
        return false;
    }
    
    // Read first frame to get dimensions
    RgbVideoDecoder::DecodedFrame first_frame;
    if (!decoder_->readNextFrame(first_frame)) {
        return false;
    }
    video_width_ = first_frame.rgb_frame.width;
    video_height_ = first_frame.rgb_frame.height;
    
    // Seek back to start
    decoder_->seekToTime(0.0);
    
    if (!initializeShaders() || !createRenderTarget(video_width_, video_height_) || !createQuad()) {
        close();
        return false;
    }
    
    return true;
}

void VideoPlayer::close() {
    decoder_.reset();
    render_target_view_.Reset();
    render_texture_.Reset();
    vertex_shader_.Reset();
    pixel_shader_.Reset();
    input_layout_.Reset();
    vertex_buffer_.Reset();
    constant_buffer_.Reset();
    sampler_state_.Reset();
    device_ = nullptr;
    context_ = nullptr;
    video_width_ = 0;
    video_height_ = 0;
    has_new_frame_ = false;
}

bool VideoPlayer::onTimer() {
    if (!decoder_) return false;
    
    RgbVideoDecoder::DecodedFrame frame;
    if (decoder_->readNextFrame(frame)) {
        renderFrame(frame.rgb_frame);
        has_new_frame_ = true;
        return true;
    }
    return false;
}

bool VideoPlayer::initializeShaders() {
    const std::string vs_code = R"(
        struct VS_INPUT {
            float4 pos : POSITION;
            float2 tex : TEXCOORD0;
        };
        struct PS_INPUT {
            float4 pos : SV_POSITION;
            float2 tex : TEXCOORD0;
        };
        PS_INPUT main(VS_INPUT input) {
            PS_INPUT output;
            output.pos = input.pos;
            output.tex = input.tex;
            return output;
        }
    )";
    
    const std::string ps_code = R"(
        Texture2D tex : register(t0);
        SamplerState sam : register(s0);
        float4 main(float4 pos : SV_POSITION, float2 texcoord : TEXCOORD0) : SV_TARGET {
            return tex.Sample(sam, texcoord);
        }
    )";
    
    ComPtr<ID3DBlob> vs_blob, error_blob;
    HRESULT hr = D3DCompile(vs_code.c_str(), vs_code.length(), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, vs_blob.GetAddressOf(), error_blob.GetAddressOf());
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "VS compile error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    hr = device_->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, vertex_shader_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    ComPtr<ID3DBlob> ps_blob;
    hr = D3DCompile(ps_code.c_str(), ps_code.length(), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, ps_blob.GetAddressOf(), error_blob.GetAddressOf());
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "PS compile error: " << (char*)error_blob->GetBufferPointer() << std::endl;
        }
        return false;
    }
    hr = device_->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, pixel_shader_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    hr = device_->CreateInputLayout(layout, ARRAYSIZE(layout), vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), input_layout_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    D3D11_SAMPLER_DESC sampler_desc = {};
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler_desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler_desc.MinLOD = 0;
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = device_->CreateSamplerState(&sampler_desc, sampler_state_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    return true;
}

bool VideoPlayer::createRenderTarget(int width, int height) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    
    HRESULT hr = device_->CreateTexture2D(&desc, nullptr, render_texture_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    D3D11_RENDER_TARGET_VIEW_DESC rtv_desc = {};
    rtv_desc.Format = desc.Format;
    rtv_desc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    hr = device_->CreateRenderTargetView(render_texture_.Get(), &rtv_desc, render_target_view_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    return true;
}

bool VideoPlayer::createQuad() {
    struct Vertex {
        float x, y, z;
        float u, v;
    };
    Vertex vertices[] = {
        { -1.0f, -1.0f, 0.0f, 0.0f, 1.0f },
        { -1.0f,  1.0f, 0.0f, 0.0f, 0.0f },
        {  1.0f, -1.0f, 0.0f, 1.0f, 1.0f },
        {  1.0f,  1.0f, 0.0f, 1.0f, 0.0f }
    };
    
    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(vertices);
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA init = {};
    init.pSysMem = vertices;
    
    HRESULT hr = device_->CreateBuffer(&bd, &init, vertex_buffer_.GetAddressOf());
    if (FAILED(hr)) return false;
    
    return true;
}

void VideoPlayer::renderFrame(const RgbVideoDecoder::RgbFrame& rgb_frame) {
    if (!context_) return;
    
    FLOAT clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    context_->ClearRenderTargetView(render_target_view_.Get(), clear_color);
    
    context_->OMSetRenderTargets(1, render_target_view_.GetAddressOf(), nullptr);
    
    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(video_width_);
    vp.Height = static_cast<float>(video_height_);
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    context_->RSSetViewports(1, &vp);
    
    context_->VSSetShader(vertex_shader_.Get(), nullptr, 0);
    context_->PSSetShader(pixel_shader_.Get(), nullptr, 0);
    context_->IASetInputLayout(input_layout_.Get());
    context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    
    UINT stride = sizeof(float) * 5;
    UINT offset = 0;
    context_->IASetVertexBuffers(0, 1, vertex_buffer_.GetAddressOf(), &stride, &offset);
    
    context_->PSSetShaderResources(0, 1, rgb_frame.rgb_srv.GetAddressOf());
    context_->PSSetSamplers(0, 1, sampler_state_.GetAddressOf());
    
    context_->Draw(4, 0);
    
    ID3D11ShaderResourceView* null_srv = nullptr;
    context_->PSSetShaderResources(0, 1, &null_srv);
}
