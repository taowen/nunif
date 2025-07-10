#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <memory>
#include "rgb_video_decoder.h"

using Microsoft::WRL::ComPtr;

class VideoPlayer {
public:
    VideoPlayer();
    ~VideoPlayer();

    bool open(const std::string& filepath);
    void close();
    
    bool onTimer();  // Called periodically to update and render frame
    
    // Getters for testing
    ID3D11Texture2D* getRenderTexture() const { return render_texture_.Get(); }
    ID3D11Device* getDevice() const { return device_; }
    ID3D11DeviceContext* getContext() const { return context_; }

private:
    std::unique_ptr<RgbVideoDecoder> decoder_;
    
    ID3D11Device* device_;
    ID3D11DeviceContext* context_;
    
    ComPtr<ID3D11Texture2D> render_texture_;
    ComPtr<ID3D11RenderTargetView> render_target_view_;
    ComPtr<ID3D11VertexShader> vertex_shader_;
    ComPtr<ID3D11PixelShader> pixel_shader_;
    ComPtr<ID3D11InputLayout> input_layout_;
    ComPtr<ID3D11Buffer> vertex_buffer_;
    ComPtr<ID3D11Buffer> constant_buffer_;
    ComPtr<ID3D11SamplerState> sampler_state_;
    
    int video_width_;
    int video_height_;
    bool has_new_frame_;
    
    bool initializeShaders();
    bool createRenderTarget(int width, int height);
    bool createQuad();
    void renderFrame(const RgbVideoDecoder::RgbFrame& rgb_frame);
};
