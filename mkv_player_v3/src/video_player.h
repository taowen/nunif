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
    VideoPlayer(ID3D11Device* existing_device, ID3D11DeviceContext* existing_context);
    ~VideoPlayer();

    bool open(const std::string& filepath);
    void close();
    
    bool onTimer();  // Called periodically to update and render frame
    
    // Getters for testing
    ID3D11Texture2D* getRenderTexture() const { return render_texture_.Get(); }
    ID3D11Device* getDevice() const { return device_.Get(); }
    ID3D11DeviceContext* getContext() const { return context_.Get(); }

private:
    std::unique_ptr<RgbVideoDecoder> decoder_;
    
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    
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
    
    void initialize();
    bool initializeDevice();
    bool initializeShaders();
    bool createQuad();
    bool createRenderTarget(int width, int height);
    void renderFrame(const RgbVideoDecoder::RgbFrame& rgb_frame);
};

// Helper function for rendering
HRESULT RenderRgbFrame(ID3D11DeviceContext* context, ID3D11RenderTargetView* rtv, const RgbVideoDecoder::RgbFrame& rgb_frame, int video_width, int video_height, ID3D11VertexShader* vertex_shader, ID3D11PixelShader* pixel_shader, ID3D11InputLayout* input_layout, ID3D11Buffer* vertex_buffer, ID3D11SamplerState* sampler_state);
