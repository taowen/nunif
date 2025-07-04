#pragma once

#include "media_sink.h"
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <Windows.h>
#include <dsound.h>
#include <chrono>

/**
 * GUI媒体播放Sink实现
 * 使用DirectX11渲染视频，DirectSound播放音频
 */
class GUIMediaSink : public IMediaSink {
public:
    GUIMediaSink();
    ~GUIMediaSink() override;

    // IMediaSink接口实现
    bool initialize(int video_width, int video_height, 
                   int audio_sample_rate, int audio_channels) override;
    
    void onVideoFrame(ID3D11Texture2D* rgb_texture, 
                     ID3D11ShaderResourceView* rgb_srv,
                     double timestamp, int width, int height) override;
    
    void onAudioFrame(const int16_t* samples, int sample_count,
                     double timestamp, int sample_rate, int channels) override;
    
    double getCurrentTime() const override;
    bool shouldSkipFrame(double timestamp) const override;
    
    void pause() override;
    void resume() override;
    bool isPaused() const override;
    
    void close() override;

    // GUI特有方法
    HWND getWindowHandle() const { return window_handle_; }
    bool processWindowMessages(); // 返回false表示应该退出

private:
    // 窗口相关
    HWND window_handle_;
    bool window_should_close_;
    
    // DirectX11渲染组件
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    IDXGISwapChain* swap_chain_;
    ID3D11RenderTargetView* render_target_view_;
    
    // 渲染管线
    ID3D11VertexShader* vertex_shader_;
    ID3D11PixelShader* pixel_shader_;
    ID3D11InputLayout* input_layout_;
    ID3D11Buffer* vertex_buffer_;
    ID3D11SamplerState* sampler_state_;
    D3D11_VIEWPORT viewport_;
    
    // DirectSound音频组件
    IDirectSound8* dsound_;
    IDirectSoundBuffer8* sound_buffer_;
    DWORD sound_buffer_size_;
    DWORD write_position_;
    
    // 时间同步
    std::chrono::high_resolution_clock::time_point start_time_;
    double audio_clock_;
    bool is_paused_;
    
    // 视频信息
    int video_width_;
    int video_height_;
    int audio_sample_rate_;
    int audio_channels_;
    
    // 内部方法
    bool createWindow();
    bool initializeDirectX11();
    bool initializeDirectSound();
    bool createRenderPipeline();
    void renderTextureToScreen(ID3D11Texture2D* texture, ID3D11ShaderResourceView* srv);
    void releaseResources();
    
    // 窗口过程
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    
    // 着色器源码
    static const char* getVertexShaderSource();
    static const char* getPixelShaderSource();
};