#pragma once

#include "media_sink.h"
#include "frame_queue.h"
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <memory>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>

using Microsoft::WRL::ComPtr;

/**
 * DirectX11 GUI媒体播放器
 * 实现IMediaSink接口，提供实时视频播放和音频播放
 */
// 前向声明
class MediaPlayer;

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
    void setVideoDimensions(int width, int height); // 设置视频尺寸
    bool createWindow(const std::string& title = "Video Player");
    void showWindow();
    bool processMessages(); // 返回false表示应该退出
    void present();
    void renderFrame();  // 公开渲染方法用于测试
    
    // 多线程播放控制
    bool startPlayback(MediaPlayer* player, const std::string& filepath);
    void stopPlayback();
    bool isPlaybackRunning() const;
    
    // 测试模式相关
    void setTestMode(bool enabled, int auto_close_ms = 1000);
    bool isTestMode() const { return test_mode_; }

protected:
    // 媒体信息 (protected for testing)
    int video_width_;
    int video_height_;
    int audio_sample_rate_;
    int audio_channels_;
    
    // 播放状态 (protected for testing)
    bool is_paused_;
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point pause_time_;
    double paused_duration_;
    
    // 测试模式 (protected for testing)
    bool test_mode_;
    int auto_close_ms_;
    std::chrono::steady_clock::time_point test_start_time_;
    
private:
    // 窗口相关
    HWND window_handle_;
    bool should_close_;
    std::string window_title_;
    
    // DirectX11相关
    ComPtr<ID3D11Device> d3d11_device_;
    ComPtr<ID3D11DeviceContext> d3d11_context_;
    ComPtr<IDXGISwapChain> swap_chain_;
    ComPtr<ID3D11RenderTargetView> render_target_view_;
    
    // 视频渲染相关
    ComPtr<ID3D11Texture2D> video_texture_;
    ComPtr<ID3D11ShaderResourceView> video_srv_;
    ComPtr<ID3D11Buffer> vertex_buffer_;
    ComPtr<ID3D11Buffer> index_buffer_;
    ComPtr<ID3D11VertexShader> vertex_shader_;
    ComPtr<ID3D11PixelShader> pixel_shader_;
    ComPtr<ID3D11InputLayout> input_layout_;
    ComPtr<ID3D11SamplerState> sampler_state_;
    
    // 最新帧信息
    double last_video_timestamp_;
    bool has_new_frame_;
    
    // 多线程相关
    std::unique_ptr<FrameQueue> frame_queue_;
    std::unique_ptr<std::thread> decoder_thread_;
    std::atomic<bool> should_stop_decoder_;
    MediaPlayer* media_player_; // 不拥有所有权
    std::string current_filepath_;
    
    // 私有方法
    bool createWindowClass();
    bool initializeDirectX11();
    bool createRenderTargets();
    bool createShaders();
    bool createGeometry();
    void updateVideoTexture(ID3D11Texture2D* source_texture);
    void cleanup();
    
    // 多线程解码循环
    void decoderThreadLoop();
    
    // 渲染线程消费帧
    void processFrameQueue();
    
    // 窗口消息处理
    static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
    LRESULT handleMessage(UINT uMsg, WPARAM wParam, LPARAM lParam);
};