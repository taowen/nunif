#include "video_player.h"
#include "async_rgb_video_decoder.h"
#include <iostream>

// VideoPlayer实现
VideoPlayer::VideoPlayer() 
    : device_(nullptr)
    , context_(nullptr)
    , swap_chain_(nullptr)
    , render_target_view_(nullptr)
    , video_decoder_(std::make_unique<AsyncRgbVideoDecoder>())
    , render_count_(0)
    , last_stats_time_(std::chrono::high_resolution_clock::now())
{
}

VideoPlayer::~VideoPlayer() {
    // 不负责释放外部传入的D3D11资源
}

bool VideoPlayer::initialize(ID3D11Device* device, 
                           ID3D11DeviceContext* context, 
                           ID3D11RenderTargetView* render_target_view,
                           IDXGISwapChain* swap_chain) {
    if (!device || !context || !render_target_view) {
        return false;
    }
    
    device_ = device;
    context_ = context;
    render_target_view_ = render_target_view;
    swap_chain_ = swap_chain;
    
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

void VideoPlayer::onTimer() {
    // 如果未初始化，直接返回
    if (!render_target_view_ || !video_decoder_) {
        return;
    }
    
    // 获取视频帧（阻塞调用）
    AsyncRgbVideoDecoder::DecodedFrame frame;
    if (video_decoder_->readNextFrame(frame) && frame.is_valid) {
        // 渲染真实视频纹理
        renderVideoTexture(frame.rgb_frame.rgb_texture.Get(), frame.rgb_frame.rgb_srv.Get());
        
        std::cout << "渲染视频帧 #" << render_count_ << std::endl;
    }
    
    // 如果有swap_chain则Present到屏幕
    if (swap_chain_) {
        HRESULT hr = swap_chain_->Present(0, 0);
        if (FAILED(hr)) {
            std::cerr << "Present failed: " << std::hex << hr << std::endl;
            return;
        }
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
    // TODO: 实现真正的纹理渲染
    // 目前先用简单的清屏替代
    float clear_color[4] = { 0.0f, 1.0f, 0.0f, 1.0f };  // 绿色表示有视频帧
    context_->ClearRenderTargetView(render_target_view_, clear_color);
    context_->OMSetRenderTargets(1, &render_target_view_, nullptr);
}

