#include "video_player.h"
#include <iostream>

// 假的视频信号源 - 三色轮替24fps
class FakeVideoSignal {
public:
    struct Frame {
        float color[3];  // RGB
        double timestamp;
    };
    
    FakeVideoSignal() : frame_count_(0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    bool getNextFrame(Frame& frame) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        
        // 24fps = 1/24 = 0.041667秒每帧
        double frame_duration = 1.0 / 24.0;
        int target_frame = static_cast<int>(elapsed / frame_duration);
        
        if (target_frame <= frame_count_) {
            return false;  // 还没到下一帧时间
        }
        
        frame_count_ = target_frame;
        frame.timestamp = frame_count_ * frame_duration;
        
        // 三色轮替：红->绿->蓝->红...
        int color_index = frame_count_ % 3;
        frame.color[0] = (color_index == 0) ? 1.0f : 0.0f;  // R
        frame.color[1] = (color_index == 1) ? 1.0f : 0.0f;  // G
        frame.color[2] = (color_index == 2) ? 1.0f : 0.0f;  // B
        
        return true;
    }
    
    int getFrameCount() const { return frame_count_; }
    
private:
    int frame_count_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

// 帧率转换器 - 24fps到60Hz
class FrameRateConverter {
public:
    FrameRateConverter() : frame_time_accumulator_(0.0), current_display_count_(0) {}
    
    bool shouldDisplayFrame(const FakeVideoSignal::Frame&) {
        // 24fps → 60Hz: 每个视频帧需要显示 60/24 = 2.5 次
        double source_fps = 24.0;
        double target_fps = 60.0;
        double frame_duration = 1.0 / source_fps;
        double display_interval = 1.0 / target_fps;
        
        if (current_display_count_ == 0) {
            // 新的视频帧，计算需要显示多少次
            frame_time_accumulator_ += frame_duration;
            required_display_count_ = static_cast<int>(frame_time_accumulator_ / display_interval + 0.5);
            frame_time_accumulator_ -= required_display_count_ * display_interval;
        }
        
        current_display_count_++;
        
        if (current_display_count_ >= required_display_count_) {
            current_display_count_ = 0;
            return false;  // 这个视频帧显示完了，需要新帧
        }
        
        return true;  // 继续显示当前帧
    }
    
private:
    double frame_time_accumulator_;
    int current_display_count_;
    int required_display_count_;
};

// VideoPlayer实现
VideoPlayer::VideoPlayer() 
    : hwnd_(nullptr)
    , device_(nullptr)
    , context_(nullptr)
    , swap_chain_(nullptr)
    , render_target_view_(nullptr)
    , video_signal_(std::make_unique<FakeVideoSignal>())
    , frame_converter_(std::make_unique<FrameRateConverter>())
    , has_frame_(false)
    , render_count_(0)
    , last_stats_time_(std::chrono::high_resolution_clock::now())
{
}

VideoPlayer::~VideoPlayer() {
    cleanup();
}

bool VideoPlayer::initialize(HWND hwnd) {
    hwnd_ = hwnd;
    
    // 创建D3D11设备和交换链
    DXGI_SWAP_CHAIN_DESC scd = {};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = 800;
    scd.BufferDesc.Height = 600;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferDesc.RefreshRate.Numerator = 60;
    scd.BufferDesc.RefreshRate.Denominator = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
    scd.SampleDesc.Quality = 0;
    scd.Windowed = TRUE;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 
        D3D11_CREATE_DEVICE_DEBUG,
        nullptr, 0, D3D11_SDK_VERSION,
        &scd, &swap_chain_, &device_, nullptr, &context_);
        
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建渲染目标
    ID3D11Texture2D* back_buffer = nullptr;
    hr = swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to get back buffer: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = device_->CreateRenderTargetView(back_buffer, nullptr, &render_target_view_);
    back_buffer->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create render target view: " << std::hex << hr << std::endl;
        return false;
    }
    
    // 设置视口
    D3D11_VIEWPORT vp = {};
    vp.Width = 800.0f;
    vp.Height = 600.0f;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    context_->RSSetViewports(1, &vp);
    
    std::cout << "VideoPlayer初始化成功 - 支持onTimer()回调架构" << std::endl;
    return true;
}

void VideoPlayer::onTimer() {
    // 检查是否需要新的视频帧
    FakeVideoSignal::Frame temp_frame;
    if (!has_frame_ || !frame_converter_->shouldDisplayFrame(temp_frame)) {
        if (video_signal_->getNextFrame(temp_frame)) {
            // 转换为VideoPlayer::Frame格式
            current_frame_.color[0] = temp_frame.color[0];
            current_frame_.color[1] = temp_frame.color[1];
            current_frame_.color[2] = temp_frame.color[2];
            current_frame_.timestamp = temp_frame.timestamp;
            has_frame_ = true;
            std::cout << "新视频帧 #" << video_signal_->getFrameCount() 
                      << " - 颜色: RGB(" << current_frame_.color[0] << "," 
                      << current_frame_.color[1] << "," << current_frame_.color[2] << ")" << std::endl;
        }
    }
    
    // 渲染当前帧
    if (has_frame_) {
        renderFrame(current_frame_);
    }
    
    // 立即Present，不等待VSync
    HRESULT hr = swap_chain_->Present(0, 0);
    if (FAILED(hr)) {
        std::cerr << "Present failed: " << std::hex << hr << std::endl;
        return;
    }
    
    render_count_++;
    
    // 每秒统计一次
    auto now = std::chrono::high_resolution_clock::now();
    auto stats_elapsed = std::chrono::duration<double>(now - last_stats_time_).count();
    if (stats_elapsed >= 1.0) {
        std::cout << "渲染统计: " << render_count_ << " 帧/秒, 视频帧: " 
                  << video_signal_->getFrameCount() << std::endl;
        render_count_ = 0;
        last_stats_time_ = now;
    }
}

void VideoPlayer::renderFrame(const Frame& frame) {
    // 清屏为指定颜色
    float clear_color[4] = { frame.color[0], frame.color[1], frame.color[2], 1.0f };
    context_->ClearRenderTargetView(render_target_view_, clear_color);
    context_->OMSetRenderTargets(1, &render_target_view_, nullptr);
}

void VideoPlayer::cleanup() {
    if (render_target_view_) {
        render_target_view_->Release();
        render_target_view_ = nullptr;
    }
    if (swap_chain_) {
        swap_chain_->Release();
        swap_chain_ = nullptr;
    }
    if (context_) {
        context_->Release();
        context_ = nullptr;
    }
    if (device_) {
        device_->Release();
        device_ = nullptr;
    }
}