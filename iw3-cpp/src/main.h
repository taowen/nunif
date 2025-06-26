#pragma once

#include <d3d11.h>
#include <queue>
#include <mutex>
#include <condition_variable>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavcodec/avcodec.h>
}

struct ColorSpaceInfo {
    AVColorSpace color_space = AVCOL_SPC_UNSPECIFIED;
    AVColorPrimaries color_primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
    AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
    int bit_depth = 8;
    bool is_hdr = false;
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;
};

// Frame data structure for queue communication
struct DecodedFrame {
    AVFrame* frame;
    bool is_end_signal;
    
    DecodedFrame() : frame(nullptr), is_end_signal(false) {}
    DecodedFrame(AVFrame* f) : frame(f), is_end_signal(false) {}
    static DecodedFrame end_signal() {
        DecodedFrame data;
        data.is_end_signal = true;
        return data;
    }
};

// Thread-safe queue for frame communication
class FrameQueue {
private:
    std::queue<DecodedFrame> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;
    
public:
    FrameQueue(size_t max_size = 10) : max_size_(max_size) {}
    
    void push(const DecodedFrame& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(data);
        condition_.notify_one();
    }
    
    DecodedFrame pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        DecodedFrame data = queue_.front();
        queue_.pop();
        condition_.notify_one();
        return data;
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
}; 