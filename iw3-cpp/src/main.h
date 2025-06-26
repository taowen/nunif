#pragma once

#include <d3d11.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime_api.h>

#include "ffmpeg_wrapper.h"

struct ColorSpaceInfo {
    AVColorSpace color_space = AVCOL_SPC_UNSPECIFIED;
    AVColorPrimaries color_primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
    AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
    int bit_depth = 8;
    bool is_hdr = false;
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;
};

// Color space conversion constants
struct ConversionConstants {
    float matrix[16];  // 4x4 color matrix
    float luma_coeffs[4];
    float chroma_coeffs[4];
    float offset[4];
    int input_format;
    int color_space;
    int bit_depth;
    int is_hdr;
};

// Decoder state structure - contains all decode-related state
struct DecoderState {
    // FFmpeg decoder resources
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_index = -1;
    
    // Thread synchronization
    std::atomic<bool> decode_finished_{false};
    
    // Color space detection state
    ColorSpaceInfo video_color_info;
    bool color_info_detected = false;
    
    // Cleanup method
    void cleanup() {
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
            codec_ctx = nullptr;
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
            format_ctx = nullptr;
        }
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
            hw_device_ctx = nullptr;
        }
        
        decode_finished_ = false;
        color_info_detected = false;
        video_stream_index = -1;
    }
};

// Color conversion state structure - dedicated to color conversion only
struct ColorConversionState {
    // DirectX shader resources
    ID3D11ComputeShader* color_conversion_shader = nullptr;
    ID3D11Buffer* conversion_constants_buffer = nullptr;
    ID3D11ShaderResourceView* input_srv_y = nullptr;
    ID3D11ShaderResourceView* input_srv_uv = nullptr;
    ID3D11Texture2D* intermediate_texture = nullptr;
    
    std::atomic<bool> process_finished_{false};
    
    // Cleanup method
    void cleanup() {
        if (intermediate_texture) {
            intermediate_texture->Release();
            intermediate_texture = nullptr;
        }
        
        if (input_srv_y) {
            input_srv_y->Release();
            input_srv_y = nullptr;
        }
        if (input_srv_uv) {
            input_srv_uv->Release();
            input_srv_uv = nullptr;
        }
        if (conversion_constants_buffer) {
            conversion_constants_buffer->Release();
            conversion_constants_buffer = nullptr;
        }
        if (color_conversion_shader) {
            color_conversion_shader->Release();
            color_conversion_shader = nullptr;
        }
        process_finished_ = false;
    }
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

// Converted frame data structure for AI processing
struct ColorConvertedFrame {
    ID3D11Texture2D* texture;
    UINT width;
    UINT height;
    bool is_end_signal;

    ColorConvertedFrame() : texture(nullptr), width(0), height(0), is_end_signal(false) {}
    ColorConvertedFrame(ID3D11Texture2D* tex, UINT w, UINT h)
        : texture(tex), width(w), height(h), is_end_signal(false) {
        if (texture) {
            texture->AddRef();
        }
    }

    ~ColorConvertedFrame() {
        if (texture && !is_end_signal) {
            texture->Release();
        }
    }

    // Move constructor
    ColorConvertedFrame(ColorConvertedFrame&& other) noexcept
        : texture(other.texture), width(other.width), height(other.height),
          is_end_signal(other.is_end_signal) {
        other.texture = nullptr;
    }

    // Move assignment
    ColorConvertedFrame& operator=(ColorConvertedFrame&& other) noexcept {
        if (this != &other) {
            if (texture && !is_end_signal) {
                texture->Release();
            }
            texture = other.texture;
            width = other.width;
            height = other.height;
            is_end_signal = other.is_end_signal;
            other.texture = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and assignment
    ColorConvertedFrame(const ColorConvertedFrame&) = delete;
    ColorConvertedFrame& operator=(const ColorConvertedFrame&) = delete;

    static ColorConvertedFrame end_signal() {
        ColorConvertedFrame data;
        data.is_end_signal = true;
        return data;
    }
};

// Thread-safe queue for frame communication
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t max_size_;
    
public:
    ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}
    
    void push(T&& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return queue_.size() < max_size_; });
        queue_.push(std::move(data));
        condition_.notify_one();
    }
    
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return !queue_.empty(); });
        T data = std::move(queue_.front());
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

// Type aliases for specific queue types
using DecodedFrameQueue = ThreadSafeQueue<DecodedFrame>;
using ColorConvertedFrameQueue = ThreadSafeQueue<ColorConvertedFrame>; 