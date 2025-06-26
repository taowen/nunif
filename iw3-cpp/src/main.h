#pragma once

#include <d3d11.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime_api.h>

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

// Color conversion state structure
struct ColorConversionState {
    // DirectX shader resources
    ID3D11ComputeShader* color_conversion_shader = nullptr;
    ID3D11Buffer* conversion_constants_buffer = nullptr;
    ID3D11ShaderResourceView* input_srv_y = nullptr;
    ID3D11ShaderResourceView* input_srv_uv = nullptr;
    ID3D11UnorderedAccessView* output_uav = nullptr;
    ID3D11Texture2D* output_texture = nullptr;
    
    // CUDA interop resources
    ID3D11Texture2D* cuda_interop_texture = nullptr;
    cudaGraphicsResource* cuda_resource = nullptr;
    
    // Color space information
    ColorSpaceInfo video_color_info;
    bool color_info_detected = false;
    
    // Cleanup method
    void cleanup() {
        if (cuda_resource) {
            cudaGraphicsUnregisterResource(cuda_resource);
            cuda_resource = nullptr;
        }
        
        if (cuda_interop_texture) {
            cuda_interop_texture->Release();
            cuda_interop_texture = nullptr;
        }
        
        if (input_srv_y) {
            input_srv_y->Release();
            input_srv_y = nullptr;
        }
        if (input_srv_uv) {
            input_srv_uv->Release();
            input_srv_uv = nullptr;
        }
        if (output_uav) {
            output_uav->Release();
            output_uav = nullptr;
        }
        if (output_texture) {
            output_texture->Release();
            output_texture = nullptr;
        }
        if (conversion_constants_buffer) {
            conversion_constants_buffer->Release();
            conversion_constants_buffer = nullptr;
        }
        if (color_conversion_shader) {
            color_conversion_shader->Release();
            color_conversion_shader = nullptr;
        }
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