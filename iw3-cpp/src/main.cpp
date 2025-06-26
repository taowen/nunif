#include "main.h"
#include "decode_thread.h"
#include "convert_color_thread.h"
#include "infer_depth_thread.h"
#include "video_file_opener.h"
#include <iostream>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>
#include <d3d11.h>
#include <d3dcompiler.h>

// D3D11VA 头文件已包含在 ffmpeg_wrapper.h 中

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")\n";
        throw std::runtime_error("CUDA error occurred");
    }
}

class VideoDecoder {
private:
    // Decoder state
    DecoderState decoder_state_;
    
    // CUDA D3D11 interop members
    ID3D11Device* d3d11_device = nullptr;
    ID3D11DeviceContext* d3d11_context = nullptr;
    
    // Thread management
    DecodedFrameQueue decode_thread_output;
    ColorConvertedFrameQueue convert_color_output;
    
public:
    VideoDecoder() = default;
    
    ~VideoDecoder() {
        cleanup();
    }
    
    void cleanup() {
        // Clean up decoder state
        decoder_state_.cleanup();
        
        if (d3d11_context) {
            d3d11_context->Release();
        }
        
        if (d3d11_device) {
            d3d11_device->Release();
        }
    }
    
    bool initialize_d3d11va() {
        int ret = av_hwdevice_ctx_create(&decoder_state_.hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << "Failed to create D3D11VA device context: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        return true;
    }
    
    bool setup_d3d11_device() {
        if (!decoder_state_.hw_device_ctx) {
            std::cerr << "D3D11VA device context not initialized\n";
            return false;
        }
        
        AVD3D11VADeviceContext* d3d11va_ctx = (AVD3D11VADeviceContext*)((AVHWDeviceContext*)decoder_state_.hw_device_ctx->data)->hwctx;
        d3d11_device = d3d11va_ctx->device;
        d3d11_context = d3d11va_ctx->device_context;
        
        d3d11_device->AddRef();
        d3d11_context->AddRef();
        
        return true;
    }
    
    bool open_video_file(const std::string& filename) {
        return ::open_video_file(filename, decoder_state_);
    }
    
    bool setup_decoder() {
        AVStream* video_stream = decoder_state_.format_ctx->streams[decoder_state_.video_stream_index];
        
        const AVCodec* decoder = nullptr;
        
        if (!decoder_state_.hw_device_ctx) {
            std::cerr << "D3D11VA device context not initialized\n";
            return false;
        }
        
        const AVCodec* codec = nullptr;
        void* opaque = nullptr;
        
        while ((codec = av_codec_iterate(&opaque))) {
            if (codec->type == AVMEDIA_TYPE_VIDEO && 
                av_codec_is_decoder(codec) &&
                codec->id == video_stream->codecpar->codec_id) {
                
                for (int i = 0; ; i++) {
                    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
                    if (!config) {
                        break;
                    }
                    if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
                        decoder = codec;
                        break;
                    }
                }
                
                if (decoder) break;
            }
        }
        
        if (!decoder) {
            std::cerr << "No D3D11VA capable decoder found\n";
            return false;
        }
        
        decoder_state_.codec_ctx = avcodec_alloc_context3(decoder);
        if (!decoder_state_.codec_ctx) {
            std::cerr << "Failed to allocate codec context\n";
            return false;
        }
        
        int ret = avcodec_parameters_to_context(decoder_state_.codec_ctx, video_stream->codecpar);
        if (ret < 0) {
            std::cerr << "Failed to copy codec parameters: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        decoder_state_.codec_ctx->hw_device_ctx = av_buffer_ref(decoder_state_.hw_device_ctx);
        
        decoder_state_.codec_ctx->get_format = [](AVCodecContext* /* ctx */, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
            const enum AVPixelFormat* p;
            for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                if (*p == AV_PIX_FMT_D3D11) {
                    return *p;
                }
            }
            std::cerr << "D3D11 format not available\n";
            return AV_PIX_FMT_NONE;
        };
        
        ret = avcodec_open2(decoder_state_.codec_ctx, decoder, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open codec: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        return true;
    }
    
    // Main function to run all threads
    void decode_and_convert_color_threaded() {
        std::cout << "=== Starting Multi-threaded Processing ===\n";
        
        // Start decode thread
        std::thread decode_th([this]() {
            start_decode_thread(decoder_state_, decode_thread_output);
        });
        
        // Start color conversion thread with its own CUDA stream
        std::thread convert_color_th([this]() {
            start_convert_color_thread(
                decode_thread_output,
                convert_color_output,
                decoder_state_.video_color_info,
                d3d11_device,
                d3d11_context
            );
        });
        
        // Start depth inference thread with its own CUDA stream
        std::thread infer_depth_th([this]() {
            start_infer_depth_thread(convert_color_output);
        });
        
        // Wait for all threads to complete
        decode_th.join();
        convert_color_th.join();
        infer_depth_th.join();
        
        std::cout << "=== Multi-threaded Processing Completed ===\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: iw3_cpp <video_file>\n";
        return 1;
    }
    
    std::string video_file = argv[1];
    
    try {
        VideoDecoder decoder;
        
        if (!decoder.initialize_d3d11va()) {
            std::cerr << "Failed to initialize D3D11VA, will use software decoding\n";
        }
        
        if (!decoder.setup_d3d11_device()) {
            std::cerr << "Failed to setup CUDA D3D11 interop\n";
            return 1;
        }
        
        if (!decoder.open_video_file(video_file)) {
            return 1;
        }
        
        if (!decoder.setup_decoder()) {
            return 1;
        }
        
        // Use threaded processing
        decoder.decode_and_convert_color_threaded();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}