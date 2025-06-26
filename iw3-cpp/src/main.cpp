#include "main.h"
#include "decode_thread.h"
#include "convert_color_thread.h"
#include "infer_sbs.h"
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


class MainProgram {
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
    MainProgram() = default;
    
    ~MainProgram() {
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
    
    void run_all_threads() {
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
        std::thread infer_sbs_th([this]() {
            start_infer_sbs(convert_color_output);
        });
        
        // Wait for all threads to complete
        decode_th.join();
        convert_color_th.join();
        infer_sbs_th.join();
        
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
        MainProgram program;
        
        if (!program.initialize_d3d11va()) {
            std::cerr << "Failed to initialize D3D11VA, will use software decoding\n";
        }
        
        if (!program.setup_d3d11_device()) {
            std::cerr << "Get d3d11 device for cuda d3d11 interop failed\n";
            return 1;
        }
        
        if (!program.open_video_file(video_file)) {
            return 1;
        }
        
        program.run_all_threads();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}