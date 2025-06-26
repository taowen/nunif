#include "main.h"
#include "decode_thread.h"
#include "convert_color_thread.h"
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

// D3D11VA 头文件需要在 extern "C" 之外包含
#include <libavutil/hwcontext_d3d11va.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")\n";
        throw std::runtime_error("CUDA error occurred");
    }
}

std::string av_err_to_string(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

class VideoDecoder {
private:
    // Decoder state
    DecoderState decoder_state_;
    
    // CUDA D3D11 interop members
    ID3D11Device* d3d11_device = nullptr;
    ID3D11DeviceContext* d3d11_context = nullptr;
    cudaStream_t cuda_stream = nullptr;
    bool cuda_d3d11_initialized = false;
    
    // Color conversion state
    ColorConversionState color_conversion_state_;
    
    // Thread management
    FrameQueue frame_queue_;
    
public:
    VideoDecoder() = default;
    
    ~VideoDecoder() {
        cleanup();
    }
    
    void cleanup() {
        // Clean up decoder state
        decoder_state_.cleanup();
        
        // Clean up color conversion state
        color_conversion_state_.cleanup();
        
        if (cuda_stream) {
            cudaStreamDestroy(cuda_stream);
        }
        
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
    
    bool setup_cuda_d3d11_interop() {
        if (!decoder_state_.hw_device_ctx) {
            std::cerr << "D3D11VA device context not initialized\n";
            return false;
        }
        
        AVD3D11VADeviceContext* d3d11va_ctx = (AVD3D11VADeviceContext*)((AVHWDeviceContext*)decoder_state_.hw_device_ctx->data)->hwctx;
        d3d11_device = d3d11va_ctx->device;
        d3d11_context = d3d11va_ctx->device_context;
        
        d3d11_device->AddRef();
        d3d11_context->AddRef();
        
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA capable devices found\n";
            return false;
        }
        
        int cuda_device = -1;
        
        IDXGIDevice* dxgi_device = nullptr;
        HRESULT hr = d3d11_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
        if (FAILED(hr)) {
            std::cerr << "Failed to get DXGI device\n";
            return false;
        }
        
        IDXGIAdapter* dxgi_adapter = nullptr;
        hr = dxgi_device->GetAdapter(&dxgi_adapter);
        dxgi_device->Release();
        
        if (FAILED(hr)) {
            std::cerr << "Failed to get DXGI adapter\n";
            return false;
        }
        
        cudaError_t cuda_status = cudaD3D11GetDevice(&cuda_device, dxgi_adapter);
        dxgi_adapter->Release();
        
        if (cuda_status != cudaSuccess) {
            std::cerr << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(cuda_status) << "\n";
            return false;
        }
        
        checkCudaErrors(cudaSetDevice(cuda_device));
        
        checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
        
        cuda_d3d11_initialized = true;
        
        return true;
    }
    
    bool open_video_file(const std::string& filename) {
        int ret = avformat_open_input(&decoder_state_.format_ctx, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open video file: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        ret = avformat_find_stream_info(decoder_state_.format_ctx, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to find stream info: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        decoder_state_.video_stream_index = av_find_best_stream(decoder_state_.format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (decoder_state_.video_stream_index < 0) {
            std::cerr << "No video stream found\n";
            return false;
        }
        
        return true;
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
        
        decoder_state_.codec_ctx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
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
    
    ColorSpaceInfo detect_color_info(AVFrame* frame) {
        ColorSpaceInfo info;
        
        // Get DXGI format from D3D11 texture
        if (frame->format == AV_PIX_FMT_D3D11) {
            ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)frame->data[0];
            D3D11_TEXTURE2D_DESC texture_desc;
            d3d11_texture->GetDesc(&texture_desc);
            info.dxgi_format = texture_desc.Format;
            
            std::cout << "=== Texture Format Detection ===\n";
            std::cout << "DXGI Format: " << static_cast<int>(texture_desc.Format) << " (";
            switch (texture_desc.Format) {
                case DXGI_FORMAT_NV12: std::cout << "NV12"; break;
                case DXGI_FORMAT_P010: std::cout << "P010"; info.bit_depth = 10; break;
                case DXGI_FORMAT_P016: std::cout << "P016"; info.bit_depth = 16; break;
                case DXGI_FORMAT_YUY2: std::cout << "YUY2"; break;
                case DXGI_FORMAT_AYUV: std::cout << "AYUV"; break;
                default: std::cout << "Unknown"; break;
            }
            std::cout << ")\n";
        }
        
        // Get color space information from codec context and frame
        info.color_space = decoder_state_.codec_ctx->colorspace != AVCOL_SPC_UNSPECIFIED ? 
                          decoder_state_.codec_ctx->colorspace : frame->colorspace;
        info.color_primaries = decoder_state_.codec_ctx->color_primaries != AVCOL_PRI_UNSPECIFIED ? 
                              decoder_state_.codec_ctx->color_primaries : frame->color_primaries;
        info.color_trc = decoder_state_.codec_ctx->color_trc != AVCOL_TRC_UNSPECIFIED ? 
                        decoder_state_.codec_ctx->color_trc : frame->color_trc;
        info.color_range = decoder_state_.codec_ctx->color_range != AVCOL_RANGE_UNSPECIFIED ? 
                          decoder_state_.codec_ctx->color_range : frame->color_range;
        
        // Detect HDR content
        info.is_hdr = (info.color_trc == AVCOL_TRC_SMPTE2084 ||  // PQ
                       info.color_trc == AVCOL_TRC_ARIB_STD_B67 || // HLG
                       info.color_primaries == AVCOL_PRI_BT2020);
        
        // Detect bit depth from DXGI format if not already set
        if (info.bit_depth == 8) {
            switch (info.dxgi_format) {
                case DXGI_FORMAT_P010:
                    info.bit_depth = 10;
                    break;
                case DXGI_FORMAT_P016:
                    info.bit_depth = 16;
                    break;
                default:
                    info.bit_depth = 8;
                    break;
            }
        }
        
        std::cout << "=== Video Color Space Information (Detected Once) ===\n";
        std::cout << "Color Space: " << av_color_space_name(info.color_space) << " (" << static_cast<int>(info.color_space) << ")\n";
        std::cout << "Color Primaries: " << av_color_primaries_name(info.color_primaries) << " (" << static_cast<int>(info.color_primaries) << ")\n";
        std::cout << "Transfer Characteristics: " << av_color_transfer_name(info.color_trc) << " (" << static_cast<int>(info.color_trc) << ")\n";
        std::cout << "Color Range: " << av_color_range_name(info.color_range) << " (" << static_cast<int>(info.color_range) << ")\n";
        std::cout << "Bit Depth: " << info.bit_depth << "\n";
        std::cout << "Is HDR: " << (info.is_hdr ? "Yes" : "No") << "\n";
        std::cout << "====================================================\n";
        
        return info;
    }
    
    // Main function to run both threads
    void decode_and_convert_color_threaded() {
        std::cout << "=== Starting Multi-threaded Processing ===\n";
        
        // Start both threads using the new decode thread function
        std::thread decode_th([this]() {
            start_decode_thread(decoder_state_, frame_queue_);
        });
        
        std::thread process_th([this]() {
            start_convert_color_thread(
                frame_queue_,
                color_conversion_state_,
                decoder_state_.video_color_info,
                d3d11_device,
                d3d11_context,
                cuda_stream
            );
        });
        
        // Wait for both threads to complete
        decode_th.join();
        process_th.join();
        
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
        
        if (!decoder.setup_cuda_d3d11_interop()) {
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