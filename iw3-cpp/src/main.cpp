#include "main.h"
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

inline void checkCudaErrors(cudaError_t result) {
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
    
    bool create_cuda_interop_texture(UINT width, UINT height, DXGI_FORMAT format) {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED; // For CUDA interop
        
        HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_conversion_state_.cuda_interop_texture);
        if (FAILED(hr)) {
            std::cerr << "Failed to create CUDA interop texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
            &color_conversion_state_.cuda_resource, color_conversion_state_.cuda_interop_texture, cudaGraphicsRegisterFlagsNone);
        
        if (cuda_status != cudaSuccess) {
            std::cerr << "Failed to register interop texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
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
    
    bool create_color_conversion_shader(const ColorSpaceInfo& color_info) {
        // Generate shader source based on color space info
        std::string shader_source = generate_shader_source(color_info);
        
        ID3DBlob* shader_blob = nullptr;
        ID3DBlob* error_blob = nullptr;
        
        HRESULT hr = D3DCompile(
            shader_source.c_str(),
            shader_source.length(),
            nullptr,
            nullptr,
            nullptr,
            "CSMain",
            "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS,
            0,
            &shader_blob,
            &error_blob
        );
        
        if (FAILED(hr)) {
            if (error_blob) {
                std::cerr << "Shader compilation error: " << (char*)error_blob->GetBufferPointer() << "\n";
                error_blob->Release();
            }
            return false;
        }
        
        hr = d3d11_device->CreateComputeShader(
            shader_blob->GetBufferPointer(),
            shader_blob->GetBufferSize(),
            nullptr,
            &color_conversion_state_.color_conversion_shader
        );
        
        shader_blob->Release();
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create compute shader: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        // Create constants buffer
        D3D11_BUFFER_DESC buffer_desc = {};
        buffer_desc.ByteWidth = sizeof(ConversionConstants);
        buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
        buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        
        hr = d3d11_device->CreateBuffer(&buffer_desc, nullptr, &color_conversion_state_.conversion_constants_buffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to create constants buffer: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        std::cout << "✓ Color conversion shader created successfully\n";
        return true;
    }
    
    std::string generate_shader_source(const ColorSpaceInfo& color_info) {
        std::ostringstream shader;
        
        shader << R"(
cbuffer ConversionConstants : register(b0)
{
    float4x4 ColorMatrix;
    float4 LumaCoeffs;
    float4 ChromaCoeffs;
    float4 Offset;
    int InputFormat;
    int ColorSpace;
    int BitDepth;
    int IsHDR;
};

Texture2D<float> LumaTexture : register(t0);
Texture2D<float2> ChromaTexture : register(t1);
RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    OutputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height)
        return;
    
    float3 yuv;
    
)";

        // Handle different input formats
        switch (color_info.dxgi_format) {
            case DXGI_FORMAT_NV12:
                shader << R"(
    // NV12 format handling
    yuv.x = LumaTexture.Load(int3(id.xy, 0)); // Y
    float2 uv = ChromaTexture.Load(int3(id.xy / 2, 0)); // UV
    yuv.y = uv.x; // U
    yuv.z = uv.y; // V
)";
                break;
            case DXGI_FORMAT_P010:
            case DXGI_FORMAT_P016:
                shader << R"(
    // P010/P016 format handling (10/16-bit)
    yuv.x = yuv_sample.r;
    yuv.y = yuv_sample.g;
    yuv.z = yuv_sample.b;
    
    // Scale from 10/16-bit to full range
    float scale_factor = )";
                shader << (color_info.bit_depth == 10 ? "1023.0" : "65535.0");
                shader << R"( / 255.0;
    yuv *= scale_factor;
)";
                break;
            default:
                shader << R"(
    // Default YUV handling
    yuv.x = yuv_sample.r;  // Y
    yuv.y = yuv_sample.g;  // U
    yuv.z = yuv_sample.b;  // V
)";
                break;
        }
        
        // Color space conversion
        shader << R"(
    
    // Apply color range expansion
    if (ColorSpace != 0) { // Not full range
        yuv.x = (yuv.x - 16.0/255.0) * 255.0/219.0;
        yuv.yz = (yuv.yz - 128.0/255.0) * 255.0/224.0;
    }
    
    // YUV to RGB conversion using color matrix
    float3 rgb = mul(ColorMatrix, float4(yuv, 1.0)).rgb;
    
)";

        // HDR tone mapping if needed
        if (color_info.is_hdr) {
            shader << R"(
    // HDR tone mapping
    if (IsHDR != 0) {
        // Simple tone mapping - you may want to implement more sophisticated methods
        rgb = rgb / (rgb + 1.0); // Reinhard tone mapping
    }
)";
        }
        
        shader << R"(
    
    // Clamp to [0, 1] range for model input
    rgb = saturate(rgb);
    
    // Output in RGBA format for model compatibility
    OutputTexture[id.xy] = float4(rgb, 1.0);
}
)";
        
        return shader.str();
    }
    
    ConversionConstants generate_conversion_constants(const ColorSpaceInfo& color_info) {
        ConversionConstants constants = {};
        
        // Set basic info
        constants.input_format = static_cast<int>(color_info.dxgi_format);
        constants.color_space = static_cast<int>(color_info.color_space);
        constants.bit_depth = color_info.bit_depth;
        constants.is_hdr = color_info.is_hdr ? 1 : 0;
        
        // Generate color conversion matrix based on color space
        float matrix[16] = {0};
        
        switch (color_info.color_space) {
            case AVCOL_SPC_BT709:
                // BT.709 YUV to RGB matrix
                matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.5748f;   matrix[3] = 0.0f;
                matrix[4] = 1.0f;    matrix[5] = -0.1873f;  matrix[6] = -0.4681f;  matrix[7] = 0.0f;
                matrix[8] = 1.0f;    matrix[9] = 1.8556f;   matrix[10] = 0.0f;     matrix[11] = 0.0f;
                matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
                break;
            case AVCOL_SPC_BT2020_NCL:
            case AVCOL_SPC_BT2020_CL:
                // BT.2020 YUV to RGB matrix
                matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.7166f;   matrix[3] = 0.0f;
                matrix[4] = 1.0f;    matrix[5] = -0.1916f;  matrix[6] = -0.6657f;  matrix[7] = 0.0f;
                matrix[8] = 1.0f;    matrix[9] = 2.1415f;   matrix[10] = 0.0f;     matrix[11] = 0.0f;
                matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
                break;
            default:
                // Default to BT.601
                matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.402f;    matrix[3] = 0.0f;
                matrix[4] = 1.0f;    matrix[5] = -0.344f;   matrix[6] = -0.714f;   matrix[7] = 0.0f;
                matrix[8] = 1.0f;    matrix[9] = 1.772f;    matrix[10] = 0.0f;     matrix[11] = 0.0f;
                matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
                break;
        }
        
        memcpy(constants.matrix, matrix, sizeof(matrix));
        
        return constants;
    }
    
    bool create_output_texture(UINT width, UINT height) {
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // Float32 RGBA for model input
        desc.SampleDesc.Count = 1;
        desc.SampleDesc.Quality = 0;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = 0;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        
        HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_conversion_state_.output_texture);
        if (FAILED(hr)) {
            std::cerr << "Failed to create output texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        // Create UAV
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
        uav_desc.Format = desc.Format;
        uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        uav_desc.Texture2D.MipSlice = 0;
        
        hr = d3d11_device->CreateUnorderedAccessView(color_conversion_state_.output_texture, &uav_desc, &color_conversion_state_.output_uav);
        if (FAILED(hr)) {
            std::cerr << "Failed to create output UAV: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        return true;
    }
    
    bool create_input_srv_once(ID3D11Texture2D* input_texture) {
        // Only create SRV if not already created
        if (color_conversion_state_.input_srv_y) {
            return true; // Already created
        }

        D3D11_TEXTURE2D_DESC desc;
        input_texture->GetDesc(&desc);
        
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MostDetailedMip = 0;
        srv_desc.Texture2D.MipLevels = 1;
        
        if (desc.Format == DXGI_FORMAT_NV12) {
            // Create Y plane view
            srv_desc.Format = DXGI_FORMAT_R8_UNORM;
            HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_conversion_state_.input_srv_y);
            if (FAILED(hr)) {
                std::cerr << "Failed to create input SRV for Y plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
                return false;
            }

            // Create UV plane view
            srv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
            hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_conversion_state_.input_srv_uv);
            if (FAILED(hr)) {
                std::cerr << "Failed to create input SRV for UV plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
                return false;
            }
        } else {
            srv_desc.Format = desc.Format;
            HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_conversion_state_.input_srv_y);
            if (FAILED(hr)) {
                std::cerr << "Failed to create input SRV: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
                return false;
            }
        }
        
        return true;
    }
    
    bool process_d3d11_frame_with_cuda(AVFrame* d3d11_frame) {
        if (!cuda_d3d11_initialized) {
            std::cerr << "CUDA D3D11 interop not initialized\n";
            return false;
        }
        
        // Detect color space and format information
        ColorSpaceInfo color_info = detect_color_info(d3d11_frame);
        
        ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)d3d11_frame->data[0];
        int texture_index = (int)(intptr_t)d3d11_frame->data[1];
        
        D3D11_TEXTURE2D_DESC texture_desc;
        d3d11_texture->GetDesc(&texture_desc);
        
        if (!color_conversion_state_.cuda_interop_texture) {
            if (!create_cuda_interop_texture(texture_desc.Width, texture_desc.Height, texture_desc.Format)) {
                return false;
            }
        }
        
        // Create shader and output resources if not already created
        if (!color_conversion_state_.color_conversion_shader) {
            if (!create_color_conversion_shader(color_info)) {
                return false;
            }
        }
        
        if (!color_conversion_state_.output_texture) {
            if (!create_output_texture(texture_desc.Width, texture_desc.Height)) {
                return false;
            }
        }
        
        // Copy from original texture to intermediate texture
        UINT src_subresource = D3D11CalcSubresource(0, texture_index, 1);
        UINT dst_subresource = D3D11CalcSubresource(0, 0, 1);
        
        d3d11_context->CopySubresourceRegion(
            color_conversion_state_.cuda_interop_texture, dst_subresource, 0, 0, 0,
            d3d11_texture, src_subresource, nullptr);
        
        std::cout << "✓ Texture copied to intermediate buffer\n";
        
        // === DirectX Shader Color Conversion ===
        
        // Create input SRV for shader (only once)
        if (!create_input_srv_once(color_conversion_state_.cuda_interop_texture)) {
            return false;
        }
        
        // Update constants buffer
        ConversionConstants constants = generate_conversion_constants(color_info);
        
        D3D11_MAPPED_SUBRESOURCE mapped_resource;
        HRESULT hr = d3d11_context->Map(color_conversion_state_.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
        if (SUCCEEDED(hr)) {
            memcpy(mapped_resource.pData, &constants, sizeof(constants));
            d3d11_context->Unmap(color_conversion_state_.conversion_constants_buffer, 0);
        }
        
        // Set shader resources
        d3d11_context->CSSetShader(color_conversion_state_.color_conversion_shader, nullptr, 0);
        if (color_conversion_state_.input_srv_uv) {
            ID3D11ShaderResourceView* srvs[] = { color_conversion_state_.input_srv_y, color_conversion_state_.input_srv_uv };
            d3d11_context->CSSetShaderResources(0, 2, srvs);
        } else {
            d3d11_context->CSSetShaderResources(0, 1, &color_conversion_state_.input_srv_y);
        }
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &color_conversion_state_.output_uav, nullptr);
        d3d11_context->CSSetConstantBuffers(0, 1, &color_conversion_state_.conversion_constants_buffer);
        
        // Dispatch shader
        UINT dispatch_x = (texture_desc.Width + 7) / 8;
        UINT dispatch_y = (texture_desc.Height + 7) / 8;
        d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);
        
        // Unbind resources
        ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* null_uav = nullptr;
        d3d11_context->CSSetShaderResources(0, 2, null_srvs);
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        
        d3d11_context->Flush();
        
        std::cout << "✓ Color conversion shader executed successfully\n";
        std::cout << "✓ Output format: RGBA32F (0-1 range) for model input\n";
        
        // === CUDA Processing on Converted Data ===
        
        // Now register the output texture with CUDA for further processing
        cudaGraphicsResource* output_cuda_resource = nullptr;
        cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
            &output_cuda_resource, color_conversion_state_.output_texture, cudaGraphicsRegisterFlagsNone);
        
        if (cuda_status == cudaSuccess) {
            checkCudaErrors(cudaGraphicsMapResources(1, &output_cuda_resource, cuda_stream));
            
            try {
                cudaArray_t cuda_array;
                checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, output_cuda_resource, 0, 0));
                
                // Now cuda_array contains RGBA32F data in 0-1 range, ready for model input
                std::cout << "✓ Converted texture mapped to CUDA successfully\n";
                std::cout << "✓ Data format: RGBA32F, Range: [0.0, 1.0]\n";
                std::cout << "✓ Ready for model inference (BCHW format expected by model)\n";
                
                // Here you can process the converted data with CUDA kernels
                // The data is now in the correct format for model input
                
                checkCudaErrors(cudaStreamSynchronize(cuda_stream));
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing converted CUDA data: " << e.what() << "\n";
            }
            
            checkCudaErrors(cudaGraphicsUnmapResources(1, &output_cuda_resource, cuda_stream));
            cudaGraphicsUnregisterResource(output_cuda_resource);
            
            std::cout << "✓ Converted texture processing completed\n";
        } else {
            std::cerr << "Failed to register output texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
        }
        
        return true;
    }
    
    // Decode thread function
    void decode_thread() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        
        if (!packet || !frame) {
            std::cerr << "Failed to allocate packet or frame\n";
            frame_queue_.push(DecodedFrame::end_signal());
            return;
        }
        
        int frame_count = 0;
        
        std::cout << "=== Decode Thread Started ===\n";
        
        // Read and decode frames
        while (av_read_frame(decoder_state_.format_ctx, packet) >= 0) {
            if (packet->stream_index == decoder_state_.video_stream_index) {
                int ret = avcodec_send_packet(decoder_state_.codec_ctx, packet);
                if (ret < 0) {
                    std::cerr << "Error sending packet: " << av_err_to_string(ret) << "\n";
                    break;
                }
                
                while (ret >= 0) {
                    ret = avcodec_receive_frame(decoder_state_.codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        std::cerr << "Error receiving frame: " << av_err_to_string(ret) << "\n";
                        break;
                    }
                    
                    frame_count++;
                    
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        // Detect color space info only for the first frame
                        if (!decoder_state_.color_info_detected) {
                            decoder_state_.video_color_info = detect_color_info(frame);
                            decoder_state_.color_info_detected = true;
                        }
                        
                        // Create a copy of the frame for the queue
                        AVFrame* frame_copy = av_frame_alloc();
                        if (av_frame_ref(frame_copy, frame) < 0) {
                            std::cerr << "Failed to reference frame\n";
                            av_frame_free(&frame_copy);
                            continue;
                        }
                        
                        // Push to queue (no color info needed, using shared one)
                        DecodedFrame decoded_frame(frame_copy);
                        frame_queue_.push(decoded_frame);
                        
                        std::cout << "✓ Frame " << frame_count << " decoded and queued\n";
                        
                    } else {
                        std::cerr << "Unexpected frame format: " << av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)) << " - hardware decoding may have failed\n";
                        continue;
                    }
                    
                    if (frame_count >= 5) {
                        goto decode_cleanup;
                    }
                }
            }
            av_packet_unref(packet);
        }
        
    decode_cleanup:
        av_frame_free(&frame);
        av_packet_free(&packet);
        
        // Signal end of decoding
        frame_queue_.push(DecodedFrame::end_signal());
        decoder_state_.decode_finished_ = true;
        
        std::cout << "=== Decode Thread Finished ===\n";
        std::cout << "Total frames decoded: " << frame_count << "\n";
    }
    
    // Process thread function
    void convert_color_thread() {
        std::cout << "=== Process Thread Started ===\n";
        
        int processed_count = 0;
        
        while (true) {
            DecodedFrame decoded_frame = frame_queue_.pop();
            
            // Check for end signal
            if (decoded_frame.is_end_signal) {
                std::cout << "=== Process Thread Received End Signal ===\n";
                break;
            }
            
            if (decoded_frame.frame) {
                processed_count++;
                std::cout << ">>> Processing frame " << processed_count << "\n";
                
                // Process the D3D11 frame using shared color info
                if (cuda_d3d11_initialized && decoder_state_.color_info_detected) {
                    process_d3d11_frame_with_cuda_threaded(decoded_frame.frame, decoder_state_.video_color_info);
                }
                
                // Clean up the frame
                av_frame_free(&decoded_frame.frame);
                
                std::cout << ">>> Frame " << processed_count << " processing completed\n";
            }
        }
        
        color_conversion_state_.process_finished_ = true;
        std::cout << "=== Process Thread Finished ===\n";
        std::cout << "Total frames processed: " << processed_count << "\n";
    }
    
    // Modified process function for threaded execution
    bool process_d3d11_frame_with_cuda_threaded(AVFrame* d3d11_frame, const ColorSpaceInfo& color_info) {
        if (!cuda_d3d11_initialized) {
            std::cerr << "CUDA D3D11 interop not initialized\n";
            return false;
        }
        
        ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)d3d11_frame->data[0];
        int texture_index = (int)(intptr_t)d3d11_frame->data[1];
        
        D3D11_TEXTURE2D_DESC texture_desc;
        d3d11_texture->GetDesc(&texture_desc);
        
        if (!color_conversion_state_.cuda_interop_texture) {
            if (!create_cuda_interop_texture(texture_desc.Width, texture_desc.Height, texture_desc.Format)) {
                return false;
            }
        }
        
        // Create shader and output resources if not already created
        if (!color_conversion_state_.color_conversion_shader) {
            if (!create_color_conversion_shader(color_info)) {
                return false;
            }
        }
        
        if (!color_conversion_state_.output_texture) {
            if (!create_output_texture(texture_desc.Width, texture_desc.Height)) {
                return false;
            }
        }
        
        // Copy from original texture to intermediate texture
        UINT src_subresource = D3D11CalcSubresource(0, texture_index, 1);
        UINT dst_subresource = D3D11CalcSubresource(0, 0, 1);
        
        d3d11_context->CopySubresourceRegion(
            color_conversion_state_.cuda_interop_texture, dst_subresource, 0, 0, 0,
            d3d11_texture, src_subresource, nullptr);
        
        std::cout << "  ✓ Texture copied to intermediate buffer\n";
        
        // === DirectX Shader Color Conversion ===
        
        // Create input SRV for shader (only once)
        if (!create_input_srv_once(color_conversion_state_.cuda_interop_texture)) {
            return false;
        }
        
        // Update constants buffer
        ConversionConstants constants = generate_conversion_constants(color_info);
        
        D3D11_MAPPED_SUBRESOURCE mapped_resource;
        HRESULT hr = d3d11_context->Map(color_conversion_state_.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
        if (SUCCEEDED(hr)) {
            memcpy(mapped_resource.pData, &constants, sizeof(constants));
            d3d11_context->Unmap(color_conversion_state_.conversion_constants_buffer, 0);
        }
        
        // Set shader resources
        d3d11_context->CSSetShader(color_conversion_state_.color_conversion_shader, nullptr, 0);
        if (color_conversion_state_.input_srv_uv) {
            ID3D11ShaderResourceView* srvs[] = { color_conversion_state_.input_srv_y, color_conversion_state_.input_srv_uv };
            d3d11_context->CSSetShaderResources(0, 2, srvs);
        } else {
            d3d11_context->CSSetShaderResources(0, 1, &color_conversion_state_.input_srv_y);
        }
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &color_conversion_state_.output_uav, nullptr);
        d3d11_context->CSSetConstantBuffers(0, 1, &color_conversion_state_.conversion_constants_buffer);
        
        // Dispatch shader
        UINT dispatch_x = (texture_desc.Width + 7) / 8;
        UINT dispatch_y = (texture_desc.Height + 7) / 8;
        d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);
        
        // Unbind resources
        ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* null_uav = nullptr;
        d3d11_context->CSSetShaderResources(0, 2, null_srvs);
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        
        d3d11_context->Flush();
        
        std::cout << "  ✓ Color conversion shader executed successfully\n";
        
        // === CUDA Processing on Converted Data ===
        
        // Now register the output texture with CUDA for further processing
        cudaGraphicsResource* output_cuda_resource = nullptr;
        cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
            &output_cuda_resource, color_conversion_state_.output_texture, cudaGraphicsRegisterFlagsNone);
        
        if (cuda_status == cudaSuccess) {
            checkCudaErrors(cudaGraphicsMapResources(1, &output_cuda_resource, cuda_stream));
            
            try {
                cudaArray_t cuda_array;
                checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, output_cuda_resource, 0, 0));
                
                std::cout << "  ✓ Converted texture mapped to CUDA successfully\n";
                
                // Here you can process the converted data with CUDA kernels
                
                checkCudaErrors(cudaStreamSynchronize(cuda_stream));
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing converted CUDA data: " << e.what() << "\n";
            }
            
            checkCudaErrors(cudaGraphicsUnmapResources(1, &output_cuda_resource, cuda_stream));
            cudaGraphicsUnregisterResource(output_cuda_resource);
            
            std::cout << "  ✓ Converted texture processing completed\n";
        } else {
            std::cerr << "Failed to register output texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
        }
        
        return true;
    }
    
    // Main function to run both threads
    void decode_and_convert_color_threaded() {
        std::cout << "=== Starting Multi-threaded Processing ===\n";
        
        // Start both threads
        std::thread decode_th(&VideoDecoder::decode_thread, this);
        std::thread process_th(&VideoDecoder::convert_color_thread, this);
        
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
        
        // Use threaded processing instead of single-threaded
        decoder.decode_and_convert_color_threaded();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}