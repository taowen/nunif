#include <iostream>
#include <string_view>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// CUDA D3D11 interop headers
#include <cuda_d3d11_interop.h>
#include <cuda_runtime_api.h>

// D3D11VA 头文件需要在 extern "C" 之外包含
#include <libavutil/hwcontext_d3d11va.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

// 替代 helper_cuda.h 中的 checkCudaErrors 函数
inline void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")\n";
        throw std::runtime_error("CUDA error occurred");
    }
}

// 辅助函数：处理 av_err2str 在 MSVC 中的问题
std::string av_err_to_string(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

class VideoDecoder {
private:
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_index = -1;
    
    // CUDA D3D11 interop members
    ID3D11Device* d3d11_device = nullptr;
    ID3D11DeviceContext* d3d11_context = nullptr;
    cudaStream_t cuda_stream = nullptr;
    bool cuda_d3d11_initialized = false;
    
    // 中间纹理用于 CUDA 互操作
    ID3D11Texture2D* cuda_interop_texture = nullptr;
    cudaGraphicsResource* cuda_resource = nullptr;
    
public:
    VideoDecoder() = default;
    
    ~VideoDecoder() {
        cleanup();
    }
    
    void cleanup() {
        if (cuda_resource) {
            cudaGraphicsUnregisterResource(cuda_resource);
        }
        
        if (cuda_interop_texture) {
            cuda_interop_texture->Release();
        }
        
        if (cuda_stream) {
            cudaStreamDestroy(cuda_stream);
        }
        
        if (d3d11_context) {
            d3d11_context->Release();
        }
        
        if (d3d11_device) {
            d3d11_device->Release();
        }
        
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
        }
    }
    
    bool initialize_d3d11va() {
        // 创建D3D11VA硬件设备上下文
        int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << "Failed to create D3D11VA device context: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        // D3D11VA hardware acceleration initialized successfully
        return true;
    }
    
    bool setup_cuda_d3d11_interop() {
        if (!hw_device_ctx) {
            std::cerr << "D3D11VA device context not initialized\n";
            return false;
        }
        
        // 从 FFmpeg D3D11VA 上下文获取 D3D11 设备
        AVD3D11VADeviceContext* d3d11va_ctx = (AVD3D11VADeviceContext*)((AVHWDeviceContext*)hw_device_ctx->data)->hwctx;
        d3d11_device = d3d11va_ctx->device;
        d3d11_context = d3d11va_ctx->device_context;
        
        // 增加引用计数
        d3d11_device->AddRef();
        d3d11_context->AddRef();
        
        // Retrieved D3D11 device from FFmpeg context
        
        // 查找 CUDA 设备
        int deviceCount = 0;
        checkCudaErrors(cudaGetDeviceCount(&deviceCount));
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA capable devices found\n";
            return false;
        }
        
        // Found N CUDA capable device(s)
        
        // 查找与 D3D11 设备对应的 CUDA 设备
        int cuda_device = -1;
        
        // 获取 DXGI 适配器
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
        
        // 查找对应的 CUDA 设备
        cudaError_t cuda_status = cudaD3D11GetDevice(&cuda_device, dxgi_adapter);
        dxgi_adapter->Release();
        
        if (cuda_status != cudaSuccess) {
            std::cerr << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(cuda_status) << "\n";
            return false;
        }
        
        // Found matching CUDA device: N
        
        // 设置 CUDA 设备
        checkCudaErrors(cudaSetDevice(cuda_device));
        
        // 创建 CUDA 流
        checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
        
        cuda_d3d11_initialized = true;
        // CUDA D3D11 interop initialized successfully
        
        return true;
    }
    
    bool open_video_file(const std::string& filename) {
        // 打开视频文件
        int ret = avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open video file: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        // 获取流信息
        ret = avformat_find_stream_info(format_ctx, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to find stream info: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        // 找到视频流
        video_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index < 0) {
            std::cerr << "No video stream found\n";
            return false;
        }
        
        // Found video stream at index: N
        return true;
    }
    
    bool setup_decoder() {
        AVStream* video_stream = format_ctx->streams[video_stream_index];
        
        // Video codec: name (id)
        
        // 只查找D3D11VA硬件解码器
        const AVCodec* decoder = nullptr;
        
        if (!hw_device_ctx) {
            std::cerr << "D3D11VA device context not initialized\n";
            return false;
        }
        
        // Looking for D3D11VA capable decoders...
        
        // 枚举所有可用的解码器，查找支持D3D11VA的
        const AVCodec* codec = nullptr;
        void* opaque = nullptr;
        
        while ((codec = av_codec_iterate(&opaque))) {
            if (codec->type == AVMEDIA_TYPE_VIDEO && 
                av_codec_is_decoder(codec) &&
                codec->id == video_stream->codecpar->codec_id) {
                
                // 检查是否支持 D3D11VA
                for (int i = 0; ; i++) {
                    const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
                    if (!config) {
                        break;
                    }
                    if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
                        decoder = codec;
                        // Found D3D11VA capable decoder: name
                        break;
                    }
                }
                
                if (decoder) break;
            }
        }
        
        // 如果没有找到硬件解码器，直接失败
        if (!decoder) {
            std::cerr << "No D3D11VA capable decoder found\n";
            return false;
        }
        
        // Using hardware decoder: name
        
        // 创建解码器上下文
        codec_ctx = avcodec_alloc_context3(decoder);
        if (!codec_ctx) {
            std::cerr << "Failed to allocate codec context\n";
            return false;
        }
        
        // 复制流参数到解码器上下文
        int ret = avcodec_parameters_to_context(codec_ctx, video_stream->codecpar);
        if (ret < 0) {
            std::cerr << "Failed to copy codec parameters: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        // 设置硬件设备上下文
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        
        // 设置硬件帧获取回调
        codec_ctx->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
            const enum AVPixelFormat* p;
            for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                if (*p == AV_PIX_FMT_D3D11) {
                    // Selected D3D11 pixel format for hardware decoding
                    return *p;
                }
            }
            std::cerr << "D3D11 format not available\n";
            return AV_PIX_FMT_NONE;  // 返回错误格式
        };
        
        // Set hardware device context and format callback for D3D11VA decoder
        
        // 打开解码器
        ret = avcodec_open2(codec_ctx, decoder, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open codec: " << av_err_to_string(ret) << "\n";
            return false;
        }
        
        // Decoder setup completed - Resolution: WxH, Pixel Format: format
        
        return true;
    }
    
    // 纹理调试信息的辅助函数 - 现在只用于内部调试，不输出日志
    void log_texture_info(ID3D11Texture2D* texture, const std::string& texture_name) {
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);
        
        // === Texture Info ===
        // Size: WxH
        // Format: N (DXGI_FORMAT)
        // MipLevels: N
        // ArraySize: N
        // Usage: N (D3D11_USAGE)
        // BindFlags: 0xN
        // CPUAccessFlags: 0xN
        // MiscFlags: 0xN
        // BindFlags details: VERTEX_BUFFER, INDEX_BUFFER, etc.
        // ===================
    }
    
    bool create_cuda_interop_texture(UINT width, UINT height, DXGI_FORMAT format) {
        // 创建用于 CUDA 互操作的纹理
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
        
        HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &cuda_interop_texture);
        if (FAILED(hr)) {
            std::cerr << "Failed to create CUDA interop texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
        
        // Created CUDA interop texture successfully
        // Log texture info for debugging
        
        // 注册纹理到 CUDA
        cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
            &cuda_resource, cuda_interop_texture, cudaGraphicsRegisterFlagsNone);
        
        if (cuda_status != cudaSuccess) {
            std::cerr << "Failed to register interop texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
            return false;
        }
        
        // Successfully registered interop texture with CUDA
        return true;
    }
    
    bool process_d3d11_frame_with_cuda(AVFrame* d3d11_frame) {
        if (!cuda_d3d11_initialized) {
            std::cerr << "CUDA D3D11 interop not initialized\n";
            return false;
        }
        
        // 获取 D3D11 纹理
        ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)d3d11_frame->data[0];
        int texture_index = (int)(intptr_t)d3d11_frame->data[1];
        
        // Processing D3D11 texture: address, index: N
        
        // 记录源纹理信息（调试用，不输出）
        // log_texture_info(d3d11_texture, "FFmpeg Source Texture");
        
        // 获取纹理描述
        D3D11_TEXTURE2D_DESC texture_desc;
        d3d11_texture->GetDesc(&texture_desc);
        
        // 如果还没有创建中间纹理，创建新的
        if (!cuda_interop_texture) {
            // Creating CUDA interop texture...
            if (!create_cuda_interop_texture(texture_desc.Width, texture_desc.Height, texture_desc.Format)) {
                return false;
            }
        }
        
        // 复制纹理内容到中间纹理
        // Copying texture to CUDA interop texture...
        
        UINT src_subresource = D3D11CalcSubresource(0, texture_index, 1);
        UINT dst_subresource = D3D11CalcSubresource(0, 0, 1);
        
        // Copying subresource N to N
        
        d3d11_context->CopySubresourceRegion(
            cuda_interop_texture, dst_subresource, 0, 0, 0,
            d3d11_texture, src_subresource, nullptr);
        
        // 等待复制完成
        d3d11_context->Flush();
        
        // Texture copy completed
        
        // 映射 CUDA 资源
        // Mapping CUDA resource...
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, cuda_stream));
        std::cout << "✓ CUDA resource mapped successfully\n";
        
        // 查询CUDA资源信息（调试用，不输出）
        cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream);
        // query_cuda_resource_info();
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, cuda_stream));
        std::cout << "✓ CUDA resource remapped successfully\n";
        
        // 处理NV12格式的CUDA数据
        try {
            // 获取 CUDA 数组
            cudaArray_t cuda_array;
            checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0));
            // Successfully mapped to CUDA array
            std::cout << "✓ CUDA array obtained successfully - Address: 0x" << std::hex << reinterpret_cast<uintptr_t>(cuda_array) << std::dec << "\n";

            // 创建surface对象来访问NV12数据
            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuda_array;
            
            cudaSurfaceObject_t surface;
            cudaError_t surf_status = cudaCreateSurfaceObject(&surface, &resDesc);
            if (surf_status == cudaSuccess) {
                // Created CUDA surface object for NV12 processing
                std::cout << "✓ CUDA surface object created successfully - Handle: 0x" << std::hex << static_cast<unsigned int>(surface) << std::dec << "\n";
                
                // 这里可以添加实际的CUDA kernel处理代码
                // 现在可以通过surface访问完整的NV12数据
                
                // 获取CUDA数组的详细信息作为映射成功的进一步证据
                cudaChannelFormatDesc desc;
                cudaExtent extent;
                unsigned int flags;
                if (cudaArrayGetInfo(&desc, &extent, &flags, cuda_array) == cudaSuccess) {
                    std::cout << "✓ CUDA array info retrieved - Size: " << static_cast<int>(extent.width) << "x" << static_cast<int>(extent.height) << "x" << static_cast<int>(extent.depth) << ", Channels: " << (desc.x + desc.y + desc.z + desc.w) << ", Format: " << static_cast<int>(desc.f) << "\n";
                }
                
                // 处理完成后销毁surface
                cudaDestroySurfaceObject(surface);
                std::cout << "✓ CUDA surface object destroyed successfully\n";
            } else {
                std::cerr << "Failed to create surface object: " << cudaGetErrorString(surf_status) << "\n";
            }
            
            // 同步 CUDA 流
            checkCudaErrors(cudaStreamSynchronize(cuda_stream));
            std::cout << "✓ CUDA stream synchronized successfully\n";
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing CUDA arrays: " << e.what() << "\n";
        }
        
        // 取消映射资源
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
        std::cout << "✓ CUDA resource unmapped successfully\n";
        
        // D3D11 frame processed with CUDA successfully
        return true;
    }
    
    // 查询CUDA图形资源的信息 - 调试用，不输出日志
    void query_cuda_resource_info() {
        if (!cuda_resource) {
            // No CUDA resource to query
            return;
        }
        
        // === CUDA Resource Info ===
        
        // 尝试映射资源来获取信息
        cudaError_t map_status = cudaGraphicsMapResources(1, &cuda_resource, cuda_stream);
        if (map_status != cudaSuccess) {
            std::cerr << "Failed to map resource for query: " << cudaGetErrorString(map_status) << "\n";
            return;
        }
        
        // 尝试查询所有可能的subresource
        for (int subresource = 0; subresource < 4; ++subresource) {
            for (int miplevel = 0; miplevel < 2; ++miplevel) {
                cudaArray_t array;
                cudaError_t status = cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, subresource, miplevel);
                if (status == cudaSuccess && array) {
                    cudaChannelFormatDesc desc;
                    cudaExtent extent;
                    unsigned int flags;
                    if (cudaArrayGetInfo(&desc, &extent, &flags, array) == cudaSuccess) {
                        // Subresource N, Mip N: WxHxD, channels: N
                    }
                }
            }
        }
        
        cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream);
        // ===========================
    }
    
    void decode_frames() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        
        if (!packet || !frame) {
            std::cerr << "Failed to allocate packet or frame\n";
            return;
        }
        
        int frame_count = 0;
        
        // 读取和解码帧
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream_index) {
                int ret = avcodec_send_packet(codec_ctx, packet);
                if (ret < 0) {
                    std::cerr << "Error sending packet: " << av_err_to_string(ret) << "\n";
                    break;
                }
                
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        std::cerr << "Error receiving frame: " << av_err_to_string(ret) << "\n";
                        break;
                    }
                    
                    frame_count++;
                    
                    // === FRAME N ===
                    // Frame format: format_name (format_id)
                    
                    // 只处理D3D11硬件帧
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        // Processing D3D11 hardware frame with CUDA...
                        
                        if (cuda_d3d11_initialized) {
                            process_d3d11_frame_with_cuda(frame);
                        }
                    } else {
                        // 如果不是D3D11格式，说明硬件解码失败
                        std::cerr << "Unexpected frame format: " << av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)) << " - hardware decoding may have failed\n";
                        continue;
                    }
                    
                    // 限制解码帧数
                    if (frame_count >= 5) {
                        // Processed 5 frames, stopping...
                        goto cleanup_decode;
                    }
                }
            }
            av_packet_unref(packet);
        }
        
    cleanup_decode:
        av_frame_free(&frame);
        av_packet_free(&packet);
        
        // Total frames decoded: N
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
        
        // 初始化D3D11VA硬件加速
        if (!decoder.initialize_d3d11va()) {
            std::cerr << "Failed to initialize D3D11VA, will use software decoding\n";
        }
        
        // 设置 CUDA D3D11 interop
        if (!decoder.setup_cuda_d3d11_interop()) {
            std::cerr << "Failed to setup CUDA D3D11 interop\n";
            return 1;
        }
        
        // 打开视频文件
        if (!decoder.open_video_file(video_file)) {
            return 1;
        }
        
        // 设置解码器
        if (!decoder.setup_decoder()) {
            return 1;
        }
        
        // 解码视频帧
        decoder.decode_frames();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}