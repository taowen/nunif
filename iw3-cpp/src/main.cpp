#include <iostream>
#include <format>
#include <string_view>
#include <memory>
#include <stdexcept>

// D3D11VA 头文件需要在 extern "C" 之外包含
#include <libavutil/hwcontext_d3d11va.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// CUDA预处理上下文结构
struct CudaPreprocessContext {
    // CUDA内存
    void* d_input_nv12 = nullptr;      // NV12纹理数据
    void* d_output_rgb = nullptr;      // RGB输出 (float32)
    
    // D3D11-CUDA互操作
    cudaGraphicsResource_t cuda_nv12_resource = nullptr;
    
    // 张量尺寸
    int input_h = 0, input_w = 0;      // 输入NV12尺寸(带padding)
    int target_h = 0, target_w = 0;    // 目标输出尺寸(去padding)
    
    ~CudaPreprocessContext() {
        cleanup();
    }
    
    void cleanup() {
        if (d_input_nv12) cudaFree(d_input_nv12);
        if (d_output_rgb) cudaFree(d_output_rgb);
        
        if (cuda_nv12_resource) {
            cudaGraphicsUnregisterResource(cuda_nv12_resource);
        }
    }
};

// CUDA kernel 声明
extern "C" {
    void launch_nv12_to_rgb_kernel(
        const uint8_t* nv12_data,
        float* rgb_output,
        int input_width, int input_height,
        int target_width, int target_height,
        cudaStream_t stream = 0
    );
}

// CUDA kernel 实现 (内联在同一文件中)
__global__ void nv12_to_rgb_kernel(
    const uint8_t* nv12_data,
    float* rgb_output,
    int input_width, int input_height,
    int target_width, int target_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= target_width || y >= target_height) return;
    
    // NV12 格式布局：
    // - Y 平面：前 input_height 行
    // - UV 平面：后 input_height/2 行，交错存储 (UVUV...)
    
    // 获取 Y 值
    uint8_t Y = nv12_data[y * input_width + x];
    
    // 获取对应的 U, V 值 (UV平面在Y平面之后)
    int uv_y = y / 2;  // UV平面高度是Y平面的一半
    int uv_x = (x / 2) * 2;  // UV是2x2子采样，且交错存储
    int uv_offset = input_height * input_width + uv_y * input_width + uv_x;
    
    uint8_t U = nv12_data[uv_offset];     // U在偶数位置
    uint8_t V = nv12_data[uv_offset + 1]; // V在奇数位置
    
    // YUV to RGB 转换 (BT.709)
    float y_norm = Y / 255.0f;
    float u_norm = (U / 255.0f) - 0.5f;
    float v_norm = (V / 255.0f) - 0.5f;
    
    float r = y_norm + 1.5748f * v_norm;
    float g = y_norm - 0.1873f * u_norm - 0.4681f * v_norm;
    float b = y_norm + 1.8556f * u_norm;
    
    // 钳制到 [0, 1] 范围
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // 输出为 CHW 格式 (channels first)
    int pixel_idx = y * target_width + x;
    rgb_output[pixel_idx] = r;                                          // R channel
    rgb_output[target_height * target_width + pixel_idx] = g;           // G channel  
    rgb_output[2 * target_height * target_width + pixel_idx] = b;       // B channel
}

// CUDA kernel 启动函数
extern "C" void launch_nv12_to_rgb_kernel(
    const uint8_t* nv12_data,
    float* rgb_output,
    int input_width, int input_height,
    int target_width, int target_height,
    cudaStream_t stream
) {
    dim3 blockSize(16, 16);
    dim3 gridSize((target_width + blockSize.x - 1) / blockSize.x,
                  (target_height + blockSize.y - 1) / blockSize.y);
    
    nv12_to_rgb_kernel<<<gridSize, blockSize, 0, stream>>>(
        nv12_data, rgb_output, input_width, input_height, target_width, target_height
    );
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
    
    CudaPreprocessContext cuda_ctx;
    
public:
    VideoDecoder() = default;
    
    ~VideoDecoder() {
        cleanup();
    }
    
    void cleanup() {
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
            std::cerr << std::format("Failed to create D3D11VA device context: {}\n", av_err_to_string(ret));
            return false;
        }
        
        std::cout << "D3D11VA hardware acceleration initialized successfully\n";
        return true;
    }
    
    bool open_video_file(const std::string& filename) {
        // 打开视频文件
        int ret = avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to open video file: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 获取流信息
        ret = avformat_find_stream_info(format_ctx, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to find stream info: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 找到视频流
        video_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index < 0) {
            std::cerr << "No video stream found\n";
            return false;
        }
        
        std::cout << std::format("Found video stream at index: {}\n", video_stream_index);
        return true;
    }
    
    bool setup_decoder() {
        AVStream* video_stream = format_ctx->streams[video_stream_index];
        
        // 查找支持D3D11VA的解码器
        const AVCodec* decoder = nullptr;
        
        // 首先尝试查找硬件加速解码器
        if (video_stream->codecpar->codec_id == AV_CODEC_ID_H264) {
            decoder = avcodec_find_decoder_by_name("h264_d3d11va");
        } else if (video_stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            decoder = avcodec_find_decoder_by_name("hevc_d3d11va");
        }
        
        // 如果没有找到硬件解码器，使用软件解码器
        if (!decoder) {
            decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
            std::cout << "Using software decoder\n";
        } else {
            std::cout << std::format("Using hardware decoder: {}\n", decoder->name);
        }
        
        if (!decoder) {
            std::cerr << "Decoder not found\n";
            return false;
        }
        
        // 创建解码器上下文
        codec_ctx = avcodec_alloc_context3(decoder);
        if (!codec_ctx) {
            std::cerr << "Failed to allocate codec context\n";
            return false;
        }
        
        // 复制流参数到解码器上下文
        int ret = avcodec_parameters_to_context(codec_ctx, video_stream->codecpar);
        if (ret < 0) {
            std::cerr << std::format("Failed to copy codec parameters: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 设置硬件设备上下文
        if (hw_device_ctx) {
            codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        }
        
        // 打开解码器
        ret = avcodec_open2(codec_ctx, decoder, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to open codec: {}\n", av_err_to_string(ret));
            return false;
        }
        
        std::cout << std::format("Decoder setup completed - Resolution: {}x{}, Pixel Format: {}\n", 
                                codec_ctx->width, codec_ctx->height, 
                                av_get_pix_fmt_name(codec_ctx->pix_fmt));
        
        // === 添加详细的像素格式日志 ===
        std::cout << std::format("=== PIXEL FORMAT DEBUG INFO ===\n");
        std::cout << std::format("Codec context pix_fmt: {} ({})\n", 
                                av_get_pix_fmt_name(codec_ctx->pix_fmt), 
                                static_cast<int>(codec_ctx->pix_fmt));
        
        // 检查是否是硬件格式
        if (codec_ctx->pix_fmt == AV_PIX_FMT_D3D11) {
            std::cout << "Hardware format: D3D11 detected\n";
        } else {
            std::cout << "Software format detected\n";
        }
        
        return true;
    }
    
    bool initialize_cuda_preprocess() {
        // 设置张量尺寸
        cuda_ctx.input_h = 1634;   // 带padding的高度
        cuda_ctx.input_w = 3840;   // 宽度
        cuda_ctx.target_h = 1608;  // 目标高度（去padding）
        cuda_ctx.target_w = 3840;  // 目标宽度
        
        // NV12 完整纹理尺寸 (H*1.5)
        int nv12_total_h = static_cast<int>(cuda_ctx.input_h * 1.5);
        
        // 分配输入内存（完整的NV12纹理）
        size_t nv12_texture_size = nv12_total_h * cuda_ctx.input_w * sizeof(uint8_t);
        size_t output_size = cuda_ctx.target_h * cuda_ctx.target_w * 3 * sizeof(float);
        
        if (cudaMalloc(&cuda_ctx.d_input_nv12, nv12_texture_size) != cudaSuccess) {
            std::cerr << "Failed to allocate NV12 texture memory\n";
            return false;
        }
        
        if (cudaMalloc(&cuda_ctx.d_output_rgb, output_size) != cudaSuccess) {
            std::cerr << "Failed to allocate output RGB memory\n";
            return false;
        }
        
        std::cout << "CUDA preprocessing initialized successfully\n";
        return true;
    }
    
    void decode_frames() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        AVFrame* sw_frame = av_frame_alloc();
        
        if (!packet || !frame || !sw_frame) {
            std::cerr << "Failed to allocate packet or frame\n";
            return;
        }
        
        int frame_count = 0;
        
        // 读取和解码帧
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream_index) {
                int ret = avcodec_send_packet(codec_ctx, packet);
                if (ret < 0) {
                    std::cerr << std::format("Error sending packet: {}\n", av_err_to_string(ret));
                    break;
                }
                
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        std::cerr << std::format("Error receiving frame: {}\n", av_err_to_string(ret));
                        break;
                    }
                    
                    frame_count++;
                    
                    // 如果是硬件帧，直接处理
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        std::cout << std::format("Processing D3D11 frame {} with CUDA...\n", frame_count);
                        
                        // 使用CUDA处理帧
                        if (!process_frame_with_cuda(frame)) {
                            std::cerr << "Failed to process frame with CUDA\n";
                            continue;
                        }
                    } else {
                        // 软件解码的情况
                        if (frame->format == AV_PIX_FMT_YUV420P) {
                            std::cout << ">>> DETECTED FORMAT: YUV420P (Software) <<<\n";
                        } else if (frame->format == AV_PIX_FMT_NV12) {
                            std::cout << ">>> DETECTED FORMAT: NV12 (Software) <<<\n";
                        } else {
                            std::cout << std::format(">>> DETECTED FORMAT: OTHER (Software) ({}) <<<\n", 
                                                    av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
                        }
                        
                        std::cout << std::format("Frame {}: {}x{} (Software decoded)\n", 
                                                frame_count, frame->width, frame->height);
                    }
                    
                    std::cout << "================================\n";
                    
                    // 限制解码帧数，避免处理整个视频
                    if (frame_count >= 3) {  // 减少到3帧，方便查看日志
                        std::cout << "Processed 3 frames, stopping...\n";
                        goto cleanup_decode;
                    }
                }
            }
            av_packet_unref(packet);
        }
        
    cleanup_decode:
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        av_packet_free(&packet);
        
        std::cout << std::format("Total frames decoded: {}\n", frame_count);
        std::cout << "=== END OF PIXEL FORMAT DEBUG ===\n";
    }
    
private:
    bool process_frame_with_cuda(AVFrame* frame) {
        if (!frame || frame->format != AV_PIX_FMT_D3D11) {
            std::cerr << "Frame is not D3D11 format\n";
            return false;
        }
        
        // 1. 从AVFrame提取D3D11纹理
        ID3D11Texture2D* pTexture = (ID3D11Texture2D*)frame->data[0];
        int subresourceIndex = (int)(intptr_t)frame->data[1];
        
        if (!pTexture) {
            std::cerr << "No D3D11 texture in frame\n";
            return false;
        }
        
        // 2. 将D3D11纹理映射到CUDA
        if (!map_d3d11_texture_to_cuda(pTexture, subresourceIndex)) {
            return false;
        }
        
        // 3. 执行CUDA预处理
        if (!run_cuda_preprocessing()) {
            return false;
        }
        
        // 4. 取消映射
        unmap_cuda_resources();
        
        return true;
    }
    
    bool map_d3d11_texture_to_cuda(ID3D11Texture2D* pTexture, int subresourceIndex) {
        // 获取纹理描述信息
        D3D11_TEXTURE2D_DESC desc;
        pTexture->GetDesc(&desc);
        
        // === 添加详细的D3D11纹理日志 ===
        std::cout << "=== D3D11 TEXTURE DEBUG INFO ===\n";
        std::cout << std::format("Texture Width: {}\n", desc.Width);
        std::cout << std::format("Texture Height: {}\n", desc.Height);
        std::cout << std::format("Texture Format: {} ({})\n", 
                                desc.Format == DXGI_FORMAT_NV12 ? "DXGI_FORMAT_NV12" : "OTHER",
                                static_cast<int>(desc.Format));
        std::cout << std::format("Array Size: {}\n", desc.ArraySize);
        std::cout << std::format("MipLevels: {}\n", desc.MipLevels);
        std::cout << std::format("Usage: {}\n", static_cast<int>(desc.Usage));
        std::cout << std::format("Bind Flags: {}\n", desc.BindFlags);
        std::cout << std::format("CPU Access Flags: {}\n", desc.CPUAccessFlags);
        std::cout << std::format("Subresource Index: {}\n", subresourceIndex);
        
        if (desc.Format == DXGI_FORMAT_NV12) {
            std::cout << ">>> CONFIRMED: NV12 format detected <<<\n";
            std::cout << std::format("Expected Y plane size: {}x{}\n", desc.Width, desc.Height);
            std::cout << std::format("Expected UV plane size: {}x{}\n", desc.Width, desc.Height / 2);
            std::cout << std::format("Total texture height: {} (H * 1.5 = {})\n", 
                                    desc.Height, static_cast<int>(desc.Height * 1.5));
            
            // 注册D3D11纹理到CUDA
            cudaError_t result = cudaGraphicsD3D11RegisterResource(
                &cuda_ctx.cuda_nv12_resource, 
                pTexture, 
                cudaGraphicsRegisterFlagsReadOnly
            );
            
            if (result != cudaSuccess) {
                std::cerr << "Failed to register D3D11 texture: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 映射资源
            result = cudaGraphicsMapResources(1, &cuda_ctx.cuda_nv12_resource);
            if (result != cudaSuccess) {
                std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 获取CUDA数组
            cudaArray_t cudaArray;
            result = cudaGraphicsSubResourceGetMappedArray(&cudaArray, cuda_ctx.cuda_nv12_resource, subresourceIndex, 0);
            if (result != cudaSuccess) {
                std::cerr << "Failed to get mapped array: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 将CUDA数组数据复制到线性内存
            cudaMemcpy2DFromArray(
                cuda_ctx.d_input_nv12, 
                cuda_ctx.input_w * sizeof(uint8_t),
                cudaArray,
                0, 0,
                cuda_ctx.input_w * sizeof(uint8_t),
                static_cast<int>(cuda_ctx.input_h * 1.5),
                cudaMemcpyDeviceToDevice
            );
            
            // 添加CUDA映射成功的确认日志
            std::cout << "D3D11 to CUDA mapping and copy successful\n";
        } else {
            std::cout << std::format(">>> WARNING: Unexpected format {} <<<\n", static_cast<int>(desc.Format));
        }
        
        std::cout << "=== END D3D11 TEXTURE DEBUG ===\n";
        
        return true;
    }
    
    bool run_cuda_preprocessing() {
        // 启动 CUDA kernel 进行 NV12 到 RGB 转换
        launch_nv12_to_rgb_kernel(
            static_cast<const uint8_t*>(cuda_ctx.d_input_nv12),
            static_cast<float*>(cuda_ctx.d_output_rgb),
            cuda_ctx.input_w, cuda_ctx.input_h,
            cuda_ctx.target_w, cuda_ctx.target_h
        );
        
        // 检查CUDA错误
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) {
            std::cerr << "CUDA kernel failed: " << cudaGetErrorString(result) << "\n";
            return false;
        }
        
        // 等待kernel完成
        result = cudaDeviceSynchronize();
        if (result != cudaSuccess) {
            std::cerr << "CUDA synchronization failed: " << cudaGetErrorString(result) << "\n";
            return false;
        }
        
        std::cout << "CUDA preprocessing completed successfully\n";
        return true;
    }
    
    void unmap_cuda_resources() {
        if (cuda_ctx.cuda_nv12_resource) {
            cudaGraphicsUnmapResources(1, &cuda_ctx.cuda_nv12_resource);
            cudaGraphicsUnregisterResource(cuda_ctx.cuda_nv12_resource);
            cuda_ctx.cuda_nv12_resource = nullptr;
        }
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
            std::cerr << "Failed to initialize D3D11VA\n";
            return 1;
        }
        
        // 初始化CUDA预处理
        if (!decoder.initialize_cuda_preprocess()) {
            std::cerr << "Failed to initialize CUDA preprocessing\n";
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
        
        std::cout << "Video decoding completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }
    
    return 0;
}