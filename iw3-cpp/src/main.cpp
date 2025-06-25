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

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// TensorRT Logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};

static Logger gLogger;

// TensorRT上下文结构
struct TensorRTContext {
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    
    // CUDA内存
    void* d_input_nv12 = nullptr;      // NV12纹理
    void* d_target_template = nullptr; // 目标尺寸模板
    void* d_output_rgb = nullptr;   // RGB输出
    
    // D3D11-CUDA互操作
    cudaGraphicsResource_t cuda_nv12_resource = nullptr;
    
    // 张量尺寸
    int input_h = 0, input_w = 0;
    int target_h = 0, target_w = 0;
    
    ~TensorRTContext() {
        cleanup();
    }
    
    void cleanup() {
        if (d_input_nv12) cudaFree(d_input_nv12);
        if (d_target_template) cudaFree(d_target_template);
        if (d_output_rgb) cudaFree(d_output_rgb);
        
        if (cuda_nv12_resource) {
            cudaGraphicsUnregisterResource(cuda_nv12_resource);
        }
        
        // In TensorRT 10, these objects are automatically managed
        // No need to call destroy() explicitly
        delete context;
        delete engine;
        delete runtime;
    }
};

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
    
    TensorRTContext trt_ctx;
    
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
    
    bool initialize_tensorrt(const std::string& onnx_model_path) {
        // 1. 创建TensorRT运行时
        trt_ctx.runtime = nvinfer1::createInferRuntime(gLogger);
        if (!trt_ctx.runtime) {
            std::cerr << "Failed to create TensorRT runtime\n";
            return false;
        }
        
        // 2. 加载ONNX模型并构建引擎
        if (!build_engine_from_onnx(onnx_model_path)) {
            return false;
        }
        
        // 3. 创建执行上下文
        trt_ctx.context = trt_ctx.engine->createExecutionContext();
        if (!trt_ctx.context) {
            std::cerr << "Failed to create execution context\n";
            return false;
        }
        
        // 4. 分配GPU内存
        if (!allocate_gpu_memory()) {
            return false;
        }
        
        std::cout << "TensorRT initialized successfully\n";
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
                        std::cout << std::format("Processing D3D11 frame {} with TensorRT...\n", frame_count);
                        
                        // 使用TensorRT处理帧
                        if (!process_frame_with_tensorrt(frame)) {
                            std::cerr << "Failed to process frame with TensorRT\n";
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
                    
                    // 这里可以处理解码后的帧数据
                    // sw_frame 包含了解码后的原始视频数据
                    
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
    bool build_engine_from_onnx(const std::string& onnx_path) {
        // 创建构建器
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            std::cerr << "Failed to create builder\n";
            return false;
        }
        
        // 创建网络
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create network\n";
            return false;
        }
        
        // 创建ONNX解析器
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser\n";
            return false;
        }
        
        // 解析ONNX模型
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX model\n";
            return false;
        }
        
        // 配置构建器
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config\n";
            return false;
        }
        
        // Use setMemoryPoolLimit instead of setMaxWorkspaceSize for TensorRT 10
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // 启用FP16
        
        // 构建引擎
        trt_ctx.engine = builder->buildEngineWithConfig(*network, *config);
        if (!trt_ctx.engine) {
            std::cerr << "Failed to build TensorRT engine\n";
            return false;
        }
        
        return true;
    }
    
    bool allocate_gpu_memory() {
        // 设置张量尺寸
        trt_ctx.input_h = 1634;   // 带padding的高度
        trt_ctx.input_w = 3840;   // 宽度
        trt_ctx.target_h = 1608;  // 目标高度（去padding）
        trt_ctx.target_w = 3840;  // 目标宽度
        
        // NV12 完整纹理尺寸 (H*1.5)
        int nv12_total_h = static_cast<int>(trt_ctx.input_h * 1.5);
        
        // 分配输入内存（完整的NV12纹理）
        size_t nv12_texture_size = nv12_total_h * trt_ctx.input_w * sizeof(uint8_t);
        size_t template_size = trt_ctx.target_h * trt_ctx.target_w * 3 * sizeof(float);
        size_t output_size = trt_ctx.target_h * trt_ctx.target_w * 3 * sizeof(float);
        
        // 只需要一个完整的 NV12 纹理内存
        if (cudaMalloc(&trt_ctx.d_input_nv12, nv12_texture_size) != cudaSuccess) {
            std::cerr << "Failed to allocate NV12 texture memory\n";
            return false;
        }
        
        if (cudaMalloc(&trt_ctx.d_target_template, template_size) != cudaSuccess) {
            std::cerr << "Failed to allocate target template memory\n";
            return false;
        }
        
        if (cudaMalloc(&trt_ctx.d_output_rgb, output_size) != cudaSuccess) {
            std::cerr << "Failed to allocate output RGB memory\n";
            return false;
        }
        
        // 初始化目标模板（零填充）
        cudaMemset(trt_ctx.d_target_template, 0, template_size);
        
        return true;
    }
    
    bool process_frame_with_tensorrt(AVFrame* frame) {
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
        
        // 3. 执行TensorRT推理
        if (!run_tensorrt_inference()) {
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
                &trt_ctx.cuda_nv12_resource, 
                pTexture, 
                cudaGraphicsRegisterFlagsReadOnly
            );
            
            if (result != cudaSuccess) {
                std::cerr << "Failed to register D3D11 texture: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 映射资源
            result = cudaGraphicsMapResources(1, &trt_ctx.cuda_nv12_resource);
            if (result != cudaSuccess) {
                std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 获取CUDA数组
            cudaArray_t cudaArray;
            result = cudaGraphicsSubResourceGetMappedArray(&cudaArray, trt_ctx.cuda_nv12_resource, subresourceIndex, 0);
            if (result != cudaSuccess) {
                std::cerr << "Failed to get mapped array: " << cudaGetErrorString(result) << "\n";
                return false;
            }
            
            // 添加CUDA数组信息日志
            cudaChannelFormatDesc channelDesc;
            cudaExtent extent;
            unsigned int flags;
            
            result = cudaArrayGetInfo(&channelDesc, &extent, &flags, cudaArray);
            if (result == cudaSuccess) {
                std::cout << "=== CUDA ARRAY DEBUG INFO ===\n";
                std::cout << std::format("CUDA Array dimensions: {}x{}x{}\n", 
                                        extent.width, extent.height, extent.depth);
                std::cout << std::format("Channel format: x={}, y={}, z={}, w={}\n",
                                        channelDesc.x, channelDesc.y, channelDesc.z, channelDesc.w);
                std::cout << std::format("Channel kind: {}\n", static_cast<int>(channelDesc.f));
                std::cout << "=== END CUDA ARRAY DEBUG ===\n";
            }
            
            // 添加CUDA映射成功的确认日志
            std::cout << "D3D11 to CUDA mapping successful\n";
        } else {
            std::cout << std::format(">>> WARNING: Unexpected format {} <<<\n", static_cast<int>(desc.Format));
        }
        
        std::cout << "=== END D3D11 TEXTURE DEBUG ===\n";
        
        return true;
    }
    
    bool run_tensorrt_inference() {
        // 获取张量名称（对应新的ONNX模型）
        const char* nv12_input_name = "nv12_texture";
        const char* template_input_name = "target_template";
        const char* output_name = "rgb_output";
        
        // 设置张量地址
        trt_ctx.context->setTensorAddress(nv12_input_name, trt_ctx.d_input_nv12);
        trt_ctx.context->setTensorAddress(template_input_name, trt_ctx.d_target_template);
        trt_ctx.context->setTensorAddress(output_name, trt_ctx.d_output_rgb);
        
        // 执行推理
        bool status = trt_ctx.context->executeV2(nullptr);
        if (!status) {
            std::cerr << "TensorRT inference failed\n";
            return false;
        }
        
        std::cout << "TensorRT inference completed successfully\n";
        return true;
    }
    
    void unmap_cuda_resources() {
        if (trt_ctx.cuda_nv12_resource) {
            cudaGraphicsUnmapResources(1, &trt_ctx.cuda_nv12_resource);
            cudaGraphicsUnregisterResource(trt_ctx.cuda_nv12_resource);
            trt_ctx.cuda_nv12_resource = nullptr;
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
        
        // 初始化TensorRT
        if (!decoder.initialize_tensorrt("d3d11va_direct_preprocess.onnx")) {
            std::cerr << "Failed to initialize TensorRT\n";
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