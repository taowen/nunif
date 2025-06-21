#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <string>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

// CUDA kernel for uint8 RGB24 to float32 CHW, normalized to [0,1]
__global__ void rgb24_to_chw_float_kernel(const uint8_t* __restrict__ src, float* __restrict__ dst, int W, int H, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H * C;
    if (idx >= total) return;
    int hw = idx / C;
    int c = idx % C;
    int h = hw / W;
    int w = hw % W;
    // src: HWC, dst: CHW
    dst[c * H * W + h * W + w] = src[(h * W + w) * C + c] / 255.0f;
}

// CUDA kernel for float32 CHW to uint8 RGB24, [0,1] → [0,255]
__global__ void chw_float_to_rgb24_kernel(const float* __restrict__ src, uint8_t* __restrict__ dst, int W, int H, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H * C;
    if (idx >= total) return;
    int hw = idx / C;
    int c = idx % C;
    int h = hw / W;
    int w = hw % W;
    float v = src[c * H * W + h * W + w];
    v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    dst[(h * W + w) * C + c] = static_cast<uint8_t>(v * 255.0f + 0.5f);
}

// CUDA kernel for NV12 to RGB24 CHW conversion
__global__ void nv12_to_chw_float_kernel(const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane, 
                                          float* __restrict__ dst, int W, int H, int y_pitch, int uv_pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (idx >= total) return;
    
    int h = idx / W;
    int w = idx % W;
    
    // Get Y value
    float y = y_plane[h * y_pitch + w] / 255.0f;
    
    // Get UV values (subsampled)
    int uv_h = h / 2;
    int uv_w = (w / 2) * 2; // Ensure even position
    float u = uv_plane[uv_h * uv_pitch + uv_w] / 255.0f - 0.5f;
    float v = uv_plane[uv_h * uv_pitch + uv_w + 1] / 255.0f - 0.5f;
    
    // YUV to RGB conversion
    float r = y + 1.402f * v;
    float g = y - 0.344136f * u - 0.714136f * v;
    float b = y + 1.772f * u;
    
    // Clamp to [0,1]
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // Store in CHW format
    dst[0 * H * W + h * W + w] = r; // R channel
    dst[1 * H * W + h * W + w] = g; // G channel
    dst[2 * H * W + h * W + w] = b; // B channel
}

// CUDA kernel for CHW Float to YUV420P conversion (direct GPU output)
__global__ void chw_float_to_yuv420p_kernel(const float* __restrict__ src, 
                                             uint8_t* __restrict__ y_plane,
                                             uint8_t* __restrict__ u_plane,
                                             uint8_t* __restrict__ v_plane,
                                             int W, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (idx >= total) return;
    
    int h = idx / W;
    int w = idx % W;
    
    // Get RGB values from CHW format
    float r = src[0 * H * W + h * W + w]; // R channel
    float g = src[1 * H * W + h * W + w]; // G channel  
    float b = src[2 * H * W + h * W + w]; // B channel
    
    // Clamp to [0,1]
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // RGB to YUV conversion
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float u = -0.169f * r - 0.331f * g + 0.5f * b + 0.5f;
    float v = 0.5f * r - 0.419f * g - 0.081f * b + 0.5f;
    
    // Convert to 8-bit and store Y plane
    y_plane[h * W + w] = static_cast<uint8_t>(y * 255.0f + 0.5f);
    
    // For U and V planes (4:2:0 subsampling - only process every 2x2 block)
    if (h % 2 == 0 && w % 2 == 0) {
        int uv_h = h / 2;
        int uv_w = w / 2;
        int uv_W = W / 2;
        
        // Clamp UV values to [0,1] and convert to 8-bit
        u = fmaxf(0.0f, fminf(1.0f, u));
        v = fmaxf(0.0f, fminf(1.0f, v));
        
        u_plane[uv_h * uv_W + uv_w] = static_cast<uint8_t>(u * 255.0f + 0.5f);
        v_plane[uv_h * uv_W + uv_w] = static_cast<uint8_t>(v * 255.0f + 0.5f);
    }
}

// 在现有CUDA kernel函数之后添加新的kernel函数
__global__ void chw_float_to_nv12_kernel(const float* __restrict__ src, 
                                          uint8_t* __restrict__ y_plane,
                                          uint8_t* __restrict__ uv_plane,
                                          int W, int H, int y_pitch, int uv_pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = W * H;
    if (idx >= total) return;
    
    int h = idx / W;
    int w = idx % W;
    
    // 获取RGB值
    float r = src[0 * H * W + h * W + w]; // R channel
    float g = src[1 * H * W + h * W + w]; // G channel  
    float b = src[2 * H * W + h * W + w]; // B channel
    
    // 限制到[0,1]
    r = fmaxf(0.0f, fminf(1.0f, r));
    g = fmaxf(0.0f, fminf(1.0f, g));
    b = fmaxf(0.0f, fminf(1.0f, b));
    
    // RGB到YUV转换
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    float u = -0.169f * r - 0.331f * g + 0.5f * b + 0.5f;
    float v = 0.5f * r - 0.419f * g - 0.081f * b + 0.5f;
    
    // 存储Y值
    y_plane[h * y_pitch + w] = static_cast<uint8_t>(y * 255.0f + 0.5f);
    
    // 对于UV分量（4:2:0子采样）
    if (h % 2 == 0 && w % 2 == 0) {
        int uv_h = h / 2;
        int uv_w = w / 2;
        
        // 限制UV值到[0,1]并转换为8位
        u = fmaxf(0.0f, fminf(1.0f, u));
        v = fmaxf(0.0f, fminf(1.0f, v));
        
        // NV12格式：UV交错存储
        uv_plane[uv_h * uv_pitch + uv_w * 2] = static_cast<uint8_t>(u * 255.0f + 0.5f);     // U
        uv_plane[uv_h * uv_pitch + uv_w * 2 + 1] = static_cast<uint8_t>(v * 255.0f + 0.5f); // V
    }
}

// 统一的GPU缓冲区管理器
struct GPUBufferManager {
    float* d_input_buffer = nullptr;
    float* d_output_buffer = nullptr;
    uint8_t* d_yuv_y_buffer = nullptr;
    uint8_t* d_yuv_u_buffer = nullptr;
    uint8_t* d_yuv_v_buffer = nullptr;
    
    size_t input_buffer_size = 0;
    size_t output_buffer_size = 0;
    size_t yuv_y_buffer_size = 0;
    size_t yuv_u_buffer_size = 0;
    size_t yuv_v_buffer_size = 0;
    
    ~GPUBufferManager() {
        cleanup();
    }
    
    void cleanup() {
        // 等待所有CUDA操作完成
        cudaDeviceSynchronize();
        
        if (d_input_buffer) { cudaFree(d_input_buffer); d_input_buffer = nullptr; }
        if (d_output_buffer) { cudaFree(d_output_buffer); d_output_buffer = nullptr; }
        if (d_yuv_y_buffer) { cudaFree(d_yuv_y_buffer); d_yuv_y_buffer = nullptr; }
        if (d_yuv_u_buffer) { cudaFree(d_yuv_u_buffer); d_yuv_u_buffer = nullptr; }
        if (d_yuv_v_buffer) { cudaFree(d_yuv_v_buffer); d_yuv_v_buffer = nullptr; }
        
        input_buffer_size = output_buffer_size = 0;
        yuv_y_buffer_size = yuv_u_buffer_size = yuv_v_buffer_size = 0;
        
        // 检查CUDA错误
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error during cleanup: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    float* getInputBuffer(size_t required_size) {
        if (required_size > input_buffer_size) {
            if (d_input_buffer) cudaFree(d_input_buffer);
            cudaMalloc(&d_input_buffer, required_size);
            input_buffer_size = required_size;
        }
        return d_input_buffer;
    }
    
    float* getOutputBuffer(size_t required_size) {
        if (required_size > output_buffer_size) {
            if (d_output_buffer) cudaFree(d_output_buffer);
            cudaMalloc(&d_output_buffer, required_size);
            output_buffer_size = required_size;
        }
        return d_output_buffer;
    }
    
    uint8_t* getYUVYBuffer(size_t required_size) {
        if (required_size > yuv_y_buffer_size) {
            if (d_yuv_y_buffer) cudaFree(d_yuv_y_buffer);
            cudaMalloc(&d_yuv_y_buffer, required_size);
            yuv_y_buffer_size = required_size;
        }
        return d_yuv_y_buffer;
    }
    
    uint8_t* getYUVUBuffer(size_t required_size) {
        if (required_size > yuv_u_buffer_size) {
            if (d_yuv_u_buffer) cudaFree(d_yuv_u_buffer);
            cudaMalloc(&d_yuv_u_buffer, required_size);
            yuv_u_buffer_size = required_size;
        }
        return d_yuv_u_buffer;
    }
    
    uint8_t* getYUVVBuffer(size_t required_size) {
        if (required_size > yuv_v_buffer_size) {
            if (d_yuv_v_buffer) cudaFree(d_yuv_v_buffer);
            cudaMalloc(&d_yuv_v_buffer, required_size);
            yuv_v_buffer_size = required_size;
        }
        return d_yuv_v_buffer;
    }
};

// 简化的TensorRT引擎结构
struct TensorRTEngine {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    std::string input_tensor_name;
    std::string output_tensor_name;
    
    GPUBufferManager gpu_buffers;
    
    int output_height = 0;
    int output_width = 0;
    int output_channels = 0;
    
    // 添加显式析构函数确保清理顺序
    ~TensorRTEngine() {
        // 显式释放TensorRT资源
        context.reset();
        engine.reset();
        runtime.reset();
        // GPU buffers会在GPUBufferManager析构时自动清理
    }
};

// Simple logger class
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger g_logger;

// 加载TensorRT引擎
static bool loadEngine(const std::string& engine_path, TensorRTEngine& engine_data) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_buffer(size);
    file.read(engine_buffer.data(), size);
    file.close();
    
    engine_data.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!engine_data.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    engine_data.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        engine_data.runtime->deserializeCudaEngine(engine_buffer.data(), size)
    );
    if (!engine_data.engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    engine_data.context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_data.engine->createExecutionContext()
    );
    if (!engine_data.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // 获取输入输出张量名称
    int num_bindings = engine_data.engine->getNbIOTensors();
    for (int i = 0; i < num_bindings; ++i) {
        const char* tensor_name = engine_data.engine->getIOTensorName(i);
        auto tensor_mode = engine_data.engine->getTensorIOMode(tensor_name);
        
        if (tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
            engine_data.input_tensor_name = tensor_name;
        } else {
            engine_data.output_tensor_name = tensor_name;
        }
    }
    
    if (engine_data.input_tensor_name.empty() || engine_data.output_tensor_name.empty()) {
        std::cerr << "Failed to find input/output tensors" << std::endl;
        return false;
    }
    
    std::cout << "Engine loaded successfully!" << std::endl;
    std::cout << "Input: " << engine_data.input_tensor_name << std::endl;
    std::cout << "Output: " << engine_data.output_tensor_name << std::endl;
    
    auto output_dims = engine_data.engine->getTensorShape(engine_data.output_tensor_name.c_str());
    engine_data.output_height = output_dims.d[2];
    engine_data.output_width = output_dims.d[3]; 
    engine_data.output_channels = output_dims.d[1];
    
    return true;
}

// 统一的批处理推理函数 - 完全GPU处理，直接输出CUDA格式AVFrame
static std::vector<AVFrame*> processBatchOnGPU(TensorRTEngine& engine_data, const std::vector<AVFrame*>& frames, AVBufferRef* hw_device_ctx) {
    using namespace std::chrono;
    auto t0 = high_resolution_clock::now();

    if (!engine_data.context || frames.empty()) {
        std::cerr << "Engine not initialized or empty frame batch" << std::endl;
        return {};
    }

    const int batch_size = frames.size();
    const int height = frames[0]->height;
    const int width = frames[0]->width;
    const int channels = 3;

    // 1. 预处理：NV12 -> CHW Float (完全在GPU上)
    auto t1 = high_resolution_clock::now();
    
    size_t input_size = batch_size * channels * height * width * sizeof(float);
    float* d_input = engine_data.gpu_buffers.getInputBuffer(input_size);
    
    for (int i = 0; i < batch_size; ++i) {
        AVFrame* frame = frames[i];
        
        if (frame->format != AV_PIX_FMT_CUDA) {
            std::cerr << "Error: Expected CUDA frame but got format: " << frame->format << std::endl;
            return {};
        }
        
        uint8_t* y_plane = frame->data[0];
        uint8_t* uv_plane = frame->data[1];
        int y_pitch = frame->linesize[0];
        int uv_pitch = frame->linesize[1];
        
        float* batch_offset = d_input + i * channels * height * width;
        
        int threads = 256;
        int blocks = (width * height + threads - 1) / threads;
        nv12_to_chw_float_kernel<<<blocks, threads>>>(
            y_plane, uv_plane, batch_offset, width, height, y_pitch, uv_pitch
        );
    }
    
    cudaDeviceSynchronize();
    auto t2 = high_resolution_clock::now();

    // 2. TensorRT推理设置
    const char* input_name = engine_data.input_tensor_name.c_str();
    const char* output_name = engine_data.output_tensor_name.c_str();

    // 设置输入形状
    nvinfer1::Dims input_shape;
    input_shape.nbDims = 4;
    input_shape.d[0] = batch_size;
    input_shape.d[1] = channels;
    input_shape.d[2] = height;
    input_shape.d[3] = width;

    if (!engine_data.context->setInputShape(input_name, input_shape)) {
        std::cerr << "Failed to set input shape!" << std::endl;
        return {};
    }

    // 获取输出形状并分配输出缓冲区
    auto output_dims = engine_data.context->getTensorShape(output_name);
    size_t output_elements = 1;
    for(int j = 0; j < output_dims.nbDims; ++j) {
        output_elements *= output_dims.d[j];
    }
    size_t output_size = output_elements * sizeof(float);
    float* d_output = engine_data.gpu_buffers.getOutputBuffer(output_size);

    // 设置输入输出指针
    engine_data.context->setTensorAddress(input_name, d_input);
    engine_data.context->setTensorAddress(output_name, d_output);

    // 3. 执行推理
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    if (!engine_data.context->enqueueV3(stream)) {
        std::cerr << "Inference failed" << std::endl;
        cudaStreamDestroy(stream);
        return {};
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    auto t3 = high_resolution_clock::now();

    // 4. 后处理：CHW Float -> CUDA NV12 (完全在GPU上)
    int out_C = output_dims.d[1];
    int out_H = output_dims.d[2]; 
    int out_W = output_dims.d[3];
    
    // 创建硬件帧上下文用于分配CUDA帧
    AVBufferRef* hw_frames_ref = nullptr;
    AVHWFramesContext* hw_frames_ctx = nullptr;
    
    hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ref) {
        std::cerr << "Failed to allocate hardware frames context" << std::endl;
        return {};
    }
    
    hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
    hw_frames_ctx->format = AV_PIX_FMT_CUDA;
    hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
    hw_frames_ctx->width = out_W;
    hw_frames_ctx->height = out_H;
    hw_frames_ctx->initial_pool_size = batch_size + 2; // 预分配一些额外的帧
    
    if (av_hwframe_ctx_init(hw_frames_ref) < 0) {
        std::cerr << "Failed to initialize hardware frames context" << std::endl;
        av_buffer_unref(&hw_frames_ref);
        return {};
    }
    
    std::vector<AVFrame*> processed_frames;
    processed_frames.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        AVFrame* cuda_frame = av_frame_alloc();
        if (!cuda_frame) {
            std::cerr << "Failed to allocate CUDA frame" << std::endl;
            // 清理已分配的frames
            for (auto* frame : processed_frames) {
                av_frame_free(&frame);
            }
            av_buffer_unref(&hw_frames_ref);
            return {};
        }
        
        // 使用硬件帧上下文分配缓冲区
        if (av_hwframe_get_buffer(hw_frames_ref, cuda_frame, 0) < 0) {
            std::cerr << "Failed to allocate CUDA frame buffer" << std::endl;
            av_frame_free(&cuda_frame);
            // 清理已分配的frames
            for (auto* frame : processed_frames) {
                av_frame_free(&frame);
            }
            av_buffer_unref(&hw_frames_ref);
            return {};
        }
        
        // 直接将处理结果转换为NV12格式存储在GPU上
        float* frame_chw = d_output + i * out_C * out_H * out_W;
        uint8_t* y_plane = cuda_frame->data[0];
        uint8_t* uv_plane = cuda_frame->data[1];
        int y_pitch = cuda_frame->linesize[0];
        int uv_pitch = cuda_frame->linesize[1];
        
        // 使用CUDA kernel直接转换到NV12格式
        int threads = 256;
        int blocks = (out_W * out_H + threads - 1) / threads;
        chw_float_to_nv12_kernel<<<blocks, threads>>>(frame_chw, y_plane, uv_plane, out_W, out_H, y_pitch, uv_pitch);
        
        processed_frames.push_back(cuda_frame);
    }
    
    // 清理硬件帧上下文
    av_buffer_unref(&hw_frames_ref);
    
    cudaDeviceSynchronize();
    auto t4 = high_resolution_clock::now();

    std::cout << "[GPU Timing] preprocess: " << duration_cast<milliseconds>(t2-t1).count() << " ms, "
              << "inference: " << duration_cast<milliseconds>(t3-t2).count() << " ms, "
              << "postprocess: " << duration_cast<milliseconds>(t4-t3).count() << " ms"
              << std::endl;

    return processed_frames;
}

// 简化的编码函数 - 直接处理CUDA格式输入
static void encode_and_write_frame(AVCodecContext *enc_ctx, AVFormatContext *ofmt_ctx, AVFrame *frame) {
    // 直接发送CUDA格式的帧给编码器，NVENC会自动处理GPU内存
    int ret = avcodec_send_frame(enc_ctx, frame);

    if (ret < 0 && frame) {
        std::cerr << "avcodec_send_frame failed" << std::endl;
        return;
    }

    while (true) {
        AVPacket *pkt = av_packet_alloc();
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_packet_free(&pkt);
            return;
        } else if (ret < 0) {
            av_packet_free(&pkt);
            return;
        }
        av_packet_rescale_ts(pkt, enc_ctx->time_base, ofmt_ctx->streams[0]->time_base);
        pkt->stream_index = 0;
        av_interleaved_write_frame(ofmt_ctx, pkt);
        av_packet_free(&pkt);
    }
}

// 提取编码器初始化逻辑 - 配置为支持CUDA输入
static bool initializeEncoder(AVCodecContext*& enc_ctx, AVFormatContext* ofmt_ctx, 
                             AVStream* out_stream, const AVFrame* sample_frame, 
                             AVFormatContext* ifmt_ctx, int video_stream_idx,
                             AVBufferRef* hw_device_ctx) {
    const AVCodec *encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder) {
        std::cerr << "Could not find H.265 NVENC encoder (hevc_nvenc)" << std::endl;
        return false;
    }
    
    enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->height = sample_frame->height;
    enc_ctx->width = sample_frame->width;
    enc_ctx->pix_fmt = AV_PIX_FMT_CUDA; // 配置为接受CUDA输入
    enc_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx); // 设置硬件设备上下文
    
    // 创建硬件帧上下文用于编码器
    AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
    if (!hw_frames_ref) {
        std::cerr << "Failed to allocate hardware frames context for encoder" << std::endl;
        return false;
    }
    
    AVHWFramesContext* hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
    hw_frames_ctx->format = AV_PIX_FMT_CUDA;
    hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
    hw_frames_ctx->width = sample_frame->width;
    hw_frames_ctx->height = sample_frame->height;
    hw_frames_ctx->initial_pool_size = 20; // 足够的缓冲池大小
    
    if (av_hwframe_ctx_init(hw_frames_ref) < 0) {
        std::cerr << "Failed to initialize hardware frames context for encoder" << std::endl;
        av_buffer_unref(&hw_frames_ref);
        return false;
    }
    
    enc_ctx->hw_frames_ctx = hw_frames_ref; // 设置硬件帧上下文
    
    enc_ctx->time_base = ifmt_ctx->streams[video_stream_idx]->time_base;
    enc_ctx->gop_size = 12;
    enc_ctx->max_b_frames = 2;
    
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // 根据分辨率计算更高的码率
    int pixels = enc_ctx->width * enc_ctx->height;
    if (pixels >= 3840 * 2160) {
        enc_ctx->bit_rate = 30000000;  // 4k: 30Mbps
    } else if (pixels >= 1920 * 1080) {
        enc_ctx->bit_rate = 12000000;  // 1080p: 12Mbps
    } else if (pixels >= 1280 * 720) {
        enc_ctx->bit_rate = 8000000;   // 720p: 8Mbps
    } else {
        enc_ctx->bit_rate = 5000000;   // 480p: 5Mbps
    }

    // 使用更高质量的设置
    av_opt_set(enc_ctx->priv_data, "preset", "p7", 0);
    av_opt_set(enc_ctx->priv_data, "rc", "vbr", 0);
    av_opt_set(enc_ctx->priv_data, "cq", "18", 0);
    av_opt_set(enc_ctx->priv_data, "profile", "main", 0);
    av_opt_set(enc_ctx->priv_data, "level", "auto", 0);
    av_opt_set(enc_ctx->priv_data, "multipass", "fullres", 0);
    av_opt_set(enc_ctx->priv_data, "lookahead", "32", 0);

    out_stream->time_base = enc_ctx->time_base;

    if (avcodec_open2(enc_ctx, encoder, nullptr) < 0) {
        std::cerr << "Could not open encoder" << std::endl;
        return false;
    }
    
    if (avcodec_parameters_from_context(out_stream->codecpar, enc_ctx) < 0) {
        std::cerr << "Could not copy encoder parameters to output stream" << std::endl;
        return false;
    }
    
    std::cout << "Output resolution: " << enc_ctx->width << "x" << enc_ctx->height << std::endl;
    std::cout << "Bitrate: " << enc_ctx->bit_rate << " bps" << std::endl;
    std::cout << "CUDA input format configured for GPU-to-GPU encoding" << std::endl;
    
    return true;
}

// 提取输出文件初始化逻辑
static bool initializeOutputFile(AVFormatContext* ofmt_ctx, const std::string& output_path) {
    if (avio_open(&ofmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE) < 0) {
        std::cerr << "Could not open output file" << std::endl;
        return false;
    }
    
    if (avformat_write_header(ofmt_ctx, nullptr) < 0) {
        std::cerr << "Error occurred when opening output file" << std::endl;
        return false;
    }
    
    return true;
}

// 修改processFrameBatch函数，确保完全清理
static bool processFrameBatch(TensorRTEngine& engine_data, 
                             std::vector<AVFrame*>& frame_buffer,
                             AVCodecContext* enc_ctx, 
                             AVFormatContext* ofmt_ctx,
                             AVFormatContext* ifmt_ctx,
                             int video_stream_idx,
                             int& frame_count,
                             AVBufferRef* hw_device_ctx) {
    if (frame_buffer.empty()) return true;
    
    std::vector<AVFrame*> processed_frames = processBatchOnGPU(engine_data, frame_buffer, hw_device_ctx);
    
    if (processed_frames.empty()) {
        std::cerr << "Failed to process frame batch" << std::endl;
        // 清理输入frame_buffer
        for(auto f : frame_buffer) {
            av_frame_free(&f);
        }
        frame_buffer.clear();
        return false;
    }
    
    // 编码处理好的CUDA格式帧
    AVRational in_tb = ifmt_ctx->streams[video_stream_idx]->time_base;
    AVRational out_tb = enc_ctx->time_base;
    
    for (size_t i = 0; i < processed_frames.size(); i++) {
        processed_frames[i]->pts = av_rescale_q(frame_buffer[i]->pts, in_tb, out_tb);
        processed_frames[i]->pkt_dts = frame_buffer[i]->pkt_dts != AV_NOPTS_VALUE ? 
            av_rescale_q(frame_buffer[i]->pkt_dts, in_tb, out_tb) : AV_NOPTS_VALUE;
        processed_frames[i]->duration = frame_buffer[i]->duration > 0 ? 
            av_rescale_q(frame_buffer[i]->duration, in_tb, out_tb) : 0;

        encode_and_write_frame(enc_ctx, ofmt_ctx, processed_frames[i]);
        
        // 确保frame被完全释放
        av_frame_free(&processed_frames[i]);
    }
    
    frame_count += frame_buffer.size();
    std::cout << "Processed " << frame_count << " frames" << std::endl;
    
    // 确保processed_frames vector被完全清理
    processed_frames.clear();
    
    // 清理输入frames
    for(auto f : frame_buffer) {
        av_frame_free(&f);
    }
    frame_buffer.clear();
    
    // 强制同步CUDA操作，确保所有GPU操作完成
    cudaDeviceSynchronize();
    
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <input_video_path> <output_video_path> [batch_size]" << std::endl;
        return 1;
    }
    
    std::string engine_path = argv[1];
    std::string input_video_path = argv[2];
    std::string output_video_path = argv[3];
    int batch_size = (argc > 4) ? std::stoi(argv[4]) : 4;
    
    TensorRTEngine engine_data;
    AVBufferRef* hw_device_ctx = nullptr;
    AVFormatContext *ifmt_ctx = nullptr;
    AVFormatContext *ofmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    AVCodecContext *enc_ctx = nullptr;
    AVPacket *pkt = nullptr;
    AVFrame *frame = nullptr;
    
    try {
        // 初始化TensorRT引擎
        if (!loadEngine(engine_path, engine_data)) {
            return 1;
        }

        // 创建CUDA硬件设备上下文
        int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << "Failed to create CUDA device context" << std::endl;
            return 1;
        }

        // 设置输入解码器
        if (avformat_open_input(&ifmt_ctx, input_video_path.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Could not open input file " << input_video_path << std::endl;
            return 1;
        }
        
        if (avformat_find_stream_info(ifmt_ctx, nullptr) < 0) {
            std::cerr << "Could not find stream info" << std::endl;
            return 1;
        }
        
        int video_stream_idx = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx < 0) {
            std::cerr << "Could not find video stream" << std::endl;
            return 1;
        }
        
        AVCodecParameters* codecpar = ifmt_ctx->streams[video_stream_idx]->codecpar;
        
        // 选择解码器
        const AVCodec* decoder = nullptr;
        if (codecpar->codec_id == AV_CODEC_ID_H264) {
            decoder = avcodec_find_decoder_by_name("h264_cuvid");
            std::cout << "Using h264_cuvid decoder" << std::endl;
        } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
            decoder = avcodec_find_decoder_by_name("hevc_cuvid");
            std::cout << "Using hevc_cuvid decoder" << std::endl;
        } else {
            decoder = avcodec_find_decoder(codecpar->codec_id);
            std::cout << "Using software decoder for codec_id " << codecpar->codec_id << std::endl;
        }
        
        if (!decoder) {
            std::cerr << "Could not find decoder for codec_id " << codecpar->codec_id << std::endl;
            return 1;
        }
        
        // 初始化解码器上下文
        dec_ctx = avcodec_alloc_context3(decoder);
        if (!dec_ctx) {
            std::cerr << "Could not allocate decoder context" << std::endl;
            return 1;
        }
        
        if (avcodec_parameters_to_context(dec_ctx, codecpar) < 0) {
            std::cerr << "Could not copy codec parameters to decoder context" << std::endl;
            return 1;
        }
        
        // 配置硬件解码器
        if (strstr(decoder->name, "cuvid")) {
            dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
            dec_ctx->pkt_timebase = ifmt_ctx->streams[video_stream_idx]->time_base;
            av_opt_set_int(dec_ctx, "surfaces", 20, 0);
            av_opt_set_int(dec_ctx, "drop_second_field", 1, 0);
        }
        
        if (avcodec_open2(dec_ctx, decoder, nullptr) < 0) {
            std::cerr << "Could not open decoder" << std::endl;
            return 1;
        }

        // 设置输出格式
        avformat_alloc_output_context2(&ofmt_ctx, nullptr, nullptr, output_video_path.c_str());
        const AVCodec *encoder = avcodec_find_encoder_by_name("hevc_nvenc");
        if (!encoder) {
            std::cerr << "Could not find H.265 NVENC encoder (hevc_nvenc)" << std::endl;
            return 1;
        }
        AVStream *out_stream = avformat_new_stream(ofmt_ctx, encoder);

        // 主处理循环
        pkt = av_packet_alloc();
        frame = av_frame_alloc();
        std::vector<AVFrame*> frame_buffer;
        int frame_count = 0;
        bool encoder_initialized = false;
        
        std::cout << "Input resolution: " << dec_ctx->width << "x" << dec_ctx->height << std::endl;
        
        while (av_read_frame(ifmt_ctx, pkt) >= 0) {
            if (pkt->stream_index != video_stream_idx) {
                av_packet_unref(pkt);
                continue;
            }
            
            if (avcodec_send_packet(dec_ctx, pkt) != 0) {
                av_packet_unref(pkt);
                continue;
            }
            
            while (avcodec_receive_frame(dec_ctx, frame) == 0) {
                AVFrame* ref_frame = av_frame_alloc();
                av_frame_ref(ref_frame, frame);  // 使用引用而不是拷贝
                frame_buffer.push_back(ref_frame);
                
                if (frame_buffer.size() >= batch_size) {
                    // 初始化编码器（第一批处理后）
                    if (!encoder_initialized) {
                        std::vector<AVFrame*> sample_frames = processBatchOnGPU(engine_data, frame_buffer, hw_device_ctx);
                        if (sample_frames.empty()) {
                            std::cerr << "Failed to process first batch" << std::endl;
                            return 1;
                        }
                        
                        if (!initializeEncoder(enc_ctx, ofmt_ctx, out_stream, sample_frames[0], ifmt_ctx, video_stream_idx, hw_device_ctx)) {
                            // 清理sample_frames
                            for (auto* frame : sample_frames) {
                                av_frame_free(&frame);
                            }
                            return 1;
                        }
                        
                        if (!initializeOutputFile(ofmt_ctx, output_video_path)) {
                            // 清理sample_frames
                            for (auto* frame : sample_frames) {
                                av_frame_free(&frame);
                            }
                            return 1;
                        }
                        
                        encoder_initialized = true;
                        
                        // 编码第一批帧
                        for (size_t i = 0; i < sample_frames.size(); i++) {
                            AVRational in_tb = ifmt_ctx->streams[video_stream_idx]->time_base;
                            AVRational out_tb = enc_ctx->time_base;
                            sample_frames[i]->pts = av_rescale_q(frame_buffer[i]->pts, in_tb, out_tb);
                            sample_frames[i]->pkt_dts = frame_buffer[i]->pkt_dts != AV_NOPTS_VALUE ? 
                                av_rescale_q(frame_buffer[i]->pkt_dts, in_tb, out_tb) : AV_NOPTS_VALUE;
                            sample_frames[i]->duration = frame_buffer[i]->duration > 0 ? 
                                av_rescale_q(frame_buffer[i]->duration, in_tb, out_tb) : 0;
                            
                            encode_and_write_frame(enc_ctx, ofmt_ctx, sample_frames[i]);
                            av_frame_free(&sample_frames[i]); // 立即释放每个frame
                        }
                        
                        frame_count += frame_buffer.size();
                        std::cout << "Processed " << frame_count << " frames" << std::endl;
                        
                        // 清理输入frames和sample_frames vector
                        for(auto f : frame_buffer) av_frame_free(&f);
                        frame_buffer.clear();
                        sample_frames.clear(); // 清理vector
                        
                        // 强制同步CUDA操作
                        cudaDeviceSynchronize();
                    } else {
                        // 处理后续批次
                        if (!processFrameBatch(engine_data, frame_buffer, enc_ctx, ofmt_ctx, ifmt_ctx, video_stream_idx, frame_count, hw_device_ctx)) {
                            return 1;
                        }
                    }
                }
            }
            av_packet_unref(pkt);
        }

        // 处理最后一批帧
        if (!frame_buffer.empty() && encoder_initialized) {
            processFrameBatch(engine_data, frame_buffer, enc_ctx, ofmt_ctx, ifmt_ctx, video_stream_idx, frame_count, hw_device_ctx);
        }

        // 刷新编码器
        encode_and_write_frame(enc_ctx, ofmt_ctx, nullptr);
        
        // 写入文件尾部
        if (av_write_trailer(ofmt_ctx) < 0) {
            std::cerr << "Error occurred when writing output trailer" << std::endl;
        }
        
        // 清理资源
        avio_closep(&ofmt_ctx->pb);
        avformat_free_context(ofmt_ctx);
        avformat_close_input(&ifmt_ctx);
        avcodec_free_context(&dec_ctx);
        avcodec_free_context(&enc_ctx);
        av_packet_free(&pkt);
        av_frame_free(&frame);
        av_buffer_unref(&hw_device_ctx);
        
        // 清理剩余的frame_buffer中的frames
        // 注意：这个需要在适当的作用域中处理
        
        cudaDeviceReset();
        
        return 0;
    } catch (...) {
        std::cerr << "Exception occurred, cleaning up resources..." << std::endl;
    }
    
    // 统一的资源清理
    if (pkt) av_packet_free(&pkt);
    if (frame) av_frame_free(&frame);
    if (enc_ctx) avcodec_free_context(&enc_ctx);
    if (dec_ctx) avcodec_free_context(&dec_ctx);
    if (ofmt_ctx) {
        if (ofmt_ctx->pb) avio_closep(&ofmt_ctx->pb);
        avformat_free_context(ofmt_ctx);
    }
    if (ifmt_ctx) avformat_close_input(&ifmt_ctx);
    if (hw_device_ctx) av_buffer_unref(&hw_device_ctx);
    
    // 清理剩余的frame_buffer中的frames
    // 注意：这个需要在适当的作用域中处理
    
    cudaDeviceReset();
    
    return 0;
}