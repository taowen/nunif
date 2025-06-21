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
        if (d_input_buffer) cudaFree(d_input_buffer); d_input_buffer = nullptr;
        if (d_output_buffer) cudaFree(d_output_buffer); d_output_buffer = nullptr;
        if (d_yuv_y_buffer) cudaFree(d_yuv_y_buffer); d_yuv_y_buffer = nullptr;
        if (d_yuv_u_buffer) cudaFree(d_yuv_u_buffer); d_yuv_u_buffer = nullptr;
        if (d_yuv_v_buffer) cudaFree(d_yuv_v_buffer); d_yuv_v_buffer = nullptr;
        input_buffer_size = output_buffer_size = 0;
        yuv_y_buffer_size = yuv_u_buffer_size = yuv_v_buffer_size = 0;
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

// 统一的批处理推理函数 - 完全GPU处理，直接输出YUV420P
static std::vector<AVFrame*> processBatchOnGPU(TensorRTEngine& engine_data, const std::vector<AVFrame*>& frames) {
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
    
    // 添加同步确保预处理完成
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

    // 4. 后处理：CHW Float -> YUV420P (直接在GPU上)
    int out_C = output_dims.d[1];
    int out_H = output_dims.d[2]; 
    int out_W = output_dims.d[3];
    
    // 分配YUV420P缓冲区
    size_t y_plane_size = out_W * out_H;
    size_t uv_plane_size = (out_W / 2) * (out_H / 2);
    size_t total_y_size = batch_size * y_plane_size;
    size_t total_u_size = batch_size * uv_plane_size;
    size_t total_v_size = batch_size * uv_plane_size;
    
    uint8_t* d_yuv_y = engine_data.gpu_buffers.getYUVYBuffer(total_y_size);
    uint8_t* d_yuv_u = engine_data.gpu_buffers.getYUVUBuffer(total_u_size);
    uint8_t* d_yuv_v = engine_data.gpu_buffers.getYUVVBuffer(total_v_size);

    std::vector<AVFrame*> processed_frames;
    processed_frames.reserve(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        float* frame_chw = d_output + i * out_C * out_H * out_W;
        uint8_t* frame_y = d_yuv_y + i * y_plane_size;
        uint8_t* frame_u = d_yuv_u + i * uv_plane_size;
        uint8_t* frame_v = d_yuv_v + i * uv_plane_size;
        
        int threads = 256;
        int blocks = (out_W * out_H + threads - 1) / threads;
        chw_float_to_yuv420p_kernel<<<blocks, threads>>>(frame_chw, frame_y, frame_u, frame_v, out_W, out_H);
    }
    
    // 同步并拷贝到CPU用于编码
    cudaDeviceSynchronize();
    
    std::vector<uint8_t> cpu_y_buffer(total_y_size);
    std::vector<uint8_t> cpu_u_buffer(total_u_size);
    std::vector<uint8_t> cpu_v_buffer(total_v_size);
    
    cudaMemcpy(cpu_y_buffer.data(), d_yuv_y, total_y_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_u_buffer.data(), d_yuv_u, total_u_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_v_buffer.data(), d_yuv_v, total_v_size, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < batch_size; ++i) {
        AVFrame* frame = av_frame_alloc();
        frame->format = AV_PIX_FMT_YUV420P;
        frame->width = out_W;
        frame->height = out_H;
        av_frame_get_buffer(frame, 0);
        
        // 复制Y平面
        uint8_t* src_y = cpu_y_buffer.data() + i * y_plane_size;
        memcpy(frame->data[0], src_y, y_plane_size);
        
        // 复制U平面
        uint8_t* src_u = cpu_u_buffer.data() + i * uv_plane_size;
        memcpy(frame->data[1], src_u, uv_plane_size);
        
        // 复制V平面
        uint8_t* src_v = cpu_v_buffer.data() + i * uv_plane_size;
        memcpy(frame->data[2], src_v, uv_plane_size);
        
        processed_frames.push_back(frame);
    }

    auto t4 = high_resolution_clock::now();

    std::cout << "[GPU Timing] preprocess: " << duration_cast<milliseconds>(t2-t1).count() << " ms, "
              << "inference: " << duration_cast<milliseconds>(t3-t2).count() << " ms, "
              << "postprocess: " << duration_cast<milliseconds>(t4-t3).count() << " ms"
              << std::endl;

    return processed_frames;
}

// 简化的编码函数 - 直接处理YUV420P输入
static void encode_and_write_frame(AVCodecContext *enc_ctx, AVFormatContext *ofmt_ctx, AVFrame *frame) {
    AVFrame* target_frame = frame;
    
    // 如果输入已经是YUV420P格式，直接使用；否则进行转换
    if (frame && frame->format != enc_ctx->pix_fmt) {
        target_frame = av_frame_alloc();
        target_frame->format = enc_ctx->pix_fmt;
        target_frame->width = enc_ctx->width;
        target_frame->height = enc_ctx->height;
        av_frame_get_buffer(target_frame, 0);
        
        SwsContext* sws_ctx = sws_getContext(
            frame->width, frame->height, (AVPixelFormat)frame->format,
            enc_ctx->width, enc_ctx->height, enc_ctx->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, 
                  target_frame->data, target_frame->linesize);
        sws_freeContext(sws_ctx);
        
        target_frame->pts = frame->pts;
        target_frame->pkt_dts = frame->pkt_dts;
        target_frame->duration = frame->duration;
    }
    
    int ret = avcodec_send_frame(enc_ctx, target_frame);
    if (target_frame != frame) av_frame_free(&target_frame);

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

// 提取编码器初始化逻辑
static bool initializeEncoder(AVCodecContext*& enc_ctx, AVFormatContext* ofmt_ctx, 
                             AVStream* out_stream, const AVFrame* sample_frame, 
                             AVFormatContext* ifmt_ctx, int video_stream_idx) {
    const AVCodec *encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder) {
        std::cerr << "Could not find H.265 NVENC encoder (hevc_nvenc)" << std::endl;
        return false;
    }
    
    enc_ctx = avcodec_alloc_context3(encoder);
    enc_ctx->height = sample_frame->height;
    enc_ctx->width = sample_frame->width;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx->time_base = ifmt_ctx->streams[video_stream_idx]->time_base;
    enc_ctx->gop_size = 12;
    enc_ctx->max_b_frames = 2;
    
    if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    // 根据分辨率计算更高的码率
    int pixels = enc_ctx->width * enc_ctx->height;
    if (pixels >= 1920 * 1080) {
        enc_ctx->bit_rate = 12000000;  // 1080p: 12Mbps (提高from 8Mbps)
    } else if (pixels >= 1280 * 720) {
        enc_ctx->bit_rate = 8000000;   // 720p: 8Mbps (提高from 5Mbps)
    } else {
        enc_ctx->bit_rate = 5000000;   // 480p: 5Mbps (提高from 3Mbps)
    }

    // 使用更高质量的设置
    av_opt_set(enc_ctx->priv_data, "preset", "p7", 0);        // 更高质量preset
    av_opt_set(enc_ctx->priv_data, "rc", "vbr", 0);
    av_opt_set(enc_ctx->priv_data, "cq", "18", 0);            // 更低CQ值，更高质量
    av_opt_set(enc_ctx->priv_data, "profile", "main", 0);
    av_opt_set(enc_ctx->priv_data, "level", "auto", 0);
    av_opt_set(enc_ctx->priv_data, "multipass", "fullres", 0); // 启用多遍编码
    av_opt_set(enc_ctx->priv_data, "lookahead", "32", 0);     // 增加前瞻帧数

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
    std::cout << "CQ: 18, Preset: p7, Multipass: fullres" << std::endl;
    
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

// 提取帧处理逻辑
static bool processFrameBatch(TensorRTEngine& engine_data, 
                             std::vector<AVFrame*>& frame_buffer,
                             AVCodecContext* enc_ctx, 
                             AVFormatContext* ofmt_ctx,
                             AVFormatContext* ifmt_ctx,
                             int video_stream_idx,
                             int& frame_count) {
    if (frame_buffer.empty()) return true;
    
    // 使用统一的GPU批处理函数
    std::vector<AVFrame*> processed_frames = processBatchOnGPU(engine_data, frame_buffer);
    
    if (processed_frames.empty()) {
        std::cerr << "Failed to process frame batch" << std::endl;
        return false;
    }
    
    // 编码处理好的帧
    AVRational in_tb = ifmt_ctx->streams[video_stream_idx]->time_base;
    AVRational out_tb = enc_ctx->time_base;
    
    for (size_t i = 0; i < processed_frames.size(); i++) {
        // 时间戳转换
        processed_frames[i]->pts = av_rescale_q(frame_buffer[i]->pts, in_tb, out_tb);
        processed_frames[i]->pkt_dts = frame_buffer[i]->pkt_dts != AV_NOPTS_VALUE ? 
            av_rescale_q(frame_buffer[i]->pkt_dts, in_tb, out_tb) : AV_NOPTS_VALUE;
        processed_frames[i]->duration = frame_buffer[i]->duration > 0 ? 
            av_rescale_q(frame_buffer[i]->duration, in_tb, out_tb) : 0;

        encode_and_write_frame(enc_ctx, ofmt_ctx, processed_frames[i]);
        av_frame_free(&processed_frames[i]);
    }
    
    frame_count += frame_buffer.size();
    std::cout << "Processed " << frame_count << " frames" << std::endl;
    
    // 清理缓冲区
    for(auto f : frame_buffer) {
        av_frame_free(&f);
    }
    frame_buffer.clear();
    
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
    
    // 初始化TensorRT引擎
    TensorRTEngine engine_data;
    if (!loadEngine(engine_path, engine_data)) {
        return 1;
    }

    // 创建CUDA硬件设备上下文
    AVBufferRef* hw_device_ctx = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create CUDA device context" << std::endl;
        return 1;
    }

    // 设置输入解码器
    AVFormatContext *ifmt_ctx = nullptr;
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
    AVCodecContext* dec_ctx = avcodec_alloc_context3(decoder);
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
    AVFormatContext *ofmt_ctx = nullptr;
    avformat_alloc_output_context2(&ofmt_ctx, nullptr, nullptr, output_video_path.c_str());
    const AVCodec *encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder) {
        std::cerr << "Could not find H.265 NVENC encoder (hevc_nvenc)" << std::endl;
        return 1;
    }
    AVStream *out_stream = avformat_new_stream(ofmt_ctx, encoder);
    AVCodecContext *enc_ctx = nullptr;

    // 主处理循环
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
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
            AVFrame* cloned_frame = av_frame_clone(frame);
            frame_buffer.push_back(cloned_frame);

            if (frame_buffer.size() >= batch_size) {
                // 初始化编码器（第一批处理后）
                if (!encoder_initialized) {
                    std::vector<AVFrame*> sample_frames = processBatchOnGPU(engine_data, frame_buffer);
                    if (sample_frames.empty()) {
                        std::cerr << "Failed to process first batch" << std::endl;
                        return 1;
                    }
                    
                    if (!initializeEncoder(enc_ctx, ofmt_ctx, out_stream, sample_frames[0], ifmt_ctx, video_stream_idx)) {
                        return 1;
                    }
                    
                    if (!initializeOutputFile(ofmt_ctx, output_video_path)) {
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
                        av_frame_free(&sample_frames[i]);
                    }
                    
                    frame_count += frame_buffer.size();
                    std::cout << "Processed " << frame_count << " frames" << std::endl;
                    
                    for(auto f : frame_buffer) av_frame_free(&f);
                    frame_buffer.clear();
                } else {
                    // 处理后续批次
                    if (!processFrameBatch(engine_data, frame_buffer, enc_ctx, ofmt_ctx, ifmt_ctx, video_stream_idx, frame_count)) {
                        return 1;
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }

    // 处理最后一批帧
    if (!frame_buffer.empty() && encoder_initialized) {
        processFrameBatch(engine_data, frame_buffer, enc_ctx, ofmt_ctx, ifmt_ctx, video_stream_idx, frame_count);
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
    
    std::cout << "Finished processing. Output saved to " << output_video_path << std::endl;
    return 0;
}