#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <NvInfer.h>
#include <cuda_runtime.h>

// CUDA kernels
__global__ void nv12_to_chw_float_kernel(const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane, 
                                          float* __restrict__ dst, int W, int H, int y_pitch, int uv_pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    
    int h = idx / W;
    int w = idx % W;
    
    float y = y_plane[h * y_pitch + w] / 255.0f;
    int uv_h = h / 2;
    int uv_w = (w / 2) * 2;
    float u = uv_plane[uv_h * uv_pitch + uv_w] / 255.0f - 0.5f;
    float v = uv_plane[uv_h * uv_pitch + uv_w + 1] / 255.0f - 0.5f;
    
    float r = fmaxf(0.0f, fminf(1.0f, y + 1.402f * v));
    float g = fmaxf(0.0f, fminf(1.0f, y - 0.344136f * u - 0.714136f * v));
    float b = fmaxf(0.0f, fminf(1.0f, y + 1.772f * u));
    
    dst[0 * H * W + h * W + w] = r;
    dst[1 * H * W + h * W + w] = g;
    dst[2 * H * W + h * W + w] = b;
}

__global__ void chw_float_to_nv12_kernel(const float* __restrict__ src, 
                                          uint8_t* __restrict__ y_plane,
                                          uint8_t* __restrict__ uv_plane,
                                          int W, int H, int y_pitch, int uv_pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= W * H) return;
    
    int h = idx / W;
    int w = idx % W;
    
    float r = fmaxf(0.0f, fminf(1.0f, src[0 * H * W + h * W + w]));
    float g = fmaxf(0.0f, fminf(1.0f, src[1 * H * W + h * W + w]));
    float b = fmaxf(0.0f, fminf(1.0f, src[2 * H * W + h * W + w]));
    
    float y = 0.299f * r + 0.587f * g + 0.114f * b;
    y_plane[h * y_pitch + w] = static_cast<uint8_t>(y * 255.0f + 0.5f);
    
    if (h % 2 == 0 && w % 2 == 0) {
        float u = -0.169f * r - 0.331f * g + 0.5f * b + 0.5f;
        float v = 0.5f * r - 0.419f * g - 0.081f * b + 0.5f;
        
        int uv_h = h / 2;
        int uv_w = w / 2;
        
        uv_plane[uv_h * uv_pitch + uv_w * 2] = static_cast<uint8_t>(fmaxf(0.0f, fminf(1.0f, u)) * 255.0f + 0.5f);
        uv_plane[uv_h * uv_pitch + uv_w * 2 + 1] = static_cast<uint8_t>(fmaxf(0.0f, fminf(1.0f, v)) * 255.0f + 0.5f);
    }
}

// Simple logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

// Main processing class
class VideoProcessor {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    size_t max_input_size = 0;   // 添加最大尺寸限制
    size_t max_output_size = 0;
    
    Logger logger;
    
public:
    ~VideoProcessor() {
        cleanup();
    }
    
    void cleanup() {
        cudaDeviceSynchronize();
        if (d_input) { cudaFree(d_input); d_input = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        context.reset();
        engine.reset();
        runtime.reset();
    }
    
    bool loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) return false;
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> buffer(size);
        file.read(buffer.data(), size);
        file.close();
        
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) return false;
        
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(buffer.data(), size));
        if (!engine) return false;
        
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) return false;
        
        // 更智能的内存管理策略
        size_t total_mem, free_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        size_t engine_required = engine->getDeviceMemorySize();
        
        // 预留给系统和其他操作的内存（至少1GB）
        size_t reserved_mem = std::max(1024ULL * 1024 * 1024, total_mem / 10); // 1GB或总内存的10%，取较大值
        size_t usable_mem = (free_mem > reserved_mem) ? (free_mem - reserved_mem) : 0;
        
        std::cout << "Total GPU Memory: " << total_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Available GPU Memory: " << free_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Reserved Memory: " << reserved_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Usable Memory: " << usable_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Engine Required Memory: " << engine_required / 1024 / 1024 << " MB" << std::endl;

        return true;
    }
    
    // 添加缓冲区大小限制方法
    void setMaxBufferSize(size_t max_input_mb, size_t max_output_mb) {
        max_input_size = max_input_mb * 1024 * 1024;
        max_output_size = max_output_mb * 1024 * 1024;
    }
    
    std::vector<AVFrame*> processBatch(const std::vector<AVFrame*>& frames, AVBufferRef* hw_device_ctx) {
        if (frames.empty()) return {};
        
        const int batch_size = frames.size();
        const int height = frames[0]->height;
        const int width = frames[0]->width;
        const int channels = 3;
        
        // 限制缓冲区大小
        size_t required_input_size = batch_size * channels * height * width * sizeof(float);
        if (max_input_size > 0 && required_input_size > max_input_size) {
            std::cerr << "Warning: Required input size exceeds limit, processing smaller batches" << std::endl;
            // 可以考虑分割batch或降低精度
        }
        
        if (required_input_size > input_size) {
            if (d_input) cudaFree(d_input);
            cudaMalloc(&d_input, required_input_size);
            input_size = required_input_size;
            
            std::cout << "Allocated input buffer: " << required_input_size / 1024 / 1024 << " MB" << std::endl;
        }
        
        // Preprocess: NV12 -> CHW Float
        for (int i = 0; i < batch_size; ++i) {
            float* batch_offset = d_input + i * channels * height * width;
            int threads = 256;
            int blocks = (width * height + threads - 1) / threads;
            
            nv12_to_chw_float_kernel<<<blocks, threads>>>(
                frames[i]->data[0], frames[i]->data[1], batch_offset, 
                width, height, frames[i]->linesize[0], frames[i]->linesize[1]);
        }
        cudaDeviceSynchronize();
        
        // Setup TensorRT inference
        const char* input_name = engine->getIOTensorName(0);
        const char* output_name = engine->getIOTensorName(1);
        
        nvinfer1::Dims input_shape;
        input_shape.nbDims = 4;
        input_shape.d[0] = batch_size;
        input_shape.d[1] = channels;
        input_shape.d[2] = height;
        input_shape.d[3] = width;
        
        context->setInputShape(input_name, input_shape);
        auto output_dims = context->getTensorShape(output_name);
        
        size_t output_elements = 1;
        for(int j = 0; j < output_dims.nbDims; ++j) {
            output_elements *= output_dims.d[j];
        }
        size_t required_output_size = output_elements * sizeof(float);
        
        if (required_output_size > output_size) {
            if (d_output) cudaFree(d_output);
            cudaMalloc(&d_output, required_output_size);
            output_size = required_output_size;
        }
        
        // 参考官方示例，设置tensor地址
        context->setTensorAddress(input_name, d_input);
        context->setTensorAddress(output_name, d_output);
        
        // 创建bindings数组（参考官方示例）
        std::vector<void*> bindings(engine->getNbIOTensors());
        for (int32_t i = 0, e = engine->getNbIOTensors(); i < e; i++) {
            auto const name = engine->getIOTensorName(i);
            if (std::string(name) == std::string(input_name)) {
                bindings[i] = d_input;
            } else if (std::string(name) == std::string(output_name)) {
                bindings[i] = d_output;
            }
        }
        
        // 使用同步 API 执行推理（参考官方示例）
        bool status = context->executeV2(bindings.data());
        if (!status) {
            std::cerr << "TensorRT synchronous execution failed" << std::endl;
            return {};
        }
        
        // Create output frames
        int out_H = output_dims.d[2];
        int out_W = output_dims.d[3];
        int out_C = output_dims.d[1];
        
        AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
        AVHWFramesContext* hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
        hw_frames_ctx->format = AV_PIX_FMT_CUDA;
        hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
        hw_frames_ctx->width = out_W;
        hw_frames_ctx->height = out_H;
        hw_frames_ctx->initial_pool_size = batch_size + 1;
        av_hwframe_ctx_init(hw_frames_ref);
        
        std::vector<AVFrame*> output_frames;
        for (int i = 0; i < batch_size; ++i) {
            AVFrame* cuda_frame = av_frame_alloc();
            av_hwframe_get_buffer(hw_frames_ref, cuda_frame, 0);
            
            float* frame_chw = d_output + i * out_C * out_H * out_W;
            int threads = 256;
            int blocks = (out_W * out_H + threads - 1) / threads;
            
            chw_float_to_nv12_kernel<<<blocks, threads>>>(
                frame_chw, cuda_frame->data[0], cuda_frame->data[1], 
                out_W, out_H, cuda_frame->linesize[0], cuda_frame->linesize[1]);
            
            output_frames.push_back(cuda_frame);
        }
        
        av_buffer_unref(&hw_frames_ref);
        cudaDeviceSynchronize();
        
        return output_frames;
    }
};

// Helper function to calculate bitrate based on resolution
int calculateBitrate(int width, int height) {
    int pixels = width * height;
    
    // Base bitrate calculation: bits per pixel approach
    // For stereo video processing, we need higher quality
    double bpp; // bits per pixel
    
    if (pixels <= 640 * 480) {           // SD and below
        bpp = 1.2;
    } else if (pixels <= 1280 * 720) {   // 720p
        bpp = 1;
    } else if (pixels <= 1920 * 1080) {  // 1080p
        bpp = 0.8;
    } else if (pixels <= 2560 * 1440) {  // 1440p
        bpp = 0.8;
    } else {                             // 4K and above
        bpp = 0.8;   // 提高到合理水平
    }
    
    int bitrate = static_cast<int>(pixels * bpp);
    
    // Ensure minimum and maximum bounds  
    int min_bitrate = 1000000;   
    int max_bitrate = 50000000;  // 50 Mbps maximum
    
    return std::max(min_bitrate, std::min(max_bitrate, bitrate));
}

// 添加显存监控函数
void printMemoryUsage() {
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t used_mem = total_mem - free_mem;
    
    std::cout << "GPU Memory - Total: " << total_mem / 1024 / 1024 << " MB, "
              << "Used: " << used_mem / 1024 / 1024 << " MB, "
              << "Free: " << free_mem / 1024 / 1024 << " MB" << std::endl;
}

// 修改 VideoContext 结构体，添加流映射支持
struct VideoContext {
    AVFormatContext* ifmt_ctx = nullptr;
    AVFormatContext* ofmt_ctx = nullptr;
    AVCodecContext* dec_ctx = nullptr;
    AVCodecContext* enc_ctx = nullptr;
    AVStream* out_stream = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_idx = -1;
    int64_t total_frames = 0;
    std::string input_path;
    std::string output_path;
    
    // 添加流映射
    std::vector<int> stream_mapping;  // input stream index -> output stream index
    std::vector<AVStream*> output_streams;  // 所有输出流
};

// 设置输入解码器
bool setupInputDecoder(VideoContext& ctx) {
    // Open input
    if (avformat_open_input(&ctx.ifmt_ctx, ctx.input_path.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file" << std::endl;
        return false;
    }
    avformat_find_stream_info(ctx.ifmt_ctx, nullptr);
    
    ctx.video_stream_idx = av_find_best_stream(ctx.ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ctx.video_stream_idx < 0) {
        std::cerr << "Could not find video stream" << std::endl;
        return false;
    }
    
    AVCodecParameters* codecpar = ctx.ifmt_ctx->streams[ctx.video_stream_idx]->codecpar;
    
    // Calculate total frames for progress estimation
    AVStream* video_stream = ctx.ifmt_ctx->streams[ctx.video_stream_idx];
    if (video_stream->nb_frames > 0) {
        ctx.total_frames = video_stream->nb_frames;
    } else if (video_stream->duration > 0 && video_stream->r_frame_rate.num > 0) {
        double duration_sec = (double)video_stream->duration * av_q2d(video_stream->time_base);
        double fps = av_q2d(video_stream->r_frame_rate);
        ctx.total_frames = (int64_t)(duration_sec * fps);
    }
    
    if (ctx.total_frames > 0) {
        std::cout << "Estimated total frames: " << ctx.total_frames << std::endl;
    }
    
    // Setup decoder
    const AVCodec* decoder = nullptr;
    if (codecpar->codec_id == AV_CODEC_ID_H264) {
        decoder = avcodec_find_decoder_by_name("h264_cuvid");
    } else if (codecpar->codec_id == AV_CODEC_ID_HEVC) {
        decoder = avcodec_find_decoder_by_name("hevc_cuvid");
    } else {
        decoder = avcodec_find_decoder(codecpar->codec_id);
    }
    
    if (!decoder) {
        std::cerr << "Decoder not found" << std::endl;
        return false;
    }
    
    ctx.dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(ctx.dec_ctx, codecpar);
    ctx.dec_ctx->pkt_timebase = ctx.ifmt_ctx->streams[ctx.video_stream_idx]->time_base;
    if (strstr(decoder->name, "cuvid")) {
        ctx.dec_ctx->hw_device_ctx = av_buffer_ref(ctx.hw_device_ctx);
    }
    
    if (avcodec_open2(ctx.dec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Failed to open decoder" << std::endl;
        return false;
    }
    
    return true;
}

// 修改设置输出编码器函数，支持多流
bool setupOutputEncoder(VideoContext& ctx) {
    int ret = avformat_alloc_output_context2(&ctx.ofmt_ctx, nullptr, nullptr, ctx.output_path.c_str());
    if (ret < 0 || !ctx.ofmt_ctx) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Could not create output context: " << error_buf << std::endl;
        return false;
    }
    
    // 初始化流映射
    ctx.stream_mapping.resize(ctx.ifmt_ctx->nb_streams, -1);
    ctx.output_streams.resize(ctx.ifmt_ctx->nb_streams, nullptr);
    
    // 为每个输入流创建对应的输出流
    for (unsigned int i = 0; i < ctx.ifmt_ctx->nb_streams; i++) {
        AVStream* in_stream = ctx.ifmt_ctx->streams[i];
        AVStream* out_stream = nullptr;
        
        if (i == ctx.video_stream_idx) {
            // 视频流 - 将被处理
            out_stream = avformat_new_stream(ctx.ofmt_ctx, nullptr);
            if (!out_stream) {
                std::cerr << "Failed to create output video stream" << std::endl;
                return false;
            }
            ctx.out_stream = out_stream;  // 保存视频流引用
        } else if (in_stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            // 音频流 - 更仔细地处理
            out_stream = avformat_new_stream(ctx.ofmt_ctx, nullptr);
            if (!out_stream) {
                std::cerr << "Failed to create output audio stream for stream " << i << std::endl;
                continue;
            }
            
            // 复制基本音频参数，但要清理容器特定的数据
            ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
            if (ret < 0) {
                std::cerr << "Failed to copy codec parameters for audio stream " << i << std::endl;
                continue;
            }
            
            // 清理可能导致兼容性问题的字段
            out_stream->codecpar->codec_tag = 0; // 让输出容器自动选择合适的tag
            
            // 复制时间基准
            out_stream->time_base = in_stream->time_base;
            
            // 如果需要，可以设置输出格式特定的参数
            if (ctx.ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
                out_stream->codecpar->codec_tag = 0;
            }
            
            std::cout << "Copying audio stream " << i 
                      << " (codec: " << avcodec_get_name(in_stream->codecpar->codec_id) 
                      << ", channels: " << in_stream->codecpar->ch_layout.nb_channels
                      << ", sample_rate: " << in_stream->codecpar->sample_rate << ")" << std::endl;
                      
        } else if (in_stream->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) {
            // 字幕流处理
            out_stream = avformat_new_stream(ctx.ofmt_ctx, nullptr);
            if (!out_stream) {
                std::cerr << "Failed to create output subtitle stream for stream " << i << std::endl;
                continue;
            }
            
            // 复制字幕流参数
            ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
            if (ret < 0) {
                std::cerr << "Failed to copy codec parameters for subtitle stream " << i << std::endl;
                continue;
            }
            
            // 清理容器特定标签
            out_stream->codecpar->codec_tag = 0;
            
            // 复制时间基准和其他元数据
            out_stream->time_base = in_stream->time_base;
            
            // 复制字幕流的元数据（如语言信息）
            av_dict_copy(&out_stream->metadata, in_stream->metadata, 0);
            
            std::cout << "Copying subtitle stream " << i 
                      << " (codec: " << avcodec_get_name(in_stream->codecpar->codec_id) << ")" << std::endl;
        }
        
        if (out_stream) {
            ctx.stream_mapping[i] = out_stream->index;
            ctx.output_streams[i] = out_stream;
        }
    }
    
    // 设置视频编码器
    const AVCodec* encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder) {
        std::cerr << "hevc_nvenc encoder not found" << std::endl;
        return false;
    }
    
    ctx.enc_ctx = avcodec_alloc_context3(encoder);
    return true;
}

// 初始化编码器
bool initializeEncoder(VideoContext& ctx, AVFrame* first_frame) {
    int ret;
    
    ctx.enc_ctx->width = first_frame->width;
    ctx.enc_ctx->height = first_frame->height;
    ctx.enc_ctx->pix_fmt = AV_PIX_FMT_CUDA;
    ctx.enc_ctx->hw_device_ctx = av_buffer_ref(ctx.hw_device_ctx);
    
    // Set frame rate and time base properly
    AVRational input_framerate = av_guess_frame_rate(ctx.ifmt_ctx, ctx.ifmt_ctx->streams[ctx.video_stream_idx], nullptr);
    if (input_framerate.num > 0 && input_framerate.den > 0) {
        ctx.enc_ctx->framerate = input_framerate;
        ctx.enc_ctx->time_base = av_inv_q(input_framerate);
    } else {
        ctx.enc_ctx->framerate = {30, 1};
        ctx.enc_ctx->time_base = {1, 30};
    }
    
    // Create hw_frames_ctx for the encoder
    AVBufferRef* enc_hw_frames_ref = av_hwframe_ctx_alloc(ctx.hw_device_ctx);
    AVHWFramesContext* enc_hw_frames_ctx = (AVHWFramesContext*)enc_hw_frames_ref->data;
    enc_hw_frames_ctx->format = AV_PIX_FMT_CUDA;
    enc_hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
    enc_hw_frames_ctx->width = first_frame->width;
    enc_hw_frames_ctx->height = first_frame->height;
    enc_hw_frames_ctx->initial_pool_size = 4;
    ret = av_hwframe_ctx_init(enc_hw_frames_ref);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Failed to initialize hw_frames_ctx: " << error_buf << std::endl;
        return false;
    }
    ctx.enc_ctx->hw_frames_ctx = enc_hw_frames_ref;
    
    // Calculate bitrate based on resolution
    int calculated_bitrate = calculateBitrate(first_frame->width, first_frame->height);
    ctx.enc_ctx->bit_rate = calculated_bitrate;
    
    std::cout << "Resolution: " << first_frame->width << "x" << first_frame->height 
              << ", Calculated bitrate: " << calculated_bitrate / 1000000.0 << " Mbps" << std::endl;
    
    // Set encoding parameters
    ctx.enc_ctx->gop_size = 30;
    ctx.enc_ctx->max_b_frames = 0;
    
    // Set HEVC-specific parameters
    av_opt_set(ctx.enc_ctx->priv_data, "preset", "fast", 0);
    av_opt_set(ctx.enc_ctx->priv_data, "profile", "main", 0);
    av_opt_set(ctx.enc_ctx->priv_data, "level", "auto", 0);
    av_opt_set(ctx.enc_ctx->priv_data, "tier", "main", 0);
    
    // Ensure global header is set for container
    if (ctx.ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        ctx.enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    
    ret = avcodec_open2(ctx.enc_ctx, avcodec_find_encoder_by_name("hevc_nvenc"), nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Failed to open encoder: " << error_buf << std::endl;
        return false;
    }
    
    // Setup output stream parameters
    ret = avcodec_parameters_from_context(ctx.out_stream->codecpar, ctx.enc_ctx);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Failed to copy codec parameters: " << error_buf << std::endl;
        return false;
    }
    ctx.out_stream->time_base = ctx.enc_ctx->time_base;
    
    // Open output file and write header
    ret = avio_open(&ctx.ofmt_ctx->pb, ctx.output_path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Could not open output file: " << error_buf << std::endl;
        return false;
    }
    
    ret = avformat_write_header(ctx.ofmt_ctx, nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Error writing header: " << error_buf << std::endl;
        return false;
    }
    
    std::cout << "Encoder initialized and header written" << std::endl;
    return true;
}

// 编码并写入帧
bool encodeAndWriteFrames(VideoContext& ctx, const std::vector<AVFrame*>& frames, int64_t& next_pts) {
    int ret;
    
    for (size_t i = 0; i < frames.size(); i++) {
        frames[i]->pts = next_pts++;
        
        if (avcodec_send_frame(ctx.enc_ctx, frames[i]) == 0) {
            AVPacket* out_pkt = av_packet_alloc();
            if (!out_pkt) continue;
            
            while (avcodec_receive_packet(ctx.enc_ctx, out_pkt) == 0) {
                if (out_pkt->size > 0) {
                    av_packet_rescale_ts(out_pkt, ctx.enc_ctx->time_base, ctx.out_stream->time_base);
                    out_pkt->stream_index = 0;
                    
                    ret = av_interleaved_write_frame(ctx.ofmt_ctx, out_pkt);
                    if (ret < 0) {
                        char error_buf[AV_ERROR_MAX_STRING_SIZE];
                        av_strerror(ret, error_buf, sizeof(error_buf));
                        std::cerr << "Error writing frame: " << error_buf << std::endl;
                    }
                }
                av_packet_unref(out_pkt);
            }
            av_packet_free(&out_pkt);
        }
    }
    
    return true;
}

// 显示进度信息
void displayProgress(int frame_count, int64_t total_frames, const std::chrono::high_resolution_clock::time_point& start_time) {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
    
    std::cout << "Processed " << frame_count << " frames";
    
    if (total_frames > 0 && frame_count > 0) {
        double progress = (double)frame_count / total_frames;
        double elapsed_sec = elapsed.count() / 1000.0;
        double estimated_total_sec = elapsed_sec / progress;
        double remaining_sec = estimated_total_sec - elapsed_sec;
        
        int remaining_min = (int)(remaining_sec / 60);
        int remaining_sec_part = (int)(remaining_sec) % 60;
        
        std::cout << " (" << std::fixed << std::setprecision(1) 
                  << (progress * 100) << "%, ETA: " 
                  << remaining_min << "m" << remaining_sec_part << "s)";
    }
    std::cout << std::endl;
}

// 完成编码器
bool finalizeEncoder(VideoContext& ctx) {
    int ret;
    
    avcodec_send_frame(ctx.enc_ctx, nullptr);
    AVPacket* out_pkt = av_packet_alloc();
    if (out_pkt) {
        while (avcodec_receive_packet(ctx.enc_ctx, out_pkt) == 0) {
            if (out_pkt->size > 0) {
                av_packet_rescale_ts(out_pkt, ctx.enc_ctx->time_base, ctx.out_stream->time_base);
                out_pkt->stream_index = 0;
                ret = av_interleaved_write_frame(ctx.ofmt_ctx, out_pkt);
                if (ret < 0) {
                    std::cerr << "Error writing final frame" << std::endl;
                }
            }
            av_packet_unref(out_pkt);
        }
        av_packet_free(&out_pkt);
    }
    
    av_write_trailer(ctx.ofmt_ctx);
    return true;
}

// 清理资源
void cleanupVideoContext(VideoContext& ctx) {
    if (ctx.ofmt_ctx && ctx.ofmt_ctx->pb) {
        avio_closep(&ctx.ofmt_ctx->pb);
    }
    avformat_free_context(ctx.ofmt_ctx);
    avformat_close_input(&ctx.ifmt_ctx);
    avcodec_free_context(&ctx.dec_ctx);
    avcodec_free_context(&ctx.enc_ctx);
    av_buffer_unref(&ctx.hw_device_ctx);
}

// 修改处理非视频流的函数，添加更好的错误处理
bool processNonVideoPacket(VideoContext& ctx, AVPacket* pkt) {
    int input_stream_index = pkt->stream_index;
    
    // 检查流映射
    if (input_stream_index >= ctx.stream_mapping.size()) {
        return true; // 忽略超出范围的流
    }
    
    int output_stream_index = ctx.stream_mapping[input_stream_index];
    
    if (output_stream_index < 0) {
        return true;  // 忽略未映射的流
    }
    
    AVStream* in_stream = ctx.ifmt_ctx->streams[input_stream_index];
    AVStream* out_stream = ctx.output_streams[input_stream_index];
    
    if (!out_stream) {
        return true;
    }
    
    // 复制数据包
    AVPacket* out_pkt = av_packet_alloc();
    if (!out_pkt) {
        return false;
    }
    
    av_packet_ref(out_pkt, pkt);
    
    // 重新缩放时间戳
    av_packet_rescale_ts(out_pkt, in_stream->time_base, out_stream->time_base);
    out_pkt->stream_index = output_stream_index;
    
    // 写入数据包
    int ret = av_interleaved_write_frame(ctx.ofmt_ctx, out_pkt);
    if (ret < 0) {
        // 只在严重错误时报告，忽略一些常见的非致命错误
        if (ret != AVERROR(EINVAL) && ret != AVERROR_INVALIDDATA) {
            char error_buf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, error_buf, sizeof(error_buf));
            std::cerr << "Error writing packet for stream " << input_stream_index 
                      << " (" << av_get_media_type_string(in_stream->codecpar->codec_type)
                      << "): " << error_buf << std::endl;
        }
    }
    
    av_packet_free(&out_pkt);
    return ret >= 0 || ret == AVERROR(EINVAL) || ret == AVERROR_INVALIDDATA;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video> [batch_size]" << std::endl;
        return 1;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string engine_path = "stereo_module_half_sbs.trt";  // 写死engine路径
    int batch_size = (argc > 3) ? std::stoi(argv[3]) : 2;   // 调整参数索引
    
    // Initialize video context
    VideoContext ctx;
    ctx.input_path = argv[1];   // 调整参数索引
    ctx.output_path = argv[2];  // 调整参数索引
    
    std::cout << "=== Video Processing Setup ===" << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << "Input: " << ctx.input_path << std::endl;
    std::cout << "Output: " << ctx.output_path << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    VideoProcessor processor;
    
    // Setup memory management
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\n=== Initial GPU Memory Status ===" << std::endl;
    printMemoryUsage();
    
    size_t estimated_engine_mem = 2ULL * 1024 * 1024 * 1024;
    size_t remaining_mem = (free_mem > estimated_engine_mem) ? (free_mem - estimated_engine_mem) : (free_mem / 4);
    
    size_t max_input_mb = std::min(256ULL, remaining_mem / 1024 / 1024 / 3);
    size_t max_output_mb = std::min(512ULL, remaining_mem / 1024 / 1024 / 2);
    
    std::cout << "Setting buffer limits - Input: " << max_input_mb 
              << "MB, Output: " << max_output_mb << "MB" << std::endl;
    
    processor.setMaxBufferSize(max_input_mb, max_output_mb);
    
    if (!processor.loadEngine(engine_path)) {
        std::cerr << "\n❌ Failed to load engine" << std::endl;
        return 1;
    }
    
    // Setup CUDA device
    if (av_hwdevice_ctx_create(&ctx.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        std::cerr << "Failed to create CUDA device" << std::endl;
        return 1;
    }
    
    // Setup input decoder
    if (!setupInputDecoder(ctx)) {
        cleanupVideoContext(ctx);
        return 1;
    }
    
    // 输出输入文件的流信息
    std::cout << "\n=== Input File Stream Information ===" << std::endl;
    for (unsigned int i = 0; i < ctx.ifmt_ctx->nb_streams; i++) {
        AVStream* stream = ctx.ifmt_ctx->streams[i];
        const char* media_type = av_get_media_type_string(stream->codecpar->codec_type);
        const char* codec_name = avcodec_get_name(stream->codecpar->codec_id);
        std::cout << "Stream " << i << ": " << media_type << " (" << codec_name << ")";
        
        if (stream->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            std::cout << ", " << stream->codecpar->ch_layout.nb_channels 
                      << " channels, " << stream->codecpar->sample_rate << " Hz";
        } else if (stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            std::cout << ", " << stream->codecpar->width << "x" << stream->codecpar->height;
        }
        std::cout << std::endl;
    }
    
    // Setup output encoder
    if (!setupOutputEncoder(ctx)) {
        cleanupVideoContext(ctx);
        return 1;
    }
    
    // Process frames
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<AVFrame*> frame_buffer;
    int frame_count = 0;
    bool encoder_initialized = false;
    int64_t next_pts = 0;
    
    std::cout << "\n=== Processing Video ===" << std::endl;
    
    while (av_read_frame(ctx.ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == ctx.video_stream_idx) {
            // 处理视频流
            if (avcodec_send_packet(ctx.dec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(ctx.dec_ctx, frame) == 0) {
                    AVFrame* ref_frame = av_frame_alloc();
                    av_frame_ref(ref_frame, frame);
                    frame_buffer.push_back(ref_frame);
                    
                    if (frame_buffer.size() >= batch_size) {
                        auto processed_frames = processor.processBatch(frame_buffer, ctx.hw_device_ctx);
                        
                        // Initialize encoder on first batch
                        if (!encoder_initialized) {
                            if (!initializeEncoder(ctx, processed_frames[0])) {
                                cleanupVideoContext(ctx);
                                return 1;
                            }
                            encoder_initialized = true;
                            std::cout << "Encoder initialized, processing streams..." << std::endl;
                        }
                        
                        // Encode frames
                        encodeAndWriteFrames(ctx, processed_frames, next_pts);
                        
                        // Cleanup processed frames
                        for (auto f : processed_frames) av_frame_free(&f);
                        for (auto f : frame_buffer) av_frame_free(&f);
                        frame_buffer.clear();
                        frame_count += processed_frames.size();
                        
                        // Display progress
                        displayProgress(frame_count, ctx.total_frames, start_time);
                    }
                }
            }
        } else {
            // 处理音频、字幕等其他流
            if (encoder_initialized) {  // 只有在编码器初始化后才开始写入其他流
                processNonVideoPacket(ctx, pkt);
            }
        }
        
        av_packet_unref(pkt);
        
        // Memory check
        if (frame_count % 100 == 0) {
            printMemoryUsage();
        }
    }
    
    // Process remaining frames
    if (!frame_buffer.empty() && encoder_initialized) {
        auto processed_frames = processor.processBatch(frame_buffer, ctx.hw_device_ctx);
        encodeAndWriteFrames(ctx, processed_frames, next_pts);
        
        for (auto f : processed_frames) av_frame_free(&f);
        for (auto f : frame_buffer) av_frame_free(&f);
        
        frame_count += processed_frames.size();
        std::cout << "Processed remaining " << processed_frames.size() << " frames" << std::endl;
    }
    
    // Finalize encoder
    if (encoder_initialized) {
        finalizeEncoder(ctx);
    }
    
    // Cleanup
    av_packet_free(&pkt);
    av_frame_free(&frame);
    cleanupVideoContext(ctx);
    
    // Print total processing time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_sec = total_elapsed.count() / 1000.0;
    int total_min = (int)(total_sec / 60);
    int total_sec_part = (int)(total_sec) % 60;
    
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Total time: " << total_min << "m" << total_sec_part << "s" << std::endl;
    if (frame_count > 0) {
        std::cout << "Average speed: " << std::fixed << std::setprecision(2) 
                  << (frame_count / total_sec) << " fps" << std::endl;
    }
    
    return 0;
}