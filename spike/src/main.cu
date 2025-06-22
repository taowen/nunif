#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>  // 添加这个头文件用于文件状态检查

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

// Simple logger
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

// 添加异步处理结构
struct AsyncFrame {
    AVFrame* input_frame = nullptr;
    AVFrame* output_frame = nullptr;
    float* d_input_buffer = nullptr;
    float* d_output_buffer = nullptr;
    cudaEvent_t decode_event;
    cudaEvent_t inference_event;
    cudaEvent_t encode_event;
    int64_t pts = 0;
    bool in_use = false;
};

// Main processing class
class VideoProcessor {
public:
    static const int PIPELINE_DEPTH = 3;  // pipeline深度
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // 异步处理相关
    std::vector<AsyncFrame> async_frames;
    cudaStream_t decode_stream;
    cudaStream_t inference_stream;
    cudaStream_t encode_stream;
    
    size_t buffer_size_per_frame = 0;
    size_t max_input_size = 0;
    size_t max_output_size = 0;
    
    Logger logger;
    
    // 添加缓存相关方法
    std::string getCacheFilePath(const std::string& onnx_path) {
        // 基于ONNX文件路径生成缓存文件路径
        std::string cache_path = onnx_path;
        size_t pos = cache_path.find_last_of('.');
        if (pos != std::string::npos) {
            cache_path = cache_path.substr(0, pos);
        }
        cache_path += ".trt_cache";
        return cache_path;
    }
    
    bool isCacheValid(const std::string& onnx_path, const std::string& cache_path) {
        // 检查缓存文件是否存在
        std::ifstream cache_file(cache_path, std::ios::binary);
        if (!cache_file.good()) {
            return false;
        }
        cache_file.close();
        
        // 检查ONNX文件和缓存文件的修改时间
        struct stat onnx_stat, cache_stat;
        if (stat(onnx_path.c_str(), &onnx_stat) != 0 || stat(cache_path.c_str(), &cache_stat) != 0) {
            return false;
        }
        
        // 如果ONNX文件比缓存文件新，则缓存无效
        return cache_stat.st_mtime >= onnx_stat.st_mtime;
    }
    
    bool loadEngineFromCache(const std::string& cache_path) {
        std::cout << "Loading TensorRT engine from cache: " << cache_path << std::endl;
        
        std::ifstream cache_file(cache_path, std::ios::binary);
        if (!cache_file.good()) {
            std::cerr << "Failed to open cache file for reading" << std::endl;
            return false;
        }
        
        // 读取文件大小
        cache_file.seekg(0, std::ios::end);
        size_t engine_size = cache_file.tellg();
        cache_file.seekg(0, std::ios::beg);
        
        if (engine_size == 0) {
            std::cerr << "Cache file is empty" << std::endl;
            return false;
        }
        
        // 读取引擎数据
        std::vector<char> engine_data(engine_size);
        cache_file.read(engine_data.data(), engine_size);
        cache_file.close();
        
        // 创建runtime并反序列化引擎
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), engine_size));
        if (!engine) {
            std::cerr << "Failed to deserialize cached engine" << std::endl;
            return false;
        }
        
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context from cached engine" << std::endl;
            return false;
        }
        
        std::cout << "Successfully loaded engine from cache" << std::endl;
        return true;
    }
    
    bool saveEngineToCache(const std::string& cache_path, nvinfer1::IHostMemory* serialized_engine) {
        std::cout << "Saving TensorRT engine to cache: " << cache_path << std::endl;
        
        std::ofstream cache_file(cache_path, std::ios::binary);
        if (!cache_file.good()) {
            std::cerr << "Failed to create cache file for writing" << std::endl;
            return false;
        }
        
        cache_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        cache_file.close();
        
        if (cache_file.good()) {
            std::cout << "Engine cached successfully (" << serialized_engine->size() / 1024 / 1024 << " MB)" << std::endl;
            return true;
        } else {
            std::cerr << "Failed to write engine cache" << std::endl;
            return false;
        }
    }
    
public:
    ~VideoProcessor() {
        cleanup();
    }
    
    void cleanup() {
        cudaDeviceSynchronize();
        
        // 清理异步帧
        for (auto& async_frame : async_frames) {
            if (async_frame.d_input_buffer) cudaFree(async_frame.d_input_buffer);
            if (async_frame.d_output_buffer) cudaFree(async_frame.d_output_buffer);
            if (async_frame.input_frame) av_frame_free(&async_frame.input_frame);
            if (async_frame.output_frame) av_frame_free(&async_frame.output_frame);
            cudaEventDestroy(async_frame.decode_event);
            cudaEventDestroy(async_frame.inference_event);
            cudaEventDestroy(async_frame.encode_event);
        }
        async_frames.clear();
        
        // 清理CUDA流
        cudaStreamDestroy(decode_stream);
        cudaStreamDestroy(inference_stream);
        cudaStreamDestroy(encode_stream);
        
        context.reset();
        engine.reset();
        runtime.reset();
    }
    
    bool loadOnnx(const std::string& onnx_path) {
        std::string cache_path = getCacheFilePath(onnx_path);
        
        if (isCacheValid(onnx_path, cache_path)) {
            if (loadEngineFromCache(cache_path)) {
                return initializeAsyncPipeline();
            }
        }
        
        if (buildEngineFromOnnx(onnx_path, cache_path)) {
            return initializeAsyncPipeline();
        }
        return false;
    }
    
    // 初始化异步pipeline
    bool initializeAsyncPipeline() {
        // 创建CUDA流
        cudaStreamCreate(&decode_stream);
        cudaStreamCreate(&inference_stream);
        cudaStreamCreate(&encode_stream);
        
        // 初始化异步帧缓冲
        async_frames.resize(PIPELINE_DEPTH);
        for (int i = 0; i < PIPELINE_DEPTH; i++) {
            AsyncFrame& frame = async_frames[i];
            
            // 创建事件
            cudaEventCreate(&frame.decode_event);
            cudaEventCreate(&frame.inference_event);
            cudaEventCreate(&frame.encode_event);
            
            frame.in_use = false;
        }
        
        std::cout << "Async pipeline initialized with depth " << PIPELINE_DEPTH << std::endl;
        return true;
    }
    
    // 设置缓冲区大小
    void setMaxBufferSize(size_t max_input_mb, size_t max_output_mb) {
        max_input_size = max_input_mb * 1024 * 1024;
        max_output_size = max_output_mb * 1024 * 1024;
    }
    
    // 获取可用的异步帧槽位
    AsyncFrame* getAvailableFrame() {
        for (auto& frame : async_frames) {
            if (!frame.in_use) {
                frame.in_use = true;
                return &frame;
            }
        }
        return nullptr;
    }
    
    // 异步解码阶段
    bool asyncDecode(AsyncFrame* async_frame, AVFrame* input_frame, AVBufferRef* hw_device_ctx) {
        const int height = input_frame->height;
        const int width = input_frame->width;
        
        // 计算NV12格式的数据大小
        size_t nv12_elements = height * width + (height / 2) * width;
        size_t nv12_size_uint8 = nv12_elements * sizeof(uint8_t);
        
        // 分配或重用缓冲区
        if (!async_frame->d_input_buffer || buffer_size_per_frame != nv12_size_uint8) {
            if (async_frame->d_input_buffer) cudaFree(async_frame->d_input_buffer);
            cudaMalloc(&async_frame->d_input_buffer, nv12_size_uint8);
            buffer_size_per_frame = nv12_size_uint8;
        }
        
        // 异步复制NV12数据到GPU
        // Y平面
        cudaMemcpy2DAsync(async_frame->d_input_buffer, width,
                         input_frame->data[0], input_frame->linesize[0],
                         width, height,
                         cudaMemcpyDeviceToDevice, decode_stream);
        
        // UV平面
        cudaMemcpy2DAsync((uint8_t*)async_frame->d_input_buffer + height * width, width,
                         input_frame->data[1], input_frame->linesize[1],
                         width, height / 2,
                         cudaMemcpyDeviceToDevice, decode_stream);
        
        // 记录解码完成事件
        cudaEventRecord(async_frame->decode_event, decode_stream);
        
        // 保存帧信息
        async_frame->pts = input_frame->pts;
        
        return true;
    }
    
    // 异步推理阶段
    bool asyncInference(AsyncFrame* async_frame, int width, int height) {
        // 等待解码完成
        cudaStreamWaitEvent(inference_stream, async_frame->decode_event, 0);
        
        // Setup TensorRT inference
        const char* input_name = engine->getIOTensorName(0);
        const char* output_name = engine->getIOTensorName(1);
        
        // 设置输入形状
        nvinfer1::Dims input_shape;
        input_shape.nbDims = 2;
        input_shape.d[0] = height + height / 2;
        input_shape.d[1] = width;
        
        context->setInputShape(input_name, input_shape);
        auto output_dims = context->getTensorShape(output_name);
        
        // 计算输出大小
        size_t output_elements = 1;
        for(int j = 0; j < output_dims.nbDims; ++j) {
            output_elements *= output_dims.d[j];
        }
        size_t required_output_size = output_elements * sizeof(uint8_t);
        
        // 分配输出缓冲区
        if (!async_frame->d_output_buffer) {
            cudaMalloc(&async_frame->d_output_buffer, required_output_size);
        }
        
        // 设置tensor地址
        context->setTensorAddress(input_name, async_frame->d_input_buffer);
        context->setTensorAddress(output_name, async_frame->d_output_buffer);
        
        // 异步执行推理
        context->enqueueV3(inference_stream);
        
        // 记录推理完成事件
        cudaEventRecord(async_frame->inference_event, inference_stream);
        
        return true;
    }
    
    // 异步编码准备阶段
    AVFrame* asyncEncodePrep(AsyncFrame* async_frame, int width, int height, AVBufferRef* hw_device_ctx) {
        // 等待推理完成
        cudaStreamWaitEvent(encode_stream, async_frame->inference_event, 0);
        
        // 创建输出帧
        int out_H = height;
        int out_W = width;
        
        AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
        AVHWFramesContext* hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
        hw_frames_ctx->format = AV_PIX_FMT_CUDA;
        hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
        hw_frames_ctx->width = out_W;
        hw_frames_ctx->height = out_H;
        hw_frames_ctx->initial_pool_size = 2;
        av_hwframe_ctx_init(hw_frames_ref);
        
        if (!async_frame->output_frame) {
            async_frame->output_frame = av_frame_alloc();
        }
        
        av_hwframe_get_buffer(hw_frames_ref, async_frame->output_frame, 0);
        
        // 异步复制推理结果到输出帧
        // Y平面
        cudaMemcpy2DAsync(async_frame->output_frame->data[0], async_frame->output_frame->linesize[0],
                         async_frame->d_output_buffer, out_W,
                         out_W, out_H,
                         cudaMemcpyDeviceToDevice, encode_stream);
        
        // UV平面
        cudaMemcpy2DAsync(async_frame->output_frame->data[1], async_frame->output_frame->linesize[1],
                         (uint8_t*)async_frame->d_output_buffer + out_H * out_W, out_W,
                         out_W, out_H / 2,
                         cudaMemcpyDeviceToDevice, encode_stream);
        
        // 记录编码准备完成事件
        cudaEventRecord(async_frame->encode_event, encode_stream);
        
        // 设置PTS
        async_frame->output_frame->pts = async_frame->pts;
        
        av_buffer_unref(&hw_frames_ref);
        return async_frame->output_frame;
    }
    
    // 检查异步帧是否准备好编码
    bool isFrameReadyForEncode(AsyncFrame* async_frame) {
        return cudaEventQuery(async_frame->encode_event) == cudaSuccess;
    }
    
    // 释放异步帧
    void releaseAsyncFrame(AsyncFrame* async_frame) {
        if (async_frame->output_frame) {
            av_frame_free(&async_frame->output_frame);
            async_frame->output_frame = nullptr;
        }
        async_frame->in_use = false;
    }

private:
    bool buildEngineFromOnnx(const std::string& onnx_path, const std::string& cache_path) {
        // Create builder, network and parser
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            std::cerr << "Failed to create TensorRT builder" << std::endl;
            return false;
        }
        
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create TensorRT network" << std::endl;
            return false;
        }
        
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        // Parse ONNX file
        std::cout << "Parsing ONNX file: " << onnx_path << std::endl;
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file" << std::endl;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
            }
            return false;
        }
        
        // Build engine
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config" << std::endl;
            return false;
        }
        
        // Get GPU memory info for better memory management
        size_t total_mem, free_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        
        // Set memory pool size (use 80% of available memory, leave some for other operations)
        size_t workspace_size = (free_mem * 8) / 10;
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
        
        // Enable FP16 precision
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "FP16 mode enabled" << std::endl;
        }
        
        // Set optimization profile for dynamic shapes
        auto profile = builder->createOptimizationProfile();
        if (!profile) {
            std::cerr << "Failed to create optimization profile" << std::endl;
            return false;
        }
        
        auto input = network->getInput(0);
        auto inputName = input->getName();
        
        // NV12格式的形状约束：(height + height/2, width)
        // Min: 392 + 196 = 588, Opt: 392 + 196 = 588, Max: 2160 + 1080 = 3240
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{588, 392});
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{588, 392});
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{3240, 3840});
        
        config->addOptimizationProfile(profile);
        
        std::cout << "Building TensorRT engine (this may take several minutes)..." << std::endl;
        auto start_build = std::chrono::high_resolution_clock::now();
        
        auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized_engine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        auto end_build = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::seconds>(end_build - start_build);
        std::cout << "Engine built successfully in " << build_time.count() << " seconds" << std::endl;
        
        // Save engine to cache
        saveEngineToCache(cache_path, serialized_engine.get());
        
        // Create runtime and deserialize engine
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Print memory usage after engine creation
        cudaMemGetInfo(&free_mem, &total_mem);
        size_t engine_required = engine->getDeviceMemorySize();
        
        std::cout << "Total GPU Memory: " << total_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Available GPU Memory: " << free_mem / 1024 / 1024 << " MB" << std::endl;
        std::cout << "Engine Required Memory: " << engine_required / 1024 / 1024 << " MB" << std::endl;

        return true;
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
        std::cerr << "Usage: " << argv[0] << " <input_video> <output_video>" << std::endl;
        return 1;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string onnx_path = "stereo_module_nv12.onnx";  // 使用新的NV12模型
    
    // Initialize video context
    VideoContext ctx;
    ctx.input_path = argv[1];
    ctx.output_path = argv[2];
    
    std::cout << "=== Async Pipeline Video Processing Setup ===" << std::endl;
    std::cout << "ONNX Model: " << onnx_path << std::endl;
    std::cout << "Input: " << ctx.input_path << std::endl;
    std::cout << "Output: " << ctx.output_path << std::endl;
    
    VideoProcessor processor;
    
    // Setup memory management
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\n=== Initial GPU Memory Status ===" << std::endl;
    printMemoryUsage();
    
    // 为pipeline分配更多内存
    size_t estimated_engine_mem = 2ULL * 1024 * 1024 * 1024;
    size_t remaining_mem = (free_mem > estimated_engine_mem) ? (free_mem - estimated_engine_mem) : (free_mem / 4);
    
    // 为3个pipeline阶段分配缓冲区
    size_t max_input_mb = std::min(128ULL, remaining_mem / 1024 / 1024 / 6);
    size_t max_output_mb = std::min(256ULL, remaining_mem / 1024 / 1024 / 4);
    
    std::cout << "Setting async buffer limits - Input: " << max_input_mb 
              << "MB, Output: " << max_output_mb << "MB" << std::endl;
    
    processor.setMaxBufferSize(max_input_mb, max_output_mb);
    
    std::cout << "\n=== Loading/Building TensorRT Engine ===" << std::endl;
    if (!processor.loadOnnx(onnx_path)) {
        std::cerr << "\n❌ Failed to load ONNX model" << std::endl;
        return 1;
    }
    
    // Setup CUDA device
    if (av_hwdevice_ctx_create(&ctx.hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        std::cerr << "Failed to create CUDA device" << std::endl;
        return 1;
    }
    
    // Setup input decoder and output encoder
    if (!setupInputDecoder(ctx)) {
        cleanupVideoContext(ctx);
        return 1;
    }
    
    if (!setupOutputEncoder(ctx)) {
        cleanupVideoContext(ctx);
        return 1;
    }
    
    // Pipeline处理循环
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int frame_count = 0;
    bool encoder_initialized = false;
    int64_t next_pts = 0;
    
    std::vector<AsyncFrame*> pending_frames;  // 等待编码的帧
    
    std::cout << "\n=== Processing Video with Async Pipeline ===" << std::endl;
    
    while (av_read_frame(ctx.ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == ctx.video_stream_idx) {
            // 处理视频流
            if (avcodec_send_packet(ctx.dec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(ctx.dec_ctx, frame) == 0) {
                    // 获取可用的异步帧槽位
                    AsyncFrame* async_frame = processor.getAvailableFrame();
                    if (async_frame) {
                        // 启动异步解码
                        if (processor.asyncDecode(async_frame, frame, ctx.hw_device_ctx)) {
                            // 启动异步推理
                            processor.asyncInference(async_frame, frame->width, frame->height);
                            
                            // 准备异步编码
                            AVFrame* output_frame = processor.asyncEncodePrep(async_frame, frame->width, frame->height, ctx.hw_device_ctx);
                            
                            if (output_frame) {
                                if (!encoder_initialized) {
                                    if (!initializeEncoder(ctx, output_frame)) {
                                        cleanupVideoContext(ctx);
                                        return 1;
                                    }
                                    encoder_initialized = true;
                                    std::cout << "Encoder initialized with async pipeline" << std::endl;
                                }
                                
                                // 添加到待编码队列
                                pending_frames.push_back(async_frame);
                            }
                        }
                    } else {
                        // 没有可用槽位，需要等待编码完成
                        std::cout << "Pipeline full, waiting for encode completion..." << std::endl;
                        cudaDeviceSynchronize();  // 强制同步，等待所有操作完成
                        
                        // 处理所有待编码帧
                        for (auto it = pending_frames.begin(); it != pending_frames.end();) {
                            AsyncFrame* pending_frame = *it;
                            if (processor.isFrameReadyForEncode(pending_frame)) {
                                // 编码帧
                                pending_frame->output_frame->pts = next_pts++;
                                std::vector<AVFrame*> single_frame = {pending_frame->output_frame};
                                encodeAndWriteFrames(ctx, single_frame, next_pts);
                                
                                // 释放异步帧
                                processor.releaseAsyncFrame(pending_frame);
                                it = pending_frames.erase(it);
                                frame_count++;
                            } else {
                                ++it;
                            }
                        }
                    }
                }
            }
        } else {
            // 处理音频、字幕等其他流
            if (encoder_initialized) {
                processNonVideoPacket(ctx, pkt);
            }
        }
        
        // 检查并编码已准备好的帧
        for (auto it = pending_frames.begin(); it != pending_frames.end();) {
            AsyncFrame* pending_frame = *it;
            if (processor.isFrameReadyForEncode(pending_frame)) {
                // 编码帧
                pending_frame->output_frame->pts = next_pts++;
                std::vector<AVFrame*> single_frame = {pending_frame->output_frame};
                encodeAndWriteFrames(ctx, single_frame, next_pts);
                
                // 释放异步帧
                processor.releaseAsyncFrame(pending_frame);
                it = pending_frames.erase(it);
                frame_count++;
                
                // Display progress
                if (frame_count % 30 == 0) {
                    displayProgress(frame_count, ctx.total_frames, start_time);
                }
            } else {
                ++it;
            }
        }
        
        av_packet_unref(pkt);
        
        // Memory check
        if (frame_count % 100 == 0) {
            printMemoryUsage();
        }
    }
    
    // 处理剩余的待编码帧
    std::cout << "Processing remaining frames..." << std::endl;
    cudaDeviceSynchronize();  // 确保所有GPU操作完成
    
    for (AsyncFrame* pending_frame : pending_frames) {
        pending_frame->output_frame->pts = next_pts++;
        std::vector<AVFrame*> single_frame = {pending_frame->output_frame};
        encodeAndWriteFrames(ctx, single_frame, next_pts);
        processor.releaseAsyncFrame(pending_frame);
        frame_count++;
    }
    
    // Finalize encoder
    if (encoder_initialized) {
        finalizeEncoder(ctx);
    }
    
    // Cleanup
    av_packet_free(&pkt);
    av_frame_free(&frame);
    cleanupVideoContext(ctx);
    
    // Print performance statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_sec = total_elapsed.count() / 1000.0;
    int total_min = (int)(total_sec / 60);
    int total_sec_part = (int)(total_sec) % 60;
    
    std::cout << "\n=== Async Pipeline Processing Complete ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Total time: " << total_min << "m" << total_sec_part << "s" << std::endl;
    if (frame_count > 0) {
        std::cout << "Average speed: " << std::fixed << std::setprecision(2) 
                  << (frame_count / total_sec) << " fps" << std::endl;
    }
    std::cout << "Pipeline depth: " << VideoProcessor::PIPELINE_DEPTH << " frames" << std::endl;
    
    return 0;
}