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
        
        std::cout << "Engine loaded successfully!" << std::endl;
        return true;
    }
    
    std::vector<AVFrame*> processBatch(const std::vector<AVFrame*>& frames, AVBufferRef* hw_device_ctx) {
        if (frames.empty()) return {};
        
        const int batch_size = frames.size();
        const int height = frames[0]->height;
        const int width = frames[0]->width;
        const int channels = 3;
        
        // Allocate GPU buffers
        size_t required_input_size = batch_size * channels * height * width * sizeof(float);
        if (required_input_size > input_size) {
            if (d_input) cudaFree(d_input);
            cudaMalloc(&d_input, required_input_size);
            input_size = required_input_size;
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
        
        context->setTensorAddress(input_name, d_input);
        context->setTensorAddress(output_name, d_output);
        
        // Run inference
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        
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
        hw_frames_ctx->initial_pool_size = batch_size + 2;
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <engine_path> <input_video> <output_video> [batch_size]" << std::endl;
        return 1;
    }
    
    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::string engine_path = argv[1];
    std::string input_path = argv[2];
    std::string output_path = argv[3];
    int batch_size = (argc > 4) ? std::stoi(argv[4]) : 4;
    
    VideoProcessor processor;
    if (!processor.loadEngine(engine_path)) {
        std::cerr << "Failed to load engine" << std::endl;
        return 1;
    }
    
    // Setup CUDA device
    AVBufferRef* hw_device_ctx = nullptr;
    if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        std::cerr << "Failed to create CUDA device" << std::endl;
        return 1;
    }
    
    // Open input
    AVFormatContext* ifmt_ctx = nullptr;
    if (avformat_open_input(&ifmt_ctx, input_path.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file" << std::endl;
        return 1;
    }
    avformat_find_stream_info(ifmt_ctx, nullptr);
    
    int video_stream_idx = av_find_best_stream(ifmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    AVCodecParameters* codecpar = ifmt_ctx->streams[video_stream_idx]->codecpar;
    
    // Calculate total frames for progress estimation
    AVStream* video_stream = ifmt_ctx->streams[video_stream_idx];
    int64_t total_frames = 0;
    if (video_stream->nb_frames > 0) {
        total_frames = video_stream->nb_frames;
    } else if (video_stream->duration > 0 && video_stream->r_frame_rate.num > 0) {
        // Estimate from duration and frame rate
        double duration_sec = (double)video_stream->duration * av_q2d(video_stream->time_base);
        double fps = av_q2d(video_stream->r_frame_rate);
        total_frames = (int64_t)(duration_sec * fps);
    }
    
    if (total_frames > 0) {
        std::cout << "Estimated total frames: " << total_frames << std::endl;
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
    
    AVCodecContext* dec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(dec_ctx, codecpar);
    dec_ctx->pkt_timebase = ifmt_ctx->streams[video_stream_idx]->time_base;
    if (strstr(decoder->name, "cuvid")) {
        dec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    }
    avcodec_open2(dec_ctx, decoder, nullptr);
    
    // Setup output
    AVFormatContext* ofmt_ctx = nullptr;
    int ret = avformat_alloc_output_context2(&ofmt_ctx, nullptr, nullptr, output_path.c_str());
    if (ret < 0 || !ofmt_ctx) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        std::cerr << "Could not create output context: " << error_buf << std::endl;
        return 1;
    }
    
    AVStream* out_stream = avformat_new_stream(ofmt_ctx, nullptr);
    if (!out_stream) {
        std::cerr << "Failed to create output stream" << std::endl;
        return 1;
    }
    
    const AVCodec* encoder = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!encoder) {
        std::cerr << "hevc_nvenc encoder not found" << std::endl;
        return 1;
    }
    AVCodecContext* enc_ctx = avcodec_alloc_context3(encoder);
    
    // Process frames
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<AVFrame*> frame_buffer;
    int frame_count = 0;
    bool encoder_initialized = false;
    int64_t next_pts = 0;
    
    while (av_read_frame(ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index != video_stream_idx) {
            av_packet_unref(pkt);
            continue;
        }
        
        if (avcodec_send_packet(dec_ctx, pkt) == 0) {
            while (avcodec_receive_frame(dec_ctx, frame) == 0) {
                AVFrame* ref_frame = av_frame_alloc();
                av_frame_ref(ref_frame, frame);
                frame_buffer.push_back(ref_frame);
                
                if (frame_buffer.size() >= batch_size) {
                    auto processed_frames = processor.processBatch(frame_buffer, hw_device_ctx);
                    
                    // Initialize encoder on first batch
                    if (!encoder_initialized) {
                        enc_ctx->width = processed_frames[0]->width;
                        enc_ctx->height = processed_frames[0]->height;
                        enc_ctx->pix_fmt = AV_PIX_FMT_CUDA;
                        enc_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                        
                        // Set frame rate and time base properly
                        AVRational input_framerate = av_guess_frame_rate(ifmt_ctx, ifmt_ctx->streams[video_stream_idx], nullptr);
                        if (input_framerate.num > 0 && input_framerate.den > 0) {
                            enc_ctx->framerate = input_framerate;
                            enc_ctx->time_base = av_inv_q(input_framerate);
                        } else {
                            enc_ctx->framerate = {30, 1};
                            enc_ctx->time_base = {1, 30};
                        }
                        
                        // Create hw_frames_ctx for the encoder
                        AVBufferRef* enc_hw_frames_ref = av_hwframe_ctx_alloc(hw_device_ctx);
                        AVHWFramesContext* enc_hw_frames_ctx = (AVHWFramesContext*)enc_hw_frames_ref->data;
                        enc_hw_frames_ctx->format = AV_PIX_FMT_CUDA;
                        enc_hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
                        enc_hw_frames_ctx->width = processed_frames[0]->width;
                        enc_hw_frames_ctx->height = processed_frames[0]->height;
                        enc_hw_frames_ctx->initial_pool_size = 8;
                        ret = av_hwframe_ctx_init(enc_hw_frames_ref);
                        if (ret < 0) {
                            char error_buf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(ret, error_buf, sizeof(error_buf));
                            std::cerr << "Failed to initialize hw_frames_ctx: " << error_buf << std::endl;
                            return 1;
                        }
                        enc_ctx->hw_frames_ctx = enc_hw_frames_ref;
                        
                        // Calculate bitrate based on resolution
                        int calculated_bitrate = calculateBitrate(processed_frames[0]->width, processed_frames[0]->height);
                        enc_ctx->bit_rate = calculated_bitrate;
                        
                        std::cout << "Resolution: " << processed_frames[0]->width << "x" << processed_frames[0]->height 
                                  << ", Calculated bitrate: " << calculated_bitrate / 1000000.0 << " Mbps" << std::endl;
                        
                        // Set encoding parameters
                        enc_ctx->gop_size = 30;
                        enc_ctx->max_b_frames = 0;  // Disable B-frames for better compatibility
                        
                        // Set HEVC-specific parameters
                        av_opt_set(enc_ctx->priv_data, "preset", "fast", 0);
                        av_opt_set(enc_ctx->priv_data, "profile", "main", 0);
                        av_opt_set(enc_ctx->priv_data, "level", "auto", 0);
                        av_opt_set(enc_ctx->priv_data, "tier", "main", 0);
                        
                        // Ensure global header is set for container
                        if (ofmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
                            enc_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
                        }
                        
                        ret = avcodec_open2(enc_ctx, encoder, nullptr);
                        if (ret < 0) {
                            char error_buf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(ret, error_buf, sizeof(error_buf));
                            std::cerr << "Failed to open encoder: " << error_buf << std::endl;
                            return 1;
                        }
                        
                        // Setup output stream parameters
                        ret = avcodec_parameters_from_context(out_stream->codecpar, enc_ctx);
                        if (ret < 0) {
                            char error_buf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(ret, error_buf, sizeof(error_buf));
                            std::cerr << "Failed to copy codec parameters: " << error_buf << std::endl;
                            return 1;
                        }
                        out_stream->time_base = enc_ctx->time_base;
                        
                        // Open output file and write header
                        ret = avio_open(&ofmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
                        if (ret < 0) {
                            char error_buf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(ret, error_buf, sizeof(error_buf));
                            std::cerr << "Could not open output file: " << error_buf << std::endl;
                            return 1;
                        }
                        
                        ret = avformat_write_header(ofmt_ctx, nullptr);
                        if (ret < 0) {
                            char error_buf[AV_ERROR_MAX_STRING_SIZE];
                            av_strerror(ret, error_buf, sizeof(error_buf));
                            std::cerr << "Error writing header: " << error_buf << std::endl;
                            return 1;
                        }
                        
                        encoder_initialized = true;
                        std::cout << "Encoder initialized and header written" << std::endl;
                    }
                    
                    // Encode frames
                    for (size_t i = 0; i < processed_frames.size(); i++) {
                        processed_frames[i]->pts = next_pts++;
                        
                        if (avcodec_send_frame(enc_ctx, processed_frames[i]) == 0) {
                            AVPacket* out_pkt = av_packet_alloc();
                            if (!out_pkt) continue;
                            
                            while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                                if (out_pkt->size > 0) {
                                    av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                                    out_pkt->stream_index = 0;
                                    
                                    ret = av_interleaved_write_frame(ofmt_ctx, out_pkt);
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
                        
                        av_frame_free(&processed_frames[i]);
                    }
                    
                    for (auto f : frame_buffer) av_frame_free(&f);
                    frame_buffer.clear();
                    frame_count += processed_frames.size();
                    
                    // Calculate and display progress with ETA
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
            }
        }
        av_packet_unref(pkt);
    }
    
    // Process remaining frames
    if (!frame_buffer.empty() && encoder_initialized) {
        auto processed_frames = processor.processBatch(frame_buffer, hw_device_ctx);
        for (size_t i = 0; i < processed_frames.size(); i++) {
            processed_frames[i]->pts = next_pts++;
            
            if (avcodec_send_frame(enc_ctx, processed_frames[i]) == 0) {
                AVPacket* out_pkt = av_packet_alloc();
                if (out_pkt) {
                    while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                        if (out_pkt->size > 0) {
                            av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                            out_pkt->stream_index = 0;
                            ret = av_interleaved_write_frame(ofmt_ctx, out_pkt);
                            if (ret < 0) {
                                std::cerr << "Error writing frame" << std::endl;
                            }
                        }
                        av_packet_unref(out_pkt);
                    }
                    av_packet_free(&out_pkt);
                }
            }
            av_frame_free(&processed_frames[i]);
        }
        for (auto f : frame_buffer) av_frame_free(&f);
    }
    
    // Finalize encoder
    if (encoder_initialized) {
        avcodec_send_frame(enc_ctx, nullptr);
        AVPacket* out_pkt = av_packet_alloc();
        if (out_pkt) {
            while (avcodec_receive_packet(enc_ctx, out_pkt) == 0) {
                if (out_pkt->size > 0) {
                    av_packet_rescale_ts(out_pkt, enc_ctx->time_base, out_stream->time_base);
                    out_pkt->stream_index = 0;
                    ret = av_interleaved_write_frame(ofmt_ctx, out_pkt);
                    if (ret < 0) {
                        std::cerr << "Error writing final frame" << std::endl;
                    }
                }
                av_packet_unref(out_pkt);
            }
            av_packet_free(&out_pkt);
        }
        
        av_write_trailer(ofmt_ctx);
    }
    
    // Cleanup
    if (ofmt_ctx && ofmt_ctx->pb) {
        avio_closep(&ofmt_ctx->pb);
    }
    avformat_free_context(ofmt_ctx);
    avformat_close_input(&ifmt_ctx);
    avcodec_free_context(&dec_ctx);
    avcodec_free_context(&enc_ctx);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    av_buffer_unref(&hw_device_ctx);
    
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