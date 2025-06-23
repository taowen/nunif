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

#include "context.h"
#include "onnx_utils.h"
#include "codec_utils.h"


// 处理单帧的函数
AVFrame* processFrame(ProcessorContext& ctx, AVFrame* frame, AVBufferRef* hw_frames_ref) {
    if (!frame) return nullptr;
    
    const int height = frame->height;
    const int width = frame->width;
    
    // 计算NV12格式的数据大小：Y平面 + UV平面(高度的一半)
    size_t nv12_elements = height * width + (height / 2) * width;
    size_t nv12_size_uint8 = nv12_elements * sizeof(uint8_t);  // uint8 大小
    
    // 限制缓冲区大小
    if (ctx.max_input_size > 0 && nv12_size_uint8 > ctx.max_input_size) {
        std::cerr << "Warning: Required input size exceeds limit" << std::endl;
        return nullptr;
    }
    
    // 分配输入缓冲区 (直接使用 uint8_t)
    if (nv12_size_uint8 > ctx.input_size) {
        if (ctx.d_input_nv12) cudaFree(ctx.d_input_nv12);
        cudaMalloc(&ctx.d_input_nv12, nv12_size_uint8);
        ctx.input_size = nv12_size_uint8;
        std::cout << "Allocated input buffer: " << nv12_size_uint8 / 1024 / 1024 << " MB" << std::endl;
    }
    
    // 直接将NV12数据从AVFrame复制到GPU内存（无需类型转换）
    // Y平面
    cudaMemcpy2D(ctx.d_input_nv12, width, 
                 frame->data[0], frame->linesize[0], 
                 width, height, 
                 cudaMemcpyDeviceToDevice);
    
    // UV平面
    cudaMemcpy2D((uint8_t*)ctx.d_input_nv12 + height * width, width,
                 frame->data[1], frame->linesize[1],
                 width, height / 2,
                 cudaMemcpyDeviceToDevice);
    
    cudaDeviceSynchronize();
    
    // Setup TensorRT inference
    const char* input_name = ctx.engine->getIOTensorName(0);
    const char* output_name = ctx.engine->getIOTensorName(1);
    
    // 设置输入形状：NV12格式为 (height + height/2, width)
    nvinfer1::Dims input_shape;
    input_shape.nbDims = 2;
    input_shape.d[0] = height + height / 2;  // Y平面高度 + UV平面高度
    input_shape.d[1] = width;
    
    ctx.context->setInputShape(input_name, input_shape);
    auto output_dims = ctx.context->getTensorShape(output_name);
    
    // 计算输出大小 (也是 uint8)
    size_t output_elements = 1;
    for(int j = 0; j < output_dims.nbDims; ++j) {
        output_elements *= output_dims.d[j];
    }
    size_t required_output_size = output_elements * sizeof(uint8_t);  // 使用 uint8 大小
    
    if (required_output_size > ctx.output_size) {
        if (ctx.d_output_nv12) cudaFree(ctx.d_output_nv12);
        cudaMalloc(&ctx.d_output_nv12, required_output_size);
        ctx.output_size = required_output_size;
    }
    
    // 设置tensor地址
    ctx.context->setTensorAddress(input_name, ctx.d_input_nv12);
    ctx.context->setTensorAddress(output_name, ctx.d_output_nv12);
    
    // 创建bindings数组
    std::vector<void*> bindings(ctx.engine->getNbIOTensors());
    for (int32_t i = 0, e = ctx.engine->getNbIOTensors(); i < e; i++) {
        auto const name = ctx.engine->getIOTensorName(i);
        if (std::string(name) == std::string(input_name)) {
            bindings[i] = ctx.d_input_nv12;
        } else if (std::string(name) == std::string(output_name)) {
            bindings[i] = ctx.d_output_nv12;
        }
    }
    
    // 执行推理
    bool status = ctx.context->executeV2(bindings.data());
    if (!status) {
        std::cerr << "TensorRT synchronous execution failed" << std::endl;
        return nullptr;
    }
    
    // 创建输出帧
    int out_H = output_dims.d[0] * 2 / 3;  // 从NV12格式恢复原始高度
    int out_W = output_dims.d[1];
    
    AVFrame* output_frame = av_frame_alloc();
    if (!output_frame) {
        std::cerr << "Failed to allocate output frame" << std::endl;
        return nullptr;
    }
    
    output_frame->width = out_W;
    output_frame->height = out_H;
    output_frame->format = AV_PIX_FMT_NV12; // Should be this format
    
    if (av_hwframe_get_buffer(hw_frames_ref, output_frame, 0) < 0) {
        std::cerr << "Failed to get buffer for output frame" << std::endl;
        av_frame_free(&output_frame);
        return nullptr;
    }
    
    // 直接将推理结果复制到输出帧（无需类型转换）
    // Y平面
    cudaMemcpy2D(output_frame->data[0], output_frame->linesize[0],
                 ctx.d_output_nv12, out_W,
                 out_W, out_H,
                 cudaMemcpyDeviceToDevice);
    
    // UV平面
    cudaMemcpy2D(output_frame->data[1], output_frame->linesize[1],
                 (uint8_t*)ctx.d_output_nv12 + out_H * out_W, out_W,
                 out_W, out_H / 2,
                 cudaMemcpyDeviceToDevice);
    
    cudaDeviceSynchronize();
    
    return output_frame;
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
    
    std::cout << "=== Video Processing Setup ===" << std::endl;
    std::cout << "ONNX Model: " << onnx_path << std::endl;
    std::cout << "Input: " << ctx.input_path << std::endl;
    std::cout << "Output: " << ctx.output_path << std::endl;
    
    ProcessorContext proc_ctx;  // 创建处理器上下文
    
    // Setup memory management
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "\n=== Initial GPU Memory Status ===" << std::endl;
    printMemoryUsage();
    
    size_t estimated_engine_mem = 2ULL * 1024 * 1024 * 1024;
    size_t remaining_mem = (free_mem > estimated_engine_mem) ? (free_mem - estimated_engine_mem) : (free_mem / 4);
    
    size_t max_input_mb = std::min(128ULL, remaining_mem / 1024 / 1024 / 3);  // NV12占用更少内存
    size_t max_output_mb = std::min(256ULL, remaining_mem / 1024 / 1024 / 2);
    
    std::cout << "Setting buffer limits - Input: " << max_input_mb 
              << "MB, Output: " << max_output_mb << "MB" << std::endl;
    
    proc_ctx.setMaxBufferSize(max_input_mb, max_output_mb);  // 使用 ProcessorContext 方法
    
    std::cout << "\n=== Loading/Building TensorRT Engine ===" << std::endl;
    if (!loadOnnxModel(proc_ctx, onnx_path)) {  // 使用新的全局函数
        std::cerr << "\n❌ Failed to load ONNX model" << std::endl;
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
    
    // Create hw_frames_ref for output frames
    AVBufferRef* hw_frames_ref_out = nullptr;
    if (ctx.video_stream_idx >= 0) {
        hw_frames_ref_out = av_hwframe_ctx_alloc(ctx.hw_device_ctx);
        AVHWFramesContext* hw_frames_ctx = (AVHWFramesContext*)hw_frames_ref_out->data;
        AVStream* video_stream = ctx.ifmt_ctx->streams[ctx.video_stream_idx];
        hw_frames_ctx->format = AV_PIX_FMT_CUDA;
        hw_frames_ctx->sw_format = AV_PIX_FMT_NV12;
        hw_frames_ctx->width = video_stream->codecpar->width;   // Initial size, will be updated by TRT
        hw_frames_ctx->height = video_stream->codecpar->height; // Initial size, will be updated by TRT
        hw_frames_ctx->initial_pool_size = 5;
        if (av_hwframe_ctx_init(hw_frames_ref_out) < 0) {
            std::cerr << "Failed to initialize output CUDA frame context" << std::endl;
            av_buffer_unref(&hw_frames_ref_out);
            cleanupVideoContext(ctx);
            return 1;
        }
    }
    
    // 修改处理循环，使用全局函数
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int frame_count = 0;
    bool encoder_initialized = false;
    int64_t next_pts = 0;
    
    std::cout << "\n=== Processing Video ===" << std::endl;
    
    while (av_read_frame(ctx.ifmt_ctx, pkt) >= 0) {
        if (pkt->stream_index == ctx.video_stream_idx) {
            // 处理视频流
            if (avcodec_send_packet(ctx.dec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(ctx.dec_ctx, frame) == 0) {
                    // 保存原始PTS，并转换到编码器的时间基准
                    int64_t original_pts = frame->pts;
                    
                    AVFrame* processed_frame = processFrame(proc_ctx, frame, hw_frames_ref_out);
                    av_frame_unref(frame); // Unref the input frame
                    
                    if (processed_frame) {
                        if (!encoder_initialized) {
                            if (!initializeEncoder(ctx, processed_frame)) {
                                cleanupVideoContext(ctx);
                                return 1;
                            }
                            encoder_initialized = true;
                            std::cout << "Encoder initialized, processing streams..." << std::endl;
                        }
                        
                        // 使用原始PTS，转换到编码器时间基准
                        if (original_pts != AV_NOPTS_VALUE) {
                            processed_frame->pts = av_rescale_q(original_pts, 
                                                              ctx.ifmt_ctx->streams[ctx.video_stream_idx]->time_base,
                                                              ctx.enc_ctx->time_base);
                        } else {
                            processed_frame->pts = next_pts;
                        }
                        next_pts = processed_frame->pts + 1;  // 更新next_pts
                        
                        // 编码单帧
                        std::vector<AVFrame*> single_frame = {processed_frame};
                        encodeAndWriteFrames(ctx, single_frame, next_pts);
                        
                        // 清理
                        av_frame_free(&processed_frame);
                        frame_count++;
                        
                        // Display progress
                        if (frame_count % 30 == 0) {  // 每30帧显示一次进度
                            displayProgress(frame_count, ctx.total_frames, start_time);
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
        
        av_packet_unref(pkt);
        
        // Memory check
        if (frame_count > 0 && frame_count % 100 == 0) {
            printMemoryUsage();
        }
    }
    
    // Finalize encoder
    if (encoder_initialized) {
        finalizeEncoder(ctx);
    }
    
    if (hw_frames_ref_out) {
        av_buffer_unref(&hw_frames_ref_out);
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