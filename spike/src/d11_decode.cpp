#include "main.h"

// 辅助函数：初始化解码器
static bool initialize_decoder(FFMepgContext* ctx, const std::string& inputFile) {
    // 打开输入文件
    int ret = avformat_open_input(&ctx->fmt_ctx, inputFile.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cout << "Failed to open input file: " << inputFile << std::endl;
        return false;
    }
    
    // 获取流信息
    if (avformat_find_stream_info(ctx->fmt_ctx, nullptr) < 0) {
        std::cout << "Failed to find stream info" << std::endl;
        return false;
    }
    
    // 找到视频流
    ctx->video_stream_idx = av_find_best_stream(ctx->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ctx->video_stream_idx < 0) {
        std::cout << "No video stream found" << std::endl;
        return false;
    }
    
    ctx->video_stream = ctx->fmt_ctx->streams[ctx->video_stream_idx];
    
    // 创建解码器
    const AVCodec* decoder = avcodec_find_decoder(ctx->video_stream->codecpar->codec_id);
    if (!decoder) {
        std::cout << "Failed to find decoder" << std::endl;
        return false;
    }
    
    ctx->codec_ctx = avcodec_alloc_context3(decoder);
    if (!ctx->codec_ctx) {
        std::cout << "Failed to allocate codec context" << std::endl;
        return false;
    }
    
    // 复制参数并设置硬件加速
    if (avcodec_parameters_to_context(ctx->codec_ctx, ctx->video_stream->codecpar) < 0) {
        std::cout << "Failed to copy codec parameters" << std::endl;
        return false;
    }
    
    // 尝试设置D3D11硬件加速
    if (av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0) >= 0) {
        ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);
        std::cout << "D3D11 hardware acceleration enabled" << std::endl;
    } else {
        std::cout << "Failed to create D3D11 context, using software decoding" << std::endl;
    }
    
    // 打开解码器
    if (avcodec_open2(ctx->codec_ctx, decoder, nullptr) < 0) {
        std::cout << "Failed to open codec" << std::endl;
        return false;
    }
    
    ctx->current_file = inputFile;
    ctx->initialized = true;
    return true;
}

// 辅助函数：清理上下文资源
void cleanup_context(FFMepgContext* ctx) {
    if (ctx->codec_ctx) {
        avcodec_free_context(&ctx->codec_ctx);
    }
    if (ctx->fmt_ctx) {
        avformat_close_input(&ctx->fmt_ctx);
    }
    if (ctx->hw_device_ctx) {
        av_buffer_unref(&ctx->hw_device_ctx);
    }
    
    ctx->video_stream_idx = -1;
    ctx->video_stream = nullptr;
    ctx->current_file.clear();
    ctx->initialized = false;
}

AVFrame* d11_decode(FFMepgContext* ctx, const std::string& inputFile, int theIndex) {
    if (!ctx) {
        return nullptr;
    }
    
    // 检查是否需要重新初始化
    if (!ctx->initialized || ctx->current_file != inputFile) {
        cleanup_context(ctx);
        if (!initialize_decoder(ctx, inputFile)) {
            return nullptr;
        }
    }
    
    // 计算目标时间戳并跳转
    int64_t target_frame = theIndex;
    int64_t timestamp = av_rescale_q(target_frame, av_inv_q(ctx->video_stream->avg_frame_rate), ctx->video_stream->time_base);
    
    if (av_seek_frame(ctx->fmt_ctx, ctx->video_stream_idx, timestamp, AVSEEK_FLAG_BACKWARD) < 0) {
        std::cout << "Failed to seek to frame " << theIndex << std::endl;
        return nullptr;
    }
    
    avcodec_flush_buffers(ctx->codec_ctx);
    
    // 解码循环
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();
    AVFrame* result_frame = nullptr;
    
    if (!packet || !frame || !sw_frame) {
        goto cleanup;
    }
    
    while (av_read_frame(ctx->fmt_ctx, packet) >= 0) {
        if (packet->stream_index != ctx->video_stream_idx) {
            av_packet_unref(packet);
            continue;
        }
        
        if (avcodec_send_packet(ctx->codec_ctx, packet) < 0) {
            av_packet_unref(packet);
            continue;
        }
        
        while (avcodec_receive_frame(ctx->codec_ctx, frame) >= 0) {
            // 检查是否到达目标帧
            if (frame->pts != AV_NOPTS_VALUE) {
                int64_t current_pts = frame->pts;
                if (ctx->video_stream->start_time != AV_NOPTS_VALUE) {
                    current_pts -= ctx->video_stream->start_time;
                }
                
                if (current_pts >= timestamp) {
                    // 处理硬件帧传输
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        if (av_hwframe_transfer_data(sw_frame, frame, 0) >= 0) {
                            result_frame = av_frame_clone(sw_frame);
                        }
                    } else {
                        result_frame = av_frame_clone(frame);
                    }
                    
                    if (result_frame) {
                        std::cout << "Frame " << theIndex << " decoded successfully" << std::endl;
                    }
                    goto cleanup;
                }
            }
        }
        av_packet_unref(packet);
    }
    
cleanup:
    if (packet) av_packet_free(&packet);
    if (frame) av_frame_free(&frame);
    if (sw_frame) av_frame_free(&sw_frame);
    
    return result_frame;
}