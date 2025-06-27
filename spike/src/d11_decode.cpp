#include "main.h"

AVFrame* d11_decode(FFMepgContext* ctx, const std::string& inputFile, int theIndex) {
    if (!ctx) {
        return nullptr;
    }
    
    // 如果是新文件或者未初始化，需要重新设置
    if (!ctx->initialized || ctx->current_file != inputFile) {
        // 清理之前的资源
        if (ctx->codec_ctx) {
            avcodec_free_context(&ctx->codec_ctx);
        }
        if (ctx->fmt_ctx) {
            avformat_close_input(&ctx->fmt_ctx);
        }
        if (ctx->hw_device_ctx) {
            av_buffer_unref(&ctx->hw_device_ctx);
        }
        
        // 重置状态
        ctx->video_stream_idx = -1;
        ctx->video_stream = nullptr;
        ctx->initialized = false;
        
        // 打开输入文件
        int ret = avformat_open_input(&ctx->fmt_ctx, inputFile.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cout << "Failed to open input file: " << inputFile << std::endl;
            return nullptr;
        }
        
        ret = avformat_find_stream_info(ctx->fmt_ctx, nullptr);
        if (ret < 0) {
            std::cout << "Failed to find stream info" << std::endl;
            return nullptr;
        }
        
        // 找到视频流
        for (unsigned int i = 0; i < ctx->fmt_ctx->nb_streams; i++) {
            if (ctx->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                ctx->video_stream_idx = i;
                ctx->video_stream = ctx->fmt_ctx->streams[i];
                break;
            }
        }
        
        if (ctx->video_stream_idx < 0) {
            std::cout << "No video stream found" << std::endl;
            return nullptr;
        }
        
        // 创建解码器
        const AVCodec* decoder = avcodec_find_decoder(ctx->video_stream->codecpar->codec_id);
        if (!decoder) {
            std::cout << "Failed to find decoder" << std::endl;
            return nullptr;
        }
        
        ctx->codec_ctx = avcodec_alloc_context3(decoder);
        if (!ctx->codec_ctx) {
            std::cout << "Failed to allocate codec context" << std::endl;
            return nullptr;
        }
        
        ret = avcodec_parameters_to_context(ctx->codec_ctx, ctx->video_stream->codecpar);
        if (ret < 0) {
            std::cout << "Failed to copy codec parameters" << std::endl;
            return nullptr;
        }
        
        // 设置 DX11 硬件加速
        ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret >= 0) {
            ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);
            std::cout << "DX11 hardware acceleration enabled" << std::endl;
        } else {
            std::cout << "Failed to create DX11 context, falling back to software decoding" << std::endl;
        }
        
        // 打开编解码器
        ret = avcodec_open2(ctx->codec_ctx, decoder, nullptr);
        if (ret < 0) {
            std::cout << "Failed to open codec" << std::endl;
            return nullptr;
        }
        
        ctx->current_file = inputFile;
        ctx->initialized = true;
    }
    
    // 计算目标帧的时间戳并跳转
    int64_t target_frame = theIndex;
    int64_t timestamp = av_rescale_q(target_frame, av_inv_q(ctx->video_stream->avg_frame_rate), ctx->video_stream->time_base);
    
    int ret = av_seek_frame(ctx->fmt_ctx, ctx->video_stream_idx, timestamp, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        std::cout << "Failed to seek to frame " << theIndex << std::endl;
        return nullptr;
    }
    
    // 清空编解码器缓冲区
    avcodec_flush_buffers(ctx->codec_ctx);
    
    // 解码到目标帧
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();
    
    if (!packet || !frame || !sw_frame) {
        if (packet) av_packet_free(&packet);
        if (frame) av_frame_free(&frame);
        if (sw_frame) av_frame_free(&sw_frame);
        return nullptr;
    }
    
    int frame_count = 0;
    AVFrame* result_frame = nullptr;
    
    while (av_read_frame(ctx->fmt_ctx, packet) >= 0) {
        if (packet->stream_index == ctx->video_stream_idx) {
            ret = avcodec_send_packet(ctx->codec_ctx, packet);
            if (ret < 0 && ret != AVERROR(EAGAIN)) {
                if (ret == AVERROR_EOF) {
                    break;
                }
                std::cout << "Error sending packet: " << ret << std::endl;
                break;
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(ctx->codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cout << "Error receiving frame: " << ret << std::endl;
                    break;
                }
                
                frame_count++;
                
                // 找到目标帧
                if (frame_count >= target_frame) {
                    // 如果是硬件解码，转换到系统内存
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                        if (ret >= 0) {
                            result_frame = av_frame_clone(sw_frame);
                            std::cout << "Frame " << theIndex << " decoded with DX11 hardware acceleration" << std::endl;
                        } else {
                            std::cout << "Failed to transfer hardware frame to system memory" << std::endl;
                        }
                    } else {
                        result_frame = av_frame_clone(frame);
                        std::cout << "Frame " << theIndex << " decoded with software decoding" << std::endl;
                    }
                    
                    if (result_frame) {
                        std::cout << "Frame format: " << av_get_pix_fmt_name((AVPixelFormat)result_frame->format) << std::endl;
                        std::cout << "Frame size: " << result_frame->width << "x" << result_frame->height << std::endl;
                    }
                    
                    goto cleanup;
                }
            }
        }
        av_packet_unref(packet);
    }
    
cleanup:
    av_frame_free(&sw_frame);
    av_frame_free(&frame);
    av_packet_free(&packet);
    
    return result_frame;
}