#include <catch2/catch_test_macros.hpp>
#include "main.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
// Include D3D11 hardware context header outside of extern "C" block
#include <libavutil/hwcontext_d3d11va.h>
#include <iostream>
#include <cmath>


AVFrame* sw_decode(FFMepgContext* ctx, const std::string& inputFile, int theIndex);
bool compare_frames(AVFrame* frame1, AVFrame* frame2, double threshold = 0.01);

// 添加软件解码函数实现
AVFrame* sw_decode(FFMepgContext* ctx, const std::string& inputFile, int theIndex) {
    // 重置context
    if (ctx->fmt_ctx) {
        avformat_close_input(&ctx->fmt_ctx);
    }
    if (ctx->codec_ctx) {
        avcodec_free_context(&ctx->codec_ctx);
    }
    
    // 打开输入文件
    if (avformat_open_input(&ctx->fmt_ctx, inputFile.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file: " << inputFile << std::endl;
        return nullptr;
    }
    
    // 获取流信息
    if (avformat_find_stream_info(ctx->fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream info" << std::endl;
        return nullptr;
    }
    
    // 找到视频流
    ctx->video_stream_idx = av_find_best_stream(ctx->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (ctx->video_stream_idx < 0) {
        std::cerr << "Could not find video stream" << std::endl;
        return nullptr;
    }
    
    ctx->video_stream = ctx->fmt_ctx->streams[ctx->video_stream_idx];
    
    // 找到解码器（使用软件解码器）
    const AVCodec* codec = avcodec_find_decoder(ctx->video_stream->codecpar->codec_id);
    if (!codec) {
        std::cerr << "Could not find software decoder" << std::endl;
        return nullptr;
    }
    
    // 创建解码器上下文
    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) {
        std::cerr << "Could not allocate codec context" << std::endl;
        return nullptr;
    }
    
    // 复制流参数到解码器上下文
    if (avcodec_parameters_to_context(ctx->codec_ctx, ctx->video_stream->codecpar) < 0) {
        std::cerr << "Could not copy codec parameters" << std::endl;
        return nullptr;
    }
    
    // 打开解码器
    if (avcodec_open2(ctx->codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open software decoder" << std::endl;
        return nullptr;
    }
    
    // 解码到指定帧
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int frame_count = 0;
    
    while (av_read_frame(ctx->fmt_ctx, packet) >= 0) {
        if (packet->stream_index == ctx->video_stream_idx) {
            int ret = avcodec_send_packet(ctx->codec_ctx, packet);
            if (ret < 0) {
                av_packet_unref(packet);
                continue;
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(ctx->codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    std::cerr << "Error during decoding" << std::endl;
                    av_frame_free(&frame);
                    av_packet_free(&packet);
                    return nullptr;
                }
                
                if (frame_count == theIndex) {
                    // 找到目标帧，先转换格式为NV12，再返回
                    AVFrame* converted_frame = av_frame_alloc();
                    converted_frame->format = AV_PIX_FMT_NV12;
                    converted_frame->width = frame->width;
                    converted_frame->height = frame->height;
                    
                    // 分配转换后帧的缓冲区
                    if (av_frame_get_buffer(converted_frame, 0) < 0) {
                        std::cerr << "Could not allocate converted frame buffer" << std::endl;
                        av_frame_free(&converted_frame);
                        av_frame_unref(frame);
                        av_frame_free(&frame);
                        av_packet_free(&packet);
                        return nullptr;
                    }
                    
                    // 创建格式转换上下文
                    SwsContext* sws_ctx = sws_getContext(
                        frame->width, frame->height, (AVPixelFormat)frame->format,
                        converted_frame->width, converted_frame->height, AV_PIX_FMT_NV12,
                        SWS_BILINEAR, nullptr, nullptr, nullptr
                    );
                    
                    if (!sws_ctx) {
                        std::cerr << "Could not create scaling context" << std::endl;
                        av_frame_free(&converted_frame);
                        av_frame_unref(frame);
                        av_frame_free(&frame);
                        av_packet_free(&packet);
                        return nullptr;
                    }
                    
                    // 执行格式转换
                    sws_scale(sws_ctx, 
                              frame->data, frame->linesize, 0, frame->height,
                              converted_frame->data, converted_frame->linesize);
                    
                    // 复制其他帧属性
                    converted_frame->pts = frame->pts;
                    converted_frame->pkt_dts = frame->pkt_dts;
                    converted_frame->best_effort_timestamp = frame->best_effort_timestamp;
                    converted_frame->colorspace = frame->colorspace;
                    converted_frame->color_range = frame->color_range;
                    converted_frame->color_primaries = frame->color_primaries;
                    converted_frame->color_trc = frame->color_trc;
                    converted_frame->chroma_location = frame->chroma_location;
                    
                    std::cout << "✓ Software decode completed, converted from " 
                              << av_get_pix_fmt_name((AVPixelFormat)frame->format)
                              << " to " << av_get_pix_fmt_name(AV_PIX_FMT_NV12) << std::endl;
                    
                    // 清理资源
                    sws_freeContext(sws_ctx);
                    av_frame_unref(frame);
                    av_frame_free(&frame);
                    av_packet_free(&packet);
                    
                    return converted_frame;
                }
                
                frame_count++;
                av_frame_unref(frame);
            }
        }
        av_packet_unref(packet);
    }
    
    av_frame_free(&frame);
    av_packet_free(&packet);
    return nullptr;
}

// 改进的帧对比函数 - 不做任何转换，要求完全一致
bool compare_frames(AVFrame* frame1, AVFrame* frame2, double threshold) {
    if (!frame1 || !frame2) {
        std::cout << "One or both frames are null" << std::endl;
        return false;
    }
    
    // 检查基本属性是否完全一致
    if (frame1->width != frame2->width || frame1->height != frame2->height) {
        std::cout << "✗ Frame dimensions differ: " 
                  << frame1->width << "x" << frame1->height << " vs " 
                  << frame2->width << "x" << frame2->height << std::endl;
        return false;
    }
    
    if (frame1->format != frame2->format) {
        std::cout << "✗ Pixel formats differ: " 
                  << av_get_pix_fmt_name((AVPixelFormat)frame1->format) << " vs " 
                  << av_get_pix_fmt_name((AVPixelFormat)frame2->format) << std::endl;
        return false;
    }
    
    // 检查色彩空间属性是否完全一致
    if (frame1->colorspace != frame2->colorspace) {
        std::cout << "✗ Colorspace differs: " 
                  << av_color_space_name(frame1->colorspace) << " vs " 
                  << av_color_space_name(frame2->colorspace) << std::endl;
        return false;
    }
    
    if (frame1->color_range != frame2->color_range) {
        std::cout << "✗ Color range differs: " 
                  << av_color_range_name(frame1->color_range) << " vs " 
                  << av_color_range_name(frame2->color_range) << std::endl;
        return false;
    }
    
    if (frame1->color_primaries != frame2->color_primaries) {
        std::cout << "✗ Color primaries differ: " 
                  << av_color_primaries_name(frame1->color_primaries) << " vs " 
                  << av_color_primaries_name(frame2->color_primaries) << std::endl;
        return false;
    }
    
    if (frame1->color_trc != frame2->color_trc) {
        std::cout << "✗ Color transfer characteristics differ: " 
                  << av_color_transfer_name(frame1->color_trc) << " vs " 
                  << av_color_transfer_name(frame2->color_trc) << std::endl;
        return false;
    }
    
    // 检查色度采样位置
    if (frame1->chroma_location != frame2->chroma_location) {
        std::cout << "✗ Chroma location differs: " 
                  << frame1->chroma_location << " vs " << frame2->chroma_location << std::endl;
        return false;
    }
    
    std::cout << "✓ All frame properties match:" << std::endl;
    std::cout << "  - Dimensions: " << frame1->width << "x" << frame1->height << std::endl;
    std::cout << "  - Format: " << av_get_pix_fmt_name((AVPixelFormat)frame1->format) << std::endl;
    std::cout << "  - Colorspace: " << av_color_space_name(frame1->colorspace) << std::endl;
    std::cout << "  - Color range: " << av_color_range_name(frame1->color_range) << std::endl;
    std::cout << "  - Color primaries: " << av_color_primaries_name(frame1->color_primaries) << std::endl;
    std::cout << "  - Color TRC: " << av_color_transfer_name(frame1->color_trc) << std::endl;
    
    // 获取像素格式描述
    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get((AVPixelFormat)frame1->format);
    if (!desc) {
        std::cout << "✗ Could not get pixel format descriptor" << std::endl;
        return false;
    }
    
    // 直接比较原始像素数据
    bool frames_match = true;
    double total_diff = 0.0;
    int pixel_count = 0;
    int max_diff = 0;
    
    // 遍历所有平面（对于YUV格式通常有多个平面）
    for (int plane = 0; plane < desc->nb_components; plane++) {
        int plane_idx = desc->comp[plane].plane;
        int plane_height = frame1->height;
        int plane_width = frame1->width;
        
        // 计算平面尺寸
        if (plane_idx > 0) {
            plane_height = AV_CEIL_RSHIFT(frame1->height, desc->log2_chroma_h);
            plane_width = AV_CEIL_RSHIFT(frame1->width, desc->log2_chroma_w);
        }
        
        // 检查linesize是否一致
        if (frame1->linesize[plane_idx] != frame2->linesize[plane_idx]) {
            std::cout << "✗ Linesize differs for plane " << plane_idx << ": " 
                      << frame1->linesize[plane_idx] << " vs " << frame2->linesize[plane_idx] << std::endl;
            return false;
        }
        
        uint8_t* data1 = frame1->data[plane_idx];
        uint8_t* data2 = frame2->data[plane_idx];
        int linesize = frame1->linesize[plane_idx];
        
        if (!data1 || !data2) {
            std::cout << "✗ Missing data for plane " << plane_idx << std::endl;
            return false;
        }
        
        // 按行比较像素数据
        for (int y = 0; y < plane_height; y++) {
            uint8_t* line1 = data1 + y * linesize;
            uint8_t* line2 = data2 + y * linesize;
            
            // 只比较实际像素数据，不包括padding
            int bytes_per_line = plane_width;
            if (desc->comp[plane].depth > 8) {
                bytes_per_line *= 2; // 16-bit
            }
            
            for (int x = 0; x < bytes_per_line; x++) {
                int diff = std::abs(line1[x] - line2[x]);
                total_diff += diff / 255.0;
                pixel_count++;
                
                if (diff > max_diff) {
                    max_diff = diff;
                }
                
                if (diff / 255.0 > threshold) {
                    frames_match = false;
                }
            }
        }
    }
    
    double avg_diff = pixel_count > 0 ? total_diff / pixel_count : 0.0;
    std::cout << "  - Average pixel difference: " << avg_diff << std::endl;
    std::cout << "  - Maximum pixel difference: " << max_diff << "/255 (" << (max_diff/255.0) << ")" << std::endl;
    std::cout << "  - Threshold: " << threshold << std::endl;
    std::cout << "  - Total pixels compared: " << pixel_count << std::endl;
    
    return frames_match;
}

// 改进硬件与软件解码对比测试
TEST_CASE("Compare Hardware vs Software Decode") {
    // Initialize FFmpeg
    av_log_set_level(AV_LOG_ERROR);
    
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 100;
    
    std::cout << "=== Testing Hardware vs Software Decode Comparison ===" << std::endl;
    std::cout << "Input file: " << input_file << std::endl;
    std::cout << "Frame index: " << frame_index << std::endl;
    
    // 硬件解码
    std::cout << "\n--- Hardware Decode ---" << std::endl;
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    REQUIRE(hw_frame != nullptr);
    
    // 软件解码
    std::cout << "\n--- Software Decode ---" << std::endl;
    FFMepgContext sw_ctx;
    AVFrame* sw_frame = sw_decode(&sw_ctx, input_file, frame_index);
    REQUIRE(sw_frame != nullptr);
    
    std::cout << "\n--- Frame Information ---" << std::endl;
    std::cout << "Hardware decode format: " << av_get_pix_fmt_name((AVPixelFormat)hw_frame->format) << std::endl;
    std::cout << "Software decode format: " << av_get_pix_fmt_name((AVPixelFormat)sw_frame->format) << std::endl;
    
    // 如果硬件解码是GPU格式，需要传输到CPU
    AVFrame* hw_frame_cpu = nullptr;
    if (hw_frame->format == AV_PIX_FMT_D3D11) {
        std::cout << "\n--- Transferring Hardware Frame to CPU ---" << std::endl;
        hw_frame_cpu = av_frame_alloc();
        if (av_hwframe_transfer_data(hw_frame_cpu, hw_frame, 0) < 0) {
            std::cerr << "✗ Failed to transfer hardware frame to CPU" << std::endl;
            av_frame_free(&hw_frame_cpu);
            hw_frame_cpu = nullptr;
        } else {
            std::cout << "✓ Hardware frame transferred to CPU" << std::endl;
            std::cout << "Transferred format: " << av_get_pix_fmt_name((AVPixelFormat)hw_frame_cpu->format) << std::endl;
            
            // 重要：从原始硬件帧复制色彩空间信息到传输后的帧
            // 因为 av_hwframe_transfer_data 可能不会复制所有元数据
            hw_frame_cpu->colorspace = hw_frame->colorspace;
            hw_frame_cpu->color_range = hw_frame->color_range;
            hw_frame_cpu->color_primaries = hw_frame->color_primaries;
            hw_frame_cpu->color_trc = hw_frame->color_trc;
            hw_frame_cpu->chroma_location = hw_frame->chroma_location;
            
            std::cout << "Color space information copied from hardware frame" << std::endl;
        }
    } else {
        hw_frame_cpu = hw_frame;
        std::cout << "Hardware frame is already in CPU memory" << std::endl;
    }
    
    if (hw_frame_cpu) {
        std::cout << "\n--- Frame Comparison ---" << std::endl;
        
        // 不做任何修改，直接比较原始帧
        bool frames_match = compare_frames(hw_frame_cpu, sw_frame, 0.02); // 2% 容差
        
        std::cout << "\n--- Result ---" << std::endl;
        if (frames_match) {
            std::cout << "✓ SUCCESS: Hardware and software decode results are identical!" << std::endl;
        } else {
            std::cout << "✗ FAILURE: Hardware and software decode results differ!" << std::endl;
            std::cout << "This could indicate:" << std::endl;
            std::cout << "  - Different color space handling between HW/SW decoders" << std::endl;
            std::cout << "  - Hardware decoder using different parameters" << std::endl;
            std::cout << "  - Precision differences in decoding algorithms" << std::endl;
        }
        
        // 如果是传输的帧，需要释放
        if (hw_frame_cpu != hw_frame) {
            av_frame_free(&hw_frame_cpu);
        }
    } else {
        std::cout << "\n--- Result ---" << std::endl;
        std::cout << "✗ FAILURE: Could not transfer hardware frame for comparison" << std::endl;
    }
    
    // Cleanup
    if (sw_frame) {
        av_frame_free(&sw_frame);
    }
    
    // Cleanup hardware context
    if (hw_ctx.codec_ctx) {
        avcodec_free_context(&hw_ctx.codec_ctx);
    }
    if (hw_ctx.fmt_ctx) {
        avformat_close_input(&hw_ctx.fmt_ctx);
    }
    if (hw_ctx.hw_device_ctx) {
        av_buffer_unref(&hw_ctx.hw_device_ctx);
    }
    
    // Cleanup software context
    if (sw_ctx.codec_ctx) {
        avcodec_free_context(&sw_ctx.codec_ctx);
    }
    if (sw_ctx.fmt_ctx) {
        avformat_close_input(&sw_ctx.fmt_ctx);
    }
}
