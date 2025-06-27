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
                    // 找到目标帧，复制一份返回
                    AVFrame* result_frame = av_frame_alloc();
                    av_frame_ref(result_frame, frame);
                    av_frame_unref(frame);
                    av_frame_free(&frame);
                    av_packet_free(&packet);
                    return result_frame;
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

// 添加帧对比函数
bool compare_frames(AVFrame* frame1, AVFrame* frame2, double threshold) {
    if (!frame1 || !frame2) {
        return false;
    }
    
    // 检查基本属性
    if (frame1->width != frame2->width || frame1->height != frame2->height) {
        std::cout << "Frame dimensions differ: " 
                  << frame1->width << "x" << frame1->height << " vs " 
                  << frame2->width << "x" << frame2->height << std::endl;
        return false;
    }
    
    // 添加色彩空间信息打印
    std::cout << "Frame1 colorspace: " << av_color_space_name(frame1->colorspace) << std::endl;
    std::cout << "Frame2 colorspace: " << av_color_space_name(frame2->colorspace) << std::endl;
    std::cout << "Frame1 color_range: " << av_color_range_name(frame1->color_range) << std::endl;
    std::cout << "Frame2 color_range: " << av_color_range_name(frame2->color_range) << std::endl;
    std::cout << "Frame1 color_primaries: " << av_color_primaries_name(frame1->color_primaries) << std::endl;
    std::cout << "Frame2 color_primaries: " << av_color_primaries_name(frame2->color_primaries) << std::endl;
    std::cout << "Frame1 color_trc: " << av_color_transfer_name(frame1->color_trc) << std::endl;
    std::cout << "Frame2 color_trc: " << av_color_transfer_name(frame2->color_trc) << std::endl;
    
    AVFrame* converted_frame1 = nullptr;
    if (frame1->format != frame2->format) {
        converted_frame1 = av_frame_alloc();
        converted_frame1->format = frame2->format;
        converted_frame1->width = frame1->width;
        converted_frame1->height = frame1->height;
        av_frame_get_buffer(converted_frame1, 32);

        SwsContext* conv_ctx = sws_getContext(
            frame1->width, frame1->height, (AVPixelFormat)frame1->format,
            frame1->width, frame1->height, (AVPixelFormat)frame2->format,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
        
        sws_scale(conv_ctx, frame1->data, frame1->linesize, 0, frame1->height,
                  converted_frame1->data, converted_frame1->linesize);
        sws_freeContext(conv_ctx);
        frame1 = converted_frame1;
    }
    
    // 将两个帧都转换为RGB24格式进行对比，但使用相同的色彩空间参数
    SwsContext* sws_ctx1 = sws_getContext(
        frame1->width, frame1->height, (AVPixelFormat)frame1->format,
        frame1->width, frame1->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    
    SwsContext* sws_ctx2 = sws_getContext(
        frame2->width, frame2->height, (AVPixelFormat)frame2->format,
        frame2->width, frame2->height, AV_PIX_FMT_RGB24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    
    if (!sws_ctx1 || !sws_ctx2) {
        if (sws_ctx1) sws_freeContext(sws_ctx1);
        if (sws_ctx2) sws_freeContext(sws_ctx2);
        return false;
    }
    
    // 设置色彩空间参数 - 强制使用相同的色彩空间
    int* inv_table1;
    int* table1;
    int srcRange1, dstRange1, brightness1, contrast1, saturation1;
    
    int* inv_table2;
    int* table2;
    int srcRange2, dstRange2, brightness2, contrast2, saturation2;
    
    // 获取当前的色彩空间转换表
    sws_getColorspaceDetails(sws_ctx1, &inv_table1, &srcRange1, &table1, &dstRange1, 
                            &brightness1, &contrast1, &saturation1);
    sws_getColorspaceDetails(sws_ctx2, &inv_table2, &srcRange2, &table2, &dstRange2, 
                            &brightness2, &contrast2, &saturation2);
    
    std::cout << "SWS Context1 - srcRange: " << srcRange1 << ", dstRange: " << dstRange1 << std::endl;
    std::cout << "SWS Context2 - srcRange: " << srcRange2 << ", dstRange: " << dstRange2 << std::endl;
    
    // 如果色彩空间参数不同，强制使用相同的参数
    if (srcRange1 != srcRange2 || dstRange1 != dstRange2) {
        std::cout << "Color space parameters differ, forcing same parameters..." << std::endl;
        
        // 使用BT.709色彩空间和full range
        const int* bt709_coeffs = sws_getCoefficients(SWS_CS_ITU709);
        sws_setColorspaceDetails(sws_ctx1, bt709_coeffs, 1, bt709_coeffs, 1, 0, 1 << 16, 1 << 16);
        sws_setColorspaceDetails(sws_ctx2, bt709_coeffs, 1, bt709_coeffs, 1, 0, 1 << 16, 1 << 16);
    }
    
    // 分配RGB24缓冲区
    AVFrame* rgb_frame1 = av_frame_alloc();
    AVFrame* rgb_frame2 = av_frame_alloc();
    
    rgb_frame1->format = AV_PIX_FMT_RGB24;
    rgb_frame1->width = frame1->width;
    rgb_frame1->height = frame1->height;
    av_frame_get_buffer(rgb_frame1, 32);
    
    rgb_frame2->format = AV_PIX_FMT_RGB24;
    rgb_frame2->width = frame2->width;
    rgb_frame2->height = frame2->height;
    av_frame_get_buffer(rgb_frame2, 32);
    
    // 转换像素格式
    sws_scale(sws_ctx1, frame1->data, frame1->linesize, 0, frame1->height,
              rgb_frame1->data, rgb_frame1->linesize);
    sws_scale(sws_ctx2, frame2->data, frame2->linesize, 0, frame2->height,
              rgb_frame2->data, rgb_frame2->linesize);
    
    // 对比像素数据
    bool frames_match = true;
    double total_diff = 0.0;
    int pixel_count = 0;
    
    for (int y = 0; y < frame1->height; y++) {
        uint8_t* line1 = rgb_frame1->data[0] + y * rgb_frame1->linesize[0];
        uint8_t* line2 = rgb_frame2->data[0] + y * rgb_frame2->linesize[0];
        
        for (int x = 0; x < frame1->width * 3; x++) {
            double diff = std::abs(line1[x] - line2[x]) / 255.0;
            total_diff += diff;
            pixel_count++;
            
            if (diff > threshold) {
                frames_match = false;
            }
        }
    }
    
    double avg_diff = total_diff / pixel_count;
    std::cout << "Average pixel difference: " << avg_diff << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
    
    // 清理资源
    sws_freeContext(sws_ctx1);
    sws_freeContext(sws_ctx2);
    av_frame_free(&rgb_frame1);
    av_frame_free(&rgb_frame2);
    if (converted_frame1) {
        av_frame_free(&converted_frame1);
    }
    
    return frames_match;
}

// 添加硬件与软件解码对比测试
TEST_CASE("Compare Hardware vs Software Decode") {
    // Initialize FFmpeg
    av_log_set_level(AV_LOG_ERROR);
    
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 650;
    
    // 硬件解码
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    REQUIRE(hw_frame != nullptr);
    
    // 软件解码
    FFMepgContext sw_ctx;
    AVFrame* sw_frame = sw_decode(&sw_ctx, input_file, frame_index);
    REQUIRE(sw_frame != nullptr);
    
    std::cout << "Hardware decode format: " << av_get_pix_fmt_name((AVPixelFormat)hw_frame->format) << std::endl;
    std::cout << "Software decode format: " << av_get_pix_fmt_name((AVPixelFormat)sw_frame->format) << std::endl;
    
    // 如果硬件解码是GPU格式，需要传输到CPU
    AVFrame* hw_frame_cpu = nullptr;
    if (hw_frame->format == AV_PIX_FMT_D3D11) {
        hw_frame_cpu = av_frame_alloc();
        if (av_hwframe_transfer_data(hw_frame_cpu, hw_frame, 0) < 0) {
            std::cerr << "Failed to transfer hardware frame to CPU" << std::endl;
            av_frame_free(&hw_frame_cpu);
            hw_frame_cpu = nullptr;
        } else {
            std::cout << "Hardware frame transferred to CPU, format: " << av_get_pix_fmt_name((AVPixelFormat)hw_frame_cpu->format) << std::endl;
        }
    } else {
        hw_frame_cpu = hw_frame;
    }
    
    if (hw_frame_cpu) {
        // Copy color space metadata from software frame to hardware frame
        // This ensures both frames use the same color space conversion parameters
        hw_frame_cpu->colorspace = sw_frame->colorspace;
        hw_frame_cpu->color_range = sw_frame->color_range;
        hw_frame_cpu->color_primaries = sw_frame->color_primaries;
        hw_frame_cpu->color_trc = sw_frame->color_trc;
        
        std::cout << "Copied color space metadata from software to hardware frame" << std::endl;
        
        // 对比两个解码结果
        bool frames_match = compare_frames(hw_frame_cpu, sw_frame, 0.05); // 5% 容差
        
        if (frames_match) {
            std::cout << "✓ Hardware and software decode results match!" << std::endl;
        } else {
            std::cout << "✗ Hardware and software decode results differ!" << std::endl;
        }
        
        // 如果是传输的帧，需要释放
        if (hw_frame_cpu != hw_frame) {
            av_frame_free(&hw_frame_cpu);
        }
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
