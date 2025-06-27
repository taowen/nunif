#include <catch2/catch_test_macros.hpp>
#include "main.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
// Include D3D11 hardware context header outside of extern "C" block
#include <libavutil/hwcontext_d3d11va.h>
#include <iostream>

TEST_CASE("DX11 Hardware Decode Frame 650", "[ffmpeg][dx11]") {
    // Initialize FFmpeg
    av_log_set_level(AV_LOG_ERROR);
    
    // Create FFmpeg context
    FFMepgContext ctx;
    
    // Input file
    const char* input_file = "06 4k.mp4";
    
    // Decode frame 650
    AVFrame* frame = d11_decode(&ctx, input_file, 650);
    
    // Verify that frame was successfully decoded
    REQUIRE(frame != nullptr);
    REQUIRE(frame->width > 0);
    REQUIRE(frame->height > 0);
    REQUIRE(frame->data[0] != nullptr);
    
    std::cout << "Frame 650 decoded successfully" << std::endl;
    std::cout << "Frame format: " << av_get_pix_fmt_name((AVPixelFormat)frame->format) << std::endl;
    std::cout << "Frame size: " << frame->width << "x" << frame->height << std::endl;
    
    // Cleanup
    if (frame) {
        av_frame_free(&frame);
    }
    
    // Cleanup context
    if (ctx.codec_ctx) {
        avcodec_free_context(&ctx.codec_ctx);
    }
    if (ctx.fmt_ctx) {
        avformat_close_input(&ctx.fmt_ctx);
    }
    if (ctx.hw_device_ctx) {
        av_buffer_unref(&ctx.hw_device_ctx);
    }
}
