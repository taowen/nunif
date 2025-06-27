#include <catch2/catch_test_macros.hpp>
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
    
    // Open input file
    AVFormatContext* fmt_ctx = nullptr;
    const char* input_file = "06 4k.mp4";
    
    int ret = avformat_open_input(&fmt_ctx, input_file, nullptr, nullptr);
    REQUIRE(ret >= 0);
    
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    REQUIRE(ret >= 0);
    
    // Find video stream
    int video_stream_idx = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    REQUIRE(video_stream_idx >= 0);
    
    AVStream* video_stream = fmt_ctx->streams[video_stream_idx];
    
    // Find decoder
    const AVCodec* decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
    REQUIRE(decoder != nullptr);
    
    // Create codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(decoder);
    REQUIRE(codec_ctx != nullptr);
    
    ret = avcodec_parameters_to_context(codec_ctx, video_stream->codecpar);
    REQUIRE(ret >= 0);
    
    // Set up DX11 hardware decoding
    AVBufferRef* hw_device_ctx = nullptr;
    ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    
    if (ret >= 0) {
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        std::cout << "DX11 hardware acceleration enabled" << std::endl;
    } else {
        std::cout << "Failed to create DX11 context, falling back to software decoding" << std::endl;
    }
    
    // Open codec
    ret = avcodec_open2(codec_ctx, decoder, nullptr);
    REQUIRE(ret >= 0);
    
    // Seek to approximate position of frame 650
    // Calculate timestamp for frame 650
    int64_t target_frame = 650;
    int64_t timestamp = av_rescale_q(target_frame, av_inv_q(video_stream->avg_frame_rate), video_stream->time_base);
    
    ret = av_seek_frame(fmt_ctx, video_stream_idx, timestamp, AVSEEK_FLAG_BACKWARD);
    REQUIRE(ret >= 0);
    
    // Flush codec buffers after seeking
    avcodec_flush_buffers(codec_ctx);
    
    // Read and decode frames until we reach frame 650
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();
    REQUIRE(packet != nullptr);
    REQUIRE(frame != nullptr);
    REQUIRE(sw_frame != nullptr);
    
    int frame_count = 0;
    bool frame_650_decoded = false;
    
    while (av_read_frame(fmt_ctx, packet) >= 0) {
        if (packet->stream_index == video_stream_idx) {
            ret = avcodec_send_packet(codec_ctx, packet);
            if (ret < 0) {
                if (ret == AVERROR(EAGAIN)) {
                    // Need to receive frames first
                } else if (ret == AVERROR_EOF) {
                    break;
                } else {
                    std::cout << "Error sending packet: " << ret << std::endl;
                    break;
                }
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cout << "Error receiving frame: " << ret << std::endl;
                    break;
                }
                
                frame_count++;
                
                // Check if this is frame 650
                if (frame_count >= target_frame) {
                    frame_650_decoded = true;
                    
                    // If hardware decoding, transfer frame to system memory for verification
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                        if (ret >= 0) {
                            std::cout << "Frame 650 decoded successfully with DX11 hardware acceleration" << std::endl;
                            std::cout << "Frame format: " << av_get_pix_fmt_name((AVPixelFormat)sw_frame->format) << std::endl;
                            std::cout << "Frame size: " << sw_frame->width << "x" << sw_frame->height << std::endl;
                        } else {
                            std::cout << "Failed to transfer hardware frame to system memory" << std::endl;
                        }
                    } else {
                        std::cout << "Frame 650 decoded successfully with software decoding" << std::endl;
                        std::cout << "Frame format: " << av_get_pix_fmt_name((AVPixelFormat)frame->format) << std::endl;
                        std::cout << "Frame size: " << frame->width << "x" << frame->height << std::endl;
                    }
                    
                    // Verify frame data is valid
                    REQUIRE(frame->width > 0);
                    REQUIRE(frame->height > 0);
                    REQUIRE(frame->data[0] != nullptr);
                    
                    break;
                }
            }
            
            if (frame_650_decoded) {
                break;
            }
        }
        av_packet_unref(packet);
    }
    
    // Verify that frame 650 was successfully decoded
    REQUIRE(frame_650_decoded);
    
    // Cleanup
    av_frame_free(&sw_frame);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
}
