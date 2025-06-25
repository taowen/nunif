#include <iostream>
#include <format>
#include <string_view>
#include <memory>
#include <stdexcept>

// D3D11VA 头文件需要在 extern "C" 之外包含
#include <libavutil/hwcontext_d3d11va.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}

// 辅助函数：处理 av_err2str 在 MSVC 中的问题
std::string av_err_to_string(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

class VideoDecoder {
private:
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_index = -1;
    
public:
    VideoDecoder() = default;
    
    ~VideoDecoder() {
        cleanup();
    }
    
    void cleanup() {
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx) {
            avformat_close_input(&format_ctx);
        }
        if (hw_device_ctx) {
            av_buffer_unref(&hw_device_ctx);
        }
    }
    
    bool initialize_d3d11va() {
        // 创建D3D11VA硬件设备上下文
        int ret = av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << std::format("Failed to create D3D11VA device context: {}\n", av_err_to_string(ret));
            return false;
        }
        
        std::cout << "D3D11VA hardware acceleration initialized successfully\n";
        return true;
    }
    
    bool open_video_file(const std::string& filename) {
        // 打开视频文件
        int ret = avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to open video file: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 获取流信息
        ret = avformat_find_stream_info(format_ctx, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to find stream info: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 找到视频流
        video_stream_index = av_find_best_stream(format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_index < 0) {
            std::cerr << "No video stream found\n";
            return false;
        }
        
        std::cout << std::format("Found video stream at index: {}\n", video_stream_index);
        return true;
    }
    
    bool setup_decoder() {
        AVStream* video_stream = format_ctx->streams[video_stream_index];
        
        // 查找支持D3D11VA的解码器
        const AVCodec* decoder = nullptr;
        
        // 首先尝试查找硬件加速解码器
        if (video_stream->codecpar->codec_id == AV_CODEC_ID_H264) {
            decoder = avcodec_find_decoder_by_name("h264_d3d11va");
        } else if (video_stream->codecpar->codec_id == AV_CODEC_ID_HEVC) {
            decoder = avcodec_find_decoder_by_name("hevc_d3d11va");
        }
        
        // 如果没有找到硬件解码器，使用软件解码器
        if (!decoder) {
            decoder = avcodec_find_decoder(video_stream->codecpar->codec_id);
            std::cout << "Using software decoder\n";
        } else {
            std::cout << std::format("Using hardware decoder: {}\n", decoder->name);
        }
        
        if (!decoder) {
            std::cerr << "Decoder not found\n";
            return false;
        }
        
        // 创建解码器上下文
        codec_ctx = avcodec_alloc_context3(decoder);
        if (!codec_ctx) {
            std::cerr << "Failed to allocate codec context\n";
            return false;
        }
        
        // 复制流参数到解码器上下文
        int ret = avcodec_parameters_to_context(codec_ctx, video_stream->codecpar);
        if (ret < 0) {
            std::cerr << std::format("Failed to copy codec parameters: {}\n", av_err_to_string(ret));
            return false;
        }
        
        // 设置硬件设备上下文
        if (hw_device_ctx) {
            codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        }
        
        // 打开解码器
        ret = avcodec_open2(codec_ctx, decoder, nullptr);
        if (ret < 0) {
            std::cerr << std::format("Failed to open codec: {}\n", av_err_to_string(ret));
            return false;
        }
        
        std::cout << std::format("Decoder setup completed - Resolution: {}x{}, Pixel Format: {}\n", 
                                codec_ctx->width, codec_ctx->height, 
                                av_get_pix_fmt_name(codec_ctx->pix_fmt));
        
        // === 添加详细的像素格式日志 ===
        std::cout << std::format("=== PIXEL FORMAT DEBUG INFO ===\n");
        std::cout << std::format("Codec context pix_fmt: {} ({})\n", 
                                av_get_pix_fmt_name(codec_ctx->pix_fmt), 
                                static_cast<int>(codec_ctx->pix_fmt));
        
        // 检查是否是硬件格式
        if (codec_ctx->pix_fmt == AV_PIX_FMT_D3D11) {
            std::cout << "Hardware format: D3D11 detected\n";
        } else {
            std::cout << "Software format detected\n";
        }
        
        return true;
    }
    
    void decode_frames() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        AVFrame* sw_frame = av_frame_alloc();
        
        if (!packet || !frame || !sw_frame) {
            std::cerr << "Failed to allocate packet or frame\n";
            return;
        }
        
        int frame_count = 0;
        
        // 读取和解码帧
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index == video_stream_index) {
                int ret = avcodec_send_packet(codec_ctx, packet);
                if (ret < 0) {
                    std::cerr << std::format("Error sending packet: {}\n", av_err_to_string(ret));
                    break;
                }
                
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        std::cerr << std::format("Error receiving frame: {}\n", av_err_to_string(ret));
                        break;
                    }
                    
                    frame_count++;
                    
                    // === 添加帧格式详细日志 ===
                    std::cout << std::format("=== FRAME {} FORMAT DEBUG ===\n", frame_count);
                    std::cout << std::format("Frame format: {} ({})\n", 
                                            av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)), 
                                            frame->format);
                    std::cout << std::format("Frame size: {}x{}\n", frame->width, frame->height);
                    
                    // 如果是硬件帧，需要传输到系统内存
                    if (frame->format == AV_PIX_FMT_D3D11) {
                        std::cout << "D3D11 hardware frame detected, transferring to system memory...\n";
                        
                        ret = av_hwframe_transfer_data(sw_frame, frame, 0);
                        if (ret < 0) {
                            std::cerr << std::format("Error transferring frame data: {}\n", av_err_to_string(ret));
                            continue;
                        }
                        
                        // === 添加传输后格式日志 ===
                        std::cout << std::format("After transfer - SW frame format: {} ({})\n", 
                                                av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw_frame->format)), 
                                                sw_frame->format);
                        std::cout << std::format("SW frame size: {}x{}\n", sw_frame->width, sw_frame->height);
                        
                        // 检查具体的像素格式
                        if (sw_frame->format == AV_PIX_FMT_YUV420P) {
                            std::cout << ">>> DETECTED FORMAT: YUV420P <<<\n";
                        } else if (sw_frame->format == AV_PIX_FMT_NV12) {
                            std::cout << ">>> DETECTED FORMAT: NV12 <<<\n";
                        } else if (sw_frame->format == AV_PIX_FMT_YUV444P) {
                            std::cout << ">>> DETECTED FORMAT: YUV444P <<<\n";
                        } else {
                            std::cout << std::format(">>> DETECTED FORMAT: OTHER ({}) <<<\n", 
                                                    av_get_pix_fmt_name(static_cast<AVPixelFormat>(sw_frame->format)));
                        }
                        
                        std::cout << std::format("Frame {}: {}x{} (D3D11VA decoded, transferred to system memory)\n", 
                                                frame_count, sw_frame->width, sw_frame->height);
                    } else {
                        // 软件解码的情况
                        if (frame->format == AV_PIX_FMT_YUV420P) {
                            std::cout << ">>> DETECTED FORMAT: YUV420P (Software) <<<\n";
                        } else if (frame->format == AV_PIX_FMT_NV12) {
                            std::cout << ">>> DETECTED FORMAT: NV12 (Software) <<<\n";
                        } else {
                            std::cout << std::format(">>> DETECTED FORMAT: OTHER (Software) ({}) <<<\n", 
                                                    av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)));
                        }
                        
                        std::cout << std::format("Frame {}: {}x{} (Software decoded)\n", 
                                                frame_count, frame->width, frame->height);
                    }
                    
                    std::cout << "================================\n";
                    
                    // 这里可以处理解码后的帧数据
                    // sw_frame 包含了解码后的原始视频数据
                    
                    // 限制解码帧数，避免处理整个视频
                    if (frame_count >= 3) {  // 减少到3帧，方便查看日志
                        std::cout << "Processed 3 frames, stopping...\n";
                        goto cleanup_decode;
                    }
                }
            }
            av_packet_unref(packet);
        }
        
    cleanup_decode:
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        av_packet_free(&packet);
        
        std::cout << std::format("Total frames decoded: {}\n", frame_count);
        std::cout << "=== END OF PIXEL FORMAT DEBUG ===\n";
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: iw3_cpp <video_file>\n";
        return 1;
    }
    
    std::string video_file = argv[1];
    
    try {
        VideoDecoder decoder;
        
        // 初始化D3D11VA硬件加速
        if (!decoder.initialize_d3d11va()) {
            std::cerr << "Failed to initialize D3D11VA, will use software decoding\n";
        }
        
        // 打开视频文件
        if (!decoder.open_video_file(video_file)) {
            return 1;
        }
        
        // 设置解码器
        if (!decoder.setup_decoder()) {
            return 1;
        }
        
        // 解码视频帧
        decoder.decode_frames();
        
        std::cout << "Video decoding completed successfully!\n";
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Error: {}\n", e.what());
        return 1;
    }
    
    return 0;
}