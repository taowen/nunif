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