#include "video_file_opener.h"
#include <iostream>

std::string av_err_to_string(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

bool open_video_file(const std::string& filename, DecoderState& decoder_state) {
    int ret = avformat_open_input(&decoder_state.format_ctx, filename.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to open video file: " << av_err_to_string(ret) << "\n";
        return false;
    }
    
    ret = avformat_find_stream_info(decoder_state.format_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to find stream info: " << av_err_to_string(ret) << "\n";
        return false;
    }
    
    decoder_state.video_stream_index = av_find_best_stream(decoder_state.format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (decoder_state.video_stream_index < 0) {
        std::cerr << "No video stream found\n";
        return false;
    }
    
    AVStream* video_stream = decoder_state.format_ctx->streams[decoder_state.video_stream_index];
    
    const AVCodec* decoder = nullptr;
    
    if (!decoder_state.hw_device_ctx) {
        std::cerr << "D3D11VA device context not initialized\n";
        return false;
    }
    
    const AVCodec* codec = nullptr;
    void* opaque = nullptr;
    
    while ((codec = av_codec_iterate(&opaque))) {
        if (codec->type == AVMEDIA_TYPE_VIDEO && 
            av_codec_is_decoder(codec) &&
            codec->id == video_stream->codecpar->codec_id) {
            
            for (int i = 0; ; i++) {
                const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
                if (!config) {
                    break;
                }
                if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA) {
                    decoder = codec;
                    break;
                }
            }
            
            if (decoder) break;
        }
    }
    
    if (!decoder) {
        std::cerr << "No D3D11VA capable decoder found\n";
        return false;
    }
    
    decoder_state.codec_ctx = avcodec_alloc_context3(decoder);
    if (!decoder_state.codec_ctx) {
        std::cerr << "Failed to allocate codec context\n";
        return false;
    }
    
    ret = avcodec_parameters_to_context(decoder_state.codec_ctx, video_stream->codecpar);
    if (ret < 0) {
        std::cerr << "Failed to copy codec parameters: " << av_err_to_string(ret) << "\n";
        return false;
    }
    
    decoder_state.codec_ctx->hw_device_ctx = av_buffer_ref(decoder_state.hw_device_ctx);
    
    decoder_state.codec_ctx->get_format = [](AVCodecContext* /* ctx */, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
        const enum AVPixelFormat* p;
        for (p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
            if (*p == AV_PIX_FMT_D3D11) {
                return *p;
            }
        }
        std::cerr << "D3D11 format not available\n";
        return AV_PIX_FMT_NONE;
    };
    
    ret = avcodec_open2(decoder_state.codec_ctx, decoder, nullptr);
    if (ret < 0) {
        std::cerr << "Failed to open codec: " << av_err_to_string(ret) << "\n";
        return false;
    }
    
    return true;
} 