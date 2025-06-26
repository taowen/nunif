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
    
    return true;
} 