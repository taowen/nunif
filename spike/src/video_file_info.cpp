#include "video_file_info.h"
#include <filesystem>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

namespace spike {

std::optional<VideoFileInfo> VideoFileInfoReader::read_from_file(const std::string& filepath) {
    // 清除之前的错误信息
    last_error_.clear();
    
    // 检查文件名是否为空
    if (filepath.empty()) {
        last_error_ = "Filename cannot be empty";
        return std::nullopt;
    }
    
    // 检查文件是否存在
    if (!std::filesystem::exists(filepath)) {
        last_error_ = "File does not exist: " + filepath;
        return std::nullopt;
    }
    
    AVFormatContext* format_ctx = nullptr;
    
    // 打开输入文件
    int ret = avformat_open_input(&format_ctx, filepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, AV_ERROR_MAX_STRING_SIZE);
        last_error_ = "Cannot open file: " + std::string(error_buf);
        return std::nullopt;
    }
    
    // 获取流信息
    ret = avformat_find_stream_info(format_ctx, nullptr);
    if (ret < 0) {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, AV_ERROR_MAX_STRING_SIZE);
        last_error_ = "Cannot find stream info: " + std::string(error_buf);
        avformat_close_input(&format_ctx);
        return std::nullopt;
    }
    
    // 查找视频流
    int video_stream_index = -1;
    AVStream* video_stream = nullptr;
    
    for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
        if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            video_stream = format_ctx->streams[i];
            break;
        }
    }
    
    if (video_stream_index == -1) {
        last_error_ = "No video stream found in format";
        avformat_close_input(&format_ctx);
        return std::nullopt;
    }
    
    // 创建 VideoFileInfo 对象并填充信息
    VideoFileInfo info;
    
    // 获取视频尺寸
    info.width = video_stream->codecpar->width;
    info.height = video_stream->codecpar->height;
    
    // 获取时长（以秒为单位）
    if (format_ctx->duration != AV_NOPTS_VALUE) {
        info.duration_seconds = static_cast<double>(format_ctx->duration) / AV_TIME_BASE;
    } else if (video_stream->duration != AV_NOPTS_VALUE) {
        info.duration_seconds = static_cast<double>(video_stream->duration) * av_q2d(video_stream->time_base);
    }
    
    // 获取帧率
    if (video_stream->avg_frame_rate.den != 0) {
        info.frame_rate = av_q2d(video_stream->avg_frame_rate);
    } else if (video_stream->r_frame_rate.den != 0) {
        info.frame_rate = av_q2d(video_stream->r_frame_rate);
    }
    
    // 获取编解码器名称
    const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (codec) {
        info.codec_name = codec->name;
    }
    
    // 获取容器格式名称
    if (format_ctx->iformat && format_ctx->iformat->name) {
        info.format_name = format_ctx->iformat->name;
    }
    
    // 获取比特率
    if (format_ctx->bit_rate > 0) {
        info.bitrate = format_ctx->bit_rate;
    } else if (video_stream->codecpar->bit_rate > 0) {
        info.bitrate = video_stream->codecpar->bit_rate;
    }
    
    // 清理资源
    avformat_close_input(&format_ctx);
    
    // 验证信息是否有效
    if (!info.is_valid()) {
        last_error_ = "Invalid video format: missing required information";
        return std::nullopt;
    }
    
    return info;
}

} // namespace spike 