#pragma once
#include <string>
#include <optional>

namespace spike {

struct VideoFileInfo {
    int width = 0;
    int height = 0;
    double duration_seconds = 0.0;
    double frame_rate = 0.0;
    std::string codec_name;
    std::string format_name;
    int64_t bitrate = 0;
    
    bool is_valid() const {
        return width > 0 && height > 0 && duration_seconds > 0.0;
    }
};

class VideoFileInfoReader {
public:
    VideoFileInfoReader() = default;
    ~VideoFileInfoReader() = default;
    
    // 禁用拷贝和移动，保持简单
    VideoFileInfoReader(const VideoFileInfoReader&) = delete;
    VideoFileInfoReader& operator=(const VideoFileInfoReader&) = delete;
    VideoFileInfoReader(VideoFileInfoReader&&) = delete;
    VideoFileInfoReader& operator=(VideoFileInfoReader&&) = delete;
    
    // 从文件读取视频信息，返回空表示读取失败
    std::optional<VideoFileInfo> read_from_file(const std::string& filepath);
    
    // 获取最后的错误信息
    const std::string& get_last_error() const { return last_error_; }

private:
    std::string last_error_;
};

} // namespace spike 