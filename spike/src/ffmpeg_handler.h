#pragma once
#include <stdexcept>

// 先包含 DirectX 相关头文件，在 extern "C" 之外
#include <d3d11.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

// 视频信息结构
struct VideoInfo {
    int width;
    int height;
    double timeBase;
    int streamIndex;
    bool isHardwareDecoded; // 新增：标识是否使用硬件解码
};

// 音频信息结构
struct AudioInfo {
    int sampleRate;
    int channels;
    double timeBase;
    int streamIndex;
};

// 新增：获取音频配置的封装函数
struct AudioConfig;  // 前向声明

// 新增：硬件解码异常类
class HardwareDecodeException : public std::runtime_error {
public:
    explicit HardwareDecodeException(const std::string& message) 
        : std::runtime_error("Hardware decode failed: " + message) {}
};

// 函数声明 - 移除句柄参数，内部管理状态
bool createFFmpegHandler(const char* filename);
void destroyFFmpegHandler();

bool getVideoInfo(VideoInfo* info);
bool getAudioInfo(AudioInfo* info);

// 新增：获取音频配置的封装函数
bool getAudioConfig(AudioConfig* config);

AVFormatContext* getFormatContext();
AVCodecContext* getVideoCodecContext();
AVCodecContext* getAudioCodecContext();
SwsContext* getSwsContext();

// 新增：获取硬件设备上下文
AVBufferRef* getHwDeviceContext(); 