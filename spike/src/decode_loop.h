#pragma once

#include <atomic>
#include <functional>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// 解码状态结构
struct DecodeState {
    std::atomic<bool>* shouldStop;
    bool hasVideo;
    bool hasAudio;
    int videoStreamIndex;
    int audioStreamIndex;
};

// 解码回调函数类型
using VideoFrameCallback = std::function<void(AVFrame*)>;
using AudioFrameCallback = std::function<void(AVFrame*)>;
using QueueFullCheckCallback = std::function<bool()>;

// 解码循环接口
class DecodeLoop {
public:
    DecodeLoop(DecodeState state,
               VideoFrameCallback videoCallback,
               AudioFrameCallback audioCallback,
               QueueFullCheckCallback queueFullCallback);
    
    // 执行解码循环
    void run();
    
    // 设置 FFmpeg 上下文
    void setContexts(AVFormatContext* formatContext,
                    AVCodecContext* videoCodecContext,
                    AVCodecContext* audioCodecContext);

private:
    DecodeState m_state;
    VideoFrameCallback m_videoCallback;
    AudioFrameCallback m_audioCallback;
    QueueFullCheckCallback m_queueFullCallback;
    
    AVFormatContext* m_formatContext = nullptr;
    AVCodecContext* m_videoCodecContext = nullptr;
    AVCodecContext* m_audioCodecContext = nullptr;
    
    void processVideoPacket(AVPacket* packet, AVFrame* frame);
    void processAudioPacket(AVPacket* packet, AVFrame* frame);
}; 