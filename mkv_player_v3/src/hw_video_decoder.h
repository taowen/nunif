#pragma once

#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <memory>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
}

#include "mkv_stream_reader.h"

using Microsoft::WRL::ComPtr;

class HwVideoDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame;
        bool is_valid;
        
        DecodedFrame() : frame(nullptr), is_valid(false) {}
    };

    HwVideoDecoder();
    ~HwVideoDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    MKVStreamReader* getStreamReader() const;
    
    bool seekToTime(double seconds);
    bool seekToFrame(int64_t frame_number);

private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    
    AVCodecContext* codec_context_;
    AVFrame* hw_frames_[2];  // 双缓冲frame数组
    int current_frame_index_;  // 当前使用的frame索引 (0或1)
    
    AVBufferRef* hw_device_ctx_;
    AVBufferRef* hw_frames_ctx_;  // 硬件帧上下文，用于显存池管理
    
    bool initializeFFmpegHWDecoder();
    bool createHWFramesContext();  // 创建硬件帧上下文和显存池
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    bool fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame);
    void cleanup();
};