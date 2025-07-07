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
    AVFrame* hw_frame_;
    AVFrame* sw_frame_;
    
    AVBufferRef* hw_device_ctx_;
    
    bool is_open_;
    bool is_eof_;
    
    bool initializeFFmpegHWDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    bool fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame);
    void cleanup();
};