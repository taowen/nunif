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
        ComPtr<ID3D11Texture2D> texture;
        int64_t pts;
        int64_t duration;
        int width;
        int height;
        bool is_valid;
        
        DecodedFrame() : pts(0), duration(0), width(0), height(0), is_valid(false) {}
    };

    HwVideoDecoder();
    ~HwVideoDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    int getWidth() const;
    int getHeight() const;
    double getFPS() const;
    double getDuration() const;
    
    bool seekToTime(double seconds);
    bool seekToFrame(int64_t frame_number);

private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    
    AVCodecContext* codec_context_;
    AVFrame* hw_frame_;
    AVFrame* sw_frame_;
    
    ComPtr<ID3D11Device> d3d11_device_;
    ComPtr<ID3D11DeviceContext> d3d11_context_;
    
    AVBufferRef* hw_device_ctx_;
    AVBufferRef* hw_frames_ctx_;
    
    bool is_open_;
    bool is_eof_;
    int width_;
    int height_;
    double fps_;
    double duration_;
    
    bool initializeDirectX();
    bool initializeFFmpegHWDecoder();
    bool createHWFramesContext();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    bool convertFrameToTexture(AVFrame* frame, DecodedFrame& decoded_frame);
    void cleanup();
};