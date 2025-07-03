#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "mkv_stream_reader.h"
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
    #include <libswresample/swresample.h>
}

class AudioVideoDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame = nullptr;
        double timestamp = 0.0;
        bool is_valid = false;
    };
    
    struct DecodedFrames {
        DecodedFrame audio_frame;
        DecodedFrame video_frame;
    };

    AudioVideoDecoder();
    ~AudioVideoDecoder();

    // 初始化音视频解码器 - 打开MKV文件并设置解码器
    bool open(const std::string& filepath);
    
    // Pull-style解码接口 - 返回同步的音视频帧（以音频时间戳为准）
    bool readNextFrames(DecodedFrames& decoded_frames);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isHardwareAccelerated() const { return hw_device_ctx_ != nullptr; }
    const char* getVideoCodecName() const;
    const char* getAudioCodecName() const;
    
    // 资源管理
    void flush();
    void close();

private:
    // DirectX11设备
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // FFmpeg解码器
    AVCodecContext* video_codec_context_;
    AVCodecContext* audio_codec_context_;
    AVBufferRef* hw_device_ctx_;
    const AVCodec* video_codec_;
    const AVCodec* audio_codec_;
    
    // 音频重采样
    SwrContext* audio_resampler_;
    
    // MKV读取器
    MKVStreamReader reader_;
    
    // 同步状态
    double next_audio_timestamp_;
    bool has_buffered_video_frame_;
    AVFrame* buffered_video_frame_;
    double buffered_video_timestamp_;
    
    // 状态
    bool is_initialized_;
    
    // 内部方法
    bool createD3D11Device();
    bool createHardwareContext();
    bool findVideoHardwareDecoder(AVCodecID codec_id);
    bool findAudioDecoder(AVCodecID codec_id);
    bool configureVideoDecoder(AVCodecParameters* codec_params);
    bool configureAudioDecoder(AVCodecParameters* codec_params);
    bool initializeAudioResampler();
    
    bool decodeNextAudioFrame(DecodedFrame& audio_frame);
    bool findMatchingVideoFrame(DecodedFrame& video_frame, double target_timestamp);
    
    void releaseResources();
};