#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "packet_demuxer.h"
#include <memory>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
    #include <libswresample/swresample.h>
}

class FrameDecoder {
public:
    struct DecodedFrame {
        FrameDecoder* owner = nullptr; // 指向拥有此帧的解码器
        int pool_index = -1;           // 在池中的索引
        double timestamp = 0.0;
        bool is_valid = false;

        AVFrame* get() const {
            if (owner && pool_index != -1) {
                return owner->getFrameFromPool(pool_index, false); // false for video
            }
            return nullptr;
        }

        // 兼容音频帧的get方法
        AVFrame* getAudioFrame() const {
            if (owner && pool_index != -1) {
                return owner->getFrameFromPool(pool_index, true); // true for audio
            }
            return nullptr;
        }
    };
    
    struct DecodedFrames {
        DecodedFrame audio_frame;
        DecodedFrame video_frame;
    };

    FrameDecoder();
    ~FrameDecoder();

    // 初始化解码器 - 打开MKV文件并设置解码器
    bool open(const std::string& filepath);
    
    // Pull-style解码接口 - 返回同步的音视频帧
    bool readNextFrames(DecodedFrames& decoded_frames);
    
    // 直接解码接口（用于测试和特殊用途）
    bool decodeVideoPacket(AVPacket* packet, AVFrame* frame);
    bool decodeAudioPacket(AVPacket* packet, AVFrame* frame);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isHardwareAccelerated() const { return hw_device_ctx_ != nullptr; }
    const char* getVideoCodecName() const;
    const char* getAudioCodecName() const;
    
    // 获取内部demuxer（用于测试）
    PacketDemuxer* getDemuxer() { return &demuxer_; }
    
    // 获取内部D3D11设备（用于RGB转换器）
    ID3D11Device* getD3D11Device() { return d3d11_device_; }
    ID3D11DeviceContext* getD3D11Context() { return d3d11_context_; }
    
    // 获取池中的帧（供DecodedFrame使用）
    AVFrame* getFrameFromPool(int index, bool is_audio) const;
    
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
    
    // 包解复用器
    PacketDemuxer demuxer_;
    
    // 状态
    bool is_initialized_;
    
    // AVFrame池 - 复用避免频繁分配
    static const int AVFRAME_POOL_SIZE = 3;
    AVFrame* audio_frame_pool_[AVFRAME_POOL_SIZE];
    AVFrame* video_frame_pool_[AVFRAME_POOL_SIZE];
    int current_audio_frame_index_;
    int current_video_frame_index_;
    
    // 内部方法
    bool createHardwareContext();
    bool findVideoHardwareDecoder(AVCodecID codec_id);
    bool findAudioDecoder(AVCodecID codec_id);
    bool configureVideoDecoder(AVCodecParameters* codec_params);
    bool configureAudioDecoder(AVCodecParameters* codec_params);
    bool initializeAudioResampler();
    
    // AVFrame池管理
    void initializeFramePools();
    AVFrame* getNextAudioFrame();
    AVFrame* getNextVideoFrame();
    void releaseFramePools();
    
    void releaseResources();
};