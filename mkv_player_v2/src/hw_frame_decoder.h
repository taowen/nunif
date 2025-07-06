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

class HwFrameDecoder {
public:
    struct HwFrame {
        AVFrame* frame = nullptr;      // 直接存储AVFrame指针
        double timestamp = 0.0;
        bool is_valid = false;

        AVFrame* get() const {
            return frame;
        }

        // 兼容音频帧的get方法
        AVFrame* getAudioFrame() const {
            return frame;
        }
    };
    
    struct HwFramePair {
        HwFrame audio_frame;
        HwFrame video_frame;
        bool is_valid = false;             // 标记此pair是否包含有效数据
        
        // 重要：使用方完全不需要管理资源！
        // - 所有AVFrame由HwFrameDecoder内部池化管理
        // - 不要调用av_frame_free()或任何释放函数
        // - 帧会在下次readNextHwFramePair()调用时自动复用
        // - HwFrameDecoder析构时会自动释放所有资源
    };

    HwFrameDecoder();
    ~HwFrameDecoder();

    // 初始化解码器 - 打开MKV文件并设置解码器
    bool open(const std::string& filepath);
    
    // Pull-style解码接口 - 返回同步的音视频帧（双缓冲机制）
    // 重要：返回的HwFramePair由内部池化管理，使用方无需释放任何资源
    bool readNextHwFramePair(HwFramePair& pair);
    
    // 直接解码接口（用于测试和特殊用途）
    bool decodeVideoPacket(AVPacket* packet, AVFrame* frame);
    bool decodeAudioPacket(AVPacket* packet, AVFrame* frame);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isHardwareAccelerated() const { return hw_device_ctx_ != nullptr; }
    const char* getVideoCodecName() const;
    const char* getAudioCodecName() const;
    
    // 缓存状态查询
    bool hasValidPair() const;          // 检查是否还有有效的缓存pair
    int getValidPairCount() const;      // 获取当前有效pair数量(0-2)
    
    // 获取内部demuxer（用于测试）
    PacketDemuxer* getDemuxer() { return &demuxer_; }
    
    // 获取内部reader（便利方法）
    MKVStreamReader* getReader() { return &demuxer_.getReader(); }
    
    // 尝试解码第一帧（使用独立的PacketDemuxer实例，不影响主解码器的游标位置）
    bool tryDecodeFirstVideoFrame(AVFrame* frame);
    bool tryDecodeFirstAudioFrame(AVFrame* frame);
    
    // 从解码后的视频帧获取D3D11设备
    static ID3D11Device* getD3D11DeviceFromFrame(AVFrame* frame);
    static ID3D11DeviceContext* getD3D11ContextFromFrame(AVFrame* frame);
    
    
    // 资源管理
    void close();

private:
    
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
    std::string filepath_;
    
    // 双缓冲机制 - 支持两个HwFramePair同时存在
    HwFramePair borrowed_pairs_[2];
    int current_pair_index_;
    
    // 内部方法
    bool configureVideoDecoder(AVCodecParameters* codec_params);
    bool configureAudioDecoder(AVCodecParameters* codec_params);
    bool initializeAudioResampler();
    
    
    // 双缓冲管理
    void initializeBorrowedPairs();
    
    void releaseResources();
};