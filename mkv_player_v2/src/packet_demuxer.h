#pragma once

#include "mkv_stream_reader.h"
#include <queue>
#include <memory>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

class PacketDemuxer {
public:
    struct SyncedPackets {
        AVPacket* audio_packet = nullptr;  // 可能为nullptr
        AVPacket* video_packet = nullptr;  // 可能为nullptr  
        double timestamp = 0.0;            // 同步时间戳（基于音频）
    };

    PacketDemuxer();
    ~PacketDemuxer();

    // 打开MKV文件
    bool open(const std::string& filepath);
    
    // 读取下一组同步的音视频包
    bool readNextSyncedPackets(SyncedPackets& packets);
    
    // 获取流信息
    MKVStreamReader::StreamInfo getStreamInfo() const { return reader_.getStreamInfo(); }
    
    // 获取编解码器参数
    AVCodecParameters* getVideoCodecParameters() { return reader_.getVideoCodecParameters(); }
    AVCodecParameters* getAudioCodecParameters() { return reader_.getAudioCodecParameters(); }
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isEOF() const { return is_eof_; }
    
    // 资源管理
    void close();

private:
    // MKV读取器
    MKVStreamReader reader_;
    
    // 缓冲队列
    std::queue<AVPacket*> audio_buffer_;
    std::queue<AVPacket*> video_buffer_;
    
    // 同步阈值（秒）
    static constexpr double SYNC_THRESHOLD = 0.1;  // 100ms
    static constexpr double MAX_BUFFER_DURATION = 1.0;  // 最多缓冲1秒
    
    // 状态
    bool is_initialized_;
    bool is_eof_;
    double next_sync_timestamp_;
    
    // 内部方法
    bool fillBuffers();
    void clearBuffers();
    double getPacketTimestamp(AVPacket* packet, bool is_audio) const;
    AVPacket* findBestVideoPacket(double target_timestamp);
};