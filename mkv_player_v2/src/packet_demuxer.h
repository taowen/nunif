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
    struct PacketPair {
        AVPacket* audio_packet = nullptr;  // 可能为nullptr
        AVPacket* video_packet = nullptr;  // 可能为nullptr  
        double timestamp = 0.0;            // 同步时间戳（基于音频）
        bool is_valid = false;             // 标记此pair是否包含有效数据
        
        // 注意：使用方不要手动释放audio_packet和video_packet
        // 这些包由PacketDemuxer内部管理，会在适当时候自动释放和复用
    };

    PacketDemuxer();
    ~PacketDemuxer();

    // 打开MKV文件
    bool open(const std::string& filepath);
    
    // 读取下一组同步的音视频包
    bool readNextPacketPair(PacketPair& packets);
    
    // 获取底层的 MKVStreamReader 引用
    const MKVStreamReader& getReader() const { return reader_; }
    MKVStreamReader& getReader() { return reader_; }
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isEOF() const { return is_eof_; }
    
    // 缓存状态查询
    bool hasValidPair() const;          // 检查是否还有有效的缓存pair
    int getValidPairCount() const;      // 获取当前有效pair数量(0-2)
    
    // 资源管理
    void close();

private:
    // MKV读取器
    MKVStreamReader reader_;
    
    // 借出的PacketPair - 支持两个pair同时存在
    PacketPair borrowed_pairs_[2];
    int current_pair_index_ = 0;
    
    // 状态
    bool is_initialized_;
    bool is_eof_;
    
    // 内部方法
    void clearBorrowedPairs();
    void clearCurrentPair();
    double getPacketTimestamp(AVPacket* packet, bool is_audio) const;
};