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
    
    // 资源管理
    void close();

private:
    // MKV读取器
    MKVStreamReader reader_;
    
    // 借出的PacketPair - 每次readNextPacketPair前会清理上一次的数据
    PacketPair borrowed_pair_;
    
    // 状态
    bool is_initialized_;
    bool is_eof_;
    
    // 内部方法
    void clearBorrowedPair();
    double getPacketTimestamp(AVPacket* packet, bool is_audio) const;
};