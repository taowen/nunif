#include "packet_demuxer.h"
#include <iostream>
#include <algorithm>

PacketDemuxer::PacketDemuxer() 
    : is_initialized_(false)
    , is_eof_(false) {
}

PacketDemuxer::~PacketDemuxer() {
    close();
}

bool PacketDemuxer::open(const std::string& filepath) {
    // 清理已有资源
    close();
    
    // 打开MKV文件
    if (!reader_.open(filepath)) {
        std::cerr << "Failed to open MKV file: " << filepath << std::endl;
        return false;
    }
    
    auto stream_info = reader_.getStreamInfo();
    if (stream_info.video_stream_index < 0 || stream_info.audio_stream_index < 0) {
        std::cerr << "Missing video or audio stream in file" << std::endl;
        reader_.close();
        return false;
    }
    
    is_initialized_ = true;
    is_eof_ = false;
    
    return true;
}

bool PacketDemuxer::readNextPacketPair(PacketPair& packets) {
    if (!is_initialized_) {
        return false;
    }
    
    // 清理上一次借出的数据
    clearBorrowedPair();
    
    // 如果已经EOF，返回false
    if (is_eof_) {
        return false;
    }
    
    // 从文件中读取下一个音频包和视频包
    AVPacket* temp_packet = av_packet_alloc();
    if (!temp_packet) {
        return false;
    }
    
    AVPacket* audio_packet = nullptr;
    AVPacket* video_packet = nullptr;
    
    // 读取包直到找到音频包和视频包
    while (reader_.readNextPacket(temp_packet)) {
        if (reader_.isAudioPacket(temp_packet) && !audio_packet) {
            audio_packet = av_packet_clone(temp_packet);
        } else if (reader_.isVideoPacket(temp_packet) && !video_packet) {
            video_packet = av_packet_clone(temp_packet);
        }
        
        av_packet_unref(temp_packet);
        
        // 如果找到了音频包，就可以返回了（视频包可选）
        if (audio_packet) {
            break;
        }
    }
    
    av_packet_free(&temp_packet);
    
    // 如果没有找到音频包，标记EOF
    if (!audio_packet) {
        is_eof_ = true;
        return false;
    }
    
    // 存储到借出的pair中
    borrowed_pair_.audio_packet = audio_packet;
    borrowed_pair_.video_packet = video_packet;
    borrowed_pair_.timestamp = getPacketTimestamp(audio_packet, true);
    
    // 将借出的pair的指针返回给调用者
    packets = borrowed_pair_;
    
    return true;
}

void PacketDemuxer::close() {
    clearBorrowedPair();
    reader_.close();
    is_initialized_ = false;
    is_eof_ = false;
}

void PacketDemuxer::clearBorrowedPair() {
    if (borrowed_pair_.audio_packet) {
        av_packet_free(&borrowed_pair_.audio_packet);
        borrowed_pair_.audio_packet = nullptr;
    }
    
    if (borrowed_pair_.video_packet) {
        av_packet_free(&borrowed_pair_.video_packet);
        borrowed_pair_.video_packet = nullptr;
    }
    
    borrowed_pair_.timestamp = 0.0;
}

double PacketDemuxer::getPacketTimestamp(AVPacket* packet, bool is_audio) const {
    if (!packet || packet->pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    
    auto stream_info = reader_.getStreamInfo();
    AVRational time_base = is_audio ? stream_info.audio_time_base : stream_info.video_time_base;
    
    return packet->pts * av_q2d(time_base);
}

