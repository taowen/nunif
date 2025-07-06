#include "packet_demuxer.h"
#include <iostream>
#include <algorithm>

PacketDemuxer::PacketDemuxer() 
    : is_initialized_(false)
    , is_eof_(false)
    , current_pair_index_(0) {
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
    
    // 清理当前要使用的pair
    clearCurrentPair();
    
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
    
    // 存储到当前的pair中
    PacketPair& current_pair = borrowed_pairs_[current_pair_index_];
    current_pair.audio_packet = audio_packet;
    current_pair.video_packet = video_packet;
    current_pair.timestamp = getPacketTimestamp(audio_packet, true);
    current_pair.is_valid = true;
    
    // 将当前pair的指针返回给调用者
    packets = current_pair;
    
    // 切换到下一个pair
    current_pair_index_ = (current_pair_index_ + 1) % 2;
    
    return true;
}

void PacketDemuxer::close() {
    clearBorrowedPairs();
    reader_.close();
    is_initialized_ = false;
    is_eof_ = false;
}

void PacketDemuxer::clearBorrowedPairs() {
    for (int i = 0; i < 2; i++) {
        if (borrowed_pairs_[i].audio_packet) {
            av_packet_free(&borrowed_pairs_[i].audio_packet);
            borrowed_pairs_[i].audio_packet = nullptr;
        }
        
        if (borrowed_pairs_[i].video_packet) {
            av_packet_free(&borrowed_pairs_[i].video_packet);
            borrowed_pairs_[i].video_packet = nullptr;
        }
        
        borrowed_pairs_[i].timestamp = 0.0;
        borrowed_pairs_[i].is_valid = false;
    }
}

void PacketDemuxer::clearCurrentPair() {
    PacketPair& current_pair = borrowed_pairs_[current_pair_index_];
    if (current_pair.audio_packet) {
        av_packet_free(&current_pair.audio_packet);
        current_pair.audio_packet = nullptr;
    }
    
    if (current_pair.video_packet) {
        av_packet_free(&current_pair.video_packet);
        current_pair.video_packet = nullptr;
    }
    
    current_pair.timestamp = 0.0;
    current_pair.is_valid = false;
}

double PacketDemuxer::getPacketTimestamp(AVPacket* packet, bool is_audio) const {
    if (!packet || packet->pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    
    auto stream_info = reader_.getStreamInfo();
    AVRational time_base = is_audio ? stream_info.audio_time_base : stream_info.video_time_base;
    
    return packet->pts * av_q2d(time_base);
}

bool PacketDemuxer::hasValidPair() const {
    return borrowed_pairs_[0].is_valid || borrowed_pairs_[1].is_valid;
}

int PacketDemuxer::getValidPairCount() const {
    int count = 0;
    if (borrowed_pairs_[0].is_valid) count++;
    if (borrowed_pairs_[1].is_valid) count++;
    return count;
}

