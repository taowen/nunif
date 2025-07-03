#include "packet_demuxer.h"
#include <iostream>
#include <algorithm>

PacketDemuxer::PacketDemuxer() 
    : is_initialized_(false)
    , is_eof_(false)
    , next_sync_timestamp_(0.0) {
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
    next_sync_timestamp_ = 0.0;
    
    return true;
}

bool PacketDemuxer::readNextSyncedPackets(SyncedPackets& packets) {
    if (!is_initialized_) {
        return false;
    }
    
    // 清空输出
    packets.audio_packet = nullptr;
    packets.video_packet = nullptr;
    packets.timestamp = 0.0;
    
    // 如果已经EOF，检查缓冲区是否还有数据
    if (is_eof_ && audio_buffer_.empty() && video_buffer_.empty()) {
        return false;
    }
    
    // 填充缓冲区
    if (!is_eof_) {
        fillBuffers();
    }
    
    // 如果音频缓冲区为空，返回false
    if (audio_buffer_.empty()) {
        // 清理视频缓冲区
        clearBuffers();
        is_eof_ = true;
        return false;
    }
    
    // 获取下一个音频包作为同步基准
    packets.audio_packet = audio_buffer_.front();
    audio_buffer_.pop();
    
    // 计算音频时间戳
    double audio_timestamp = getPacketTimestamp(packets.audio_packet, true);
    packets.timestamp = audio_timestamp;
    
    // 找到最匹配的视频包
    packets.video_packet = findBestVideoPacket(audio_timestamp);
    
    return true;
}

void PacketDemuxer::close() {
    clearBuffers();
    reader_.close();
    is_initialized_ = false;
    is_eof_ = false;
    next_sync_timestamp_ = 0.0;
}

bool PacketDemuxer::fillBuffers() {
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        return false;
    }
    
    int packets_read = 0;
    
    // 读取包直到两个缓冲区都有足够的数据
    while (reader_.readNextPacket(packet)) {
        packets_read++;
        
        if (reader_.isAudioPacket(packet)) {
            // 复制音频包到缓冲区
            AVPacket* audio_pkt = av_packet_clone(packet);
            if (audio_pkt) {
                audio_buffer_.push(audio_pkt);
            }
        } else if (reader_.isVideoPacket(packet)) {
            // 复制视频包到缓冲区
            AVPacket* video_pkt = av_packet_clone(packet);
            if (video_pkt) {
                video_buffer_.push(video_pkt);
            }
        }
        
        av_packet_unref(packet);
        
        // 检查缓冲区是否已满
        if (!audio_buffer_.empty() && !video_buffer_.empty()) {
            // 计算缓冲区时间跨度
            double audio_front_ts = getPacketTimestamp(audio_buffer_.front(), true);
            double audio_back_ts = getPacketTimestamp(audio_buffer_.back(), true);
            double audio_duration = audio_back_ts - audio_front_ts;
            
            if (audio_duration >= MAX_BUFFER_DURATION) {
                break;
            }
        }
        
        // 防止缓冲区过大
        if (audio_buffer_.size() > 100 || video_buffer_.size() > 100) {
            break;
        }
    }
    
    av_packet_free(&packet);
    
    // 如果没有读到任何包，标记EOF
    if (packets_read == 0 && reader_.isEOF()) {
        is_eof_ = true;
    }
    
    return packets_read > 0;
}

void PacketDemuxer::clearBuffers() {
    // 清理音频缓冲区
    while (!audio_buffer_.empty()) {
        AVPacket* pkt = audio_buffer_.front();
        audio_buffer_.pop();
        av_packet_free(&pkt);
    }
    
    // 清理视频缓冲区
    while (!video_buffer_.empty()) {
        AVPacket* pkt = video_buffer_.front();
        video_buffer_.pop();
        av_packet_free(&pkt);
    }
}

double PacketDemuxer::getPacketTimestamp(AVPacket* packet, bool is_audio) const {
    if (!packet || packet->pts == AV_NOPTS_VALUE) {
        return 0.0;
    }
    
    auto stream_info = getStreamInfo();
    AVRational time_base = is_audio ? stream_info.audio_time_base : stream_info.video_time_base;
    
    return packet->pts * av_q2d(time_base);
}

AVPacket* PacketDemuxer::findBestVideoPacket(double target_timestamp) {
    if (video_buffer_.empty()) {
        return nullptr;
    }
    
    AVPacket* best_packet = nullptr;
    double min_diff = std::numeric_limits<double>::max();
    
    // 创建临时队列来存储检查过的包
    std::queue<AVPacket*> temp_queue;
    
    // 遍历视频缓冲区找到最佳匹配
    while (!video_buffer_.empty()) {
        AVPacket* pkt = video_buffer_.front();
        video_buffer_.pop();
        
        double video_ts = getPacketTimestamp(pkt, false);
        double diff = std::abs(video_ts - target_timestamp);
        
        // 如果这个包更接近目标时间戳
        if (diff < min_diff) {
            // 释放之前的最佳包（如果有）
            if (best_packet) {
                av_packet_free(&best_packet);
            }
            best_packet = pkt;
            min_diff = diff;
        } else {
            // 如果时间戳已经超过目标太多，保留在缓冲区
            if (video_ts > target_timestamp + SYNC_THRESHOLD) {
                temp_queue.push(pkt);
                break;
            } else {
                // 丢弃过时的包
                av_packet_free(&pkt);
            }
        }
        
        // 如果找到足够好的匹配，停止搜索
        if (min_diff < SYNC_THRESHOLD / 2) {
            break;
        }
    }
    
    // 将剩余的包放回缓冲区
    while (!video_buffer_.empty()) {
        temp_queue.push(video_buffer_.front());
        video_buffer_.pop();
    }
    
    // 恢复缓冲区
    video_buffer_ = std::move(temp_queue);
    
    // 如果没有找到合适的视频包，但差异太大，返回nullptr
    if (best_packet && min_diff > SYNC_THRESHOLD) {
        // 如果视频包太新，放回缓冲区
        if (getPacketTimestamp(best_packet, false) > target_timestamp) {
            // 创建新队列，将best_packet放在前面
            std::queue<AVPacket*> new_queue;
            new_queue.push(best_packet);
            while (!video_buffer_.empty()) {
                new_queue.push(video_buffer_.front());
                video_buffer_.pop();
            }
            video_buffer_ = std::move(new_queue);
            best_packet = nullptr;
        }
        // 如果视频包太旧，直接丢弃
        else {
            av_packet_free(&best_packet);
            best_packet = nullptr;
        }
    }
    
    return best_packet;
}