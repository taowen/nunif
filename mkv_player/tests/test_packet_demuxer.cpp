#include <catch2/catch_test_macros.hpp>
#include "../src/packet_demuxer.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample.mkv";

TEST_CASE("PacketDemuxer basic functionality", "[packet_demuxer]") {
    PacketDemuxer demuxer;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(demuxer.isInitialized());
    }
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(demuxer.open("non_existent_file.mkv"));
        REQUIRE_FALSE(demuxer.isInitialized());
    }
}

TEST_CASE("PacketDemuxer with valid MKV file", "[packet_demuxer][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    PacketDemuxer demuxer;
    
    SECTION("Open valid MKV file") {
        REQUIRE(demuxer.open(TEST_MKV_FILE));
        REQUIRE(demuxer.isInitialized());
        
        // 获取流信息
        auto stream_info = demuxer.getStreamInfo();
        REQUIRE(stream_info.video_stream_index >= 0);
        REQUIRE(stream_info.audio_stream_index >= 0);
        
        // 获取编解码器参数
        REQUIRE(demuxer.getVideoCodecParameters() != nullptr);
        REQUIRE(demuxer.getAudioCodecParameters() != nullptr);
        
        demuxer.close();
    }
}

TEST_CASE("PacketDemuxer synchronized packet reading", "[packet_demuxer][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    PacketDemuxer demuxer;
    
    if (!demuxer.open(TEST_MKV_FILE)) {
        SKIP("Failed to open test file");
    }
    
    SECTION("Read synchronized packets") {
        PacketDemuxer::SyncedPackets synced_packets;
        int packets_read = 0;
        double last_timestamp = -1.0;
        
        // 读取前50组同步包
        while (demuxer.readNextSyncedPackets(synced_packets) && packets_read < 50) {
            packets_read++;
            
            // 验证时间戳递增
            REQUIRE(synced_packets.timestamp >= last_timestamp);
            last_timestamp = synced_packets.timestamp;
            
            // 至少有一个包
            REQUIRE((synced_packets.audio_packet != nullptr || synced_packets.video_packet != nullptr));
            
            // 验证音频包
            if (synced_packets.audio_packet) {
                REQUIRE(synced_packets.audio_packet->size > 0);
                REQUIRE(synced_packets.audio_packet->data != nullptr);
                
                std::cout << "Audio packet: size=" << synced_packets.audio_packet->size 
                         << ", pts=" << synced_packets.audio_packet->pts
                         << ", sync_ts=" << synced_packets.timestamp << std::endl;
                
                av_packet_free(&synced_packets.audio_packet);
            }
            
            // 验证视频包  
            if (synced_packets.video_packet) {
                REQUIRE(synced_packets.video_packet->size > 0);
                REQUIRE(synced_packets.video_packet->data != nullptr);
                
                std::cout << "Video packet: size=" << synced_packets.video_packet->size
                         << ", pts=" << synced_packets.video_packet->pts  
                         << ", sync_ts=" << synced_packets.timestamp << std::endl;
                
                av_packet_free(&synced_packets.video_packet);
            }
        }
        
        REQUIRE(packets_read > 0);
        std::cout << "Total synced packet groups read: " << packets_read << std::endl;
    }
    
    SECTION("Audio-video synchronization") {
        PacketDemuxer::SyncedPackets synced_packets;
        int sync_checks = 0;
        
        // 读取包并检查同步
        while (demuxer.readNextSyncedPackets(synced_packets) && sync_checks < 20) {
            if (synced_packets.audio_packet && synced_packets.video_packet) {
                sync_checks++;
                
                // 获取实际的音视频时间戳
                auto stream_info = demuxer.getStreamInfo();
                double audio_ts = synced_packets.audio_packet->pts * av_q2d(stream_info.audio_time_base);
                double video_ts = synced_packets.video_packet->pts * av_q2d(stream_info.video_time_base);
                
                // 验证同步时间戳与音频时间戳接近（因为以音频为主时钟）
                REQUIRE(std::abs(synced_packets.timestamp - audio_ts) < 0.001);
                
                // 验证音视频时间戳差异在合理范围内
                double av_diff = std::abs(audio_ts - video_ts);
                REQUIRE(av_diff < 0.2); // 200ms容差
                
                std::cout << "Sync check: audio_ts=" << audio_ts 
                         << ", video_ts=" << video_ts
                         << ", diff=" << av_diff << "s" << std::endl;
                
                av_packet_free(&synced_packets.audio_packet);
                av_packet_free(&synced_packets.video_packet);
            } else {
                // 释放单个包
                if (synced_packets.audio_packet) av_packet_free(&synced_packets.audio_packet);
                if (synced_packets.video_packet) av_packet_free(&synced_packets.video_packet);
            }
        }
        
        REQUIRE(sync_checks > 0);
        std::cout << "Synchronization checks performed: " << sync_checks << std::endl;
    }
    
    demuxer.close();
}

TEST_CASE("PacketDemuxer EOF handling", "[packet_demuxer][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    PacketDemuxer demuxer;
    
    if (!demuxer.open(TEST_MKV_FILE)) {
        SKIP("Failed to open test file");
    }
    
    SECTION("Read until EOF") {
        PacketDemuxer::SyncedPackets synced_packets;
        int total_packets = 0;
        
        // 读取所有包直到EOF
        while (demuxer.readNextSyncedPackets(synced_packets)) {
            total_packets++;
            
            // 释放包
            if (synced_packets.audio_packet) av_packet_free(&synced_packets.audio_packet);
            if (synced_packets.video_packet) av_packet_free(&synced_packets.video_packet);
            
            // 防止无限循环
            if (total_packets > 10000) {
                FAIL("Too many packets - possible infinite loop");
            }
        }
        
        // 验证读取了一定数量的包
        REQUIRE(total_packets > 0);
        std::cout << "Total packets read until EOF: " << total_packets << std::endl;
        
        // EOF后继续读取应该返回false
        REQUIRE_FALSE(demuxer.readNextSyncedPackets(synced_packets));
        REQUIRE(demuxer.isEOF());
    }
    
    demuxer.close();
}

TEST_CASE("PacketDemuxer error handling", "[packet_demuxer]") {
    PacketDemuxer demuxer;
    
    SECTION("Operations on uninitialized demuxer") {
        PacketDemuxer::SyncedPackets packets;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(demuxer.readNextSyncedPackets(packets));
        REQUIRE(demuxer.getVideoCodecParameters() == nullptr);
        REQUIRE(demuxer.getAudioCodecParameters() == nullptr);
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        demuxer.close();
        demuxer.close();
        demuxer.close();
    }
}