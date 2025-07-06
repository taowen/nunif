#include <catch2/catch_test_macros.hpp>
#include "../src/packet_demuxer.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("PacketDemuxer basic functionality", "[packet_demuxer]") {
    PacketDemuxer demuxer;
    
    SECTION("Initial state and error handling") {
        REQUIRE_FALSE(demuxer.isInitialized());
        REQUIRE(demuxer.getValidPairCount() == 0);
        REQUIRE_FALSE(demuxer.hasValidPair());
        
        // 打开不存在的文件应该失败
        REQUIRE_FALSE(demuxer.open("non_existent_file.mkv"));
        REQUIRE_FALSE(demuxer.isInitialized());
        
        // 未初始化时的操作应该安全失败
        PacketDemuxer::PacketPair packets;
        REQUIRE_FALSE(demuxer.readNextPacketPair(packets));
        
        // 多次关闭应该安全
        demuxer.close();
        demuxer.close();
    }
}

TEST_CASE("PacketDemuxer core functionality", "[packet_demuxer][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    PacketDemuxer demuxer;
    REQUIRE(demuxer.open(TEST_MKV_FILE));
    REQUIRE(demuxer.isInitialized());
    
    SECTION("File opening and stream info") {
        // 验证流信息
        auto stream_info = demuxer.getReader().getStreamInfo();
        REQUIRE(stream_info.video_stream_index >= 0);
        REQUIRE(stream_info.audio_stream_index >= 0);
        REQUIRE(demuxer.getReader().getVideoCodecParameters() != nullptr);
        REQUIRE(demuxer.getReader().getAudioCodecParameters() != nullptr);
    }
    
    SECTION("Synchronized packet reading and audio-video sync") {
        PacketDemuxer::PacketPair synced_packets;
        int packets_read = 0;
        int sync_checks = 0;
        double last_timestamp = -1.0;
        
        // 读取包并验证同步
        while (demuxer.readNextPacketPair(synced_packets) && packets_read < 20) {
            packets_read++;
            REQUIRE(synced_packets.is_valid);
            REQUIRE(synced_packets.timestamp >= last_timestamp);
            last_timestamp = synced_packets.timestamp;
            
            // 至少有一个包
            REQUIRE((synced_packets.audio_packet != nullptr || synced_packets.video_packet != nullptr));
            
            // 验证包数据有效性
            if (synced_packets.audio_packet) {
                REQUIRE(synced_packets.audio_packet->size > 0);
                REQUIRE(synced_packets.audio_packet->data != nullptr);
            }
            if (synced_packets.video_packet) {
                REQUIRE(synced_packets.video_packet->size > 0);
                REQUIRE(synced_packets.video_packet->data != nullptr);
            }
            
            // 音视频同步验证（只对有双包的情况检查）
            if (synced_packets.audio_packet && synced_packets.video_packet && sync_checks < 5) {
                sync_checks++;
                auto stream_info = demuxer.getReader().getStreamInfo();
                double audio_ts = synced_packets.audio_packet->pts * av_q2d(stream_info.audio_time_base);
                double video_ts = synced_packets.video_packet->pts * av_q2d(stream_info.video_time_base);
                
                // 验证同步时间戳与音频时间戳接近
                REQUIRE(std::abs(synced_packets.timestamp - audio_ts) < 0.001);
                // 验证音视频时间戳差异在合理范围内
                REQUIRE(std::abs(audio_ts - video_ts) < 0.5);
            }
        }
        
        REQUIRE(packets_read > 0);
        REQUIRE(sync_checks > 0);
    }
    
    demuxer.close();
}

TEST_CASE("PacketDemuxer dual buffer and EOF handling", "[packet_demuxer][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    PacketDemuxer demuxer;
    REQUIRE(demuxer.open(TEST_MKV_FILE));
    
    SECTION("Dual buffer boundary conditions") {
        PacketDemuxer::PacketPair pair1, pair2;
        
        // 初始状态
        REQUIRE(demuxer.getValidPairCount() == 0);
        
        // 读取两个pair验证双缓冲
        REQUIRE(demuxer.readNextPacketPair(pair1));
        REQUIRE(pair1.is_valid);
        REQUIRE(demuxer.getValidPairCount() == 1);
        
        REQUIRE(demuxer.readNextPacketPair(pair2));
        REQUIRE(pair2.is_valid);
        REQUIRE(demuxer.getValidPairCount() == 2);
        REQUIRE(demuxer.hasValidPair());
        
        // 验证两个pair不同
        REQUIRE(pair1.timestamp != pair2.timestamp);
        REQUIRE(pair1.audio_packet != pair2.audio_packet);
    }
    
    SECTION("EOF handling and final pairs") {
        PacketDemuxer::PacketPair pair;
        int total_packets = 0;
        
        // 读取所有包直到EOF
        while (demuxer.readNextPacketPair(pair) && total_packets < 1000) {
            total_packets++;
        }
        
        // 验证EOF状态
        REQUIRE(total_packets > 0);
        REQUIRE(demuxer.isEOF());
        
        // EOF后继续读取应该失败
        REQUIRE_FALSE(demuxer.readNextPacketPair(pair));
        
        // 可能还有有效的缓存pair
        int valid_count = demuxer.getValidPairCount();
        REQUIRE(valid_count >= 0);
        REQUIRE(valid_count <= 2);
    }
    
    demuxer.close();
}