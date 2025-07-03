#include <catch2/catch_test_macros.hpp>
#include "../src/mkv_stream_reader.h"
#include <filesystem>

// 测试用MKV文件路径 - 需要你提供一个真实的MKV文件
const std::string TEST_MKV_FILE = "test_data/sample.mkv";

TEST_CASE("MKVStreamReader basic functionality", "[mkv_reader]") {
    MKVStreamReader reader;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(reader.isOpen());
        REQUIRE_FALSE(reader.isEOF());
    }
    
    SECTION("Open non-existent file") {
        REQUIRE_FALSE(reader.open("non_existent_file.mkv"));
        REQUIRE_FALSE(reader.isOpen());
    }
}

TEST_CASE("MKVStreamReader with valid MKV file", "[mkv_reader][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    MKVStreamReader reader;
    
    SECTION("Open valid MKV file") {
        REQUIRE(reader.open(TEST_MKV_FILE));
        REQUIRE(reader.isOpen());
        REQUIRE_FALSE(reader.isEOF());
        
        // 获取流信息
        auto info = reader.getStreamInfo();
        
        // 验证基本流信息
        REQUIRE(info.video_stream_index >= 0);
        REQUIRE(info.audio_stream_index >= 0);
        REQUIRE(info.width > 0);
        REQUIRE(info.height > 0);
        REQUIRE(info.duration > 0.0);
        REQUIRE_FALSE(info.video_codec.empty());
        REQUIRE_FALSE(info.audio_codec.empty());
        REQUIRE(info.fps > 0.0);
        REQUIRE(info.audio_sample_rate > 0);
        REQUIRE(info.audio_channels > 0);
        
        reader.close();
        REQUIRE_FALSE(reader.isOpen());
    }
    
    SECTION("Read packets using pull interface") {
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        AVPacket* packet = av_packet_alloc();
        REQUIRE(packet != nullptr);
        
        int video_packets = 0;
        int audio_packets = 0;
        int total_packets = 0;
        
        // 读取前100个包进行测试
        while (reader.readNextPacket(packet) && total_packets < 100) {
            total_packets++;
            
            if (reader.isVideoPacket(packet)) {
                video_packets++;
                REQUIRE(packet->stream_index == reader.getStreamInfo().video_stream_index);
            }
            
            if (reader.isAudioPacket(packet)) {
                audio_packets++;
                REQUIRE(packet->stream_index == reader.getStreamInfo().audio_stream_index);
            }
            
            av_packet_unref(packet);
        }
        
        // 验证读取到了音视频包
        REQUIRE(video_packets > 0);
        REQUIRE(audio_packets > 0);
        REQUIRE(total_packets == (video_packets + audio_packets));
        
        av_packet_free(&packet);
        reader.close();
    }
    
    SECTION("EOF detection") {
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        AVPacket* packet = av_packet_alloc();
        REQUIRE(packet != nullptr);
        
        // 读取所有包直到EOF
        while (reader.readNextPacket(packet)) {
            av_packet_unref(packet);
        }
        
        // 验证EOF状态
        REQUIRE(reader.isEOF());
        REQUIRE_FALSE(reader.readNextPacket(packet));
        
        av_packet_free(&packet);
        reader.close();
    }
    
    SECTION("Seek functionality") {
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        auto info = reader.getStreamInfo();
        double mid_time = info.duration / 2.0;
        
        // 跳转到视频中间位置
        REQUIRE(reader.seekToTime(mid_time));
        
        // 验证能继续读取包
        AVPacket* packet = av_packet_alloc();
        REQUIRE(reader.readNextPacket(packet));
        
        av_packet_free(&packet);
        reader.close();
    }
    
    SECTION("Multiple open/close cycles") {
        // 测试多次打开关闭同一文件
        for (int i = 0; i < 3; i++) {
            REQUIRE(reader.open(TEST_MKV_FILE));
            REQUIRE(reader.isOpen());
            
            auto info = reader.getStreamInfo();
            REQUIRE(info.video_stream_index >= 0);
            
            reader.close();
            REQUIRE_FALSE(reader.isOpen());
        }
    }
    
    SECTION("Stream info consistency") {
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        auto info1 = reader.getStreamInfo();
        auto info2 = reader.getStreamInfo();
        
        // 多次获取应该返回相同信息
        REQUIRE(info1.video_stream_index == info2.video_stream_index);
        REQUIRE(info1.audio_stream_index == info2.audio_stream_index);
        REQUIRE(info1.width == info2.width);
        REQUIRE(info1.height == info2.height);
        REQUIRE(info1.duration == info2.duration);
        REQUIRE(info1.video_codec == info2.video_codec);
        REQUIRE(info1.audio_codec == info2.audio_codec);
        
        reader.close();
    }
}

TEST_CASE("MKVStreamReader error handling", "[mkv_reader]") {
    MKVStreamReader reader;
    
    SECTION("Operations on closed reader") {
        // 未打开文件时的操作应该安全失败
        REQUIRE_FALSE(reader.isOpen());
        
        AVPacket* packet = av_packet_alloc();
        REQUIRE_FALSE(reader.readNextPacket(packet));
        REQUIRE_FALSE(reader.isVideoPacket(packet));
        REQUIRE_FALSE(reader.isAudioPacket(packet));
        REQUIRE_FALSE(reader.seekToTime(10.0));
        
        av_packet_free(&packet);
    }
    
    SECTION("Invalid seek operations") {
        if (!std::filesystem::exists(TEST_MKV_FILE)) {
            SKIP("Test MKV file not found: " + TEST_MKV_FILE);
        }
        
        REQUIRE(reader.open(TEST_MKV_FILE));
        
        // 测试无效的seek操作
        REQUIRE_FALSE(reader.seekToTime(-1.0));
        REQUIRE_FALSE(reader.seekToTime(reader.getStreamInfo().duration + 100.0));
        
        reader.close();
    }
}

