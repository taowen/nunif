#include <catch2/catch_test_macros.hpp>
#include "../src/frame_decoder.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("FrameDecoder basic functionality", "[frame_decoder]") {
    FrameDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE_FALSE(decoder.isHardwareAccelerated());
    }
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isInitialized());
    }
}

TEST_CASE("FrameDecoder with valid MKV file", "[frame_decoder][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    FrameDecoder decoder;
    
    SECTION("Open valid MKV file") {
        bool open_result = decoder.open(TEST_MKV_FILE);
        
        // 如果系统支持硬件解码，应该打开成功
        if (open_result) {
            REQUIRE(decoder.isInitialized());
            REQUIRE(decoder.isHardwareAccelerated());
            REQUIRE(decoder.getVideoCodecName() != nullptr);
            REQUIRE(decoder.getAudioCodecName() != nullptr);
            
            std::cout << "Hardware video decoder initialized: " << decoder.getVideoCodecName() << std::endl;
            std::cout << "Audio decoder initialized: " << decoder.getAudioCodecName() << std::endl;
        } else {
            // 如果不支持硬件解码，应该失败（不fallback）
            REQUIRE_FALSE(decoder.isInitialized());
            REQUIRE_FALSE(decoder.isHardwareAccelerated());
            
            std::cout << "Hardware decoder not supported on this system" << std::endl;
        }
        
        decoder.close();
    }
}

TEST_CASE("FrameDecoder decoding workflow", "[frame_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    FrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("Hardware decoder initialization failed - may not be supported on this system");
    }
    
    SECTION("Decode synchronized audio and video frames") {
        int frames_decoded = 0;
        FrameDecoder::DecodedFrames decoded_frames;
        
        // 解码前20对音视频帧
        while (decoder.readNextFrames(decoded_frames) && frames_decoded < 20) {
            frames_decoded++;
            
            // 验证音频帧
            REQUIRE(decoded_frames.audio_frame.is_valid);
            REQUIRE(decoded_frames.audio_frame.getAudioFrame() != nullptr);
            REQUIRE(decoded_frames.audio_frame.getAudioFrame()->nb_samples > 0);
            REQUIRE(decoded_frames.audio_frame.getAudioFrame()->sample_rate > 0);
            REQUIRE(decoded_frames.audio_frame.timestamp >= -1.0); // 允许小的负偏移
            REQUIRE(decoded_frames.audio_frame.getAudioFrame()->data[0] != nullptr);
            
            // std::cout << "Audio frame: samples=" << decoded_frames.audio_frame.getAudioFrame()->nb_samples 
            //          << ", ts=" << decoded_frames.audio_frame.timestamp << std::endl;
            
            // 验证视频帧（同步后应该总是有效）
            if (decoded_frames.video_frame.is_valid) {
                REQUIRE(decoded_frames.video_frame.get() != nullptr);
                REQUIRE(decoded_frames.video_frame.get()->width > 0);
                REQUIRE(decoded_frames.video_frame.get()->height > 0);
                REQUIRE(decoded_frames.video_frame.timestamp >= -1.0); // 允许小的负偏移
                
                // 验证是硬件帧
                REQUIRE(decoded_frames.video_frame.get()->format == AV_PIX_FMT_D3D11);
                REQUIRE(decoded_frames.video_frame.get()->data[0] != nullptr);
                
                // std::cout << "Video frame: " << decoded_frames.video_frame.get()->width 
                //          << "x" << decoded_frames.video_frame.get()->height 
                //          << ", ts=" << decoded_frames.video_frame.timestamp << std::endl;
                
                // 验证音视频同步（时间戳应该接近）
                double av_diff = std::abs(decoded_frames.audio_frame.timestamp - decoded_frames.video_frame.timestamp);
                REQUIRE(av_diff < 0.1); // 100ms容差
                
                av_frame_unref(decoded_frames.video_frame.get());
            }
            
            // 释放音频帧
            av_frame_unref(decoded_frames.audio_frame.getAudioFrame());
        }
        
        // 验证解码了预期数量的帧
        REQUIRE(frames_decoded == 20);
        std::cout << "Total synchronized frames decoded: " << frames_decoded << std::endl;
    }
    
    decoder.close();
}

TEST_CASE("FrameDecoder packet-level decoding", "[frame_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    FrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("Hardware decoder initialization failed");
    }
    
    SECTION("Direct packet decoding") {
        // 获取内部demuxer进行测试
        auto* demuxer = decoder.getDemuxer();
        REQUIRE(demuxer != nullptr);
        
        PacketDemuxer::SyncedPackets packets;
        AVFrame* video_frame = av_frame_alloc();
        AVFrame* audio_frame = av_frame_alloc();
        
        int decoded_count = 0;
        
        // 直接测试包解码
        while (demuxer->readNextSyncedPackets(packets) && decoded_count < 10) {
            bool video_decoded = false;
            bool audio_decoded = false;
            
            if (packets.video_packet) {
                video_decoded = decoder.decodeVideoPacket(packets.video_packet, video_frame);
                if (video_decoded) {
                    REQUIRE(video_frame->format == AV_PIX_FMT_D3D11);
                    REQUIRE(video_frame->width > 0);
                    REQUIRE(video_frame->height > 0);
                    av_frame_unref(video_frame);
                }
                av_packet_free(&packets.video_packet);
            }
            
            if (packets.audio_packet) {
                audio_decoded = decoder.decodeAudioPacket(packets.audio_packet, audio_frame);
                if (audio_decoded) {
                    REQUIRE(audio_frame->nb_samples > 0);
                    REQUIRE(audio_frame->sample_rate > 0);
                    av_frame_unref(audio_frame);
                }
                av_packet_free(&packets.audio_packet);
            }
            
            if (video_decoded || audio_decoded) {
                decoded_count++;
            }
        }
        
        REQUIRE(decoded_count > 0);
        std::cout << "Packets decoded successfully: " << decoded_count << std::endl;
        
        av_frame_free(&video_frame);
        av_frame_free(&audio_frame);
    }
    
    decoder.close();
}

TEST_CASE("FrameDecoder error handling", "[frame_decoder]") {
    FrameDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        FrameDecoder::DecodedFrames frames;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextFrames(frames));
        
        // 直接包解码也应该失败
        AVPacket* dummy_packet = av_packet_alloc();
        AVFrame* dummy_frame = av_frame_alloc();
        
        REQUIRE_FALSE(decoder.decodeVideoPacket(dummy_packet, dummy_frame));
        REQUIRE_FALSE(decoder.decodeAudioPacket(dummy_packet, dummy_frame));
        
        av_packet_free(&dummy_packet);
        av_frame_free(&dummy_frame);
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        decoder.close();
        decoder.close();
        decoder.close();
    }
}