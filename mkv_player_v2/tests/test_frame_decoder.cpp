#include <catch2/catch_test_macros.hpp>
#include "../src/hw_frame_decoder.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("FrameDecoder basic functionality", "[frame_decoder]") {
    HwFrameDecoder decoder;
    
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
    
    HwFrameDecoder decoder;
    
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
    
    HwFrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("Hardware decoder initialization failed - may not be supported on this system");
    }
    
    SECTION("Decode synchronized audio and video frames") {
        int frames_decoded = 0;
        int total_read_attempts = 0;
        const int MAX_READ_ATTEMPTS = 100; // 防止无限循环
        HwFrameDecoder::HwFramePair decoded_frames;
        
        // 解码前20对音视频帧
        while (decoder.readNextHwFramePair(decoded_frames) && frames_decoded < 20 && total_read_attempts < MAX_READ_ATTEMPTS) {
            total_read_attempts++;
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


TEST_CASE("FrameDecoder error handling", "[frame_decoder]") {
    HwFrameDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        HwFrameDecoder::HwFramePair frames;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextHwFramePair(frames));
        
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