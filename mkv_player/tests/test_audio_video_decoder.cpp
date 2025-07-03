#include <catch2/catch_test_macros.hpp>
#include "../src/audio_video_decoder_old.h"
#include "../src/mkv_stream_reader.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample.mkv";

TEST_CASE("AudioVideoDecoder basic functionality", "[audio_video_decoder]") {
    AudioVideoDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE_FALSE(decoder.isHardwareAccelerated());
    }
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isInitialized());
    }
}

TEST_CASE("AudioVideoDecoder with valid MKV file", "[audio_video_decoder][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    AudioVideoDecoder decoder;
    
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

TEST_CASE("AudioVideoDecoder decoding workflow", "[audio_video_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    AudioVideoDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("Hardware decoder initialization failed - may not be supported on this system");
    }
    
    SECTION("Decode audio and video frames") {
        int frames_decoded = 0;
        AudioVideoDecoder::DecodedFrames decoded_frames;
        
        // 解码前20对音视频帧
        int attempts = 0;
        while (decoder.readNextFrames(decoded_frames) && frames_decoded < 20 && attempts < 100) {
            attempts++;
            
            // 验证音频帧（以音频时间戳为准）
            if (decoded_frames.audio_frame.is_valid) {
                frames_decoded++;
                
                // 验证音频帧
                REQUIRE(decoded_frames.audio_frame.frame != nullptr);
                REQUIRE(decoded_frames.audio_frame.frame->nb_samples > 0);
                REQUIRE(decoded_frames.audio_frame.frame->sample_rate > 0);
                REQUIRE(decoded_frames.audio_frame.timestamp >= -1.0); // 允许小的负偏移
                REQUIRE(decoded_frames.audio_frame.frame->data[0] != nullptr);
                
                std::cout << "Audio frame: samples=" << decoded_frames.audio_frame.frame->nb_samples 
                         << ", ts=" << decoded_frames.audio_frame.timestamp << std::endl;
                
                // 验证视频帧（可能为空，如果跳帧了）
                if (decoded_frames.video_frame.is_valid) {
                    REQUIRE(decoded_frames.video_frame.frame != nullptr);
                    REQUIRE(decoded_frames.video_frame.frame->width > 0);
                    REQUIRE(decoded_frames.video_frame.frame->height > 0);
                    REQUIRE(decoded_frames.video_frame.timestamp >= -1.0); // 允许小的负偏移
                    
                    // 验证是硬件帧
                    REQUIRE(decoded_frames.video_frame.frame->format == AV_PIX_FMT_D3D11);
                    REQUIRE(decoded_frames.video_frame.frame->data[0] != nullptr);
                    
                    std::cout << "Video frame: " << decoded_frames.video_frame.frame->width 
                             << "x" << decoded_frames.video_frame.frame->height 
                             << ", ts=" << decoded_frames.video_frame.timestamp << std::endl;
                    
                    av_frame_unref(decoded_frames.video_frame.frame);
                }
                
                // 释放音频帧
                av_frame_unref(decoded_frames.audio_frame.frame);
            }
        }
        
        // 验证解码了一些帧
        std::cout << "Total attempts: " << attempts << ", audio+video frames decoded: " << frames_decoded << std::endl;
        REQUIRE(frames_decoded > 0);
    }
    
    decoder.close();
}

TEST_CASE("AudioVideoDecoder error handling", "[audio_video_decoder]") {
    AudioVideoDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        AudioVideoDecoder::DecodedFrames frames;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextFrames(frames));
    }
    
    SECTION("Open non-existent file") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isInitialized());
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        decoder.close();
        decoder.close();
        decoder.close();
    }
}