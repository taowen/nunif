#include <catch2/catch_test_macros.hpp>
#include "../src/async_rgb_frame_decoder.h"
#include <filesystem>
#include <iostream>
#include <thread>
#include <chrono>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("AsyncRGBFrameDecoder basic functionality", "[async_rgb_frame_decoder]") {
    AsyncRGBFrameDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE_FALSE(decoder.isWorkerRunning());
    }
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE_FALSE(decoder.isWorkerRunning());
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        decoder.close();
        decoder.close();
        decoder.close();
    }
}

TEST_CASE("AsyncRGBFrameDecoder with valid MKV file", "[async_rgb_frame_decoder][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    AsyncRGBFrameDecoder decoder;
    
    SECTION("Open valid MKV file") {
        bool open_result = decoder.open(TEST_MKV_FILE);
        
        if (open_result) {
            REQUIRE(decoder.isInitialized());
            // 给worker线程一点时间启动
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            REQUIRE(decoder.isWorkerRunning());
            
            // 验证底层RGB解码器已初始化
            REQUIRE(decoder.getRGBDecoder()->isInitialized());
            REQUIRE(decoder.getRGBDecoder()->getFrameDecoder()->isHardwareAccelerated());
            REQUIRE(decoder.getRGBDecoder()->getFrameDecoder()->getVideoCodecName() != nullptr);
            REQUIRE(decoder.getRGBDecoder()->getFrameDecoder()->getAudioCodecName() != nullptr);
            
            std::cout << "Async RGB decoder initialized: " << decoder.getRGBDecoder()->getFrameDecoder()->getVideoCodecName() << std::endl;
        } else {
            REQUIRE_FALSE(decoder.isInitialized());
            std::cout << "Async RGB decoder initialization failed - may not support Video Processor on this system" << std::endl;
        }
        
        decoder.close();
        
        // 关闭后worker线程应该停止
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE_FALSE(decoder.isWorkerRunning());
    }
}

TEST_CASE("AsyncRGBFrameDecoder async reading workflow", "[async_rgb_frame_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    AsyncRGBFrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("Async RGB decoder initialization failed - may not support Video Processor on this system");
    }
    
    SECTION("Decode RGB frames asynchronously") {
        int frames_decoded = 0;
        RGBFrameDecoder::RGBFramePair async_pair;
        
        // 解码前10对音视频帧
        while (decoder.readNextRGBFramePair(async_pair) && frames_decoded < 10) {
            frames_decoded++;
            
            // 验证音频帧
            if (async_pair.audio_frame.is_valid) {
                REQUIRE(async_pair.audio_frame.getAudioFrame() != nullptr);
                REQUIRE(async_pair.audio_frame.getAudioFrame()->nb_samples > 0);
                REQUIRE(async_pair.audio_frame.getAudioFrame()->sample_rate > 0);
                REQUIRE(async_pair.audio_frame.timestamp >= -1.0);
                REQUIRE(async_pair.audio_frame.getAudioFrame()->data[0] != nullptr);
                
                std::cout << "Async audio frame: samples=" << async_pair.audio_frame.getAudioFrame()->nb_samples 
                         << ", ts=" << async_pair.audio_frame.timestamp << std::endl;
            }
            
            // 验证RGB帧
            if (async_pair.rgb_frame.is_valid) {
                REQUIRE(async_pair.rgb_frame.rgb_texture != nullptr);
                REQUIRE(async_pair.rgb_frame.rgb_srv != nullptr);
                REQUIRE(async_pair.rgb_frame.width > 0);
                REQUIRE(async_pair.rgb_frame.height > 0);
                REQUIRE(async_pair.rgb_frame.timestamp >= -1.0);
                
                std::cout << "Async RGB frame: " << async_pair.rgb_frame.width 
                         << "x" << async_pair.rgb_frame.height 
                         << ", ts=" << async_pair.rgb_frame.timestamp << std::endl;
                
                // 验证音视频同步（时间戳应该接近）
                if (async_pair.audio_frame.is_valid) {
                    double av_diff = std::abs(async_pair.audio_frame.timestamp - async_pair.rgb_frame.timestamp);
                    REQUIRE(av_diff < 0.15); // 150ms容差
                }
            }
            
            // 至少要有一种帧有效
            REQUIRE((async_pair.audio_frame.is_valid || async_pair.rgb_frame.is_valid));
            REQUIRE(async_pair.is_valid);
            
            // 验证worker线程仍在运行
            REQUIRE(decoder.isWorkerRunning());
        }
        
        // 验证解码了预期数量的帧
        REQUIRE(frames_decoded > 0);
        std::cout << "Total async RGB frames decoded: " << frames_decoded << std::endl;
    }
    
    decoder.close();
}

TEST_CASE("AsyncRGBFrameDecoder error handling", "[async_rgb_frame_decoder]") {
    AsyncRGBFrameDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        RGBFrameDecoder::RGBFramePair async_pair;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextRGBFramePair(async_pair));
        REQUIRE_FALSE(async_pair.audio_frame.is_valid);
        REQUIRE_FALSE(async_pair.rgb_frame.is_valid);
        REQUIRE_FALSE(async_pair.is_valid);
    }
    
    SECTION("Graceful shutdown") {
        if (std::filesystem::exists(TEST_MKV_FILE)) {
            bool open_result = decoder.open(TEST_MKV_FILE);
            
            if (open_result) {
                // 给worker线程时间启动
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                REQUIRE(decoder.isWorkerRunning());
                
                // 关闭应该优雅地停止worker线程
                decoder.close();
                
                // 给worker线程时间停止
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                REQUIRE_FALSE(decoder.isWorkerRunning());
            }
        }
    }
}