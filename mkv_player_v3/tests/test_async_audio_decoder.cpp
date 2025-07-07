#include <catch2/catch_test_macros.hpp>
#include "../src/async_audio_decoder.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

namespace fs = std::filesystem;

TEST_CASE("AsyncAudioDecoder Basic Test", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
}

TEST_CASE("AsyncAudioDecoder Read Audio Frame", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    AsyncAudioDecoder::DecodedFrame frame;
    bool success = decoder.readNextFrame(frame);
    
    if (success) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE_FALSE(frame.is_eof);
        
        // 验证音频帧属性
        AVFrame* audio_frame = frame.frame;
        REQUIRE(audio_frame->nb_samples > 0);
        REQUIRE(audio_frame->sample_rate > 0);
        REQUIRE(audio_frame->ch_layout.nb_channels > 0);
        
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AsyncAudioDecoder Performance Test", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int frame_count = 0;
    AsyncAudioDecoder::DecodedFrame frame;
    
    // 连续读取多个帧，测试异步性能
    while (frame_count < 10 && decoder.readNextFrame(frame)) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE_FALSE(frame.is_eof);
        
        frame_count++;
        
        // 模拟处理时间
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (frame_count > 0) {
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AsyncAudioDecoder Async Preloading Test", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 等待一小段时间让工作线程预读取第一帧
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 第一次读取应该很快（已经预读取）
    auto start_time = std::chrono::high_resolution_clock::now();
    AsyncAudioDecoder::DecodedFrame frame;
    bool success = decoder.readNextFrame(frame);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (success) {
        REQUIRE(frame.is_valid);
        
        // 验证预读取性能：应该在很短时间内返回
        REQUIRE(duration.count() < 10000); // 小于10ms
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AsyncAudioDecoder Thread Safety Test", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    std::atomic<int> frame_count(0);
    std::atomic<bool> has_error(false);
    
    // 创建多个读取线程测试线程安全性
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 3; i++) {
        threads.emplace_back([&]() {
            try {
                AsyncAudioDecoder::DecodedFrame frame;
                while (frame_count.load() < 5 && decoder.readNextFrame(frame)) {
                    if (frame.is_valid) {
                        frame_count++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    if (frame.is_eof) break;
                }
            } catch (...) {
                has_error = true;
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE_FALSE(has_error.load());
}

TEST_CASE("AsyncAudioDecoder Seek Test", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 读取第一帧
    AsyncAudioDecoder::DecodedFrame frame1;
    bool success1 = decoder.readNextFrame(frame1);
    
    if (success1) {
        REQUIRE(frame1.is_valid);
        
        // Seek到开头
        REQUIRE(decoder.seekToTime(0.0));
        
        // 再次读取第一帧
        AsyncAudioDecoder::DecodedFrame frame2;
        bool success2 = decoder.readNextFrame(frame2);
        
        if (success2) {
            REQUIRE(frame2.is_valid);
        }
    } else {
        WARN("No audio frames found for seek test");
    }
}

TEST_CASE("AsyncAudioDecoder Close and Reopen", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    
    // 第一次打开
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    
    // 读取一帧
    AsyncAudioDecoder::DecodedFrame frame;
    decoder.readNextFrame(frame);
    
    // 关闭
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
    
    // 重新打开
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
    
}

TEST_CASE("AsyncAudioDecoder EOF Detection", "[AsyncAudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncAudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    AsyncAudioDecoder::DecodedFrame frame;
    int frame_count = 0;
    
    // 读取所有音频帧直到EOF
    while (decoder.readNextFrame(frame) && frame_count < 1000) {  // 防止无限循环
        if (frame.is_valid) {
            frame_count++;
        }
        if (frame.is_eof) {
            break;
        }
    }
    
    // 验证EOF状态
    if (frame_count > 0) {
        REQUIRE(decoder.isEOF());
    } else {
        WARN("No audio frames found in test file");
    }
}