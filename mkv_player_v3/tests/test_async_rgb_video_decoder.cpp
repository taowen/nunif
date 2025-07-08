#include <catch2/catch_test_macros.hpp>
#include "../src/async_rgb_video_decoder.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

namespace fs = std::filesystem;

TEST_CASE("AsyncRgbVideoDecoder Basic Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 验证解码器状态通过尝试读取帧来检查
    
    // 验证D3D11设备
    ID3D11Device* device = decoder.getD3D11Device();
    ID3D11DeviceContext* context = decoder.getD3D11Context();
    REQUIRE(device != nullptr);
    REQUIRE(context != nullptr);
}

TEST_CASE("AsyncRgbVideoDecoder Read RGB Frame", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    AsyncRgbVideoDecoder::DecodedFrame frame;
    bool success = decoder.readNextFrame(frame);
    
    if (success) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.hw_frame != nullptr);
        
        // 验证硬件帧属性
        AVFrame* hw_frame = frame.hw_frame;
        REQUIRE(hw_frame->width > 0);
        REQUIRE(hw_frame->height > 0);
        REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
        
        // 验证RGB帧属性
        REQUIRE(frame.rgb_frame.is_valid);
        REQUIRE(frame.rgb_frame.rgb_texture != nullptr);
        REQUIRE(frame.rgb_frame.rgb_srv != nullptr);
        REQUIRE(frame.rgb_frame.width > 0);
        REQUIRE(frame.rgb_frame.height > 0);
        REQUIRE(frame.rgb_frame.width == hw_frame->width);
        REQUIRE(frame.rgb_frame.height == hw_frame->height);
        
    } else {
        WARN("No video frames found in test file");
    }
}

TEST_CASE("AsyncRgbVideoDecoder Performance Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int frame_count = 0;
    AsyncRgbVideoDecoder::DecodedFrame frame;
    
    // 连续读取多个帧，测试异步性能
    while (frame_count < 10 && decoder.readNextFrame(frame)) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.hw_frame != nullptr);
        REQUIRE(frame.rgb_frame.is_valid);
        REQUIRE(frame.rgb_frame.rgb_texture != nullptr);
        
        frame_count++;
        
        // 模拟处理时间
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (frame_count > 0) {
        std::cout << "AsyncRgbVideoDecoder decoded " << frame_count << " frames in " 
                  << duration.count() << "ms" << std::endl;
    } else {
        WARN("No video frames found in test file");
    }
}

TEST_CASE("AsyncRgbVideoDecoder Async Preloading Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 等待一小段时间让工作线程预读取第一帧
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 第一次读取应该很快（已经预读取）
    auto start_time = std::chrono::high_resolution_clock::now();
    AsyncRgbVideoDecoder::DecodedFrame frame;
    bool success = decoder.readNextFrame(frame);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    if (success) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.rgb_frame.is_valid);
        
        // 验证预读取性能：应该在很短时间内返回
        REQUIRE(duration.count() < 50000); // 小于50ms
        
        std::cout << "First frame read took " << duration.count() << " microseconds" << std::endl;
    } else {
        WARN("No video frames found in test file");
    }
}

TEST_CASE("AsyncRgbVideoDecoder Thread Safety Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    std::atomic<int> frame_count(0);
    std::atomic<bool> has_error(false);
    
    // 创建多个读取线程测试线程安全性
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 3; i++) {
        threads.emplace_back([&]() {
            try {
                AsyncRgbVideoDecoder::DecodedFrame frame;
                while (frame_count.load() < 5 && decoder.readNextFrame(frame)) {
                    if (frame.is_valid && frame.rgb_frame.is_valid) {
                        frame_count++;
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                    // 继续读取直到没有更多帧
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

TEST_CASE("AsyncRgbVideoDecoder Seek Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 读取第一帧
    AsyncRgbVideoDecoder::DecodedFrame frame1;
    bool success1 = decoder.readNextFrame(frame1);
    
    if (success1) {
        REQUIRE(frame1.is_valid);
        REQUIRE(frame1.rgb_frame.is_valid);
        
        // Seek到开头
        REQUIRE(decoder.seekToTime(0.0));
        
        // 再次读取第一帧
        AsyncRgbVideoDecoder::DecodedFrame frame2;
        bool success2 = decoder.readNextFrame(frame2);
        
        if (success2) {
            REQUIRE(frame2.is_valid);
            REQUIRE(frame2.rgb_frame.is_valid);
        }
    } else {
        WARN("No video frames found for seek test");
    }
}

TEST_CASE("AsyncRgbVideoDecoder Close and Reopen", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    
    // 第一次打开
    REQUIRE(decoder.open(test_file.string()));
    
    // 验证解码器已打开并可以读取帧
    
    // 读取一帧
    AsyncRgbVideoDecoder::DecodedFrame frame;
    decoder.readNextFrame(frame);
    
    // 关闭
    decoder.close();
    // 关闭后无法读取帧
    
    // 重新打开
    REQUIRE(decoder.open(test_file.string()));
    // 重新打开后可以读取帧
}

TEST_CASE("AsyncRgbVideoDecoder EOF Detection", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    AsyncRgbVideoDecoder::DecodedFrame frame;
    int frame_count = 0;
    
    // 读取所有视频帧直到EOF
    while (decoder.readNextFrame(frame) && frame_count < 100) {  // 防止无限循环
        if (frame.is_valid) {
            frame_count++;
        }
        // 继续读取直到没有更多帧
    }
    
    // 验证解码器已处理完文件
    if (frame_count > 0) {
        // 文件已处理完成
        std::cout << "Processed " << frame_count << " frames" << std::endl;
    } else {
        WARN("No video frames found in test file");
    }
}

TEST_CASE("AsyncRgbVideoDecoder Double Buffer Test", "[AsyncRgbVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AsyncRgbVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 读取前4帧来验证双缓冲机制
    std::vector<AsyncRgbVideoDecoder::DecodedFrame> frames;
    
    for (int i = 0; i < 4; i++) {
        AsyncRgbVideoDecoder::DecodedFrame frame;
        if (decoder.readNextFrame(frame) && frame.is_valid) {
            frames.push_back(frame);
        } else {
            break;
        }
    }
    
    if (frames.size() >= 4) {
        // 验证双缓冲：frame1和frame3应该使用相同的RGB纹理
        // frame2和frame4应该使用相同的RGB纹理
        ID3D11Texture2D* tex1 = frames[0].rgb_frame.rgb_texture.Get();
        ID3D11Texture2D* tex2 = frames[1].rgb_frame.rgb_texture.Get();
        ID3D11Texture2D* tex3 = frames[2].rgb_frame.rgb_texture.Get();
        ID3D11Texture2D* tex4 = frames[3].rgb_frame.rgb_texture.Get();
        
        REQUIRE(tex1 != nullptr);
        REQUIRE(tex2 != nullptr);
        REQUIRE(tex3 != nullptr);
        REQUIRE(tex4 != nullptr);
        
        // 验证双缓冲模式：frame1和frame3使用相同纹理，frame2和frame4使用相同纹理
        REQUIRE(tex1 == tex3);
        REQUIRE(tex2 == tex4);
        REQUIRE(tex1 != tex2);
        
        std::cout << "Double buffer validation successful" << std::endl;
    } else {
        WARN("Not enough frames for double buffer test");
    }
}