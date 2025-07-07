#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

TEST_CASE("HwVideoDecoder Basic Test", "[HwVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
}

TEST_CASE("HwVideoDecoder Double Buffer Memory Reuse", "[HwVideoDecoder][memory]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 双缓冲内存复用测试
    HwVideoDecoder::DecodedFrame frame1, frame2, frame3;
    
    // 第1次解码 - 使用frame[0]
    REQUIRE(decoder.readNextFrame(frame1));
    REQUIRE(frame1.is_valid);
    REQUIRE(frame1.frame != nullptr);
    AVFrame* frame1_ptr = frame1.frame;
    std::cout << "Frame1 address: " << frame1_ptr << std::endl;
    
    // 第2次解码 - 使用frame[1]
    REQUIRE(decoder.readNextFrame(frame2));
    REQUIRE(frame2.is_valid);
    REQUIRE(frame2.frame != nullptr);
    AVFrame* frame2_ptr = frame2.frame;
    std::cout << "Frame2 address: " << frame2_ptr << std::endl;
    
    // 第3次解码 - 应该复用frame[0]
    REQUIRE(decoder.readNextFrame(frame3));
    REQUIRE(frame3.is_valid);
    REQUIRE(frame3.frame != nullptr);
    AVFrame* frame3_ptr = frame3.frame;
    std::cout << "Frame3 address: " << frame3_ptr << std::endl;
    
    // 关键验证：双缓冲模式
    REQUIRE(frame1.frame != frame2.frame);  // Frame1 != Frame2
    REQUIRE(frame3.frame == frame1.frame);  // Frame3 == Frame1 (复用)
    REQUIRE(frame3.frame != frame2.frame);  // Frame3 != Frame2
    
    std::cout << "Double buffering working: Frame3 reuses Frame1 memory" << std::endl;
}

TEST_CASE("HwVideoDecoder Extended Double Buffer Pattern", "[HwVideoDecoder][memory]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 测试连续多次调用的双缓冲模式
    std::vector<AVFrame*> frame_pointers;
    
    for (int i = 0; i < 6; i++) {
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE(decoder.readNextFrame(frame));
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        
        frame_pointers.push_back(frame.frame);
        std::cout << "Frame " << i+1 << " address: " << frame.frame << std::endl;
    }
    
    // 验证双缓冲复用模式：frame[i] == frame[i+2]
    // Frame1 == Frame3 == Frame5 (偶数索引)
    REQUIRE(frame_pointers[0] == frame_pointers[2]);  // Frame1 == Frame3
    REQUIRE(frame_pointers[2] == frame_pointers[4]);  // Frame3 == Frame5
    
    // Frame2 == Frame4 == Frame6 (奇数索引)
    REQUIRE(frame_pointers[1] == frame_pointers[3]);  // Frame2 == Frame4
    REQUIRE(frame_pointers[3] == frame_pointers[5]);  // Frame4 == Frame6
    
    // 相邻的frame应该使用不同的内存
    REQUIRE(frame_pointers[0] != frame_pointers[1]);  // Frame1 != Frame2
    REQUIRE(frame_pointers[1] != frame_pointers[2]);  // Frame2 != Frame3
    REQUIRE(frame_pointers[2] != frame_pointers[3]);  // Frame3 != Frame4
    
    std::cout << "Double buffering pattern verified: Alternating between 2 frame buffers" << std::endl;
}

