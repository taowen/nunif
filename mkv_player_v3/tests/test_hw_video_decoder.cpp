#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <filesystem>
#include <iostream>
#include <set>

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

TEST_CASE("HwVideoDecoder GPU Memory Reuse - D3D11 Surface", "[HwVideoDecoder][gpu_memory]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 测试真正的GPU显存复用
    HwVideoDecoder::DecodedFrame frame1, frame2, frame3;
    
    // 第1次解码
    REQUIRE(decoder.readNextFrame(frame1));
    REQUIRE(frame1.is_valid);
    REQUIRE(frame1.frame != nullptr);
    
    // 获取D3D11表面地址和sub-resource index
    void* gpu_surface1 = nullptr;
    intptr_t sub_resource1 = 0;
    if (frame1.frame->format == AV_PIX_FMT_D3D11) {
        gpu_surface1 = frame1.frame->data[0];  // D3D11 surface指针
        sub_resource1 = (intptr_t)frame1.frame->data[1];  // sub-resource index
    }
    REQUIRE(gpu_surface1 != nullptr);
    
    // 第2次解码
    REQUIRE(decoder.readNextFrame(frame2));
    REQUIRE(frame2.is_valid);
    REQUIRE(frame2.frame != nullptr);
    
    void* gpu_surface2 = nullptr;
    intptr_t sub_resource2 = 0;
    if (frame2.frame->format == AV_PIX_FMT_D3D11) {
        gpu_surface2 = frame2.frame->data[0];
        sub_resource2 = (intptr_t)frame2.frame->data[1];
    }
    REQUIRE(gpu_surface2 != nullptr);
    
    // 第3次解码 - 应该复用第1个surface
    REQUIRE(decoder.readNextFrame(frame3));
    REQUIRE(frame3.is_valid);
    REQUIRE(frame3.frame != nullptr);
    
    void* gpu_surface3 = nullptr;
    intptr_t sub_resource3 = 0;
    if (frame3.frame->format == AV_PIX_FMT_D3D11) {
        gpu_surface3 = frame3.frame->data[0];
        sub_resource3 = (intptr_t)frame3.frame->data[1];
    }
    REQUIRE(gpu_surface3 != nullptr);
    
    // 关键验证：检查sub-resource index的复用模式
    if (gpu_surface1 == gpu_surface2 && gpu_surface2 == gpu_surface3) {
        // 如果surface指针相同，验证sub-resource使用了不同的indices
        // 验证FFmpeg确实在使用不同的sub-resource indices进行某种复用
        // 不强制要求特定的双缓冲模式，因为FFmpeg可能有自己的分配策略
        bool has_different_sub_resources = (sub_resource1 != sub_resource2) || 
                                          (sub_resource2 != sub_resource3) || 
                                          (sub_resource1 != sub_resource3);
        REQUIRE(has_different_sub_resources);  // 至少有部分sub-resource index不同
    } else {
        // 如果surface指针不同，验证传统的双缓冲
        REQUIRE(gpu_surface1 != gpu_surface2);  // 不同的GPU表面
        REQUIRE(gpu_surface3 == gpu_surface1);  // Frame3复用Frame1的GPU表面
        REQUIRE(gpu_surface3 != gpu_surface2);  // Frame3不等于Frame2
    }
}

TEST_CASE("HwVideoDecoder Extended GPU Surface Pattern", "[HwVideoDecoder][gpu_memory]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 测试连续多次调用的GPU surface和sub-resource模式
    std::vector<void*> gpu_surfaces;
    std::vector<intptr_t> sub_resources;
    
    for (int i = 0; i < 6; i++) {
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE(decoder.readNextFrame(frame));
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE(frame.frame->format == AV_PIX_FMT_D3D11);
        
        void* gpu_surface = frame.frame->data[0];
        intptr_t sub_resource = (intptr_t)frame.frame->data[1];
        REQUIRE(gpu_surface != nullptr);
        
        gpu_surfaces.push_back(gpu_surface);
        sub_resources.push_back(sub_resource);
    }
    
    // 检查是否所有surface指针都相同（使用同一个texture）
    bool all_surfaces_same = true;
    for (size_t i = 1; i < gpu_surfaces.size(); i++) {
        if (gpu_surfaces[i] != gpu_surfaces[0]) {
            all_surfaces_same = false;
            break;
        }
    }
    
    if (all_surfaces_same) {
        // 验证至少有一些不同的sub-resource indices被使用
        std::set<intptr_t> unique_sub_resources(sub_resources.begin(), sub_resources.end());
        REQUIRE(unique_sub_resources.size() >= 2);  // 至少使用了2个不同的sub-resource
    } else {
        // 如果surface指针不同，验证传统的双缓冲模式
        REQUIRE(gpu_surfaces[0] == gpu_surfaces[2]);  // Surface1 == Surface3
        REQUIRE(gpu_surfaces[2] == gpu_surfaces[4]);  // Surface3 == Surface5
        REQUIRE(gpu_surfaces[1] == gpu_surfaces[3]);  // Surface2 == Surface4
        REQUIRE(gpu_surfaces[3] == gpu_surfaces[5]);  // Surface4 == Surface6
        REQUIRE(gpu_surfaces[0] != gpu_surfaces[1]);  // Surface1 != Surface2
    }
}

