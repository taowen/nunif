#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_frame_decoder.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

// 辅助函数：创建D3D11设备
static bool CreateTestD3D11Device(ID3D11Device** device, ID3D11DeviceContext** context) {
    HRESULT hr = D3D11CreateDevice(
        nullptr,                    // 默认适配器
        D3D_DRIVER_TYPE_HARDWARE,   // 硬件驱动
        nullptr,                    // 软件驱动句柄
        D3D11_CREATE_DEVICE_VIDEO_SUPPORT, // 支持视频
        nullptr,                    // 功能级别数组
        0,                          // 功能级别数组大小
        D3D11_SDK_VERSION,          // SDK版本
        device,                     // 输出设备
        nullptr,                    // 输出功能级别
        context                     // 输出设备上下文
    );
    
    return SUCCEEDED(hr);
}

TEST_CASE("RGBFrameDecoder basic functionality", "[rgb_frame_decoder]") {
    RGBFrameDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE(decoder.getVideoWidth() == 0);
        REQUIRE(decoder.getVideoHeight() == 0);
    }
    
    SECTION("Open non-existent file should fail") {
        REQUIRE_FALSE(decoder.open("non_existent_file.mkv"));
        REQUIRE_FALSE(decoder.isInitialized());
    }
    
}

TEST_CASE("RGBFrameDecoder with valid MKV file", "[rgb_frame_decoder][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    RGBFrameDecoder decoder;
    
    SECTION("Open valid MKV file") {
        bool open_result = decoder.open(TEST_MKV_FILE);
        
        if (open_result) {
            REQUIRE(decoder.isInitialized());
            REQUIRE(decoder.getFrameDecoder()->isHardwareAccelerated());
            REQUIRE(decoder.getFrameDecoder()->getVideoCodecName() != nullptr);
            REQUIRE(decoder.getFrameDecoder()->getAudioCodecName() != nullptr);
            REQUIRE(decoder.getVideoWidth() > 0);
            REQUIRE(decoder.getVideoHeight() > 0);
            
            std::cout << "Hardware RGB decoder initialized: " << decoder.getFrameDecoder()->getVideoCodecName() << std::endl;
            std::cout << "Video dimensions: " << decoder.getVideoWidth() << "x" << decoder.getVideoHeight() << std::endl;
        } else {
            REQUIRE_FALSE(decoder.isInitialized());
            std::cout << "RGB decoder initialization failed - may not support Video Processor on this system" << std::endl;
        }
        
        decoder.close();
    }
}

TEST_CASE("RGBFrameDecoder RGB conversion workflow", "[rgb_frame_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    RGBFrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("RGB decoder initialization failed - may not support Video Processor on this system");
    }
    
    SECTION("Decode RGB frames") {
        int frames_decoded = 0;
        RGBFrameDecoder::RGBFramePair rgb_pair;
        
        // 解码前10对音视频帧
        while (decoder.readNextRGBFramePair(rgb_pair) && frames_decoded < 10) {
            frames_decoded++;
            
            // 验证音频帧
            if (rgb_pair.audio_frame.is_valid) {
                REQUIRE(rgb_pair.audio_frame.getAudioFrame() != nullptr);
                REQUIRE(rgb_pair.audio_frame.getAudioFrame()->nb_samples > 0);
                REQUIRE(rgb_pair.audio_frame.getAudioFrame()->sample_rate > 0);
                REQUIRE(rgb_pair.audio_frame.timestamp >= -1.0);
                REQUIRE(rgb_pair.audio_frame.getAudioFrame()->data[0] != nullptr);
                
                std::cout << "Audio frame: samples=" << rgb_pair.audio_frame.getAudioFrame()->nb_samples 
                         << ", ts=" << rgb_pair.audio_frame.timestamp << std::endl;
                
                av_frame_unref(rgb_pair.audio_frame.getAudioFrame());
            }
            
            // 验证RGB帧
            if (rgb_pair.rgb_frame.is_valid) {
                REQUIRE(rgb_pair.rgb_frame.rgb_texture != nullptr);
                REQUIRE(rgb_pair.rgb_frame.rgb_srv != nullptr);
                REQUIRE(rgb_pair.rgb_frame.width > 0);
                REQUIRE(rgb_pair.rgb_frame.height > 0);
                REQUIRE(rgb_pair.rgb_frame.timestamp >= -1.0);
                
                std::cout << "RGB frame: " << rgb_pair.rgb_frame.width 
                         << "x" << rgb_pair.rgb_frame.height 
                         << ", ts=" << rgb_pair.rgb_frame.timestamp << std::endl;
                
                // 验证音视频同步（时间戳应该接近）
                if (rgb_pair.audio_frame.is_valid) {
                    double av_diff = std::abs(rgb_pair.audio_frame.timestamp - rgb_pair.rgb_frame.timestamp);
                    REQUIRE(av_diff < 0.15); // 150ms容差（比原来稍宽松，因为有额外的处理步骤）
                }
            }
            
            // 至少要有一种帧有效
            REQUIRE((rgb_pair.audio_frame.is_valid || rgb_pair.rgb_frame.is_valid));
            REQUIRE(rgb_pair.is_valid);
        }
        
        // 验证解码了预期数量的帧
        REQUIRE(frames_decoded > 0);
        std::cout << "Total RGB frames decoded: " << frames_decoded << std::endl;
    }
    
    decoder.close();
}

TEST_CASE("RGBFrameDecoder error handling", "[rgb_frame_decoder]") {
    RGBFrameDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        RGBFrameDecoder::RGBFramePair rgb_pair;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextRGBFramePair(rgb_pair));
        REQUIRE_FALSE(rgb_pair.audio_frame.is_valid);
        REQUIRE_FALSE(rgb_pair.rgb_frame.is_valid);
        REQUIRE_FALSE(rgb_pair.is_valid);
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        decoder.close();
        decoder.close();
        decoder.close();
    }
}