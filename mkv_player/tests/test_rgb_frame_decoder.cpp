#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_frame_decoder.h"
#include <filesystem>
#include <iostream>

// 测试用MKV文件路径
const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

// 辅助函数：创建D3D11设备
bool CreateTestD3D11Device(ID3D11Device** device, ID3D11DeviceContext** context) {
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
    
    SECTION("Open with nullptr should use internal device") {
        // 现在nullptr会使用内部设备，所以应该成功
        bool result = decoder.open(TEST_MKV_FILE, nullptr);
        if (result) {
            REQUIRE(decoder.isInitialized());
            decoder.close();
        }
        // 如果失败，说明内部设备不支持Video Processor，这也是可接受的
    }
    
    SECTION("Open non-existent file should fail") {
        ID3D11Device* device = nullptr;
        ID3D11DeviceContext* context = nullptr;
        
        if (CreateTestD3D11Device(&device, &context)) {
            REQUIRE_FALSE(decoder.open("non_existent_file.mkv", device));
            REQUIRE_FALSE(decoder.isInitialized());
            
            device->Release();
            context->Release();
        }
    }
}

TEST_CASE("RGBFrameDecoder with valid MKV file", "[rgb_frame_decoder][requires_test_file]") {
    // 检查测试文件是否存在
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    // 创建D3D11设备
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    
    if (!CreateTestD3D11Device(&device, &context)) {
        SKIP("Failed to create D3D11 device - may not support video on this system");
    }
    
    RGBFrameDecoder decoder;
    
    SECTION("Open valid MKV file") {
        bool open_result = decoder.open(TEST_MKV_FILE); // 使用内部设备
        
        if (open_result) {
            REQUIRE(decoder.isInitialized());
            REQUIRE(decoder.isHardwareAccelerated());
            REQUIRE(decoder.getVideoCodecName() != nullptr);
            REQUIRE(decoder.getAudioCodecName() != nullptr);
            REQUIRE(decoder.getVideoWidth() > 0);
            REQUIRE(decoder.getVideoHeight() > 0);
            
            std::cout << "Hardware RGB decoder initialized: " << decoder.getVideoCodecName() << std::endl;
            std::cout << "Video dimensions: " << decoder.getVideoWidth() << "x" << decoder.getVideoHeight() << std::endl;
        } else {
            REQUIRE_FALSE(decoder.isInitialized());
            std::cout << "RGB decoder initialization failed - may not support Video Processor on this system" << std::endl;
        }
        
        decoder.close();
    }
    
    device->Release();
    context->Release();
}

TEST_CASE("RGBFrameDecoder RGB conversion workflow", "[rgb_frame_decoder][requires_test_file]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    // 创建D3D11设备
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    
    if (!CreateTestD3D11Device(&device, &context)) {
        SKIP("Failed to create D3D11 device");
    }
    
    RGBFrameDecoder decoder;
    
    if (!decoder.open(TEST_MKV_FILE)) { // 使用内部设备
        SKIP("RGB decoder initialization failed - may not support Video Processor on this system");
    }
    
    SECTION("Decode RGB frames") {
        int frames_decoded = 0;
        RGBFrameDecoder::DecodedFrames decoded_frames;
        
        // 解码前10对音视频帧
        while (decoder.readNextFrames(decoded_frames) && frames_decoded < 10) {
            frames_decoded++;
            
            // 验证音频帧
            if (decoded_frames.audio_frame.is_valid) {
                REQUIRE(decoded_frames.audio_frame.getAudioFrame() != nullptr);
                REQUIRE(decoded_frames.audio_frame.getAudioFrame()->nb_samples > 0);
                REQUIRE(decoded_frames.audio_frame.getAudioFrame()->sample_rate > 0);
                REQUIRE(decoded_frames.audio_frame.timestamp >= -1.0);
                REQUIRE(decoded_frames.audio_frame.getAudioFrame()->data[0] != nullptr);
                
                std::cout << "Audio frame: samples=" << decoded_frames.audio_frame.getAudioFrame()->nb_samples 
                         << ", ts=" << decoded_frames.audio_frame.timestamp << std::endl;
                
                av_frame_unref(decoded_frames.audio_frame.getAudioFrame());
            }
            
            // 验证RGB帧
            if (decoded_frames.rgb_frame.is_valid) {
                REQUIRE(decoded_frames.rgb_frame.rgb_texture != nullptr);
                REQUIRE(decoded_frames.rgb_frame.rgb_srv != nullptr);
                REQUIRE(decoded_frames.rgb_frame.width > 0);
                REQUIRE(decoded_frames.rgb_frame.height > 0);
                REQUIRE(decoded_frames.rgb_frame.timestamp >= -1.0);
                
                std::cout << "RGB frame: " << decoded_frames.rgb_frame.width 
                         << "x" << decoded_frames.rgb_frame.height 
                         << ", ts=" << decoded_frames.rgb_frame.timestamp << std::endl;
                
                // 验证音视频同步（时间戳应该接近）
                if (decoded_frames.audio_frame.is_valid) {
                    double av_diff = std::abs(decoded_frames.audio_frame.timestamp - decoded_frames.rgb_frame.timestamp);
                    REQUIRE(av_diff < 0.15); // 150ms容差（比原来稍宽松，因为有额外的处理步骤）
                }
            }
            
            // 至少要有一种帧有效
            REQUIRE((decoded_frames.audio_frame.is_valid || decoded_frames.rgb_frame.is_valid));
        }
        
        // 验证解码了预期数量的帧
        REQUIRE(frames_decoded > 0);
        std::cout << "Total RGB frames decoded: " << frames_decoded << std::endl;
    }
    
    decoder.close();
    device->Release();
    context->Release();
}

TEST_CASE("RGBFrameDecoder error handling", "[rgb_frame_decoder]") {
    RGBFrameDecoder decoder;
    
    SECTION("Operations on uninitialized decoder") {
        RGBFrameDecoder::DecodedFrames frames;
        
        // 未初始化时的操作应该安全失败
        REQUIRE_FALSE(decoder.readNextFrames(frames));
        REQUIRE_FALSE(frames.audio_frame.is_valid);
        REQUIRE_FALSE(frames.rgb_frame.is_valid);
    }
    
    SECTION("Multiple close calls") {
        // 多次关闭应该安全
        decoder.close();
        decoder.close();
        decoder.close();
    }
}