#include <catch2/catch_test_macros.hpp>
#include "../src/rgb_frame_decoder.h"
#include "../src/rgb_verification.h"
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

TEST_CASE("RGB verification functionality", "[rgb_verification]") {
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        SKIP("Test MKV file not found: " + TEST_MKV_FILE);
    }
    
    RGBFrameDecoder decoder;
    
    // 使用内部设备方式初始化（避免设备兼容性问题）
    if (!decoder.open(TEST_MKV_FILE)) {
        SKIP("RGB decoder initialization failed");
    }
    
    // 使用解码器内部的设备进行验证
    ID3D11Device* decoder_device = decoder.getD3D11Device();
    ID3D11DeviceContext* decoder_context = decoder.getD3D11Context();
    
    SECTION("RGB frame correctness verification") {
        RGBFrameDecoder::RGBFramePair rgb_pair;
        
        // 解码直到获得有效的RGB帧（前几帧可能需要建立GOP缓冲区）
        bool found_valid_rgb = false;
        int attempts = 0;
        const int max_attempts = 10;
        
        while (!found_valid_rgb && attempts < max_attempts) {
            std::cout << "Attempt " << (attempts + 1) << " to decode RGB frame..." << std::endl;
            bool read_success = decoder.readNextRGBFramePair(rgb_pair);
            
            if (read_success && rgb_pair.rgb_frame.is_valid) {
                found_valid_rgb = true;
                std::cout << "Successfully decoded RGB frame on attempt " << (attempts + 1) << std::endl;
                break;
            }
            attempts++;
        }
        
        if (found_valid_rgb) {
            
            std::cout << "\\n=== RGB Frame Verification ===" << std::endl;
            std::cout << "Frame dimensions: " << rgb_pair.rgb_frame.width 
                     << "x" << rgb_pair.rgb_frame.height << std::endl;
            std::cout << "Frame timestamp: " << rgb_pair.rgb_frame.timestamp << std::endl;
            
            // 1. 读取纹理数据 - 使用RGB解码器的D3D11设备，不是测试创建的设备
            std::vector<RGBVerification::PixelData> pixel_data;
            int width, height;
            
            // 使用解码器内部的设备读取纹理数据
            bool read_success = RGBVerification::readTextureData(
                decoder_device, decoder_context, rgb_pair.rgb_frame.rgb_texture.Get(), 
                pixel_data, width, height
            );
            
            REQUIRE(read_success);
            REQUIRE(width == rgb_pair.rgb_frame.width);
            REQUIRE(height == rgb_pair.rgb_frame.height);
            REQUIRE(pixel_data.size() == width * height);
            
            // 2. 验证RGB值范围
            bool valid_range = RGBVerification::isValidRGBRange(pixel_data);
            REQUIRE(valid_range);
            std::cout << "✓ RGB values are in valid range [0-255]" << std::endl;
            
            // 3. 计算图像统计信息
            auto stats = RGBVerification::calculateImageStats(pixel_data, width, height);
            RGBVerification::printImageStats(stats, "First RGB Frame");
            
            // 基本合理性检查
            REQUIRE(stats.mean_r >= 0);
            REQUIRE(stats.mean_g >= 0);
            REQUIRE(stats.mean_b >= 0);
            REQUIRE(stats.mean_r <= 255);
            REQUIRE(stats.mean_g <= 255);
            REQUIRE(stats.mean_b <= 255);
            
            // 检查不是完全黑色（我们的测试视频应该有内容）
            bool not_all_black = (stats.max_r > 10 || stats.max_g > 10 || stats.max_b > 10);
            REQUIRE(not_all_black);
            std::cout << "✓ Frame is not completely black" << std::endl;
            
            // 4. 保存第一帧为图片文件
            std::string filename = "rgb_frame_0.bmp";
            bool save_success = RGBVerification::saveTextureAsBMP(
                decoder_device, decoder_context, rgb_pair.rgb_frame.rgb_texture.Get(), filename
            );
            REQUIRE(save_success);
            std::cout << "✓ Saved RGB frame as: " << filename << std::endl;
            
            // 5. 检测是否为纯色（测试视频不应该是纯色）
            RGBVerification::PixelData solid_color;
            bool is_solid = RGBVerification::detectSolidColor(pixel_data, width, height, solid_color, 5.0);
            
            if (is_solid) {
                std::cout << "! Frame appears to be solid color: RGB(" 
                         << (int)solid_color.r << "," << (int)solid_color.g << "," << (int)solid_color.b << ")" << std::endl;
            } else {
                std::cout << "✓ Frame has varied content (not solid color)" << std::endl;
            }
            
        } else {
            FAIL("Failed to decode RGB frame after " + std::to_string(max_attempts) + " attempts");
        }
    }
    
    decoder.close();
}