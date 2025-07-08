#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
#include "../src/async_rgb_video_decoder.h"
#include <iostream>
#include <thread>

TEST_CASE("VideoPlayer basic functionality", "[video_player]") {
    SECTION("Constructor and destructor work") {
        VideoPlayer player;
        REQUIRE(true);
    }
}

TEST_CASE("VideoPlayer with real video decoder", "[video_player]") {
    VideoPlayer player;
    
    // 打开MKV文件
    bool opened = player.open("test_data/sample_hw.mkv");
    
    if (!opened) {
        std::cout << "跳过视频解码器测试 - 无法打开测试文件" << std::endl;
        return;
    }
    
    // 创建D3D11设备用于测试
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &device, nullptr, &context);
    
    if (FAILED(hr)) {
        std::cout << "跳过视频解码器测试 - 无法创建D3D11设备" << std::endl;
        return;
    }
    
    // 创建离屏纹理
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = 320;
    desc.Height = 240;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    
    hr = device->CreateTexture2D(&desc, nullptr, &texture);
    if (SUCCEEDED(hr)) {
        hr = device->CreateRenderTargetView(texture, nullptr, &rtv);
    }
    
    if (SUCCEEDED(hr)) {
        bool result = player.initialize(device, context, rtv, nullptr);
        REQUIRE(result == true);
        
        // 使用真实视频帧进行渲染测试
        int frame_count = 0;
        
        for (int i = 0; i < 5; ++i) {
            // 每帧调用onTimer，VideoPlayer内部会自动获取视频帧
            player.onTimer();
            frame_count++;
            
            // 读取渲染结果进行验证
            D3D11_MAPPED_SUBRESOURCE mapped;
            ID3D11Texture2D* staging_texture = nullptr;
            
            // 创建staging纹理用于CPU读取
            D3D11_TEXTURE2D_DESC staging_desc = desc;
            staging_desc.Usage = D3D11_USAGE_STAGING;
            staging_desc.BindFlags = 0;
            staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            
            HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
            if (SUCCEEDED(hr)) {
                // 复制渲染结果到staging纹理
                context->CopyResource(staging_texture, texture);
                
                // 映射并读取像素数据
                hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
                if (SUCCEEDED(hr)) {
                    uint8_t* pixels = (uint8_t*)mapped.pData;
                    
                    // 检查前几个像素值
                    uint32_t* pixel_data = (uint32_t*)pixels;
                    uint32_t first_pixel = pixel_data[0];
                    uint32_t middle_pixel = pixel_data[(320 * 120) + 160]; // 中心像素
                    
                    std::cout << "Frame " << i << " - 第一个像素: 0x" << std::hex << first_pixel 
                              << ", 中心像素: 0x" << middle_pixel << std::dec << std::endl;
                    
                    // 验证像素不全是0（说明有实际内容）
                    bool has_content = false;
                    for (int y = 0; y < 240 && !has_content; y++) {
                        for (int x = 0; x < 320 && !has_content; x++) {
                            uint32_t pixel = pixel_data[y * (mapped.RowPitch / 4) + x];
                            if (pixel != 0) {
                                has_content = true;
                            }
                        }
                    }
                    
                    REQUIRE(has_content == true);
                    
                    context->Unmap(staging_texture, 0);
                }
                
                staging_texture->Release();
            }
        }
        
        std::cout << "真实视频渲染测试成功，处理了 " << frame_count << " 帧，像素验证通过" << std::endl;
    }
    
    // 清理
    if (rtv) rtv->Release();
    if (texture) texture->Release();
    if (context) context->Release();
    if (device) device->Release();
    
    player.close();
}