#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
#include <iostream>
#include <thread>

TEST_CASE("VideoPlayer basic functionality", "[video_player]") {
    SECTION("Constructor and destructor work") {
        VideoPlayer player;
        REQUIRE(true);
    }
}

TEST_CASE("VideoPlayer offscreen rendering", "[video_player]") {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    ID3D11Texture2D* texture = nullptr;
    ID3D11RenderTargetView* rtv = nullptr;
    
    // 创建D3D11设备
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &device, nullptr, &context);
        
    if (SUCCEEDED(hr)) {
        // 创建离屏纹理
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = 100;
        desc.Height = 100;
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
            VideoPlayer player;
            bool result = player.initialize(device, context, rtv, nullptr);
            REQUIRE(result == true);
            
            // 渲染几帧
            for (int i = 0; i < 5; ++i) {
                player.onTimer();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            
            std::cout << "离屏渲染测试成功" << std::endl;
        }
        
        // 清理
        if (rtv) rtv->Release();
        if (texture) texture->Release();
        if (context) context->Release();
        if (device) device->Release();
    } else {
        std::cout << "跳过D3D11测试" << std::endl;
    }
}