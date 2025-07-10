#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include "../src/video_player.h"
#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <filesystem>

using Microsoft::WRL::ComPtr;
namespace fs = std::filesystem;

TEST_CASE("VideoPlayer rendering", "[video]") {
    VideoPlayer player;
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    REQUIRE(player.open(test_file.string()));
    REQUIRE(player.onTimer());
    
    ID3D11Texture2D* texture = player.getRenderTexture();
    REQUIRE(texture != nullptr);
    
    ID3D11Device* device = player.getDevice();
    ID3D11DeviceContext* context = player.getContext();
    
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    REQUIRE(desc.Width > 0);
    REQUIRE(desc.Height > 0);
    
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.MiscFlags = 0;
    
    ComPtr<ID3D11Texture2D> staging;
    HRESULT hr = device->CreateTexture2D(&desc, nullptr, staging.GetAddressOf());
    REQUIRE(SUCCEEDED(hr));
    
    context->CopyResource(staging.Get(), texture);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    REQUIRE(SUCCEEDED(hr));
    
    auto* pixels = static_cast<uint8_t*>(mapped.pData);
    bool hasNonBlackPixel = false;
    for (uint32_t y = 0; y < desc.Height; ++y) {
        auto* row = pixels + y * mapped.RowPitch;
        for (uint32_t x = 0; x < desc.Width; ++x) {
            uint8_t b = row[x * 4 + 0];
            uint8_t g = row[x * 4 + 1];
            uint8_t r = row[x * 4 + 2];
            uint8_t a = row[x * 4 + 3];
            if (r != 0 || g != 0 || b != 0) {
                hasNonBlackPixel = true;
                break;
            }
        }
        if (hasNonBlackPixel) break;
    }
    
    context->Unmap(staging.Get(), 0);
    
    REQUIRE(hasNonBlackPixel);
}
