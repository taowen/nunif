#include <catch2/catch_test_macros.hpp>
#include "main.h"

// 改进硬件与软件解码对比测试
TEST_CASE("Try convert color") {
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 650;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    // 验证解码是否成功
    REQUIRE(hw_frame != nullptr);
    
    // 验证帧格式（应该是D3D11硬件格式）
    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    
    // 验证D3D11设备是否可用
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // 测试颜色转换功能
    ID3D11Texture2D* rgba_texture = convert_color(&hw_ctx, hw_frame);
    
    // 验证颜色转换是否成功
    REQUIRE(rgba_texture != nullptr);
    
    // 验证转换后的纹理属性
    D3D11_TEXTURE2D_DESC desc;
    rgba_texture->GetDesc(&desc);
    
    // 验证纹理格式为RGBA
    REQUIRE(desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
    
    // 验证纹理尺寸与原始帧匹配
    REQUIRE(desc.Width == static_cast<UINT>(hw_frame->width));
    REQUIRE(desc.Height == static_cast<UINT>(hw_frame->height));
    
    // 验证纹理支持CUDA互操作
    REQUIRE((desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED) != 0);
    
    // 清理资源
    if (rgba_texture) {
        rgba_texture->Release();
    }
    if (hw_frame) {
        av_frame_free(&hw_frame);
    }
    cleanup_context(&hw_ctx);
}