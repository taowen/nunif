#include <catch2/catch_test_macros.hpp>
#include "main.h"
// 添加FFmpeg软件缩放库
extern "C" {
#include <libswscale/swscale.h>
}
#include <vector>
#include <cstring>
#include <fstream>
#include <string>
#include <memory>
#include <cmath>
#include <iomanip>
#include <libavutil/pixdesc.h>
#include <cuda_runtime_api.h>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif


TEST_CASE("Test from_cuda_output") {
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 200;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    // 验证解码是否成功
    REQUIRE(hw_frame != nullptr);
    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // 测试 D3D11 颜色转换
    ID3D11Texture2D* d3d11_rgba_texture = convert_color(&hw_ctx, hw_frame);
    REQUIRE(d3d11_rgba_texture != nullptr);
    
    // 使用新的ToCudaInputContext类
    ToCudaInputContext cuda_input_context;
    void* cuda_input_ptr = cuda_input_context.to_cuda_input(&hw_ctx, d3d11_rgba_texture);
    REQUIRE(cuda_input_ptr != nullptr);
    std::cout << "to_cuda_input conversion successful! CUDA pointer: " << cuda_input_ptr << std::endl;

    // Run inference using the mapped CUDA pointer directly
    // inferIW3 will allocate the output buffer.
    int width = hw_frame->width;
    int height = hw_frame->height;
    size_t cuda_infer_output_size = 0;
    void* cuda_infer_output_ptr = inferIW3(cuda_input_ptr, height, width, &cuda_infer_output_size);
    REQUIRE(cuda_infer_output_ptr != nullptr);
    std::cout << "inferIW3 successful! Output size: " << cuda_infer_output_size << " bytes" << std::endl;
    
    // 测试 from_cuda_output 功能
    FromCudaOutputContext cuda_output_context;
    
    // 修正：inferIW3 的输出宽度与输入宽度相同（half side-by-side 格式）
    // 左半部分是左眼视图，右半部分是右眼视图
    int output_width = width;  // 不是 width/2
    ID3D11Texture2D* output_texture = cuda_output_context.from_cuda_output(
        &hw_ctx, cuda_infer_output_ptr, height, output_width);
    
    REQUIRE(output_texture != nullptr);
    std::cout << "from_cuda_output conversion successful!" << std::endl;
    
    // 验证输出纹理的属性
    D3D11_TEXTURE2D_DESC texture_desc;
    output_texture->GetDesc(&texture_desc);
    
    REQUIRE(texture_desc.Width == output_width);
    REQUIRE(texture_desc.Height == height);
    REQUIRE(texture_desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM);
    
    std::cout << "Output texture verified - Width: " << texture_desc.Width 
              << ", Height: " << texture_desc.Height 
              << ", Format: DXGI_FORMAT_R8G8B8A8_UNORM (Half SBS format)" << std::endl;
    
    // 取消CUDA输入映射
    cuda_input_context.unmap_cuda_input();
    
    // Cleanup
    output_texture->Release();
    cudaFree(cuda_infer_output_ptr);
    d3d11_rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
    
    std::cout << "Test completed successfully!" << std::endl;
}