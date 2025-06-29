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

// Helper function to save RGBA data as a BMP file for debugging
void save_rgba_as_bmp(const char* filename, const std::vector<uint8_t>& rgba_data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << " for writing." << std::endl;
        return;
    }

    // BMP format requires BGRA, so we need to swizzle channels from RGBA
    std::vector<uint8_t> bgra_data(rgba_data.size());
    for (size_t i = 0; i < rgba_data.size(); i += 4) {
        bgra_data[i + 0] = rgba_data[i + 2]; // B
        bgra_data[i + 1] = rgba_data[i + 1]; // G
        bgra_data[i + 2] = rgba_data[i + 0]; // R
        bgra_data[i + 3] = rgba_data[i + 3]; // A
    }

    // BMP File Header (14 bytes)
#pragma pack(push, 1)
    struct BmpFileHeader {
        uint16_t bfType{0x4D42}; // 'BM'
        uint32_t bfSize{0};
        uint16_t bfReserved1{0};
        uint16_t bfReserved2{0};
        uint32_t bfOffBits{54};
    };

    // DIB Header (BITMAPINFOHEADER) (40 bytes)
    struct BmpInfoHeader {
        uint32_t biSize{40};
        int32_t  biWidth{0};
        int32_t  biHeight{0};
        uint16_t biPlanes{1};
        uint16_t biBitCount{32};
        uint32_t biCompression{0}; // BI_RGB
        uint32_t biSizeImage{0};
        int32_t  biXPelsPerMeter{0};
        int32_t  biYPelsPerMeter{0};
        uint32_t biClrUsed{0};
        uint32_t biClrImportant{0};
    };
#pragma pack(pop)

    BmpFileHeader file_header;
    file_header.bfSize = sizeof(BmpFileHeader) + sizeof(BmpInfoHeader) + bgra_data.size();
    
    BmpInfoHeader info_header;
    info_header.biWidth = width;
    info_header.biHeight = -height; // Negative for top-down bitmap
    info_header.biSizeImage = (uint32_t)bgra_data.size();
    
    file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
    file.write(reinterpret_cast<const char*>(bgra_data.data()), bgra_data.size());

    file.close();
    std::cout << "Saved BMP image to " << filename << std::endl;
}

// Helper function to convert NCHW float tensor to RGBA uint8 vector
std::vector<uint8_t> nchw_to_rgba(const float* nchw_data, int C, int H, int W) {
    if (C != 4) {
        std::cerr << "nchw_to_rgba requires 4 channels (RGBA)." << std::endl;
        return {};
    }
    std::vector<uint8_t> rgba_data(H * W * 4);
    const float* R_plane = nchw_data;
    const float* G_plane = nchw_data + H * W;
    const float* B_plane = nchw_data + 2 * H * W;
    const float* A_plane = nchw_data + 3 * H * W;

    for (int i = 0; i < H * W; ++i) {
        rgba_data[i * 4 + 0] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, R_plane[i] * 255.0f))); // R
        rgba_data[i * 4 + 1] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, G_plane[i] * 255.0f))); // G
        rgba_data[i * 4 + 2] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, B_plane[i] * 255.0f))); // B
        rgba_data[i * 4 + 3] = static_cast<uint8_t>(std::max(0.f, std::min(255.f, A_plane[i] * 255.0f))); // A
    }
    return rgba_data;
}

TEST_CASE("Test inferIW3") {
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
    
    // 调用 to_cuda_input 转换函数，直接获取映射的CUDA指针
    void* cuda_input_ptr = to_cuda_input(&hw_ctx, d3d11_rgba_texture);
    REQUIRE(cuda_input_ptr != nullptr);
    std::cout << "to_cuda_input conversion successful! CUDA pointer: " << cuda_input_ptr << std::endl;

    // Run inference using the mapped CUDA pointer directly
    // inferIW3 will allocate the output buffer.
    int width = hw_frame->width;
    int height = hw_frame->height;
    size_t cuda_infer_output_size = 0;
    void* cuda_infer_output_ptr = inferIW3(cuda_input_ptr, height, width, &cuda_infer_output_size);
    REQUIRE(cuda_infer_output_ptr != nullptr);
    
    // 取消CUDA资源映射
    unmap_cuda_input();
    std::cout << "Unmapped CUDA input resource." << std::endl;
    
    // Copy output from CUDA device to host
    std::vector<float> host_output_buffer(cuda_infer_output_size / sizeof(float));
    cudaError_t cuda_err = cudaMemcpy(host_output_buffer.data(), cuda_infer_output_ptr, cuda_infer_output_size, cudaMemcpyDeviceToHost);
    REQUIRE(cuda_err == cudaSuccess);
    std::cout << "Copied inference output from device to host." << std::endl;
    
    // Convert NCHW float output to RGBA8 for saving
    // The output is half side-by-side, so the width for the image is `width`.
    // The data itself contains L/R images, so total elements correspond to `width` but arranged in `width/2` for each eye in sbs format.
    // The nchw_to_rgba expects channel count and H, W of the *final image*.
    // The model output is (1, 4, H, W/2) for each eye, concatenated to (1, 4, H, W).
    // Oh, my `infer` has a bug, it should be `width/2` for half sbs. Let me check the onnx.
    // The onnx output is `half_sbs` with shape (1, 4, H, W). It is not `W/2`. My previous assumption was wrong.
    // The `nchw_to_rgba` should be correct.
    std::vector<uint8_t> output_image_data = nchw_to_rgba(host_output_buffer.data(), 4, height, width);
    REQUIRE(!output_image_data.empty());
    
    // Save output image
    save_rgba_as_bmp("test_infer_iw3_output.bmp", output_image_data, width, height);
    
    // Cleanup
    cudaFree(cuda_infer_output_ptr);
    d3d11_rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
    cleanup_cuda_input_resources();
    std::cout << "Cleaned up all resources." << std::endl;
}