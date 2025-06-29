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

TEST_CASE("Compare convert_color with FFmpeg swscale") {
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 200;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    // 验证解码是否成功
    REQUIRE(hw_frame != nullptr);

    // Print decoded frame properties for diagnostics
    std::cout << "\n--- Decoded Frame Diagnostics ---" << std::endl;
    std::cout << "Format: " << av_get_pix_fmt_name(static_cast<AVPixelFormat>(hw_frame->format)) << std::endl;
    std::cout << "Color Range: " << av_color_range_name(hw_frame->color_range) << " (" << hw_frame->color_range << ")" << std::endl;
    std::cout << "Color Space: " << av_color_space_name(hw_frame->colorspace) << " (" << hw_frame->colorspace << ")" << std::endl;
    std::cout << "Color Primaries: " << av_color_primaries_name(hw_frame->color_primaries) << " (" << hw_frame->color_primaries << ")" << std::endl;
    std::cout << "Color TRC: " << av_color_transfer_name(hw_frame->color_trc) << " (" << hw_frame->color_trc << ")" << std::endl;
    std::cout << "---------------------------------\n" << std::endl;

    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // 测试 D3D11 颜色转换
    ID3D11Texture2D* d3d11_rgba_texture = convert_color(&hw_ctx, hw_frame);
    REQUIRE(d3d11_rgba_texture != nullptr);
    
    // 分配 CUDA 内存用于输出
    // 输出格式: float32, NCHW, shape=(1, 4, height, width)
    int width = hw_frame->width;
    int height = hw_frame->height;
    size_t cuda_output_size = 1 * 4 * height * width * sizeof(float); // NCHW format
    
    void* cuda_output_ptr = nullptr;
    cudaError_t cuda_err = cudaMalloc(&cuda_output_ptr, cuda_output_size);
    REQUIRE(cuda_err == cudaSuccess);
    std::cout << "Allocated CUDA memory: " << cuda_output_size << " bytes" << std::endl;
    
    // 调用 to_cuda_input 转换函数
    bool conversion_success = to_cuda_input(&hw_ctx, d3d11_rgba_texture, cuda_output_ptr);
    REQUIRE(conversion_success);
    std::cout << "to_cuda_input conversion successful!" << std::endl;
    
    // 验证转换结果：将 CUDA 数据复制回 CPU 进行检查
    std::vector<float> cuda_result_data(1 * 4 * height * width);
    cuda_err = cudaMemcpy(cuda_result_data.data(), cuda_output_ptr, cuda_output_size, cudaMemcpyDeviceToHost);
    REQUIRE(cuda_err == cudaSuccess);
    
}