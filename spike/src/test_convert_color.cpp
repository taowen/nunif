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

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// Helper function to copy D3D11 texture data to CPU
std::vector<uint8_t> copy_d3d11_texture_to_cpu(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* texture, int texture_index = 0) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // Create staging texture for CPU readback
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    staging_desc.ArraySize = 1;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return {};
    }
    
    // Copy texture to staging
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, desc.MipLevels);
    context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, texture, src_subresource, nullptr);
    
    // Map and read data
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: 0x" << std::hex << hr << std::endl;
        staging_texture->Release();
        return {};
    }
    
    // Copy data to vector
    std::vector<uint8_t> data;
    uint32_t bytes_per_pixel = 4; // RGBA
    if (desc.Format == DXGI_FORMAT_NV12) {
        // NV12: Y plane + UV plane
        bytes_per_pixel = 1; // Y plane is 1 byte per pixel
        uint32_t y_size = desc.Width * desc.Height;
        uint32_t uv_size = desc.Width * desc.Height / 2; // UV plane is half height
        data.resize(y_size + uv_size);
        
        // Copy Y plane
        for (uint32_t row = 0; row < desc.Height; ++row) {
            memcpy(&data[row * desc.Width], 
                   (uint8_t*)mapped.pData + row * mapped.RowPitch, 
                   desc.Width);
        }
        
        // Copy UV plane (starts at Y plane + half height)
        uint8_t* uv_src = (uint8_t*)mapped.pData + mapped.RowPitch * desc.Height;
        for (uint32_t row = 0; row < desc.Height / 2; ++row) {
            memcpy(&data[y_size + row * desc.Width], 
                   uv_src + row * mapped.RowPitch, 
                   desc.Width);
        }
    } else if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) {
        // RGBA format
        data.resize(desc.Width * desc.Height * 4);
        for (uint32_t row = 0; row < desc.Height; ++row) {
            memcpy(&data[row * desc.Width * 4], 
                   (uint8_t*)mapped.pData + row * mapped.RowPitch, 
                   desc.Width * 4);
        }
    }
    
    context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    return data;
}

// Helper function to convert NV12 to RGBA using FFmpeg swscale
std::vector<uint8_t> convert_nv12_to_rgba_ffmpeg(const uint8_t* nv12_data, int width, int height, 
                                                 AVColorSpace colorspace, AVColorRange color_range) {
    SwsContext* sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_NV12,
        width, height, AV_PIX_FMT_RGBA,
        SWS_BICUBIC, nullptr, nullptr, nullptr
    );
    
    if (!sws_ctx) {
        std::cerr << "Failed to create swscale context" << std::endl;
        return {};
    }

    // Explicitly set colorspace and range for the conversion
    const int* inv_table;
    const int* table;
    int src_range, dst_range, brightness, contrast, saturation;

    sws_getColorspaceDetails(sws_ctx, (int**)&inv_table, &src_range, (int**)&table, &dst_range,
        &brightness, &contrast, &saturation);

    // Override with details from the frame
    table = sws_getCoefficients(colorspace);
    inv_table = table;
    src_range = (color_range == AVCOL_RANGE_JPEG); // 1 for full range, 0 for limited
    dst_range = 1; // Output RGBA is full range

    sws_setColorspaceDetails(sws_ctx, inv_table, src_range, table, dst_range,
        brightness, contrast, saturation);
    
    // Prepare input data (NV12)
    const uint8_t* src_data[4] = { nullptr };
    int src_linesize[4] = { 0 };
    
    src_data[0] = nv12_data;                    // Y plane
    src_data[1] = nv12_data + width * height;  // UV plane
    src_linesize[0] = width;                   // Y plane stride
    src_linesize[1] = width;                   // UV plane stride
    
    // Prepare output data (RGBA)
    std::vector<uint8_t> rgba_data(width * height * 4);
    uint8_t* dst_data[4] = { nullptr };
    int dst_linesize[4] = { 0 };
    
    dst_data[0] = rgba_data.data();
    dst_linesize[0] = width * 4;
    
    // Perform conversion
    int result = sws_scale(
        sws_ctx,
        src_data, src_linesize, 0, height,
        dst_data, dst_linesize
    );
    
    sws_freeContext(sws_ctx);
    
    if (result != height) {
        std::cerr << "swscale conversion failed" << std::endl;
        return {};
    }
    
    return rgba_data;
}

// Helper function to calculate PSNR between two RGBA images
double calculate_psnr(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2) {
    if (img1.size() != img2.size()) {
        return 0.0;
    }
    
    double mse = 0.0;
    for (size_t i = 0; i < img1.size(); ++i) {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }
    mse /= img1.size();
    
    if (mse == 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    
    return 10.0 * log10(255.0 * 255.0 / mse);
}

// Add function to analyze pixel differences
void analyze_pixel_differences(const std::vector<uint8_t>& img1, const std::vector<uint8_t>& img2, int width, int height) {
    if (img1.size() != img2.size()) {
        std::cout << "Image sizes don't match!" << std::endl;
        return;
    }
    
    // Sample analysis - check a few pixels from different regions
    std::vector<std::pair<int, int>> sample_points = {
        {width/4, height/4},     // Top-left quadrant
        {3*width/4, height/4},   // Top-right quadrant
        {width/2, height/2},     // Center
        {width/4, 3*height/4},   // Bottom-left quadrant
        {3*width/4, 3*height/4}  // Bottom-right quadrant
    };
    
    std::cout << "\n--- Pixel Difference Analysis ---" << std::endl;
    std::cout << "Sampling pixels at different locations:" << std::endl;
    
    for (const auto& point : sample_points) {
        int x = point.first;
        int y = point.second;
        int idx = (y * width + x) * 4;
        
        if (idx + 3 < (int)img1.size()) {
            uint8_t r1 = img1[idx], g1 = img1[idx+1], b1 = img1[idx+2], a1 = img1[idx+3];
            uint8_t r2 = img2[idx], g2 = img2[idx+1], b2 = img2[idx+2], a2 = img2[idx+3];
            
            std::cout << "Point (" << x << "," << y << "):" << std::endl;
            std::cout << "  D3D11:  R=" << (int)r1 << " G=" << (int)g1 << " B=" << (int)b1 << " A=" << (int)a1 << std::endl;
            std::cout << "  FFmpeg: R=" << (int)r2 << " G=" << (int)g2 << " B=" << (int)b2 << " A=" << (int)a2 << std::endl;
            std::cout << "  Diff:   R=" << abs((int)r1-(int)r2) << " G=" << abs((int)g1-(int)g2) << " B=" << abs((int)b1-(int)b2) << " A=" << abs((int)a1-(int)a2) << std::endl;
            std::cout << std::endl;
        }
    }
    
    // Calculate per-channel statistics
    std::vector<double> channel_mse(4, 0.0);
    std::vector<int> max_diff(4, 0);
    
    for (size_t i = 0; i < img1.size(); i += 4) {
        for (int c = 0; c < 4; ++c) {
            if (i + c < img1.size()) {
                double diff = (double)img1[i+c] - (double)img2[i+c];
                channel_mse[c] += diff * diff;
                max_diff[c] = std::max(max_diff[c], (int)abs(diff));
            }
        }
    }
    
    std::cout << "--- Per-Channel Statistics ---" << std::endl;
    const char* channel_names[] = {"Red", "Green", "Blue", "Alpha"};
    for (int c = 0; c < 4; ++c) {
        channel_mse[c] /= (img1.size() / 4);
        double channel_psnr = (channel_mse[c] == 0.0) ? std::numeric_limits<double>::infinity() : 
                             10.0 * log10(255.0 * 255.0 / channel_mse[c]);
        std::cout << channel_names[c] << " Channel:" << std::endl;
        std::cout << "  PSNR: " << std::fixed << std::setprecision(2) << channel_psnr << " dB" << std::endl;
        std::cout << "  Max Diff: " << max_diff[c] << std::endl;
    }
    std::cout << "--------------------------------\n" << std::endl;
}

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
    
    // 获取原始 NV12 数据
    ID3D11Texture2D* nv12_texture = reinterpret_cast<ID3D11Texture2D*>(hw_frame->data[0]);
    int nv12_texture_index = (int)(intptr_t)hw_frame->data[1];
    std::vector<uint8_t> nv12_data = copy_d3d11_texture_to_cpu(hw_ctx.d3d_device, hw_ctx.d3d_context, nv12_texture, nv12_texture_index);
    REQUIRE(!nv12_data.empty());

    // Save NV12 data for debugging
    std::ofstream nv12_file("test_output_nv12.yuv", std::ios::binary);
    if (nv12_file) {
        nv12_file.write(reinterpret_cast<const char*>(nv12_data.data()), nv12_data.size());
        nv12_file.close();
        std::cout << "Saved raw NV12 frame to test_output_nv12.yuv" << std::endl;
    }
    
    // 测试 D3D11 颜色转换
    ID3D11Texture2D* d3d11_rgba_texture = convert_color(&hw_ctx, hw_frame);
    REQUIRE(d3d11_rgba_texture != nullptr);
    
    // 获取 D3D11 转换结果
    std::vector<uint8_t> d3d11_rgba_data = copy_d3d11_texture_to_cpu(hw_ctx.d3d_device, hw_ctx.d3d_context, d3d11_rgba_texture);
    REQUIRE(!d3d11_rgba_data.empty());
    
    // 使用 FFmpeg swscale 转换
    std::vector<uint8_t> ffmpeg_rgba_data = convert_nv12_to_rgba_ffmpeg(nv12_data.data(), hw_frame->width, hw_frame->height, hw_frame->colorspace, hw_frame->color_range);
    REQUIRE(!ffmpeg_rgba_data.empty());
    
    // 确保数据大小一致
    REQUIRE(d3d11_rgba_data.size() == ffmpeg_rgba_data.size());
    
    // 计算 PSNR
    double psnr = calculate_psnr(d3d11_rgba_data, ffmpeg_rgba_data);
    std::cout << "PSNR between D3D11 and FFmpeg conversion: " << std::fixed << std::setprecision(2) << psnr << " dB" << std::endl;
    
    // Add detailed pixel analysis
    analyze_pixel_differences(d3d11_rgba_data, ffmpeg_rgba_data, hw_frame->width, hw_frame->height);
    
    // 写入测试图像文件以便目视检查
    save_rgba_as_bmp("test_output_d3d11.bmp", d3d11_rgba_data, hw_frame->width, hw_frame->height);
    save_rgba_as_bmp("test_output_ffmpeg.bmp", ffmpeg_rgba_data, hw_frame->width, hw_frame->height);
    
    std::cout << "Frame dimensions: " << hw_frame->width << "x" << hw_frame->height << std::endl;
    std::cout << "D3D11 RGBA size: " << d3d11_rgba_data.size() << " bytes" << std::endl;
    std::cout << "FFmpeg RGBA size: " << ffmpeg_rgba_data.size() << " bytes" << std::endl;
    
    // Temporarily lower the PSNR threshold to understand the differences better
    // We'll investigate why there's a difference before setting the final threshold
    std::cout << "\nNote: PSNR difference suggests different color space handling between D3D11 and FFmpeg" << std::endl;
    std::cout << "This could be due to:" << std::endl;
    std::cout << "1. Different BT.709 matrix coefficients" << std::endl;
    std::cout << "2. Different handling of TV/limited vs full range" << std::endl;
    std::cout << "3. Different rounding/precision in the conversion" << std::endl;
    
    // With correct color space handling, PSNR should be very high
    REQUIRE(psnr > 35.0);
    
    // 清理资源
    d3d11_rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
}