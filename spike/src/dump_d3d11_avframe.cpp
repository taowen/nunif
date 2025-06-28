#include "main.h"
#include <fstream>
#include <vector>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

/*
(venv) C:\games\nunif>spike\build\Debug\spike.exe
Randomness seeded to: 3908875051
D3D11 hardware acceleration enabled
Frame 200 decoded successfully
Hardware frame returned (GPU memory)
[dump_d3d11_avframe] Input Texture Desc:
  Width: 3840, Height: 1648
  MipLevels: 1, ArraySize: 20
  Format: 103 (Expected DXGI_FORMAT_NV12=103)
  Texture Index: 17
[dump_d3d11_avframe] Memory Layout Analysis:
  Frame dimensions: 3840x1634
  Row pitch: 3840 bytes
  Width vs Row pitch difference: 0 bytes (padding)
  Y plane expected size: 6274560 bytes
  Y plane actual size with padding: 6274560 bytes
  UV plane expected size: 3137280 bytes
  Total mapped data size: 9411840 bytes
  UV plane offset calculation: 1634 * 3840 = 6274560
[dump_d3d11_avframe] Y Plane Analysis:
  First row Y values (first 16 pixels): 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13
  Second row Y values (first 16 pixels): 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21 21
[dump_d3d11_avframe] UV Plane Analysis:
  UV start address offset: 6274560 bytes
  UV plane dimensions: 1920x817
  UV interleaved data (first 16 UV pairs): 125 149 125 149 125 149 125 149 125 149 125 149 125 149 125 149 125 148 125 146 125 146 125 146 125 146 125 146 125 146 125 146
  UV first row analysis:
    Raw UV data (first 16 bytes): 125 149 125 149 125 149 125 149 125 149 125 149 125 149 125 149

    Separated U values: 125 125 125 125 125 125 125 125
    Separated V values: 149 149 149 149 149 149 149 149
  UV second row analysis:
    Raw UV data (first 16 bytes): 125 153 125 153 125 153 125 153 125 153 125 153 125 153 125 154

  Memory boundary check:
    Expected total size: 9411840 bytes
    UV plane ends at offset: 9411840 bytes
[dump_d3d11_avframe] Extraction Results:
  Y plane first 16 values: 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13 13
  U plane first 8 values: 125 125 125 125 125 125 125 125
  V plane first 8 values: 149 149 149 149 149 149 149 149
Saved YUV420P to frame_3840x1634_8_yuv420p.yuv
Successfully extracted YUV data from D3D11 AVFrame (3840x1634)
[ffmpeg_d3d11_to_yuv] hw-transferred sw_frame format: nv12

--- YUV Data Details: dump_d3d11_avframe ---
Dimensions: 3840x1634
  Y plane size: 6274560. First 8 bytes: d d d d d d d d
  U plane size: 1568640. First 8 bytes: 7d 7d 7d 7d 7d 7d 7d 7d
  V plane size: 1568640. First 8 bytes: 95 95 95 95 95 95 95 95

--- YUV Data Details: ffmpeg_d3d11_to_yuv ---
Dimensions: 3840x1634
  Y plane size: 6274560. First 8 bytes: d d d d d d d d
  U plane size: 1568640. First 8 bytes: 7d 7d 7d 7d 7d 7d 7d 7d
  V plane size: 1568640. First 8 bytes: 95 95 95 95 95 95 95 95

=== YUVµò░µì«µ»öΦ╛âτ╗ôµ₧£ ===
σêåΦ╛¿τÄç: 3840x1634
Yσ╣│Θ¥ó - MSE: 0, PSNR: 100 dB, µ£Çσñºσ╖«σ╝é: 0
Uσ╣│Θ¥ó - MSE: 0, PSNR: 100 dB, µ£Çσñºσ╖«σ╝é: 0
Vσ╣│Θ¥ó - MSE: 0, PSNR: 100 dB, µ£Çσñºσ╖«σ╝é: 0
Γ£ô dump_d3d11_avframeτÜäτ╗ôµ₧£Σ╕ÄFFmpegµáçσçåµû╣µ│òσƒ║µ£¼Σ╕ÇΦç┤!
===============================================================================
All tests passed (14 assertions in 1 test case)
*/
YUVData dump_d3d11_avframe(const FFMepgContext* ctx, AVFrame* frame, bool save_to_file) {
    YUVData result;
    
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return result;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return result;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return result;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return result;
    }
    
    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

    std::cout << "[dump_d3d11_avframe] Input Texture Desc:" << std::endl;
    std::cout << "  Width: " << input_desc.Width << ", Height: " << input_desc.Height << std::endl;
    std::cout << "  MipLevels: " << input_desc.MipLevels << ", ArraySize: " << input_desc.ArraySize << std::endl;
    std::cout << "  Format: " << input_desc.Format << " (Expected DXGI_FORMAT_NV12=" << DXGI_FORMAT_NV12 << ")" << std::endl;
    std::cout << "  Texture Index: " << texture_index << std::endl;
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return result;
    }

    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = {};
    staging_desc.Width = input_desc.Width;
    staging_desc.Height = input_desc.Height;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.Format = input_desc.Format;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.BindFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return result;
    }

    // Copy from D3D11 texture to staging texture
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
    ctx->d3d_context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    
    // Map the staging texture to access CPU memory
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    hr = ctx->d3d_context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped_resource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: 0x" << std::hex << hr << std::endl;
        staging_texture->Release();
        return result;
    }

    // Extract YUV data from NV12 format
    uint8_t* mapped_data = static_cast<uint8_t*>(mapped_resource.pData);
    int width = frame->width;
    int height = frame->height;
    int row_pitch = mapped_resource.RowPitch;
    
    // 添加详细的内存布局调试信息
    std::cout << "[dump_d3d11_avframe] Memory Layout Analysis:" << std::endl;
    std::cout << "  Frame dimensions: " << width << "x" << height << std::endl;
    std::cout << "  Row pitch: " << row_pitch << " bytes" << std::endl;
    std::cout << "  Width vs Row pitch difference: " << (row_pitch - width) << " bytes (padding)" << std::endl;
    std::cout << "  Y plane expected size: " << (width * height) << " bytes" << std::endl;
    std::cout << "  Y plane actual size with padding: " << (height * row_pitch) << " bytes" << std::endl;
    std::cout << "  UV plane expected size: " << (width * height / 2) << " bytes" << std::endl;
    std::cout << "  Total mapped data size: " << (height * row_pitch + (height / 2) * row_pitch) << " bytes" << std::endl;
    
    // 计算UV平面的起始偏移
    size_t uv_offset = height * row_pitch;
    std::cout << "  UV plane offset calculation: " << height << " * " << row_pitch << " = " << uv_offset << std::endl;
    
    // Calculate plane sizes
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 4; // U and V planes are each 1/4 the size
    
    // Allocate buffers for separated planes
    result.y_plane.resize(y_plane_size);
    result.u_plane.resize(uv_plane_size);
    result.v_plane.resize(uv_plane_size);
    result.width = width;
    result.height = height;
    
    // 添加Y平面数据样本日志
    std::cout << "[dump_d3d11_avframe] Y Plane Analysis:" << std::endl;
    std::cout << "  First row Y values (first 16 pixels): ";
    for (int i = 0; i < std::min(16, width); i++) {
        std::cout << (int)mapped_data[i] << " ";
    }
    std::cout << std::endl;
    
    if (height > 1) {
        std::cout << "  Second row Y values (first 16 pixels): ";
        for (int i = 0; i < std::min(16, width); i++) {
            std::cout << (int)mapped_data[row_pitch + i] << " ";
        }
        std::cout << std::endl;
    }
    
    // Copy Y plane
    for (int y = 0; y < height; y++) {
        memcpy(result.y_plane.data() + y * width, 
               mapped_data + y * row_pitch, 
               width);
    }
    
    // Copy and separate UV plane (NV12 format has interleaved UV)
    uint8_t* uv_start = mapped_data + input_desc.Height * row_pitch;
    
    // 添加UV平面详细分析
    std::cout << "[dump_d3d11_avframe] UV Plane Analysis:" << std::endl;
    std::cout << "  UV start address offset: " << uv_offset << " bytes" << std::endl;
    std::cout << "  UV plane dimensions: " << (width/2) << "x" << (height/2) << std::endl;
    std::cout << "  UV interleaved data (first 16 UV pairs): ";
    for (int i = 0; i < std::min(32, width); i++) {
        std::cout << (int)uv_start[i] << " ";
    }
    std::cout << std::endl;
    
    // 分析UV数据的行布局
    std::cout << "  UV first row analysis:" << std::endl;
    std::cout << "    Raw UV data (first " << std::min(16, width) << " bytes): ";
    for (int i = 0; i < std::min(16, width); i++) {
        std::cout << (int)uv_start[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "    Separated U values: ";
    for (int i = 0; i < std::min(8, width/2); i++) {
        std::cout << (int)uv_start[i * 2] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "    Separated V values: ";
    for (int i = 0; i < std::min(8, width/2); i++) {
        std::cout << (int)uv_start[i * 2 + 1] << " ";
    }
    std::cout << std::endl;
    
    if (height > 2) {
        std::cout << "  UV second row analysis:" << std::endl;
        std::cout << "    Raw UV data (first " << std::min(16, width) << " bytes): ";
        for (int i = 0; i < std::min(16, width); i++) {
            std::cout << (int)uv_start[row_pitch + i] << " ";
        }
        std::cout << std::endl;
    }
    
    // 验证内存边界
    size_t expected_total_size = height * row_pitch + (height / 2) * row_pitch;
    std::cout << "  Memory boundary check:" << std::endl;
    std::cout << "    Expected total size: " << expected_total_size << " bytes" << std::endl;
    std::cout << "    UV plane ends at offset: " << (uv_offset + (height / 2) * row_pitch) << " bytes" << std::endl;
    
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            int src_idx = y * row_pitch + x * 2;
            int dst_idx = y * (width / 2) + x;
            result.u_plane[dst_idx] = uv_start[src_idx];     // U component
            result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
        }
    }
    
    result.valid = true;
    
    // 添加提取结果验证
    std::cout << "[dump_d3d11_avframe] Extraction Results:" << std::endl;
    std::cout << "  Y plane first 16 values: ";
    for (int i = 0; i < std::min(16, (int)result.y_plane.size()); i++) {
        std::cout << (int)result.y_plane[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  U plane first 8 values: ";
    for (int i = 0; i < std::min(8, (int)result.u_plane.size()); i++) {
        std::cout << (int)result.u_plane[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "  V plane first 8 values: ";
    for (int i = 0; i < std::min(8, (int)result.v_plane.size()); i++) {
        std::cout << (int)result.v_plane[i] << " ";
    }
    std::cout << std::endl;
    
    // Optionally save to files
    if (save_to_file) {
        std::string base_filename = "frame_" + std::to_string(width) + "x" + std::to_string(height) + "_8";
        
        // Save interleaved YUV420 format
        std::ofstream yuv_file(base_filename + "_yuv420p.yuv", std::ios::binary);
        if (yuv_file.is_open()) {
            yuv_file.write(reinterpret_cast<const char*>(result.y_plane.data()), result.y_plane.size());
            yuv_file.write(reinterpret_cast<const char*>(result.u_plane.data()), result.u_plane.size());
            yuv_file.write(reinterpret_cast<const char*>(result.v_plane.data()), result.v_plane.size());
            yuv_file.close();
            std::cout << "Saved YUV420P to " << base_filename << "_yuv420p.yuv" << std::endl;
        }
    }
    
    // Unmap and release resources
    ctx->d3d_context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    std::cout << "Successfully extracted YUV data from D3D11 AVFrame (" 
              << width << "x" << height << ")" << std::endl;
    
    return result;
}