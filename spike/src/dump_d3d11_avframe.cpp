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

void dump_d3d11_avframe(const FFMepgContext* ctx, AVFrame* frame) {
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return;
    }
    
    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return;
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
        return;
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
        return;
    }

    // Extract YUV data from NV12 format
    uint8_t* mapped_data = static_cast<uint8_t*>(mapped_resource.pData);
    int width = frame->width;
    int height = frame->height;
    int row_pitch = mapped_resource.RowPitch;
    
    // Calculate plane sizes
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 2; // NV12 UV plane is half height, but interleaved
    
    // Allocate buffers for separated planes
    std::vector<uint8_t> y_plane(y_plane_size);
    std::vector<uint8_t> u_plane(width * height / 4);
    std::vector<uint8_t> v_plane(width * height / 4);
    
    // Copy Y plane
    for (int y = 0; y < height; y++) {
        memcpy(y_plane.data() + y * width, 
               mapped_data + y * row_pitch, 
               width);
    }
    
    // Copy and separate UV plane (NV12 format has interleaved UV)
    uint8_t* uv_start = mapped_data + height * row_pitch;
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            int src_idx = y * row_pitch + x * 2;
            int dst_idx = y * (width / 2) + x;
            u_plane[dst_idx] = uv_start[src_idx];     // U component
            v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
        }
    }
    
    // Create filename with frame dimensions and YUV format info
    std::string base_filename = "frame_" + std::to_string(width) + "x" + std::to_string(height) + "_8";
    
    // Save Y plane
    std::ofstream y_file(base_filename + "_Y.raw", std::ios::binary);
    if (y_file.is_open()) {
        y_file.write(reinterpret_cast<const char*>(y_plane.data()), y_plane_size);
        y_file.close();
        std::cout << "Saved Y plane to " << base_filename << "_Y.raw" << std::endl;
    }
    
    // Save U plane
    std::ofstream u_file(base_filename + "_U.raw", std::ios::binary);
    if (u_file.is_open()) {
        u_file.write(reinterpret_cast<const char*>(u_plane.data()), u_plane.size());
        u_file.close();
        std::cout << "Saved U plane to " << base_filename << "_U.raw" << std::endl;
    }
    
    // Save V plane
    std::ofstream v_file(base_filename + "_V.raw", std::ios::binary);
    if (v_file.is_open()) {
        v_file.write(reinterpret_cast<const char*>(v_plane.data()), v_plane.size());
        v_file.close();
        std::cout << "Saved V plane to " << base_filename << "_V.raw" << std::endl;
    }
    
    // Save interleaved YUV420 format with proper naming
    std::ofstream yuv_file(base_filename + "_yuv420p.yuv", std::ios::binary);
    if (yuv_file.is_open()) {
        yuv_file.write(reinterpret_cast<const char*>(y_plane.data()), y_plane_size);
        yuv_file.write(reinterpret_cast<const char*>(u_plane.data()), u_plane.size());
        yuv_file.write(reinterpret_cast<const char*>(v_plane.data()), v_plane.size());
        yuv_file.close();
        std::cout << "Saved YUV420P to " << base_filename << "_yuv420p.yuv" << std::endl;
    }
    
    // Unmap and release resources
    ctx->d3d_context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    std::cout << "Successfully dumped YUV data from D3D11 AVFrame (" 
              << width << "x" << height << ")" << std::endl;
}