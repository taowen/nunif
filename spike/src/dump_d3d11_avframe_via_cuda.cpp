#include "main.h"
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <dxgi1_2.h>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#include <iostream>
#include <vector>
#include <fstream>

// Windows/D3D headers
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_6.h>

// CUDA Driver API header
#include <cuda.h>

/*
struct YUVData {
    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> u_plane;
    std::vector<uint8_t> v_plane;
    int width;
    int height;
    bool valid;
    
    YUVData() : width(0), height(0), valid(false) {}
};
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
    std::cout << "  AVFrame width: " << frame->width << ", height: " << frame->height << std::endl;
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return result;
    }

    // First, create a CPU-accessible staging texture to get the actual row pitch
    D3D11_TEXTURE2D_DESC staging_desc{};
    staging_desc.Width = input_desc.Width;
    staging_desc.Height = input_desc.Height;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.Format = input_desc.Format;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    
    ID3D11Texture2D* temp_staging_texture = nullptr;
    HRESULT hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &temp_staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create temporary staging texture: 0x" << std::hex << hr << std::endl;
        return result;
    }

    // Copy to get the actual row pitch
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
    ctx->d3d_context->CopySubresourceRegion(temp_staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    
    D3D11_MAPPED_SUBRESOURCE temp_mapped;
    hr = ctx->d3d_context->Map(temp_staging_texture, 0, D3D11_MAP_READ, 0, &temp_mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map temporary staging texture: 0x" << std::hex << hr << std::endl;
        temp_staging_texture->Release();
        return result;
    }
    
    int actual_row_pitch = temp_mapped.RowPitch;
    ctx->d3d_context->Unmap(temp_staging_texture, 0);
    temp_staging_texture->Release();

    std::cout << "Actual row pitch from D3D11: " << actual_row_pitch << std::endl;

    // Now create texture for CUDA interop
    staging_desc.Usage = D3D11_USAGE_DEFAULT;
    staging_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    staging_desc.CPUAccessFlags = 0;
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
    
    ID3D11Texture2D* staging_texture = nullptr;
    hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return result;
    }

    // Copy from D3D11 texture to staging texture
    ctx->d3d_context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    ctx->d3d_context->Flush();
    
    return result;
}