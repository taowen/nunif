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
#include <stdexcept>

// Windows/D3D headers
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_6.h>

// CUDA Driver API header
#include <cuda.h>

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));   \
        }                                                        \
    } while (0)

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
YUVData dump_d3d11_avframe(const FFMepgContext* ctx, AVFrame* frame) {
    YUVData result;
    
    ID3D11Texture2D* staging_texture = nullptr;
    IDXGIResource1* dxgi_resource = nullptr;
    HANDLE nt_handle = nullptr;
    cudaExternalMemory_t external_memory = nullptr;
    cudaMipmappedArray_t mipmapped_array_y = nullptr;
    cudaMipmappedArray_t mipmapped_array_uv = nullptr;
    cudaArray_t array_y = nullptr;
    cudaArray_t array_uv = nullptr;

    try {
        if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
            throw std::runtime_error("Invalid context or frame provided.");
        }
        if (frame->format != AV_PIX_FMT_D3D11) {
            throw std::runtime_error("Expected D3D11 format");
        }
        if (frame->width <= 0 || frame->height <= 0) {
            throw std::runtime_error("Invalid frame dimensions.");
        }
        if (!frame->data[0]) {
            throw std::runtime_error("D3D11 texture pointer is null.");
        }
        
        ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
        int texture_index = (int)(intptr_t)frame->data[1];
        D3D11_TEXTURE2D_DESC input_desc;
        input_texture->GetDesc(&input_desc);

        if (input_desc.Format != DXGI_FORMAT_NV12) {
            throw std::runtime_error("Expected DXGI_FORMAT_NV12 format");
        }

        // Create a shareable texture for CUDA interop
        D3D11_TEXTURE2D_DESC staging_desc{};
        staging_desc.Width = input_desc.Width;
        staging_desc.Height = input_desc.Height;
        staging_desc.MipLevels = 1;
        staging_desc.ArraySize = 1;
        staging_desc.Format = input_desc.Format;
        staging_desc.SampleDesc.Count = 1;
        staging_desc.Usage = D3D11_USAGE_DEFAULT;
        staging_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        staging_desc.CPUAccessFlags = 0;
        staging_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
        
        HRESULT hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create staging texture for CUDA interop");
        }

        UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
        ctx->d3d_context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
        
        // Ensure D3D11 operations complete before CUDA access
        ctx->d3d_context->Flush();
        
        // Get NT Handle for sharing
        hr = staging_texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to query IDXGIResource1 interface");
        }
        hr = dxgi_resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &nt_handle);
        if (FAILED(hr)) {
            throw std::runtime_error("Failed to create shared handle");
        }

        // Import D3D11 resource into CUDA
        cudaExternalMemoryHandleDesc mem_handle_desc = {};
        mem_handle_desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
        mem_handle_desc.handle.win32.handle = nt_handle;
        mem_handle_desc.size = (size_t)input_desc.Width * input_desc.Height * 3 /2;
        mem_handle_desc.flags |= cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&external_memory, &mem_handle_desc));

        // Map NV12 planes to CUDA arrays
        // Y plane
        cudaExternalMemoryMipmappedArrayDesc mip_map_desc_y = {};
        cudaChannelFormatDesc channel_desc_y = {};
        channel_desc_y.x = 8;
        channel_desc_y.f = cudaChannelFormatKindUnsigned;
        mip_map_desc_y.formatDesc = channel_desc_y;
        mip_map_desc_y.extent = make_cudaExtent(input_desc.Width, input_desc.Height*3/2, 0);
        mip_map_desc_y.numLevels = 1;
        mip_map_desc_y.offset = 0;
        CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipmapped_array_y, external_memory, &mip_map_desc_y));
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&array_y, mipmapped_array_y, 0));
        
        // Copy yuv_array data
        std::vector<uint8_t> yuv_array;
        yuv_array.resize((size_t)input_desc.Width * input_desc.Height * 3 /2);
        CUDA_CHECK(cudaMemcpy2DFromArray(yuv_array.data(), input_desc.Width, array_y, 0, 0, input_desc.Width, input_desc.Height * 3 / 2, cudaMemcpyDeviceToHost));

        std::cout << "  yuv_array first 16 values: ";
        for (int i = 0; i < std::min(16, (int)yuv_array.size()); i++) {
            std::cout << (int)yuv_array[i] << " ";
        }
        std::cout << std::endl;

        // Extract YUV data from NV12 format (similar to CPU version)
        int width = input_desc.Width;
        int height = input_desc.Height;
        
        // Calculate plane sizes
        int y_plane_size = width * height;
        int uv_plane_size = width * height / 4; // U and V planes are each 1/4 the size
        
        // Allocate buffers for separated planes
        result.y_plane.resize(y_plane_size);
        result.u_plane.resize(uv_plane_size);
        result.v_plane.resize(uv_plane_size);
        
        std::cout << "[dump_d3d11_avframe_via_cuda] Memory Layout Analysis:" << std::endl;
        std::cout << "  Frame dimensions: " << width << "x" << height << std::endl;
        std::cout << "  Y plane size: " << y_plane_size << " bytes" << std::endl;
        std::cout << "  UV plane size (each): " << uv_plane_size << " bytes" << std::endl;
        std::cout << "  Total YUV array size: " << yuv_array.size() << " bytes" << std::endl;
        
        // Copy Y plane (first width*height bytes)
        memcpy(result.y_plane.data(), yuv_array.data(), y_plane_size);
        
        // Extract UV plane analysis
        uint8_t* uv_start = yuv_array.data() + y_plane_size;
        std::cout << "[dump_d3d11_avframe_via_cuda] UV Plane Analysis:" << std::endl;
        std::cout << "  UV plane dimensions: " << (width/2) << "x" << (height/2) << std::endl;
        std::cout << "  UV interleaved data (first 16 UV pairs): ";
        for (int i = 0; i < std::min(32, width); i++) {
            std::cout << (int)uv_start[i] << " ";
        }
        std::cout << std::endl;
        
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
        
        // Separate interleaved UV data (NV12 format has interleaved UV)
        for (int y = 0; y < height / 2; y++) {
            for (int x = 0; x < width / 2; x++) {
                int src_idx = y * width + x * 2; // UV plane stride is width
                int dst_idx = y * (width / 2) + x;
                result.u_plane[dst_idx] = uv_start[src_idx];     // U component
                result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
            }
        }
        
        // Validation output
        std::cout << "[dump_d3d11_avframe_via_cuda] Extraction Results:" << std::endl;
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

        result.width = frame->width;
        result.height = frame->height;
        result.valid = true;
        
        std::cout << "Successfully extracted YUV data from D3D11 AVFrame via CUDA (" 
                  << width << "x" << height << ")" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in dump_d3d11_avframe: " << e.what() << std::endl;
        result = YUVData(); // Reset on error
    }

    // Cleanup
    if (mipmapped_array_y) cudaFreeMipmappedArray(mipmapped_array_y);
    if (mipmapped_array_uv) cudaFreeMipmappedArray(mipmapped_array_uv);
    if (external_memory) cudaDestroyExternalMemory(external_memory);
    if (nt_handle) CloseHandle(nt_handle);
    if (dxgi_resource) dxgi_resource->Release();
    if (staging_texture) staging_texture->Release();
    
    return result;
}