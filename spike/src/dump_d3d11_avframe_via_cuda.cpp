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
    
    // Initialize CUDA and get device
    IDXGIDevice* dxgi_device = nullptr;
    HRESULT hr_dxgi = ctx->d3d_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
    if (FAILED(hr_dxgi)) {
        std::cerr << "Failed to query IDXGIDevice from D3D11 device." << std::endl;
        staging_texture->Release();
        return result;
    }

    IDXGIAdapter* dxgi_adapter = nullptr;
    hr_dxgi = dxgi_device->GetAdapter(&dxgi_adapter);
    dxgi_device->Release();
    if (FAILED(hr_dxgi)) {
        std::cerr << "Failed to get IDXGIAdapter from IDXGIDevice." << std::endl;
        staging_texture->Release();
        return result;
    }

    int cuda_device_id = -1;
    cudaError_t init_status = cudaD3D11GetDevice(&cuda_device_id, dxgi_adapter);
    dxgi_adapter->Release();
    if (init_status != cudaSuccess) {
        std::cerr << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    init_status = cudaSetDevice(cuda_device_id);
    if (init_status != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA device " << cuda_device_id << ": " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    cudaDeviceProp device_prop;
    init_status = cudaGetDeviceProperties(&device_prop, cuda_device_id);
    if (init_status != cudaSuccess) {
        std::cerr << "Failed to get CUDA device properties: " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    std::cout << "CUDA device initialized: " << device_prop.name << std::endl;
    std::cout << "CUDA texturePitchAlignment: " << device_prop.texturePitchAlignment << " bytes" << std::endl;
    
    // Get shared handle for the staging texture
    IDXGIResource1* dxgi_resource = nullptr;
    hr_dxgi = staging_texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource);
    if (FAILED(hr_dxgi)) {
        std::cerr << "Failed to query IDXGIResource1 from staging texture: 0x" << std::hex << hr_dxgi << std::endl;
        staging_texture->Release();
        return result;
    }

    HANDLE texture_handle;
    hr_dxgi = dxgi_resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &texture_handle);
    dxgi_resource->Release();
    if (FAILED(hr_dxgi)) {
        std::cerr << "Failed to create shared handle for staging texture: 0x" << std::hex << hr_dxgi << std::endl;
        staging_texture->Release();
        return result;
    }

    // Import D3D11 texture as external memory in CUDA
    cudaExternalMemory_t ext_mem;
    cudaExternalMemoryHandleDesc mem_handle_desc = {};
    mem_handle_desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
    mem_handle_desc.flags = cudaExternalMemoryDedicated;
    // For NV12 format: Y plane (width * height) + UV plane (width * height / 2)
    mem_handle_desc.size = (size_t)frame->width * frame->height * 3 / 2;
    mem_handle_desc.handle.win32.handle = texture_handle;
    
    cudaError_t cuda_status = cudaImportExternalMemory(&ext_mem, &mem_handle_desc);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to import external memory: " << cudaGetErrorString(cuda_status) << std::endl;
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Map Y plane
    cudaMipmappedArray_t y_mipmapped_array;
    cudaExternalMemoryMipmappedArrayDesc mipmap_desc = {};
    mipmap_desc.offset = 0;
    mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    mipmap_desc.extent = make_cudaExtent(frame->width, frame->height, 0);
    mipmap_desc.numLevels = 1;

    std::cout << "[Map Y plane] mipmap_desc:" << std::endl;
    std::cout << "  offset: " << mipmap_desc.offset << std::endl;
    std::cout << "  extent: " << mipmap_desc.extent.width << "x" << mipmap_desc.extent.height << "x" << mipmap_desc.extent.depth << std::endl;
    std::cout << "  format: " << mipmap_desc.formatDesc.x << "," << mipmap_desc.formatDesc.y << "," << mipmap_desc.formatDesc.z << "," << mipmap_desc.formatDesc.w << std::endl;
    std::cout << "  numLevels: " << mipmap_desc.numLevels << std::endl;

    cuda_status = cudaExternalMemoryGetMappedMipmappedArray(&y_mipmapped_array, ext_mem, &mipmap_desc);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to map Y plane mipmapped array: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }
    cudaArray_t y_array;
    cuda_status = cudaGetMipmappedArrayLevel(&y_array, y_mipmapped_array, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get Y plane array level: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Map UV plane
    cudaMipmappedArray_t uv_mipmapped_array;
    mipmap_desc.offset = (size_t)actual_row_pitch * input_desc.Height;
    mipmap_desc.formatDesc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindUnsigned);
    mipmap_desc.extent = make_cudaExtent(frame->width / 2, frame->height / 2, 0);
    
    std::cout << "[Map UV plane] mipmap_desc:" << std::endl;
    std::cout << "  offset: " << mipmap_desc.offset << std::endl;
    std::cout << "  extent: " << mipmap_desc.extent.width << "x" << mipmap_desc.extent.height << "x" << mipmap_desc.extent.depth << std::endl;
    std::cout << "  format: " << mipmap_desc.formatDesc.x << "," << mipmap_desc.formatDesc.y << "," << mipmap_desc.formatDesc.z << "," << mipmap_desc.formatDesc.w << std::endl;
    std::cout << "  numLevels: " << mipmap_desc.numLevels << std::endl;
    
    cuda_status = cudaExternalMemoryGetMappedMipmappedArray(&uv_mipmapped_array, ext_mem, &mipmap_desc);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to map UV plane mipmapped array: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }
    cudaArray_t uv_array;
    cuda_status = cudaGetMipmappedArrayLevel(&uv_array, uv_mipmapped_array, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get UV plane array level: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFreeMipmappedArray(uv_mipmapped_array);
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Create texture objects for reading
    cudaTextureObject_t y_tex, uv_tex;
    
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = y_array;
    
    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    
    cuda_status = cudaCreateTextureObject(&y_tex, &res_desc, &tex_desc, nullptr);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to create Y texture object: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFreeMipmappedArray(uv_mipmapped_array);
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    res_desc.res.array.array = uv_array;
    cuda_status = cudaCreateTextureObject(&uv_tex, &res_desc, &tex_desc, nullptr);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to create UV texture object: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaDestroyTextureObject(y_tex);
        cudaFreeMipmappedArray(uv_mipmapped_array);
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Allocate host memory for YUV data
    result.width = frame->width;
    result.height = frame->height;
    result.y_plane.resize(frame->width * frame->height);
    result.u_plane.resize(frame->width * frame->height / 4);
    result.v_plane.resize(frame->width * frame->height / 4);

    // Copy Y plane data from CUDA to CPU
    cuda_status = cudaMemcpy2DFromArray(result.y_plane.data(), frame->width,
                                       y_array, 0, 0,
                                       frame->width, frame->height,
                                       cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy Y plane data: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaDestroyTextureObject(uv_tex);
        cudaDestroyTextureObject(y_tex);
        cudaFreeMipmappedArray(uv_mipmapped_array);
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Copy UV plane data from CUDA to CPU (interleaved U and V)
    std::vector<uint8_t> uv_interleaved(frame->width * frame->height / 2);
    cuda_status = cudaMemcpy2DFromArray(uv_interleaved.data(), frame->width,
                                       uv_array, 0, 0,
                                       frame->width, frame->height / 2,
                                       cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy UV plane data: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaDestroyTextureObject(uv_tex);
        cudaDestroyTextureObject(y_tex);
        cudaFreeMipmappedArray(uv_mipmapped_array);
        cudaFreeMipmappedArray(y_mipmapped_array);
        cudaDestroyExternalMemory(ext_mem);
        CloseHandle(texture_handle);
        staging_texture->Release();
        return result;
    }

    // Deinterleave UV data into separate U and V planes
    for (int i = 0; i < result.u_plane.size(); ++i) {
        result.u_plane[i] = uv_interleaved[i * 2];
        result.v_plane[i] = uv_interleaved[i * 2 + 1];
    }

    result.valid = true;

    // Save to file if requested
    if (save_to_file) {
        std::ofstream y_file("dump_y.raw", std::ios::binary);
        std::ofstream u_file("dump_u.raw", std::ios::binary);
        std::ofstream v_file("dump_v.raw", std::ios::binary);
        
        if (y_file.is_open()) {
            y_file.write(reinterpret_cast<const char*>(result.y_plane.data()), result.y_plane.size());
            y_file.close();
            std::cout << "Y plane saved to dump_y.raw" << std::endl;
        }
        
        if (u_file.is_open()) {
            u_file.write(reinterpret_cast<const char*>(result.u_plane.data()), result.u_plane.size());
            u_file.close();
            std::cout << "U plane saved to dump_u.raw" << std::endl;
        }
        
        if (v_file.is_open()) {
            v_file.write(reinterpret_cast<const char*>(result.v_plane.data()), result.v_plane.size());
            v_file.close();
            std::cout << "V plane saved to dump_v.raw" << std::endl;
        }
    }

    // Cleanup CUDA resources
    cudaDestroyTextureObject(uv_tex);
    cudaDestroyTextureObject(y_tex);
    cudaFreeMipmappedArray(uv_mipmapped_array);
    cudaFreeMipmappedArray(y_mipmapped_array);
    cudaDestroyExternalMemory(ext_mem);
    CloseHandle(texture_handle);
    staging_texture->Release();

    std::cout << "Successfully extracted YUV data via CUDA interop" << std::endl;
    
    return result;
}