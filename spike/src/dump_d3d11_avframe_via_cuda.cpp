#include "main.h"
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

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
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
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
    
    // Register the texture with CUDA
    cudaGraphicsResource_t cuda_resource = nullptr;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &cuda_resource, staging_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to register D3D11 texture with CUDA: " << cudaGetErrorString(cuda_status) << std::endl;
        staging_texture->Release();
        return result;
    }

    // Map the CUDA resource
    cuda_status = cudaGraphicsMapResources(1, &cuda_resource, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to map CUDA graphics resource: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Get the mapped CUDA array
    cudaArray_t cuda_array = nullptr;
    cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get mapped array from CUDA resource: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Get CUDA array information
    cudaChannelFormatDesc format_desc;
    cudaExtent extent;
    unsigned int flags;
    cuda_status = cudaArrayGetInfo(&format_desc, &extent, &flags, cuda_array);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get CUDA array info: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }
    
    std::cout << "CUDA array info:" << std::endl;
    std::cout << "  Width: " << extent.width << std::endl;
    std::cout << "  Height: " << extent.height << std::endl;
    std::cout << "  Format: x=" << format_desc.x << ", y=" << format_desc.y << ", z=" << format_desc.z << ", w=" << format_desc.w << std::endl;

    int width = frame->width;
    int height = frame->height;
    
    // Use the actual row pitch from D3D11, not the texture width
    size_t row_pitch = actual_row_pitch;
    size_t texture_height = extent.height;
    
    // Calculate total buffer size using actual row pitch
    size_t total_size_bytes = texture_height * row_pitch;

    std::cout << "Frame dimensions: " << width << "x" << height << std::endl;
    std::cout << "Texture dimensions: " << extent.width << "x" << texture_height << std::endl;
    std::cout << "Row pitch: " << row_pitch << std::endl;
    std::cout << "Total buffer size: " << total_size_bytes << std::endl;

    void* d_buffer = nullptr;
    cuda_status = cudaMalloc(&d_buffer, total_size_bytes);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA device memory: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Copy the entire texture data
    cuda_status = cudaMemcpy2DFromArray(
        d_buffer, row_pitch,           // destination and pitch
        cuda_array, 0, 0,              // source array and offset
        row_pitch, texture_height,     // width in bytes and height
        cudaMemcpyDeviceToDevice
    );
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy NV12 data from CUDA array to device buffer: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaFree(d_buffer);
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Allocate host memory and copy from CUDA
    std::vector<uint8_t> host_buffer(total_size_bytes);
    cuda_status = cudaMemcpy(host_buffer.data(), d_buffer, total_size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_buffer);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy data from CUDA memory: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Now extract YUV data using the same layout as the CPU version
    uint8_t* buffer_data = host_buffer.data();
    
    // Calculate plane sizes
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 4;  // U and V are each width/2 * height/2
    
    std::cout << "Y plane size: " << y_plane_size << std::endl;
    std::cout << "UV plane size each: " << uv_plane_size << std::endl;
    
    // Allocate buffers for separated planes
    result.y_plane.resize(y_plane_size);
    result.u_plane.resize(uv_plane_size);
    result.v_plane.resize(uv_plane_size);
    result.width = width;
    result.height = height;
    
    // Copy Y plane - exactly like CPU version
    for (int y = 0; y < height; y++) {
        if (y * row_pitch + width <= total_size_bytes && y * width + width <= y_plane_size) {
            memcpy(result.y_plane.data() + y * width, 
                   buffer_data + y * row_pitch, 
                   width);
        } else {
            std::cerr << "Error: Y plane copy out of bounds at row " << y << std::endl;
            cudaGraphicsUnmapResources(1, &cuda_resource, 0);
            cudaGraphicsUnregisterResource(cuda_resource);
            staging_texture->Release();
            return result;
        }
    }
    
    // Copy and separate UV plane - use texture description height for offset, like CPU version
    size_t uv_data_start = input_desc.Height * row_pitch;
    uint8_t* uv_start = buffer_data + uv_data_start;
    
    std::cout << "UV plane analysis:" << std::endl;
    std::cout << "  UV data start offset: " << uv_data_start << std::endl;
    std::cout << "  UV data available: " << (total_size_bytes > uv_data_start ? total_size_bytes - uv_data_start : 0) << std::endl;
    
    if (uv_data_start >= total_size_bytes) {
        std::cerr << "Error: UV data start offset exceeds buffer size" << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }
    
    // Extract UV data - exactly like CPU version
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            size_t src_idx = y * row_pitch + x * 2;
            int dst_idx = y * (width / 2) + x;
            
            if (uv_data_start + src_idx + 1 < total_size_bytes && dst_idx < uv_plane_size) {
                result.u_plane[dst_idx] = uv_start[src_idx];     // U component
                result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
            } else {
                std::cerr << "Error: UV plane copy out of bounds at (" << x << "," << y << ")" << std::endl;
                cudaGraphicsUnmapResources(1, &cuda_resource, 0);
                cudaGraphicsUnregisterResource(cuda_resource);
                staging_texture->Release();
                return result;
            }
        }
    }
    
    result.valid = true;
    
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
    
    // Cleanup CUDA resources
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    cudaGraphicsUnregisterResource(cuda_resource);
    staging_texture->Release();
    
    std::cout << "Successfully extracted YUV data from D3D11 AVFrame via CUDA (" 
              << width << "x" << height << ")" << std::endl;
    
    return result;
}