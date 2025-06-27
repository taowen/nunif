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

    // Create texture for CUDA interop (not staging)
    D3D11_TEXTURE2D_DESC staging_desc{};
    staging_desc.Width = input_desc.Width;
    staging_desc.Height = input_desc.Height;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.Format = input_desc.Format;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.Usage = D3D11_USAGE_DEFAULT;
    staging_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    staging_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return result;
    }

    // Copy from D3D11 texture to staging texture
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
    ctx->d3d_context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    
    // Flush the context to ensure copy is complete before CUDA registration
    ctx->d3d_context->Flush();
    
    // Get CUDA device corresponding to D3D11 device
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
    
    // Initialize CUDA on the correct device
    init_status = cudaSetDevice(cuda_device_id);
    if (init_status != cudaSuccess) {
        std::cerr << "Failed to initialize CUDA device " << cuda_device_id << ": " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    // Check if CUDA-D3D11 interop is supported
    int device_count = 0;
    init_status = cudaGetDeviceCount(&device_count);
    if (init_status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices available: " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    // Optionally, check if the current device supports D3D11 interop
    cudaDeviceProp device_prop;
    init_status = cudaGetDeviceProperties(&device_prop, cuda_device_id);
    if (init_status != cudaSuccess) {
        std::cerr << "Failed to get CUDA device properties: " << cudaGetErrorString(init_status) << std::endl;
        staging_texture->Release();
        return result;
    }
    
    std::cout << "CUDA device initialized: " << device_prop.name << std::endl;
    
    // Register the texture with CUDA - add more detailed error checking
    cudaGraphicsResource_t cuda_resource = nullptr;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &cuda_resource, staging_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to register D3D11 texture with CUDA: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cerr << "CUDA error code: " << cuda_status << std::endl;
        staging_texture->Release();
        return result;
    }
    
    // Verify the resource handle is valid
    if (cuda_resource == nullptr) {
        std::cerr << "CUDA resource handle is null after registration" << std::endl;
        staging_texture->Release();
        return result;
    }

    // Map the CUDA resource
    cuda_status = cudaGraphicsMapResources(1, &cuda_resource, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to map CUDA graphics resource: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cerr << "CUDA error code: " << cuda_status << std::endl;
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Get the mapped CUDA array from the resource (only one array for NV12)
    cudaArray_t cuda_array = nullptr;
    cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get mapped array from CUDA resource: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // Get CUDA array information using the correct API
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
    std::cout << "  Depth: " << extent.depth << std::endl;
    std::cout << "  Format: x=" << format_desc.x << ", y=" << format_desc.y << ", z=" << format_desc.z << ", w=" << format_desc.w << std::endl;
    std::cout << "  Kind: " << format_desc.f << std::endl;

    int width = frame->width;
    int height = frame->height;
    
    // For NV12 format, the CUDA array height includes both Y and UV planes
    // The total height should be height * 1.5 for NV12
    size_t expected_total_height = height + height / 2;  // Y plane + UV plane height
    size_t actual_width = extent.width;
    size_t actual_height = extent.height;
    
    std::cout << "Frame dimensions: " << width << "x" << height << std::endl;
    std::cout << "Expected total height for NV12: " << expected_total_height << std::endl;
    std::cout << "Actual array height: " << actual_height << std::endl;
    
    // Use the actual array dimensions for copying
    size_t bytes_per_element = 1; // For NV12, each element is 1 byte
    if (format_desc.f == cudaChannelFormatKindUnsigned) {
        if (format_desc.x == 8) bytes_per_element = 1;
        else if (format_desc.x == 16) bytes_per_element = 2;
    }
    
    // Calculate buffer size based on actual array dimensions
    size_t total_size_bytes = actual_width * actual_height * bytes_per_element;

    void* d_buffer = nullptr;
    cuda_status = cudaMalloc(&d_buffer, total_size_bytes);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to allocate CUDA device memory: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }

    // For NV12, copy the entire array including both Y and UV data
    // The pitch should match the array width
    size_t src_pitch = actual_width * bytes_per_element;
    size_t dst_pitch = actual_width * bytes_per_element;
    
    cuda_status = cudaMemcpy2DFromArray(
        d_buffer, dst_pitch,           // destination and pitch
        cuda_array, 0, 0,              // source array and offset
        actual_width * bytes_per_element, actual_height,  // width in bytes and height
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

    std::cout << "CUDA mapping and copy successful, size=" << total_size_bytes << std::endl;

    // Allocate host memory for copying from CUDA
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

    // Now extract data from host buffer
    uint8_t* buffer_data = host_buffer.data();
    
    // For NV12 format with actual array dimensions
    // Y plane: width * height bytes
    // UV plane: width * height/2 bytes (interleaved U,V)
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 4;  // U and V are each width/2 * height/2
    
    // Calculate where UV data starts based on array layout
    // If the array height is height * 1.5, then UV starts at width * height
    // If the array width includes padding, we need to account for that
    size_t y_data_end = (size_t)width * height;
    size_t uv_data_start = y_data_end;
    
    // If array has different dimensions, adjust accordingly
    if (actual_height == expected_total_height) {
        // Standard NV12 layout: Y plane followed by UV plane
        uv_data_start = (size_t)actual_width * height;
    } else if (actual_height == height) {
        // UV might be stored separately or in a different format
        std::cerr << "Warning: Unexpected array height for NV12 format" << std::endl;
        uv_data_start = (size_t)actual_width * height;
    }
    
    std::cout << "Buffer analysis:" << std::endl;
    std::cout << "  Total buffer size: " << total_size_bytes << std::endl;
    std::cout << "  Frame dimensions: " << width << "x" << height << std::endl;
    std::cout << "  Array dimensions: " << actual_width << "x" << actual_height << std::endl;
    std::cout << "  Y plane size: " << y_plane_size << std::endl;
    std::cout << "  UV data start: " << uv_data_start << std::endl;
    std::cout << "  UV data available: " << (total_size_bytes > uv_data_start ? total_size_bytes - uv_data_start : 0) << std::endl;
    
    // Allocate buffers for separated planes
    result.y_plane.resize(y_plane_size);
    result.u_plane.resize(uv_plane_size);
    result.v_plane.resize(uv_plane_size);
    result.width = width;
    result.height = height;
    
    // Copy Y plane with proper pitch handling
    if (actual_width == width) {
        // No padding, direct copy
        if (y_plane_size <= total_size_bytes) {
            memcpy(result.y_plane.data(), buffer_data, y_plane_size);
        } else {
            std::cerr << "Error: Y plane size exceeds buffer size" << std::endl;
            cudaGraphicsUnmapResources(1, &cuda_resource, 0);
            cudaGraphicsUnregisterResource(cuda_resource);
            staging_texture->Release();
            return result;
        }
    } else {
        // Handle padding in Y plane
        for (int y = 0; y < height; y++) {
            size_t src_offset = y * actual_width;
            size_t dst_offset = y * width;
            if (src_offset + width <= total_size_bytes && dst_offset + width <= y_plane_size) {
                memcpy(result.y_plane.data() + dst_offset, buffer_data + src_offset, width);
            } else {
                std::cerr << "Error: Y plane copy out of bounds at row " << y << std::endl;
                cudaGraphicsUnmapResources(1, &cuda_resource, 0);
                cudaGraphicsUnregisterResource(cuda_resource);
                staging_texture->Release();
                return result;
            }
        }
    }
    
    // Copy and separate UV plane (NV12 format has interleaved UV)
    size_t uv_plane_bytes_available = total_size_bytes > uv_data_start ? total_size_bytes - uv_data_start : 0;
    size_t uv_plane_bytes_needed = (size_t)width * height / 2;  // For NV12 UV plane
    
    std::cout << "UV plane analysis:" << std::endl;
    std::cout << "  UV bytes available: " << uv_plane_bytes_available << std::endl;
    std::cout << "  UV bytes needed: " << uv_plane_bytes_needed << std::endl;
    
    if (uv_plane_bytes_available < uv_plane_bytes_needed) {
        std::cerr << "Error: Not enough data for UV plane (available: " << uv_plane_bytes_available 
                  << ", needed: " << uv_plane_bytes_needed << ")" << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }
    
    uint8_t* uv_start = buffer_data + uv_data_start;
    
    // Handle UV plane separation with proper pitch
    if (actual_width == width) {
        // No padding in UV plane
        for (int y = 0; y < height / 2; y++) {
            for (int x = 0; x < width / 2; x++) {
                int src_idx = y * width + x * 2;  // Index in UV plane
                int dst_idx = y * (width / 2) + x;
                
                if (src_idx + 1 < uv_plane_bytes_available && dst_idx < uv_plane_size) {
                    result.u_plane[dst_idx] = uv_start[src_idx];     // U component
                    result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
                }
            }
        }
    } else {
        // Handle padding in UV plane
        for (int y = 0; y < height / 2; y++) {
            for (int x = 0; x < width / 2; x++) {
                int src_idx = y * actual_width + x * 2;  // Index in UV plane with padding
                int dst_idx = y * (width / 2) + x;
                
                if (src_idx + 1 < uv_plane_bytes_available && dst_idx < uv_plane_size) {
                    result.u_plane[dst_idx] = uv_start[src_idx];     // U component
                    result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
                }
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
    
    // Unmap and release CUDA resources
    cudaGraphicsUnmapResources(1, &cuda_resource, 0);
    cudaGraphicsUnregisterResource(cuda_resource);
    staging_texture->Release();
    
    std::cout << "Successfully extracted YUV data from D3D11 AVFrame via CUDA (" 
              << width << "x" << height << ")" << std::endl;
    
    return result;
}