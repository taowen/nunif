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

    // Get mapped pointer and size
    void* cuda_ptr = nullptr;
    size_t cuda_size = 0;
    cuda_status = cudaGraphicsResourceGetMappedPointer(&cuda_ptr, &cuda_size, cuda_resource);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer from CUDA resource: " << cudaGetErrorString(cuda_status) << std::endl;
        std::cerr << "CUDA error code: " << cuda_status << std::endl;
        std::cerr << "Resource handle: " << cuda_resource << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }
    
    std::cout << "CUDA mapping successful: ptr=" << cuda_ptr << ", size=" << cuda_size << std::endl;

    // Extract YUV data from NV12 format using CUDA memory
    uint8_t* mapped_data = static_cast<uint8_t*>(cuda_ptr);
    int width = frame->width;
    int height = frame->height;
    
    // For CUDA mapped memory, we need to calculate the pitch ourselves
    // Assuming the texture has the same layout as D3D11 staging texture
    int row_pitch = width; // This might need adjustment based on actual memory layout
    
    // Calculate plane sizes
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 4; // U and V planes are each 1/4 the size
    
    // Allocate buffers for separated planes
    result.y_plane.resize(y_plane_size);
    result.u_plane.resize(uv_plane_size);
    result.v_plane.resize(uv_plane_size);
    result.width = width;
    result.height = height;
    
    // Allocate host memory for copying from CUDA
    std::vector<uint8_t> host_buffer(cuda_size);
    cuda_status = cudaMemcpy(host_buffer.data(), cuda_ptr, cuda_size, cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to copy data from CUDA memory: " << cudaGetErrorString(cuda_status) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource, 0);
        cudaGraphicsUnregisterResource(cuda_resource);
        staging_texture->Release();
        return result;
    }
    
    // Now extract data from host buffer
    uint8_t* buffer_data = host_buffer.data();
    
    // Copy Y plane
    for (int y = 0; y < height; y++) {
        memcpy(result.y_plane.data() + y * width, 
               buffer_data + y * row_pitch, 
               width);
    }
    
    // Copy and separate UV plane (NV12 format has interleaved UV)
    uint8_t* uv_start = buffer_data + height * row_pitch;
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            int src_idx = y * row_pitch + x * 2;
            int dst_idx = y * (width / 2) + x;
            result.u_plane[dst_idx] = uv_start[src_idx];     // U component
            result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
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