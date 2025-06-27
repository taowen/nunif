#include <catch2/catch_test_macros.hpp>
#include "main.h"
// 添加FFmpeg软件缩放库
extern "C" {
#include <libswscale/swscale.h>
}
// Add CUDA headers for interoperability testing
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include <cstring>
#include <fstream>
#include <string>
#include <iostream>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

TEST_CASE("Verify RGBA texture for ONNX inference input") {
    // Input file and frame setup
    const char* input_file = "06 4k.mp4";
    const int frame_index = 0;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    REQUIRE(hw_frame != nullptr);
    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // Test color conversion
    ID3D11Texture2D* rgba_texture = convert_color(&hw_ctx, hw_frame);
    REQUIRE(rgba_texture != nullptr);
    
    ID3D11Device* d3d11_device = hw_ctx.d3d_device;
    ID3D11DeviceContext* d3d11_context = hw_ctx.d3d_context;
    
    // Verify texture properties for ONNX inference
    D3D11_TEXTURE2D_DESC texture_desc;
    rgba_texture->GetDesc(&texture_desc);
    
    // Check texture format - should be float32 RGBA for ONNX
    REQUIRE(texture_desc.Format == DXGI_FORMAT_R32G32B32A32_FLOAT);
    
    // Check dimensions match the original frame
    REQUIRE(texture_desc.Width == static_cast<UINT>(hw_frame->width));
    REQUIRE(texture_desc.Height == static_cast<UINT>(hw_frame->height));
    
    // Verify texture is shared for CUDA interop
    REQUIRE((texture_desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED) != 0);
    
    // Test CUDA-D3D11 interop capability
    REQUIRE(d3d11_device != nullptr);
    
    // Get the corresponding CUDA device
    int cuda_device = -1;
    bool cuda_device_set = false;
    
    IDXGIDevice* dxgi_device = nullptr;
    HRESULT hr = d3d11_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
    if (SUCCEEDED(hr)) {
        IDXGIAdapter* dxgi_adapter = nullptr;
        hr = dxgi_device->GetAdapter(&dxgi_adapter);
        dxgi_device->Release();
        
        if (SUCCEEDED(hr)) {
            cudaError_t cuda_status = cudaD3D11GetDevice(&cuda_device, dxgi_adapter);
            dxgi_adapter->Release();
            
            if (cuda_status == cudaSuccess) {
                cuda_status = cudaSetDevice(cuda_device);
                if (cuda_status == cudaSuccess) {
                    std::cout << "Successfully set CUDA device to " << cuda_device << std::endl;
                    cuda_device_set = true;
                } else {
                    std::cout << "Failed to set CUDA device " << cuda_device << ": " << cudaGetErrorString(cuda_status) << std::endl;
                }
            } else {
                std::cout << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(cuda_status) << std::endl;
            }
        } else {
            std::cout << "Failed to get DXGI adapter" << std::endl;
        }
    } else {
        std::cout << "Failed to get DXGI device" << std::endl;
    }
    
    if (!cuda_device_set) {
        std::cout << "Could not set proper CUDA device, skipping interop test" << std::endl;
        rgba_texture->Release();
        return;
    }
    
    // Try to register the texture with CUDA
    cudaGraphicsResource_t cuda_resource = nullptr;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &cuda_resource, rgba_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status == cudaSuccess) {
        std::cout << "Successfully registered D3D11 texture with CUDA" << std::endl;
        
        // Test mapping the resource
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cuda_status = cudaGraphicsMapResources(1, &cuda_resource, stream);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Get mapped array
        cudaArray_t cuda_array;
        cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Test data accessibility - allocate device memory and copy
        size_t pixel_size = 4 * sizeof(float); // RGBA float32
        size_t data_size = texture_desc.Width * texture_desc.Height * pixel_size;
        float* d_temp_buffer = nullptr;
        
        cuda_status = cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), data_size);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Copy from CUDA array to linear memory
        cuda_status = cudaMemcpy2DFromArrayAsync(
            d_temp_buffer, texture_desc.Width * pixel_size,
            cuda_array, 0, 0,
            texture_desc.Width * pixel_size, texture_desc.Height,
            cudaMemcpyDeviceToDevice, stream
        );
        REQUIRE(cuda_status == cudaSuccess);
        
        // Synchronize to ensure copy is complete
        cudaStreamSynchronize(stream);
        
        // Test: Copy some data back to host to verify it's valid
        std::vector<float> host_sample(16); // Sample first 4 pixels (4 components each)
        cuda_status = cudaMemcpy(host_sample.data(), d_temp_buffer, 
                                host_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Verify the data is in valid range for ONNX input (normalized [0,1])
        bool valid_range = true;
        for (float value : host_sample) {
            if (value < 0.0f || value > 1.0f) {
                valid_range = false;
                break;
            }
        }
        REQUIRE(valid_range);
        
        std::cout << "Texture data is in valid range [0,1] for ONNX inference" << std::endl;
        std::cout << "Sample pixel values: R=" << host_sample[0] << " G=" << host_sample[1] 
                << " B=" << host_sample[2] << " A=" << host_sample[3] << std::endl;
        
        // Cleanup
        cudaFree(d_temp_buffer);
        cudaGraphicsUnmapResources(1, &cuda_resource, stream);
        cudaGraphicsUnregisterResource(cuda_resource);
        cudaStreamDestroy(stream);
    } else {
        std::cout << "Failed to register texture with CUDA: " << cudaGetErrorString(cuda_status) << std::endl;
    }
    // Typical ONNX input shape: [batch, channels, height, width] or [batch, height, width, channels]
    UINT width = texture_desc.Width;
    UINT height = texture_desc.Height;
    UINT channels = 4; // RGBA
    
    // For most vision models, input size should be reasonable
    REQUIRE(width > 0);
    REQUIRE(height > 0);
    REQUIRE(width <= 8192);  // Reasonable upper bound
    REQUIRE(height <= 8192); // Reasonable upper bound
    
    size_t total_elements = static_cast<size_t>(width) * height * channels;
    size_t memory_size = total_elements * sizeof(float);
    
    std::cout << "Tensor dimensions: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "Memory requirement: " << (memory_size / (1024*1024)) << " MB" << std::endl;
    
    // Should not exceed reasonable memory limits (e.g., 1GB)
    REQUIRE(memory_size < (1024ULL * 1024 * 1024));
    
    // Cleanup
    rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
}