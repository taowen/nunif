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

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// 改进硬件与软件解码对比测试
TEST_CASE("Try convert color") {
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 0;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    // 验证解码是否成功
    REQUIRE(hw_frame != nullptr);
    
    // 验证帧格式（应该是D3D11硬件格式）
    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    
    // 验证D3D11设备是否可用
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // 测试颜色转换功能
    ID3D11Texture2D* rgba_texture = convert_color(&hw_ctx, hw_frame);
}

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
    SECTION("CUDA Interop Registration") {
        // Initialize CUDA if not already done
        cudaError_t cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            WARN("CUDA not available, skipping interop test");
            rgba_texture->Release();
            return;
        }
        
        // Try to register the texture with CUDA
        cudaGraphicsResource_t cuda_resource = nullptr;
        cuda_status = cudaGraphicsD3D11RegisterResource(
            &cuda_resource, rgba_texture, cudaGraphicsRegisterFlagsNone);
        
        if (cuda_status == cudaSuccess) {
            INFO("Successfully registered D3D11 texture with CUDA");
            
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
            
            INFO("Texture data is in valid range [0,1] for ONNX inference");
            INFO("Sample pixel values: R=" << host_sample[0] << " G=" << host_sample[1] 
                 << " B=" << host_sample[2] << " A=" << host_sample[3]);
            
            // Cleanup
            cudaFree(d_temp_buffer);
            cudaGraphicsUnmapResources(1, &cuda_resource, stream);
            cudaGraphicsUnregisterResource(cuda_resource);
            cudaStreamDestroy(stream);
        } else {
            WARN("Failed to register texture with CUDA: " << cudaGetErrorString(cuda_status));
        }
    }
    
    // Test tensor shape compatibility
    SECTION("ONNX Input Shape Compatibility") {
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
        
        INFO("Tensor dimensions: " << width << "x" << height << "x" << channels);
        INFO("Memory requirement: " << (memory_size / (1024*1024)) << " MB");
        
        // Should not exceed reasonable memory limits (e.g., 1GB)
        REQUIRE(memory_size < (1024ULL * 1024 * 1024));
    }
    
    // Cleanup
    rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
}