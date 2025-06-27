#include "main.h"
// Add CUDA headers for interoperability
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <algorithm>  // for std::min, std::max
#include <iomanip>    // for std::setw, std::setprecision, std::fixed

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

void* convert_color(const FFMepgContext* ctx, AVFrame* frame) {
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return nullptr;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return nullptr;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return nullptr;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return nullptr;
    }

    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

    // Print input texture format information
    std::cout << "Input Texture Information:" << std::endl;
    std::cout << "  Format: " << input_desc.Format;
    switch (input_desc.Format) {
        case DXGI_FORMAT_NV12:
            std::cout << " (DXGI_FORMAT_NV12)" << std::endl;
            std::cout << "    This is a planar YUV format with two planes within the same subresource:" << std::endl;
            std::cout << "    - Y plane (Luminance): " << input_desc.Width << "x" << input_desc.Height << " (accessed as DXGI_FORMAT_R8_UNORM)" << std::endl;
            std::cout << "    - UV plane (Chrominance): " << input_desc.Width / 2 << "x" << input_desc.Height / 2 << " (accessed as DXGI_FORMAT_R8G8_UNORM, interleaved)";
            break;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            std::cout << " (DXGI_FORMAT_R8G8B8A8_UNORM)";
            break;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
            std::cout << " (DXGI_FORMAT_B8G8R8A8_UNORM)";
            break;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            std::cout << " (DXGI_FORMAT_R16G16B16A16_FLOAT)";
            break;
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            std::cout << " (DXGI_FORMAT_R32G32B32A32_FLOAT)";
            break;
        default:
            std::cout << " (Unknown/Other)";
            break;
    }
    std::cout << std::endl;
    std::cout << "  Dimensions: " << input_desc.Width << "x" << input_desc.Height << std::endl;
    std::cout << "  MipLevels: " << input_desc.MipLevels << std::endl;
    std::cout << "  ArraySize: " << input_desc.ArraySize << std::endl;
    std::cout << "  Selected frame index from array: " << texture_index << std::endl;
    std::cout << "  SampleDesc: Count=" << input_desc.SampleDesc.Count 
              << ", Quality=" << input_desc.SampleDesc.Quality << std::endl;
    std::cout << "  Usage: " << input_desc.Usage << std::endl;
    std::cout << "  BindFlags: " << input_desc.BindFlags << std::endl;
    std::cout << "  CPUAccessFlags: " << input_desc.CPUAccessFlags << std::endl;
    std::cout << "  MiscFlags: " << input_desc.MiscFlags << std::endl;
    std::cout << "  Frame colorspace: " << frame->colorspace << std::endl;

    if (frame->colorspace != AVCOL_SPC_BT709) {
        std::cerr << "Error: Expected BT.709 colorspace, got " << frame->colorspace << std::endl;
        return nullptr;
    }

    // Set up CUDA device for D3D11 interop
    int cuda_device = -1;
    bool cuda_device_set = false;
    
    IDXGIDevice* dxgi_device = nullptr;
    HRESULT hr = ctx->d3d_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
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
                    std::cerr << "Failed to set CUDA device " << cuda_device << ": " << cudaGetErrorString(cuda_status) << std::endl;
                }
            } else {
                std::cerr << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(cuda_status) << std::endl;
            }
        } else {
            std::cerr << "Failed to get DXGI adapter" << std::endl;
        }
    } else {
        std::cerr << "Failed to get DXGI device" << std::endl;
    }
    
    if (!cuda_device_set) {
        std::cerr << "Could not set proper CUDA device, releasing texture" << std::endl;
        return nullptr;
    }
    
    // Register D3D11 texture with CUDA
    cudaGraphicsResource_t* cuda_resource = new cudaGraphicsResource_t;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        cuda_resource, input_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to register D3D11 texture with CUDA: " << cudaGetErrorString(cuda_status) << std::endl;
        delete cuda_resource;
        return nullptr;
    }
    
    // Verify YUV data content in CUDA resource
    std::cout << "\n=== Verifying YUV Data in CUDA Resource ===" << std::endl;
    
    // Map resource again for data verification
    cuda_status = cudaGraphicsMapResources(1, cuda_resource, 0);
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to map CUDA resource for YUV verification: " << cudaGetErrorString(cuda_status) << std::endl;
    } else {
        // Get mapped array for the entire NV12 texture
        cudaArray_t mapped_cuda_array = nullptr;
        cuda_status = cudaGraphicsSubResourceGetMappedArray(&mapped_cuda_array, *cuda_resource, texture_index, 0);
        if (cuda_status == cudaSuccess) {
            // For FFmpeg D3D11 NV12: Y and UV data are in the same texture but accessed differently
            // The actual height of the texture includes both Y and UV planes
            // Y plane occupies the first 'height' rows
            // UV plane occupies the next 'height/2' rows
            
            int sample_width = std::min(8, static_cast<int>(input_desc.Width));
            int sample_height = std::min(8, static_cast<int>(input_desc.Height));
            
            // Try to get the actual dimensions of the CUDA array
            cudaExtent extent;
            cudaChannelFormatDesc desc;
            unsigned int flags;
            cuda_status = cudaArrayGetInfo(&desc, &extent, &flags, mapped_cuda_array);
            
            if (cuda_status == cudaSuccess) {
                std::cout << "CUDA Array Info:" << std::endl;
                std::cout << "  Extent: " << extent.width << "x" << extent.height << "x" << extent.depth << std::endl;
                std::cout << "  Channel format: x=" << desc.x << ", y=" << desc.y << ", z=" << desc.z << ", w=" << desc.w << std::endl;
                std::cout << "  Channel kind: " << desc.f << std::endl;
            }
            
            // Allocate host memory for Y plane
            unsigned char* y_data = new unsigned char[input_desc.Width * input_desc.Height];
            
            // Copy Y plane data - this should work as it did before
            cudaMemcpy2DFromArray(
                y_data, 
                input_desc.Width,                    // dst pitch
                mapped_cuda_array, 
                0, 0,                               // src offset (top of texture)
                input_desc.Width,                   // width in bytes
                input_desc.Height,                  // height
                cudaMemcpyDeviceToHost
            );
            
            cuda_status = cudaGetLastError();
            if (cuda_status == cudaSuccess) {
                std::cout << "Successfully copied Y plane data from CUDA array" << std::endl;
                
                // Display sample Y plane data
                std::cout << "Y Plane Sample Data (top-left " << sample_width << "x" << sample_height << "):" << std::endl;
                for (int y = 0; y < sample_height; y++) {
                    std::cout << "  Y[" << y << "]: ";
                    for (int x = 0; x < sample_width; x++) {
                        int idx = y * static_cast<int>(input_desc.Width) + x;
                        std::cout << std::setw(3) << (int)y_data[idx] << " ";
                    }
                    std::cout << std::endl;
                }
                
                // Calculate Y plane statistics
                int min_y = 255, max_y = 0;
                long long sum_y = 0;
                for (int i = 0; i < sample_width * sample_height; i++) {
                    int val = (int)y_data[i];
                    min_y = std::min(min_y, val);
                    max_y = std::max(max_y, val);
                    sum_y += val;
                }
                float avg_y = (float)sum_y / (sample_width * sample_height);
                
                std::cout << "\nY Plane Statistics (sample area):" << std::endl;
                std::cout << "  Min: " << min_y << ", Max: " << max_y << ", Avg: " << std::fixed << std::setprecision(1) << avg_y << std::endl;
                
                if (min_y >= 16 && max_y <= 235) {
                    std::cout << "  Y values appear to be in limited range (16-235) - typical for broadcast video" << std::endl;
                } else if (min_y >= 0 && max_y <= 255) {
                    std::cout << "  Y values appear to be in full range (0-255) - typical for computer graphics" << std::endl;
                } else {
                    std::cout << "  Y values are outside expected ranges - possible data corruption" << std::endl;
                }
            } else {
                std::cerr << "Failed to copy Y plane data from CUDA array: " << cudaGetErrorString(cuda_status) << std::endl;
            }
            
            // Now try to copy UV plane data using the correct FFmpeg D3D11 approach
            std::cout << "\nReading UV plane data (FFmpeg D3D11 NV12 format)..." << std::endl;
            
            // In FFmpeg D3D11 NV12, the UV plane follows immediately after the Y plane
            // The total texture height should be height * 1.5 (height + height/2)
            size_t uv_width = input_desc.Width;        // UV plane has same width as Y for NV12
            size_t uv_height = input_desc.Height / 2;  // UV plane height is half of Y
            
            unsigned char* uv_data = new unsigned char[uv_width * uv_height];
            
            // Try copying UV plane from the area right after Y plane
            cudaMemcpy2DFromArray(
                uv_data,
                uv_width,                           // dst pitch
                mapped_cuda_array,
                0, input_desc.Height,               // src offset (starts after Y plane)
                uv_width,                          // width in bytes
                uv_height,                         // height
                cudaMemcpyDeviceToHost
            );
            
            cuda_status = cudaGetLastError();
            if (cuda_status != cudaSuccess) {
                std::cerr << "Method 1 failed: " << cudaGetErrorString(cuda_status) << std::endl;
                
                // Try alternative approach: maybe the texture actual height is height * 1.5
                std::cout << "Trying alternative UV plane access method..." << std::endl;
                
                // Reset error
                cudaGetLastError();
                
                // Try with smaller copy size first to test
                size_t test_uv_height = std::min(static_cast<size_t>(4), uv_height);
                
                cudaMemcpy2DFromArray(
                    uv_data,
                    uv_width,
                    mapped_cuda_array,
                    0, input_desc.Height,           // Still try after Y plane
                    uv_width,
                    test_uv_height,                 // Use smaller height for testing
                    cudaMemcpyDeviceToHost
                );
                
                cuda_status = cudaGetLastError();
                if (cuda_status != cudaSuccess) {
                    std::cerr << "Method 2 also failed: " << cudaGetErrorString(cuda_status) << std::endl;
                    
                    // Try one more approach: copy from a different offset or check if we need to access differently
                    std::cout << "Checking if UV data might be interleaved or at different location..." << std::endl;
                    
                    // Reset error
                    cudaGetLastError();
                    
                    // Try copying from beginning with different interpretation
                    // Maybe UV is stored differently in FFmpeg's D3D11 implementation
                    size_t total_y_size = input_desc.Width * input_desc.Height;
                    unsigned char* all_data = new unsigned char[total_y_size + uv_width * uv_height];
                    
                    // Try to copy more data to see the layout
                    cudaMemcpy2DFromArray(
                        all_data,
                        input_desc.Width,
                        mapped_cuda_array,
                        0, 0,
                        input_desc.Width,
                        input_desc.Height + uv_height,  // Try to get Y + UV
                        cudaMemcpyDeviceToHost
                    );
                    
                    cuda_status = cudaGetLastError();
                    if (cuda_status == cudaSuccess) {
                        std::cout << "Successfully copied extended data, checking UV at different offsets..." << std::endl;
                        
                        // Check if UV data is at the expected location in the extended buffer
                        unsigned char* uv_ptr = all_data + total_y_size;
                        
                        std::cout << "UV data found at offset " << total_y_size << ":" << std::endl;
                        int uv_sample_width = std::min(4, static_cast<int>(uv_width / 2));
                        int uv_sample_height = std::min(4, static_cast<int>(uv_height));
                        
                        for (int y = 0; y < uv_sample_height; y++) {
                            std::cout << "  UV[" << y << "]: ";
                            for (int x = 0; x < uv_sample_width; x++) {
                                size_t idx = y * uv_width + x * 2;
                                if (idx + 1 < uv_width * uv_height) {
                                    int u_val = (int)uv_ptr[idx];
                                    int v_val = (int)uv_ptr[idx + 1];
                                    std::cout << "U" << std::setw(3) << u_val << "V" << std::setw(3) << v_val << " ";
                                }
                            }
                            std::cout << std::endl;
                        }
                    } else {
                        std::cerr << "Extended copy also failed: " << cudaGetErrorString(cuda_status) << std::endl;
                    }
                    
                    delete[] all_data;
                } else {
                    std::cout << "Method 2 succeeded with smaller height!" << std::endl;
                }
            } else {
                std::cout << "Successfully copied UV plane data from CUDA array" << std::endl;
                std::cout << "UV plane dimensions: " << uv_width << "x" << uv_height << std::endl;
                
                // Display sample UV plane data (interleaved U/V)
                int uv_sample_width = std::min(4, static_cast<int>(uv_width / 2));
                int uv_sample_height = std::min(4, static_cast<int>(uv_height));
                
                std::cout << "UV Plane Sample Data (top-left " << uv_sample_width << "x" << uv_sample_height << ", interleaved U/V):" << std::endl;
                for (int y = 0; y < uv_sample_height; y++) {
                    std::cout << "  UV[" << y << "]: ";
                    for (int x = 0; x < uv_sample_width; x++) {
                        // In NV12, UV data is interleaved: UVUVUV...
                        size_t idx = y * uv_width + x * 2;
                        if (idx + 1 < uv_width * uv_height) {
                            int u_val = (int)uv_data[idx];
                            int v_val = (int)uv_data[idx + 1];
                            std::cout << "U" << std::setw(3) << u_val << "V" << std::setw(3) << v_val << " ";
                        }
                    }
                    std::cout << std::endl;
                }
                
                // Calculate UV plane statistics
                int min_u = 255, max_u = 0, min_v = 255, max_v = 0;
                long long sum_u = 0, sum_v = 0;
                int uv_sample_count = uv_sample_width * uv_sample_height;
                
                for (int y = 0; y < uv_sample_height; y++) {
                    for (int x = 0; x < uv_sample_width; x++) {
                        size_t idx = y * uv_width + x * 2;
                        if (idx + 1 < uv_width * uv_height) {
                            int u_val = (int)uv_data[idx];
                            int v_val = (int)uv_data[idx + 1];
                            
                            min_u = std::min(min_u, u_val);
                            max_u = std::max(max_u, u_val);
                            min_v = std::min(min_v, v_val);
                            max_v = std::max(max_v, v_val);
                            
                            sum_u += u_val;
                            sum_v += v_val;
                        }
                    }
                }
                
                if (uv_sample_count > 0) {
                    float avg_u = (float)sum_u / uv_sample_count;
                    float avg_v = (float)sum_v / uv_sample_count;
                    
                    std::cout << "\nUV Plane Statistics (sample area):" << std::endl;
                    std::cout << "  U - Min: " << min_u << ", Max: " << max_u << ", Avg: " << std::fixed << std::setprecision(1) << avg_u << std::endl;
                    std::cout << "  V - Min: " << min_v << ", Max: " << max_v << ", Avg: " << std::fixed << std::setprecision(1) << avg_v << std::endl;
                    
                    if ((min_u >= 16 && max_u <= 240) && (min_v >= 16 && max_v <= 240)) {
                        std::cout << "  UV values appear to be in limited range (16-240)" << std::endl;
                    } else if ((min_u >= 0 && max_u <= 255) && (min_v >= 0 && max_v <= 255)) {
                        std::cout << "  UV values appear to be in full range (0-255)" << std::endl;
                    } else {
                        std::cout << "  UV values are outside expected ranges" << std::endl;
                    }
                }
            }
            
            delete[] y_data;
            delete[] uv_data;
        } else {
            std::cerr << "Failed to get mapped CUDA array for YUV verification: " << cudaGetErrorString(cuda_status) << std::endl;
        }
        
        // Unmap resource after verification
        cuda_status = cudaGraphicsUnmapResources(1, cuda_resource, 0);
        if (cuda_status != cudaSuccess) {
            std::cerr << "Failed to unmap CUDA resource after YUV verification: " << cudaGetErrorString(cuda_status) << std::endl;
        }
    }
    
    std::cout << "=== YUV Data Verification Complete ===" << std::endl;
    
    std::cout << "Successfully registered and verified D3D11 texture with CUDA" << std::endl;
    std::cout << "Texture dimensions: " << input_desc.Width << "x" << input_desc.Height << std::endl;
    
    // Return the CUDA resource pointer for further use
    // Note: Caller is responsible for proper cleanup
    return cuda_resource;
}