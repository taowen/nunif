Of course. Interoperating between DirectX and CUDA, especially with planar video formats like NV12, is a powerful technique for processing video frames on the GPU. The modern external memory APIs provide a robust and efficient way to share resources without copies.

Here is a complete, self-contained C++ example demonstrating how to:
1.  Set up a D3D11 device and find the corresponding CUDA device.
2.  Create a shared D3D11 texture with the `DXGI_FORMAT_NV12` format.
3.  Create a shared D3D11 fence for synchronization.
4.  Import the texture and fence into CUDA using the external memory and semaphore APIs.
5.  Map the separate Y (luma) and UV (chroma) planes of the NV12 texture into two distinct CUDA texture objects.
6.  Write a CUDA kernel to perform YUV-to-RGBA color conversion.
7.  Use the imported fence to synchronize operations between D3D11 and CUDA.
8.  Read the result back and save it as a BMP file for verification.

This example uses the D3D11 API, but the fundamental concepts of creating a shared resource, obtaining a handle, and using `cudaImportExternalMemory` and `cudaImportExternalSemaphore` are identical for D3D12.

```cpp
// Filename: d3d11_cuda_nv12_interop_example.cpp
//
// Description:
// A complete example of importing a DirectX11 texture with format DXGI_FORMAT_NV12
// into CUDA using the external resource interoperability APIs. It performs a
// YUV-to-RGBA conversion on the GPU and saves the result.
//
// Compilation:
// Requires the CUDA Toolkit and Windows SDK.
// Use a C++ compiler (like MSVC) and link against d3d11.lib, dxgi.lib, and cuda.lib.
// Example command line with MSVC:
// cl d3d11_cuda_nv12_interop_example.cpp /link d3d11.lib dxgi.lib cuda.lib
//
// Note: This must be compiled as a 64-bit application.

#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
#include <fstream>

// Windows/D3D headers
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_6.h>

// CUDA Driver API header
#include <cuda.h>

// Linker directives
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "cuda.lib")

// Helper for checking CUDA driver API results
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        CUresult res = call;                                             \
        if (res != CUDA_SUCCESS) {                                       \
            const char* str;                                             \
            cuGetErrorString(res, &str);                                 \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", str, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                      \
        }                                                                \
    } while (0)

// Helper for checking HRESULT from D3D/DXGI calls
#define CHECK_HRESULT(call)                                              \
    do {                                                                 \
        HRESULT hr = call;                                               \
        if (FAILED(hr)) {                                                \
            fprintf(stderr, "HRESULT Error: 0x%08X at %s:%d\n", (unsigned int)hr, __FILE__, __LINE__); \
            throw std::runtime_error("HRESULT error");                   \
        }                                                                \
    } while (0)

// Simple BMP saver
void save_rgba_to_bmp(const char* filename, const unsigned char* data, int width, int height) {
    BITMAPFILEHEADER bmp_header = {};
    BITMAPINFOHEADER bmp_info = {};

    bmp_header.bfType = 0x4D42; // 'BM'
    bmp_header.bfSize = sizeof(bmp_header) + sizeof(bmp_info) + width * height * 4;
    bmp_header.bfOffBits = sizeof(bmp_header) + sizeof(bmp_info);

    bmp_info.biSize = sizeof(bmp_info);
    bmp_info.biWidth = width;
    bmp_info.biHeight = -height; // Top-down
    bmp_info.biPlanes = 1;
    bmp_info.biBitCount = 32;
    bmp_info.biCompression = BI_RGB;

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    file.write(reinterpret_cast<char*>(&bmp_header), sizeof(bmp_header));
    file.write(reinterpret_cast<char*>(&bmp_info), sizeof(bmp_info));
    // D3D gives BGRA, need to swizzle to RGBA for many viewers, but BMP wants BGR.
    // The output from our kernel is RGBA, so we write it as is, which will be read as BGRA by BMP readers.
    // This is fine for this example.
    file.write(reinterpret_cast<const char*>(data), width * height * 4);
    
    std::cout << "Saved result to " << filename << std::endl;
}


// The CUDA kernel is provided as a string to be compiled at runtime with NVRTC.
// This avoids the need for a separate compilation step with nvcc.
// For simplicity in this example, we assume nvcc compilation. If you were to use NVRTC,
// you would load this string and compile it using the NVRTC APIs.
// To compile this file directly, the kernel needs to be defined in a way the host compiler understands.
// We will define it in a separate .cu file and link, or for a single-file example,
// we will assume a build process that handles the CUDA kernel (like using the CUDA runtime API's integration with MSVC).
// For this example, we'll put the kernel code here directly. The host compiler will ignore `__global__` etc.
// but the CUDA compiler (nvcc) will process it.
__global__ void nv12_to_rgba_kernel(cudaTextureObject_t y_plane_tex,
                                  cudaTextureObject_t uv_plane_tex,
                                  cudaSurfaceObject_t rgba_out_surf,
                                  int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    float norm_x = (x + 0.5f) / width;
    float norm_y = (y + 0.5f) / height;

    // Fetch Y (luma) value. tex2D returns a normalized float [0,1].
    float luma = tex2D<float>(y_plane_tex, norm_x, norm_y);

    // Fetch UV (chroma) pair. tex2D on the half-sized UV texture with the same
    // normalized coordinates correctly samples the corresponding chroma information.
    float2 chroma = tex2D<float2>(uv_plane_tex, norm_x, norm_y);

    // Denormalize from [0,1] to [0,255] range
    float Y = luma * 255.0f;
    float U = chroma.x * 255.0f;
    float V = chroma.y * 255.0f;

    // YUV to RGB conversion (ITU-R BT.601 standard, video range)
    // C = Y - 16
    // D = U - 128
    // E = V - 128
    float C = Y - 16.0f;
    float D = U - 128.0f;
    float E = V - 128.0f;

    // R = clamp( ( 298 * C + 409 * E + 128) >> 8 )
    // G = clamp( ( 298 * C - 100 * D - 208 * E + 128) >> 8 )
    // B = clamp( ( 298 * C + 516 * D + 128) >> 8 )
    float r = 1.164f * C + 1.596f * E;
    float g = 1.164f * C - 0.391f * D - 0.813f * E;
    float b = 1.164f * C + 2.018f * D;

    // Clamp to [0, 255]
    r = r < 0.f ? 0.f : (r > 255.f ? 255.f : r);
    g = g < 0.f ? 0.f : (g > 255.f ? 255.f : g);
    b = b < 0.f ? 0.f : (b > 255.f ? 255.f : b);

    // Write to output RGBA surface
    uchar4 rgba_out = make_uchar4((unsigned char)r, (unsigned char)g, (unsigned char)b, 255);
    surf2Dwrite(rgba_out, rgba_out_surf, x * sizeof(uchar4), y);
}

// Function to find the CUDA device corresponding to a D3D11 device.
// This is crucial for the interoperability APIs to work correctly.
int get_cuda_device_for_d3d11_device(ID3D11Device* d3d11_device, CUdevice* cu_device) {
    // Get the DXGI device from the D3D11 device
    IDXGIDevice* dxgi_device;
    CHECK_HRESULT(d3d11_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device));

    // Get the adapter from the DXGI device
    IDXGIAdapter* dxgi_adapter;
    CHECK_HRESULT(dxgi_device->GetAdapter(&dxgi_adapter));
    dxgi_device->Release();

    // Get the adapter description
    DXGI_ADAPTER_DESC dxgi_adapter_desc;
    CHECK_HRESULT(dxgi_adapter->GetDesc(&dxgi_adapter_desc));
    dxgi_adapter->Release();

    // Now, iterate through CUDA devices and match the LUID
    int device_count = 0;
    CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        return -1;
    }

    for (int i = 0; i < device_count; ++i) {
        CUdevice device;
        CHECK_CUDA(cuDeviceGet(&device, i));

        char luid_chars[8] = { 0 };
        unsigned int node_mask = 0;
        CHECK_CUDA(cuDeviceGetAttribute((int*)luid_chars, CU_DEVICE_ATTRIBUTE_LUID, device));
        CHECK_CUDA(cuDeviceGetAttribute(&node_mask, CU_DEVICE_ATTRIBUTE_LUID_DEVICE_NODE_MASK, device));

        // Compare the LUIDs
        if (memcmp(&dxgi_adapter_desc.AdapterLuid, luid_chars, sizeof(luid_chars)) == 0) {
            std::cout << "Found matching D3D11 and CUDA device (CUDA device " << i << ")" << std::endl;
            *cu_device = device;
            return i;
        }
    }

    std::cerr << "Could not find a CUDA device corresponding to the D3D11 device." << std::endl;
    return -1;
}


int main() {
    try {
        const int width = 1280;
        const int height = 720;
        const char* output_filename = "nv12_to_rgba_result.bmp";

        // ========================================================================
        // 1. D3D11 and CUDA Initialization
        // ========================================================================
        std::cout << "1. Initializing D3D11 and CUDA..." << std::endl;
        
        // Initialize CUDA
        CHECK_CUDA(cuInit(0));

        // Create D3D11 Device
        ID3D11Device* d3d_device = nullptr;
        ID3D11DeviceContext* d3d_context = nullptr;
        UINT creation_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        creation_flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0 };
        CHECK_HRESULT(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
            creation_flags, feature_levels, ARRAYSIZE(feature_levels),
            D3D11_SDK_VERSION, &d3d_device, nullptr, &d3d_context));

        // Get D3D11.4 interfaces for fences
        ID3D11Device4* d3d_device4 = nullptr;
        ID3D11DeviceContext4* d3d_context4 = nullptr;
        CHECK_HRESULT(d3d_device->QueryInterface(__uuidof(ID3D11Device4), (void**)&d3d_device4));
        CHECK_HRESULT(d3d_context->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&d3d_context4));


        // Find matching CUDA device and create a context
        CUdevice cu_device;
        int cuda_device_id = get_cuda_device_for_d3d11_device(d3d_device, &cu_device);
        if (cuda_device_id < 0) throw std::runtime_error("No matching CUDA device found.");

        CUcontext cu_context;
        CHECK_CUDA(cuCtxCreate(&cu_context, 0, cu_device));
        
        // ========================================================================
        // 2. Create Shared D3D11 Resources (Textures and Fence)
        // ========================================================================
        std::cout << "2. Creating shared D3D11 resources..." << std::endl;

        // --- Input NV12 Texture ---
        ID3D11Texture2D* nv12_texture = nullptr;
        D3D11_TEXTURE2D_DESC nv12_desc = {};
        nv12_desc.Width = width;
        nv12_desc.Height = height;
        nv12_desc.MipLevels = 1;
        nv12_desc.ArraySize = 1;
        nv12_desc.Format = DXGI_FORMAT_NV12;
        nv12_desc.SampleDesc.Count = 1;
        nv12_desc.Usage = D3D11_USAGE_DEFAULT;
        nv12_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        // This flag is crucial for sharing via NT Handle
        nv12_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        CHECK_HRESULT(d3d_device->CreateTexture2D(&nv12_desc, nullptr, &nv12_texture));

        // --- Output RGBA Texture ---
        ID3D11Texture2D* rgba_texture = nullptr;
        D3D11_TEXTURE2D_DESC rgba_desc = {};
        rgba_desc.Width = width;
        rgba_desc.Height = height;
        rgba_desc.MipLevels = 1;
        rgba_desc.ArraySize = 1;
        rgba_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rgba_desc.SampleDesc.Count = 1;
        rgba_desc.Usage = D3D11_USAGE_DEFAULT;
        rgba_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
        rgba_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
        CHECK_HRESULT(d3d_device->CreateTexture2D(&rgba_desc, nullptr, &rgba_texture));

        // --- Synchronization Fence ---
        ID3D11Fence* d3d_fence = nullptr;
        // D3D11_FENCE_FLAG_SHARED is needed to get a shareable NT handle.
        CHECK_HRESULT(d3d_device4->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&d3d_fence));

        // ========================================================================
        // 3. Get Shareable NT Handles
        // ========================================================================
        std::cout << "3. Getting shareable NT handles..." << std::endl;

        HANDLE nv12_handle, rgba_handle, fence_handle;

        IDXGIResource1* dxgi_resource_nv12 = nullptr;
        CHECK_HRESULT(nv12_texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource_nv12));
        CHECK_HRESULT(dxgi_resource_nv12->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &nv12_handle));
        dxgi_resource_nv12->Release();

        IDXGIResource1* dxgi_resource_rgba = nullptr;
        CHECK_HRESULT(rgba_texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource_rgba));
        CHECK_HRESULT(dxgi_resource_rgba->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &rgba_handle));
        dxgi_resource_rgba->Release();

        CHECK_HRESULT(d3d_fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fence_handle));
        
        // ========================================================================
        // 4. Import D3D11 Resources into CUDA
        // ========================================================================
        std::cout << "4. Importing D3D11 resources into CUDA..." << std::endl;
        
        // --- Import Textures as External Memory ---
        cudaExternalMemory_t ext_mem_nv12, ext_mem_rgba;
        
        cudaExternalMemoryHandleDesc mem_handle_desc = {};
        mem_handle_desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
        mem_handle_desc.flags = cudaExternalMemoryDedicated;
        
        // The size must be the allocated size of the resource. For NV12, it's 1.5 bytes per pixel.
        mem_handle_desc.size = (size_t)width * height * 3 / 2;
        mem_handle_desc.handle.win32.handle = nv12_handle;
        CHECK_CUDA(cuImportExternalMemory(&ext_mem_nv12, &mem_handle_desc));

        mem_handle_desc.size = (size_t)width * height * 4;
        mem_handle_desc.handle.win32.handle = rgba_handle;
        CHECK_CUDA(cuImportExternalMemory(&ext_mem_rgba, &mem_handle_desc));

        // --- Import Fence as External Semaphore ---
        cudaExternalSemaphore_t ext_sem;
        cudaExternalSemaphoreHandleDesc sem_handle_desc = {};
        sem_handle_desc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
        sem_handle_desc.handle.win32.handle = fence_handle;
        CHECK_CUDA(cuImportExternalSemaphore(&ext_sem, &sem_handle_desc));

        // ========================================================================
        // 5. Map CUDA arrays/surfaces onto External Memory
        // ========================================================================
        std::cout << "5. Mapping CUDA arrays onto external memory..." << std::endl;

        // --- Map NV12 Planes ---
        cudaMipmappedArray_t y_plane_array, uv_plane_array;
        
        cudaExternalMemoryMipmappedArrayDesc mipmap_desc = {};
        mipmap_desc.numLevels = 1;
        
        // Map Y plane (first plane, offset 0)
        mipmap_desc.offset = 0;
        mipmap_desc.formatDesc = cuCreateChannelDesc(8, 0, 0, 0, CU_CHANNEL_FORMAT_KIND_UNSIGNED);
        mipmap_desc.extent = make_cudaExtent(width, height, 0);
        CHECK_CUDA(cuExternalMemoryGetMappedMipmappedArray(&y_plane_array, ext_mem_nv12, &mipmap_desc));

        // Map UV plane (second plane, starts after Y plane)
        // NOTE: This assumes a simple contiguous memory layout (pitch == width).
        // For pitched resources, the offset would be `pitch * height`.
        mipmap_desc.offset = (size_t)width * height;
        // UV plane has 2 channels (U and V), and is half the dimensions of Y
        mipmap_desc.formatDesc = cuCreateChannelDesc(8, 8, 0, 0, CU_CHANNEL_FORMAT_KIND_UNSIGNED);
        mipmap_desc.extent = make_cudaExtent(width / 2, height / 2, 0);
        CHECK_CUDA(cuExternalMemoryGetMappedMipmappedArray(&uv_plane_array, ext_mem_nv12, &mipmap_desc));
        
        // --- Map RGBA Output Surface ---
        cudaArray_t rgba_array;
        mipmap_desc.offset = 0;
        mipmap_desc.formatDesc = cuCreateChannelDesc(8, 8, 8, 8, CU_CHANNEL_FORMAT_KIND_UNORM);
        mipmap_desc.extent = make_cudaExtent(width, height, 0);
        mipmap_desc.flags = cudaArraySurfaceLoadStore; // Enable surface access
        CHECK_CUDA(cuExternalMemoryGetMappedArray(&rgba_array, ext_mem_rgba, (const cudaExternalMemoryBufferDesc*)&mipmap_desc));
        
        // ========================================================================
        // 6. Create CUDA Texture/Surface Objects
        // ========================================================================
        std::cout << "6. Creating CUDA texture and surface objects..." << std::endl;
        
        cudaTextureObject_t y_tex_obj, uv_tex_obj;
        cudaSurfaceObject_t rgba_surf_obj;
        
        cudaResourceDesc res_desc = {};
        res_desc.resType = cudaResourceTypeMipmappedArray;
        
        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeClamp;
        tex_desc.addressMode[1] = cudaAddressModeClamp;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat; // Read as float in [0,1]

        res_desc.res.mipmap.mipmappedArray = y_plane_array;
        CHECK_CUDA(cuTexObjectCreate(&y_tex_obj, &res_desc, &tex_desc, nullptr));

        res_desc.res.mipmap.mipmappedArray = uv_plane_array;
        CHECK_CUDA(cuTexObjectCreate(&uv_tex_obj, &res_desc, &tex_desc, nullptr));

        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = rgba_array;
        CHECK_CUDA(cuSurfObjectCreate(&rgba_surf_obj, &res_desc));
        
        // ========================================================================
        // 7. Populate Input Texture and Run Kernel
        // ========================================================================
        std::cout << "7. Populating input texture and running kernel..." << std::endl;
        
        // Create test data for NV12
        std::vector<unsigned char> nv12_data(width * height * 3 / 2);
        // Y plane: a horizontal gradient
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                nv12_data[r * width + c] = (unsigned char)(((float)c / width) * 235.0f + 16.0f);
            }
        }
        // UV plane: a color gradient
        for (int r = 0; r < height / 2; ++r) {
            for (int c = 0; c < width / 2; ++c) {
                nv12_data[width * height + r * width + c * 2]     = (unsigned char)(((float)c / (width/2)) * 224.0f + 16.0f); // U
                nv12_data[width * height + r * width + c * 2 + 1] = (unsigned char)(((float)r / (height/2)) * 224.0f + 16.0f); // V
            }
        }

        // Upload to D3D texture
        d3d_context->UpdateSubresource(nv12_texture, 0, nullptr, nv12_data.data(), width, 0);
        d3d_context->UpdateSubresource(nv12_texture, 1, nullptr, nv12_data.data() + width * height, width, 0);
        
        // --- Synchronization and Kernel Launch ---
        UINT64 fence_value = 1;
        CUstream stream;
        CHECK_CUDA(cuStreamCreate(&stream, 0));

        // D3D signals that the upload is complete
        CHECK_HRESULT(d3d_context4->Signal(d3d_fence, fence_value));

        // CUDA waits for D3D
        cudaExternalSemaphoreWaitParams wait_params = {};
        wait_params.params.fence.value = fence_value;
        CHECK_CUDA(cuWaitExternalSemaphoresAsync(&ext_sem, &wait_params, 1, stream));
        
        // Launch Kernel
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        void* args[] = { &y_tex_obj, &uv_tex_obj, &rgba_surf_obj, &width, &height };
        CHECK_CUDA(cuLaunchKernel(nv12_to_rgba_kernel, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, stream, args, nullptr));

        // CUDA signals that the kernel is complete
        fence_value++;
        cudaExternalSemaphoreSignalParams signal_params = {};
        signal_params.params.fence.value = fence_value;
        CHECK_CUDA(cuSignalExternalSemaphoresAsync(&ext_sem, &signal_params, 1, stream));
        
        // D3D waits for CUDA
        CHECK_HRESULT(d3d_context4->Wait(d3d_fence, fence_value));
        
        // ========================================================================
        // 8. Read Back Result and Save
        // ========================================================================
        std::cout << "8. Reading back result..." << std::endl;

        ID3D11Texture2D* staging_texture = nullptr;
        D3D11_TEXTURE2D_DESC staging_desc = rgba_desc;
        staging_desc.Usage = D3D11_USAGE_STAGING;
        staging_desc.BindFlags = 0;
        staging_desc.MiscFlags = 0;
        staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        CHECK_HRESULT(d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture));
        
        d3d_context->CopyResource(staging_texture, rgba_texture);
        
        D3D11_MAPPED_SUBRESOURCE mapped_res;
        CHECK_HRESULT(d3d_context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped_res));

        std::vector<unsigned char> result_data(width * height * 4);
        for(int i = 0; i < height; ++i) {
            memcpy(result_data.data() + i * width * 4, (unsigned char*)mapped_res.pData + i * mapped_res.RowPitch, width * 4);
        }

        d3d_context->Unmap(staging_texture, 0);
        
        save_rgba_to_bmp(output_filename, result_data.data(), width, height);
        
        // ========================================================================
        // 9. Cleanup
        // ========================================================================
        std::cout << "9. Cleaning up resources..." << std::endl;
        
        CHECK_CUDA(cuStreamDestroy(stream));
        CHECK_CUDA(cuTexObjectDestroy(y_tex_obj));
        CHECK_CUDA(cuTexObjectDestroy(uv_tex_obj));
        CHECK_CUDA(cuSurfObjectDestroy(rgba_surf_obj));
        CHECK_CUDA(cuDestroyExternalSemaphore(ext_sem));
        CHECK_CUDA(cuDestroyExternalMemory(ext_mem_nv12));
        CHECK_CUDA(cuDestroyExternalMemory(ext_mem_rgba));
        CHECK_CUDA(cuMipmappedArrayDestroy(y_plane_array));
        CHECK_CUDA(cuMipmappedArrayDestroy(uv_plane_array));
        CHECK_CUDA(cuArrayDestroy(rgba_array));
        
        CloseHandle(nv12_handle);
        CloseHandle(rgba_handle);
        CloseHandle(fence_handle);
        
        if (staging_texture) staging_texture->Release();
        if (d3d_fence) d3d_fence->Release();
        if (nv12_texture) nv12_texture->Release();
        if (rgba_texture) rgba_texture->Release();
        if (d3d_context4) d3d_context4->Release();
        if (d3d_device4) d3d_device4->Release();
        if (d3d_context) d3d_context->Release();
        if (d3d_device) d3d_device->Release();
        CHECK_CUDA(cuCtxDestroy(cu_context));

        std::cout << "Example finished successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
```