/**
 * @file d3d11_nv12_cuda_interop.cpp
 * @brief Demonstrates importing a D3D11 NV12 texture into CUDA via external memory interop.
 *
 * This example shows how to use the NT Handle sharing mechanism to import a
 * D3D11 texture (specifically, a planar NV12 format) into CUDA. The texture
 * is then processed by a CUDA kernel (YUV to RGB conversion) and the result
 * is saved to a file.
 *
 * To compile (assuming CUDA and Visual Studio build tools are in PATH):
 * nvcc -o d3d11_nv12_cuda_interop.exe d3d11_nv12_cuda_interop.cpp -ld3d11 -ldxgi
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <memory>

#include <d3d11_1.h>
#include <dxgi1_2.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// --- Error Handling ---

class HrException : public std::runtime_error {
public:
    HrException(HRESULT hr) : std::runtime_error("HRESULT Exception"), hr_(hr) {}
    HRESULT result() const { return hr_; }
private:
    const HRESULT hr_;
};

inline void HR_CHECK(HRESULT hr) {
    if (FAILED(hr)) {
        fprintf(stderr, "HRESULT check failed: 0x%08X\n", static_cast<UINT>(hr));
        throw HrException(hr);
    }
}

#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA Error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            throw std::runtime_error(cudaGetErrorString(err));   \
        }                                                        \
    } while (0)

// --- CUDA Kernel ---

__device__ __forceinline__ unsigned char clamp_val(float v) {
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return static_cast<unsigned char>(v);
}

__global__ void nv12_to_rgba_kernel(cudaSurfaceObject_t y_plane, cudaSurfaceObject_t uv_plane, uchar4* rgba_out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    // Read luma (Y) from the first plane
    unsigned char luma = surf2Dread<unsigned char>(y_plane, x * sizeof(unsigned char), y);

    // Read chroma (U,V) from the second plane (half resolution)
    uchar2 chroma = surf2Dread<uchar2>(uv_plane, (x / 2) * sizeof(uchar2), y / 2);

    // YUV to RGB conversion (BT.601)
    float y_f = luma;
    float u_f = chroma.x - 128.0f;
    float v_f = chroma.y - 128.0f;

    float r_f = y_f + 1.402f * v_f;
    float g_f = y_f - 0.344136f * u_f - 0.714136f * v_f;
    float b_f = y_f + 1.772f * u_f;

    rgba_out[y * width + x] = make_uchar4(clamp_val(r_f), clamp_val(g_f), clamp_val(b_f), 255);
}

// --- PPM Image Saver ---

void save_ppm(const char* filename, uchar4* data, int width, int height) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.write(reinterpret_cast<char*>(&data[i]), 3);
    }
    std::cout << "Saved result to " << filename << std::endl;
}


int main() {
    const int width = 256;
    const int height = 256;

    IDXGIFactory1* dxgi_factory = nullptr;
    IDXGIAdapter1* cuda_adapter = nullptr;
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* device_context = nullptr;
    ID3D11Texture2D* d3d_texture = nullptr;
    HANDLE nt_handle = nullptr;
    cudaExternalMemory_t external_memory = nullptr;
    cudaMipmappedArray_t mipmapped_array_y = nullptr;
    cudaMipmappedArray_t mipmapped_array_uv = nullptr;
    cudaArray_t array_y = nullptr;
    cudaArray_t array_uv = nullptr;
    cudaSurfaceObject_t surface_y = 0;
    cudaSurfaceObject_t surface_uv = 0;
    uchar4* d_rgba_out = nullptr;

    try {
        // 1. D3D11 Setup
        std::cout << "1. Setting up D3D11 device..." << std::endl;

        HR_CHECK(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&dxgi_factory));

        for (UINT i = 0; dxgi_factory->EnumAdapters1(i, &cuda_adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
            int cuda_device_id;
            if (cudaD3D11GetDevice(&cuda_device_id, cuda_adapter) == cudaSuccess) {
                std::cout << "   Found CUDA-compatible DXGI adapter." << std::endl;
                break;
            }
            cuda_adapter->Release();
            cuda_adapter = nullptr;
        }
        if (!cuda_adapter) {
            throw std::runtime_error("Could not find a CUDA-compatible DXGI adapter.");
        }

        D3D_FEATURE_LEVEL feature_level = D3D_FEATURE_LEVEL_11_1;
        HR_CHECK(D3D11CreateDevice(cuda_adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, &feature_level, 1, D3D11_SDK_VERSION, &device, nullptr, &device_context));

        // 2. Create shareable D3D11 NV12 texture
        std::cout << "2. Creating D3D11 NV12 texture..." << std::endl;

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = width;
        tex_desc.Height = height;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_NV12;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.Usage = D3D11_USAGE_DEFAULT;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        tex_desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

        HR_CHECK(device->CreateTexture2D(&tex_desc, nullptr, &d3d_texture));
        
        // Create a test pattern for the NV12 texture
        std::vector<uint8_t> nv12_data(width * height * 3 / 2);
        // Y plane (luma): vertical gradient
        uint8_t* y_plane_data = nv12_data.data();
        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                y_plane_data[r * width + c] = static_cast<uint8_t>((r * 255) / (height - 1));
            }
        }
        // UV plane (chroma): horizontal gradient for U, constant for V
        uint8_t* uv_plane_data = y_plane_data + width * height;
        for (int r = 0; r < height / 2; ++r) {
            for (int c = 0; c < width / 2; ++c) {
                uv_plane_data[r * width + c * 2 + 0] = static_cast<uint8_t>((c * 2 * 255) / (width - 1)); // U
                uv_plane_data[r * width + c * 2 + 1] = 128; // V
            }
        }
        device_context->UpdateSubresource(d3d_texture, 0, nullptr, nv12_data.data(), width, width * height);


        // 3. Get NT Handle for the texture
        std::cout << "3. Getting NT handle..." << std::endl;
        IDXGIResource1* dxgi_resource;
        HR_CHECK(d3d_texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgi_resource));
        HR_CHECK(dxgi_resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ, nullptr, &nt_handle));
        dxgi_resource->Release();

        // 4. Import D3D11 resource into CUDA
        std::cout << "4. Importing resource into CUDA..." << std::endl;
        cudaExternalMemoryHandleDesc mem_handle_desc = {};
        mem_handle_desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
        mem_handle_desc.handle.win32.handle = nt_handle;
        mem_handle_desc.size = width * height * 3 / 2;
        mem_handle_desc.flags |= cudaExternalMemoryDedicated;
        CUDA_CHECK(cudaImportExternalMemory(&external_memory, &mem_handle_desc));

        // 5. Map NV12 planes to CUDA arrays
        std::cout << "5. Mapping NV12 planes to CUDA arrays..." << std::endl;
        cudaExternalMemoryMipmappedArrayDesc mip_map_desc_y = {}, mip_map_desc_uv = {};
        
        // Map Y plane
        mip_map_desc_y.formatDesc = cudaCreateChannelDesc<unsigned char>();
        mip_map_desc_y.extent = make_cudaExtent(width, height, 0);
        mip_map_desc_y.numLevels = 1;
        mip_map_desc_y.offset = 0;
        CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipmapped_array_y, external_memory, &mip_map_desc_y));

        // Map UV plane
        mip_map_desc_uv.formatDesc = cudaCreateChannelDesc<uchar2>();
        mip_map_desc_uv.extent = make_cudaExtent(width / 2, height / 2, 0);
        mip_map_desc_uv.numLevels = 1;
        mip_map_desc_uv.offset = width * height;
        CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&mipmapped_array_uv, external_memory, &mip_map_desc_uv));

        // 6. Create CUDA surfaces for kernel access
        std::cout << "6. Creating CUDA surfaces..." << std::endl;
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&array_y, mipmapped_array_y, 0));
        CUDA_CHECK(cudaGetMipmappedArrayLevel(&array_uv, mipmapped_array_uv, 0));
        
        cudaResourceDesc res_desc_y = {}, res_desc_uv = {};
        res_desc_y.resType = cudaResourceTypeArray;
        res_desc_y.res.array.array = array_y;
        res_desc_uv.resType = cudaResourceTypeArray;
        res_desc_uv.res.array.array = array_uv;

        CUDA_CHECK(cudaCreateSurfaceObject(&surface_y, &res_desc_y));
        CUDA_CHECK(cudaCreateSurfaceObject(&surface_uv, &res_desc_uv));

        // 7. Execute the conversion kernel
        std::cout << "7. Launching CUDA kernel..." << std::endl;
        CUDA_CHECK(cudaMalloc(&d_rgba_out, width * height * sizeof(uchar4)));

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        
        nv12_to_rgba_kernel<<<grid, block>>>(surface_y, surface_uv, d_rgba_out, width, height);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 8. Save result
        std::cout << "8. Saving result..." << std::endl;
        std::vector<uchar4> h_rgba_out(width * height);
        CUDA_CHECK(cudaMemcpy(h_rgba_out.data(), d_rgba_out, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));
        save_ppm("nv12_output.ppm", h_rgba_out.data(), width, height);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    // 9. Cleanup
    std::cout << "9. Cleaning up resources..." << std::endl;
    if (d_rgba_out) cudaFree(d_rgba_out);
    if (surface_y) cudaDestroySurfaceObject(surface_y);
    if (surface_uv) cudaDestroySurfaceObject(surface_uv);
    if (mipmapped_array_y) cudaFreeMipmappedArray(mipmapped_array_y);
    if (mipmapped_array_uv) cudaFreeMipmappedArray(mipmapped_array_uv);
    if (external_memory) cudaDestroyExternalMemory(external_memory);
    if (nt_handle) CloseHandle(nt_handle);
    if (d3d_texture) d3d_texture->Release();
    if (device_context) device_context->Release();
    if (device) device->Release();
    if (cuda_adapter) cuda_adapter->Release();
    if (dxgi_factory) dxgi_factory->Release();

    return 0;
} 