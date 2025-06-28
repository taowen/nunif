#include <d3d11.h>
#include <dxgi1_2.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <vector>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

#define CHECK_HRESULT(call) \
    do { \
        HRESULT hr = call; \
        if (FAILED(hr)) { \
            std::cerr << "DirectX Error: 0x" << std::hex << hr << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Structure to hold external memory resources
struct ExternalMemoryResource {
    cudaExternalMemory_t externalMemory;
    cudaMipmappedArray_t mipmappedArrayY;  // Y plane
    cudaMipmappedArray_t mipmappedArrayUV; // UV plane
    cudaArray_t arrayY;
    cudaArray_t arrayUV;
    cudaSurfaceObject_t surfaceY;
    cudaSurfaceObject_t surfaceUV;
};

// Function to get CUDA channel format descriptor for DXGI format
cudaChannelFormatDesc getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT dxgiFormat) {
    cudaChannelFormatDesc d;
    memset(&d, 0, sizeof(d));
    switch (dxgiFormat) {
        case DXGI_FORMAT_R8_UNORM:
            d.x = 8; d.y = 0; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned;
            break;
        case DXGI_FORMAT_R8G8_UNORM:
            d.x = 8; d.y = 8; d.z = 0; d.w = 0; d.f = cudaChannelFormatKindUnsigned;
            break;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
            d.x = 8; d.y = 8; d.z = 8; d.w = 8; d.f = cudaChannelFormatKindUnsigned;
            break;
        default:
            std::cerr << "Unsupported DXGI format: " << dxgiFormat << std::endl;
            exit(1);
    }
    return d;
}

// Function to create DirectX11 NV12 texture with NT handle sharing
ID3D11Texture2D* createD3D11NV12Texture(ID3D11Device* device, UINT width, UINT height, HANDLE* sharedHandle) {
    D3D11_TEXTURE2D_DESC textureDesc = {};
    textureDesc.Width = width;
    textureDesc.Height = height;
    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.Format = DXGI_FORMAT_NV12;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.Usage = D3D11_USAGE_DEFAULT;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    textureDesc.CPUAccessFlags = 0;
    textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

    ID3D11Texture2D* texture = nullptr;
    CHECK_HRESULT(device->CreateTexture2D(&textureDesc, nullptr, &texture));

    // Get the shared handle
    IDXGIResource1* dxgiResource = nullptr;
    CHECK_HRESULT(texture->QueryInterface(__uuidof(IDXGIResource1), (void**)&dxgiResource));
    
    CHECK_HRESULT(dxgiResource->CreateSharedHandle(
        nullptr, 
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, 
        nullptr, 
        sharedHandle
    ));

    dxgiResource->Release();
    return texture;
}

// Function to import D3D11 resource from NT handle
cudaExternalMemory_t importD3D11ResourceFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = nullptr;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));
    desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
    desc.handle.win32.handle = (void*)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    CHECK_CUDA(cudaImportExternalMemory(&extMem, &desc));
    
    // Close the handle as it's no longer needed
    CloseHandle(handle);
    
    return extMem;
}

// Function to map NV12 texture onto external memory
void mapNV12TextureOntoExternalMemory(ExternalMemoryResource& resource, 
                                      cudaExternalMemory_t extMem, 
                                      UINT width, UINT height) {
    // NV12 format has two planes:
    // - Y plane: full resolution, 8-bit per pixel
    // - UV plane: half resolution, 16-bit per pixel (8-bit U + 8-bit V)
    
    // Map Y plane (luminance)
    cudaChannelFormatDesc formatDescY = getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT_R8_UNORM);
    cudaExtent extentY = make_cudaExtent(width, height, 0);
    
    cudaExternalMemoryMipmappedArrayDesc mipmapDescY = {};
    mipmapDescY.offset = 0;  // Y plane starts at offset 0
    mipmapDescY.formatDesc = formatDescY;
    mipmapDescY.extent = extentY;
    mipmapDescY.flags = 0;
    mipmapDescY.numLevels = 1;

    CHECK_CUDA(cudaExternalMemoryGetMappedMipmappedArray(&resource.mipmappedArrayY, extMem, &mipmapDescY));
    CHECK_CUDA(cudaGetMipmappedArrayLevel(&resource.arrayY, resource.mipmappedArrayY, 0));

    // Map UV plane (chrominance)
    cudaChannelFormatDesc formatDescUV = getCudaChannelFormatDescForDxgiFormat(DXGI_FORMAT_R8G8_UNORM);
    cudaExtent extentUV = make_cudaExtent(width / 2, height / 2, 0);
    
    cudaExternalMemoryMipmappedArrayDesc mipmapDescUV = {};
    mipmapDescUV.offset = width * height;  // UV plane starts after Y plane
    mipmapDescUV.formatDesc = formatDescUV;
    mipmapDescUV.extent = extentUV;
    mipmapDescUV.flags = 0;
    mipmapDescUV.numLevels = 1;

    CHECK_CUDA(cudaExternalMemoryGetMappedMipmappedArray(&resource.mipmappedArrayUV, extMem, &mipmapDescUV));
    CHECK_CUDA(cudaGetMipmappedArrayLevel(&resource.arrayUV, resource.mipmappedArrayUV, 0));

    // Create surface objects for kernel access
    cudaResourceDesc resDescY = {};
    resDescY.resType = cudaResourceTypeArray;
    resDescY.res.array.array = resource.arrayY;
    CHECK_CUDA(cudaCreateSurfaceObject(&resource.surfaceY, &resDescY));

    cudaResourceDesc resDescUV = {};
    resDescUV.resType = cudaResourceTypeArray;
    resDescUV.res.array.array = resource.arrayUV;
    CHECK_CUDA(cudaCreateSurfaceObject(&resource.surfaceUV, &resDescUV));

    resource.externalMemory = extMem;
}

// CUDA kernel to process NV12 data
__global__ void processNV12Kernel(cudaSurfaceObject_t surfaceY, 
                                 cudaSurfaceObject_t surfaceUV,
                                 int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Read Y value
        unsigned char yValue = surf2Dread<unsigned char>(surfaceY, x, y);
        
        // Modify Y value (example: increase brightness)
        yValue = min(255, (int)yValue + 20);
        
        // Write back Y value
        surf2Dwrite(yValue, surfaceY, x, y);
        
        // Process UV plane (half resolution)
        if (x < width / 2 && y < height / 2) {
            uchar2 uvValue = surf2Dread<uchar2>(surfaceUV, x, y);
            // Modify UV values (example: adjust saturation)
            uvValue.x = min(255, max(0, (int)uvValue.x + 10));
            uvValue.y = min(255, max(0, (int)uvValue.y + 10));
            surf2Dwrite(uvValue, surfaceUV, x, y);
        }
    }
}

// Function to cleanup external memory resources
void cleanupExternalMemoryResource(ExternalMemoryResource& resource) {
    if (resource.surfaceY) {
        CHECK_CUDA(cudaDestroySurfaceObject(resource.surfaceY));
    }
    if (resource.surfaceUV) {
        CHECK_CUDA(cudaDestroySurfaceObject(resource.surfaceUV));
    }
    if (resource.mipmappedArrayY) {
        CHECK_CUDA(cudaFreeMipmappedArray(resource.mipmappedArrayY));
    }
    if (resource.mipmappedArrayUV) {
        CHECK_CUDA(cudaFreeMipmappedArray(resource.mipmappedArrayUV));
    }
    if (resource.externalMemory) {
        CHECK_CUDA(cudaDestroyExternalMemory(resource.externalMemory));
    }
}

// Main example function
int main() {
    // Initialize DirectX11
    ID3D11Device* d3d11Device = nullptr;
    ID3D11DeviceContext* d3d11Context = nullptr;
    
    D3D_FEATURE_LEVEL featureLevel;
    CHECK_HRESULT(D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_DEBUG,
        nullptr, 0, D3D11_SDK_VERSION,
        &d3d11Device, &featureLevel, &d3d11Context
    ));

    std::cout << "DirectX11 device created successfully" << std::endl;

    // Initialize CUDA
    CHECK_CUDA(cudaSetDevice(0));
    std::cout << "CUDA device initialized" << std::endl;

    // Create NV12 texture
    const UINT width = 1920;
    const UINT height = 1080;
    HANDLE sharedHandle;
    
    ID3D11Texture2D* nv12Texture = createD3D11NV12Texture(d3d11Device, width, height, &sharedHandle);
    std::cout << "NV12 texture created with shared handle" << std::endl;

    // Calculate texture size for NV12 format
    // NV12: Y plane (width * height) + UV plane (width * height / 2)
    unsigned long long textureSize = width * height + (width * height / 2);

    // Import external memory
    cudaExternalMemory_t externalMemory = importD3D11ResourceFromNTHandle(sharedHandle, textureSize);
    std::cout << "External memory imported successfully" << std::endl;

    // Map texture onto external memory
    ExternalMemoryResource resource = {};
    mapNV12TextureOntoExternalMemory(resource, externalMemory, width, height);
    std::cout << "NV12 texture mapped onto external memory" << std::endl;

    // Example: Process the NV12 data with CUDA
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    processNV12Kernel<<<gridSize, blockSize>>>(resource.surfaceY, resource.surfaceUV, width, height);
    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "CUDA kernel executed successfully" << std::endl;

    // Cleanup
    cleanupExternalMemoryResource(resource);
    nv12Texture->Release();
    d3d11Context->Release();
    d3d11Device->Release();

    std::cout << "Resources cleaned up successfully" << std::endl;
    return 0;
}