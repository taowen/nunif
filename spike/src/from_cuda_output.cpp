#include "main.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <d3dcompiler.h>
#include <string>
#include <vector>

// DirectCompute 着色器代码 - 将 float32 NCHW 转换为 RGBA8 纹理格式
const char* g_FromCudaCSSource = R"(
// 输入: float32 结构化缓冲区 (NCHW 格式)
// 输出: RGBA8 纹理 (交错格式)
StructuredBuffer<float> InputBuffer : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

cbuffer Constants : register(b0)
{
    uint Width;
    uint Height;
    uint ChannelSize; // Width * Height，用于计算NCHW偏移
    uint Padding;
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height)
        return;
    
    // 计算 NCHW 布局的基础索引
    uint baseIndex = id.y * Width + id.x; // H*W 位置
    
    // 从 NCHW 格式读取 R, G, B, A 通道
    // NCHW: [batch, channel, height, width] where batch=0
    float r = InputBuffer[0 * ChannelSize + baseIndex]; // R 通道
    float g = InputBuffer[1 * ChannelSize + baseIndex]; // G 通道  
    float b = InputBuffer[2 * ChannelSize + baseIndex]; // B 通道
    float a = InputBuffer[3 * ChannelSize + baseIndex]; // A 通道
    
    // 确保值在 [0,1] 范围内并写入纹理
    OutputTexture[id.xy] = float4(
        saturate(r),
        saturate(g), 
        saturate(b),
        saturate(a)
    );
}
)";

struct FromCudaComputeShaderResources {
    ID3D11ComputeShader* computeShader = nullptr;
    ID3D11Buffer* constantBuffer = nullptr;
    ID3D11Buffer* inputBuffer = nullptr;
    ID3D11ShaderResourceView* inputSRV = nullptr;
    ID3D11Texture2D* outputTexture = nullptr;
    ID3D11UnorderedAccessView* outputUAV = nullptr;
    
    // CUDA 互操作资源
    cudaGraphicsResource* cudaResource = nullptr;
    
    ~FromCudaComputeShaderResources() {
        cleanup();
    }
    
    void cleanup() {
        if (cudaResource) {
            cudaGraphicsUnregisterResource(cudaResource);
            cudaResource = nullptr;
        }
        if (outputUAV) { outputUAV->Release(); outputUAV = nullptr; }
        if (outputTexture) { outputTexture->Release(); outputTexture = nullptr; }
        if (inputSRV) { inputSRV->Release(); inputSRV = nullptr; }
        if (inputBuffer) { inputBuffer->Release(); inputBuffer = nullptr; }
        if (constantBuffer) { constantBuffer->Release(); constantBuffer = nullptr; }
        if (computeShader) { computeShader->Release(); computeShader = nullptr; }
    }
};

// FromCudaOutputContext implementation
class FromCudaOutputContext::FromCudaOutputContextImpl {
private:
    FromCudaComputeShaderResources csResources;
    int lastWidth = 0, lastHeight = 0;
    bool isMapped = false;
    void* mappedPtr = nullptr;
    size_t mappedSize = 0;

    bool initializeComputeShader(const FFMepgContext* ctx, int width, int height) {
        HRESULT hr;
        
        // 如果尺寸没变且资源已初始化，只需要重新创建输出纹理
        if (csResources.computeShader && lastWidth == width && lastHeight == height) {
            return true;
        }
        
        // 清理旧资源 - 但保留输出纹理
        ID3D11Texture2D* preservedOutputTexture = csResources.outputTexture;
        ID3D11UnorderedAccessView* preservedOutputUAV = csResources.outputUAV;
        csResources.outputTexture = nullptr;
        csResources.outputUAV = nullptr;
        
        csResources.cleanup();
        
        // 恢复输出纹理
        csResources.outputTexture = preservedOutputTexture;
        csResources.outputUAV = preservedOutputUAV;
        
        // 编译计算着色器
        ID3DBlob* csBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;
        
        hr = D3DCompile(
            g_FromCudaCSSource, strlen(g_FromCudaCSSource),
            "CSMain", nullptr, nullptr,
            "CSMain", "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS, 0,
            &csBlob, &errorBlob
        );
        
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "From CUDA compute shader compilation error: " 
                          << (char*)errorBlob->GetBufferPointer() << std::endl;
                errorBlob->Release();
            }
            return false;
        }
        
        // 创建计算着色器
        hr = ctx->d3d_device->CreateComputeShader(
            csBlob->GetBufferPointer(), csBlob->GetBufferSize(),
            nullptr, &csResources.computeShader
        );
        csBlob->Release();
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA compute shader. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建常量缓冲区
        struct Constants {
            UINT Width;
            UINT Height;
            UINT ChannelSize;
            UINT Padding;
        } constants = { (UINT)width, (UINT)height, (UINT)(width * height), 0 };
        
        D3D11_BUFFER_DESC cbDesc = {};
        cbDesc.ByteWidth = sizeof(Constants);
        cbDesc.Usage = D3D11_USAGE_DEFAULT;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        
        D3D11_SUBRESOURCE_DATA cbData = {};
        cbData.pSysMem = &constants;
        
        hr = ctx->d3d_device->CreateBuffer(&cbDesc, &cbData, &csResources.constantBuffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA constant buffer. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建输入缓冲区 (NCHW float32 格式)
        // 大小: batch(1) * channels(4) * height * width * sizeof(float)
        UINT bufferSize = 1 * 4 * width * height * sizeof(float);
        
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.ByteWidth = bufferSize;
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED | D3D11_RESOURCE_MISC_SHARED;
        bufferDesc.StructureByteStride = sizeof(float);
        
        hr = ctx->d3d_device->CreateBuffer(&bufferDesc, nullptr, &csResources.inputBuffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA input buffer. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建输入SRV
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;  // 对于structured buffer使用UNKNOWN
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = 1 * 4 * width * height; // 总的float元素数量
        
        hr = ctx->d3d_device->CreateShaderResourceView(
            csResources.inputBuffer, &srvDesc, &csResources.inputSRV
        );
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA input SRV. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 注册CUDA互操作资源
        cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
            &csResources.cudaResource, csResources.inputBuffer,
            cudaGraphicsRegisterFlagsNone
        );
        
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to register from CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        
        lastWidth = width;
        lastHeight = height;
        
        std::cout << "From CUDA compute shader initialized successfully for " << width << "x" << height << std::endl;
        return true;
    }

    bool createOutputTexture(const FFMepgContext* ctx, int width, int height) {
        HRESULT hr;
        
        // 清理旧纹理
        if (csResources.outputUAV) { csResources.outputUAV->Release(); csResources.outputUAV = nullptr; }
        if (csResources.outputTexture) { csResources.outputTexture->Release(); csResources.outputTexture = nullptr; }
        
        // 创建输出纹理 (RGBA8 格式)
        D3D11_TEXTURE2D_DESC textureDesc = {};
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        textureDesc.CPUAccessFlags = 0;
        textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        
        hr = ctx->d3d_device->CreateTexture2D(&textureDesc, nullptr, &csResources.outputTexture);
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA output texture. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 创建输出UAV
        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
        uavDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
        uavDesc.Texture2D.MipSlice = 0;
        
        hr = ctx->d3d_device->CreateUnorderedAccessView(
            csResources.outputTexture, &uavDesc, &csResources.outputUAV
        );
        if (FAILED(hr)) {
            std::cerr << "Failed to create from CUDA output UAV. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        return true;
    }

public:
    FromCudaOutputContextImpl() = default;
    
    ~FromCudaOutputContextImpl() {
        cleanup();
    }

    // 禁止拷贝构造和赋值
    FromCudaOutputContextImpl(const FromCudaOutputContextImpl&) = delete;
    FromCudaOutputContextImpl& operator=(const FromCudaOutputContextImpl&) = delete;

    /**
     * @brief 将CUDA float32 NCHW格式转换为D3D11 RGBA8纹理
     */
    ID3D11Texture2D* from_cuda_output(const FFMepgContext* ctx, void* cudaOutputPtr, int height, int width) {
        if (!ctx || !cudaOutputPtr) {
            std::cerr << "Error: Invalid parameters for from_cuda_output" << std::endl;
            return nullptr;
        }
        
        // 初始化compute shader (如果需要) - 先初始化再创建输出纹理
        if (!initializeComputeShader(ctx, width, height)) {
            return nullptr;
        }
        
        // 创建输出纹理
        if (!createOutputTexture(ctx, width, height)) {
            return nullptr;
        }
        
        // 直接将CUDA输出指针注册为D3D11互操作资源
        cudaGraphicsResource* cudaOutputResource = nullptr;
        HRESULT hr;
        
        // 创建一个临时的D3D11缓冲区来包装CUDA内存
        size_t dataSize = 1 * 4 * width * height * sizeof(float);
        
        // 方法1：使用外部CUDA内存创建D3D11缓冲区
        // 创建共享的D3D11缓冲区
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.ByteWidth = (UINT)dataSize;
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED | D3D11_RESOURCE_MISC_SHARED;
        bufferDesc.StructureByteStride = sizeof(float);
        
        ID3D11Buffer* sharedBuffer = nullptr;
        hr = ctx->d3d_device->CreateBuffer(&bufferDesc, nullptr, &sharedBuffer);
        if (FAILED(hr)) {
            std::cerr << "Failed to create shared buffer. HRESULT: 0x" << std::hex << hr << std::endl;
            return nullptr;
        }
        
        // 注册CUDA互操作资源
        cudaError_t cudaErr = cudaGraphicsD3D11RegisterResource(
            &cudaOutputResource, sharedBuffer, cudaGraphicsRegisterFlagsNone
        );
        
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to register CUDA output resource: " << cudaGetErrorString(cudaErr) << std::endl;
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 映射D3D11缓冲区到CUDA
        cudaErr = cudaGraphicsMapResources(1, &cudaOutputResource, 0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to map CUDA output resource: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnregisterResource(cudaOutputResource);
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 获取映射的CUDA指针
        void* mappedPtr;
        size_t mappedSize;
        cudaErr = cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, cudaOutputResource);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnmapResources(1, &cudaOutputResource, 0);
            cudaGraphicsUnregisterResource(cudaOutputResource);
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 直接将CUDA输出数据复制到映射的缓冲区（这是唯一必要的复制）
        cudaErr = cudaMemcpy(mappedPtr, cudaOutputPtr, dataSize, cudaMemcpyDeviceToDevice);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to copy CUDA output data: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnmapResources(1, &cudaOutputResource, 0);
            cudaGraphicsUnregisterResource(cudaOutputResource);
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 取消映射
        cudaErr = cudaGraphicsUnmapResources(1, &cudaOutputResource, 0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to unmap CUDA output resource: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnregisterResource(cudaOutputResource);
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 创建SRV用于compute shader
        ID3D11ShaderResourceView* inputSRV = nullptr;
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = DXGI_FORMAT_UNKNOWN;
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srvDesc.Buffer.FirstElement = 0;
        srvDesc.Buffer.NumElements = 1 * 4 * width * height;
        
        hr = ctx->d3d_device->CreateShaderResourceView(sharedBuffer, &srvDesc, &inputSRV);
        if (FAILED(hr)) {
            std::cerr << "Failed to create input SRV. HRESULT: 0x" << std::hex << hr << std::endl;
            cudaGraphicsUnregisterResource(cudaOutputResource);
            sharedBuffer->Release();
            return nullptr;
        }
        
        // 设置计算着色器资源
        ctx->d3d_context->CSSetShader(csResources.computeShader, nullptr, 0);
        ctx->d3d_context->CSSetConstantBuffers(0, 1, &csResources.constantBuffer);
        ctx->d3d_context->CSSetShaderResources(0, 1, &inputSRV);
        ctx->d3d_context->CSSetUnorderedAccessViews(0, 1, &csResources.outputUAV, nullptr);
        
        // 执行计算着色器
        UINT groupX = (width + 15) / 16;
        UINT groupY = (height + 15) / 16;
        ctx->d3d_context->Dispatch(groupX, groupY, 1);
        
        // 等待GPU完成
        ctx->d3d_context->Flush();
        
        // 清理着色器绑定
        ID3D11ShaderResourceView* nullSRV = nullptr;
        ID3D11UnorderedAccessView* nullUAV = nullptr;
        ctx->d3d_context->CSSetShaderResources(0, 1, &nullSRV);
        ctx->d3d_context->CSSetUnorderedAccessViews(0, 1, &nullUAV, nullptr);
        ctx->d3d_context->CSSetShader(nullptr, nullptr, 0);
        
        // 清理临时资源
        inputSRV->Release();
        cudaGraphicsUnregisterResource(cudaOutputResource);
        sharedBuffer->Release();
        
        // 返回输出纹理（调用者负责Release）
        if (csResources.outputTexture) {
            csResources.outputTexture->AddRef();
            std::cout << "Successfully converted CUDA float32 NCHW to D3D11 RGBA8 texture (direct method)" << std::endl;
            std::cout << "Output texture size: " << width << "x" << height << std::endl;
            return csResources.outputTexture;
        } else {
            std::cerr << "Error: Output texture is null" << std::endl;
            return nullptr;
        }
    }

    /**
     * @brief 清理所有资源
     */
    void cleanup() {
        csResources.cleanup();
        lastWidth = 0;
        lastHeight = 0;
        isMapped = false;
        mappedPtr = nullptr;
        mappedSize = 0;
    }
};

// FromCudaOutputContext implementation - using pimpl pattern
FromCudaOutputContext::FromCudaOutputContext() : pImpl(std::make_unique<FromCudaOutputContextImpl>()) {}

FromCudaOutputContext::~FromCudaOutputContext() = default;

ID3D11Texture2D* FromCudaOutputContext::from_cuda_output(const FFMepgContext* ctx, void* cudaOutputPtr, int height, int width) {
    return pImpl->from_cuda_output(ctx, cudaOutputPtr, height, width);
}

void FromCudaOutputContext::cleanup() {
    pImpl->cleanup();
}

// 全局函数接口
ID3D11Texture2D* from_cuda_output(FromCudaOutputContext* context, const FFMepgContext* ctx, 
                                   void* cudaOutputPtr, int height, int width) {
    if (!context) {
        std::cerr << "Error: FromCudaOutputContext is null" << std::endl;
        return nullptr;
    }
    return context->from_cuda_output(ctx, cudaOutputPtr, height, width);
}
