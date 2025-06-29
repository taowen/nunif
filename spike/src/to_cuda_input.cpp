#include "main.h"
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <d3dcompiler.h>
#include <string>
#include <vector>

// DirectCompute 着色器代码 - 将 RGBA8 转换为 float32 NCHW 格式
const char* g_CSSource = R"(
// 输入: RGBA8 纹理 (交错格式)
// 输出: float32 缓冲区 (NCHW 格式)
Texture2D<float4> InputTexture : register(t0);
RWStructuredBuffer<float> OutputBuffer : register(u0);

cbuffer Constants : register(b0)
{
    uint Width;
    uint Height;
    uint ChannelOffset; // 用于 NCHW 布局的通道偏移
};

[numthreads(16, 16, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height)
        return;
    
    // 读取 RGBA 像素 (已经是 [0,1] 范围的 float4)
    float4 pixel = InputTexture[id.xy];
    
    // 计算 NCHW 布局的索引
    // NCHW: [batch, channel, height, width]
    // batch = 0 (固定), channel = 0,1,2,3 (R,G,B,A)
    uint baseIndex = id.y * Width + id.x; // H*W 位置
    uint channelSize = Width * Height;     // 每个通道的大小
    
    // 分别写入 R, G, B, A 通道到对应的 NCHW 位置
    OutputBuffer[0 * channelSize + baseIndex] = pixel.r; // R 通道
    OutputBuffer[1 * channelSize + baseIndex] = pixel.g; // G 通道  
    OutputBuffer[2 * channelSize + baseIndex] = pixel.b; // B 通道
    OutputBuffer[3 * channelSize + baseIndex] = pixel.a; // A 通道
}
)";

struct ComputeShaderResources {
    ID3D11ComputeShader* computeShader = nullptr;
    ID3D11Buffer* constantBuffer = nullptr;
    ID3D11Buffer* outputBuffer = nullptr;
    ID3D11UnorderedAccessView* outputUAV = nullptr;
    ID3D11ShaderResourceView* inputSRV = nullptr;
    
    // CUDA 互操作资源
    cudaGraphicsResource* cudaResource = nullptr;
    
    ~ComputeShaderResources() {
        cleanup();
    }
    
    void cleanup() {
        if (cudaResource) {
            cudaGraphicsUnregisterResource(cudaResource);
            cudaResource = nullptr;
        }
        if (inputSRV) { inputSRV->Release(); inputSRV = nullptr; }
        if (outputUAV) { outputUAV->Release(); outputUAV = nullptr; }
        if (outputBuffer) { outputBuffer->Release(); outputBuffer = nullptr; }
        if (constantBuffer) { constantBuffer->Release(); constantBuffer = nullptr; }
        if (computeShader) { computeShader->Release(); computeShader = nullptr; }
    }
};

// 全局资源缓存 (可以优化为每个上下文一个)
static ComputeShaderResources g_CSResources;
static int g_lastWidth = 0, g_lastHeight = 0;

bool initializeComputeShader(const FFMepgContext* ctx, int width, int height) {
    HRESULT hr;
    
    // 如果尺寸没变且资源已初始化，直接返回
    if (g_CSResources.computeShader && g_lastWidth == width && g_lastHeight == height) {
        return true;
    }
    
    // 清理旧资源
    g_CSResources.cleanup();
    
    // 编译计算着色器
    ID3DBlob* csBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    
    hr = D3DCompile(
        g_CSSource, strlen(g_CSSource),
        "CSMain", nullptr, nullptr,
        "CSMain", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS, 0,
        &csBlob, &errorBlob
    );
    
    if (FAILED(hr)) {
        if (errorBlob) {
            std::cerr << "Compute shader compilation error: " 
                      << (char*)errorBlob->GetBufferPointer() << std::endl;
            errorBlob->Release();
        }
        return false;
    }
    
    // 创建计算着色器
    hr = ctx->d3d_device->CreateComputeShader(
        csBlob->GetBufferPointer(), csBlob->GetBufferSize(),
        nullptr, &g_CSResources.computeShader
    );
    csBlob->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建常量缓冲区
    struct Constants {
        UINT Width;
        UINT Height;
        UINT ChannelOffset;
        UINT Padding;
    } constants = { (UINT)width, (UINT)height, 0, 0 };
    
    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(Constants);
    cbDesc.Usage = D3D11_USAGE_DEFAULT;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    
    D3D11_SUBRESOURCE_DATA cbData = {};
    cbData.pSysMem = &constants;
    
    hr = ctx->d3d_device->CreateBuffer(&cbDesc, &cbData, &g_CSResources.constantBuffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to create constant buffer. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建输出缓冲区 (NCHW float32 格式)
    // 大小: batch(1) * channels(4) * height * width * sizeof(float)
    UINT bufferSize = 1 * 4 * width * height * sizeof(float);
    
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.ByteWidth = bufferSize;
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
    bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED | D3D11_RESOURCE_MISC_SHARED;
    bufferDesc.StructureByteStride = sizeof(float);
    
    hr = ctx->d3d_device->CreateBuffer(&bufferDesc, nullptr, &g_CSResources.outputBuffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output buffer. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建输出UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;  // 对于structured buffer使用UNKNOWN
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.FirstElement = 0;
    uavDesc.Buffer.NumElements = 1 * 4 * width * height; // 总的float元素数量
    
    hr = ctx->d3d_device->CreateUnorderedAccessView(
        g_CSResources.outputBuffer, &uavDesc, &g_CSResources.outputUAV
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create output UAV. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 注册CUDA互操作资源之前进行验证
    std::cout << "Attempting to register CUDA resource..." << std::endl;
    std::cout << "Buffer size: " << bufferSize << " bytes" << std::endl;
    std::cout << "D3D11 device: " << ctx->d3d_device << std::endl;
    std::cout << "Output buffer: " << g_CSResources.outputBuffer << std::endl;
    
    // 检查CUDA设备
    int deviceCount;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    std::cout << "CUDA devices available: " << deviceCount << std::endl;
    
    // 注册CUDA互操作资源
    cudaErr = cudaGraphicsD3D11RegisterResource(
        &g_CSResources.cudaResource, g_CSResources.outputBuffer,
        cudaGraphicsRegisterFlagsNone
    );
    
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to register CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    g_lastWidth = width;
    g_lastHeight = height;
    
    std::cout << "Compute shader initialized successfully for " << width << "x" << height << std::endl;
    return true;
}

/**
 * @brief 将D3D11 RGBA8纹理转换为CUDA float32 NCHW格式，用作IW3推理输入
 * 
 * @param ctx FFMepg上下文，包含D3D11设备和上下文
 * @param rgbaTexture convert_color输出的RGBA8纹理
 * @param cudaOutputPtr 输出的CUDA设备指针，用于IW3推理
 *                      格式要求：float32, NCHW布局, shape=(1,4,H,W)
 * @return bool 转换是否成功
 */
bool to_cuda_input(const FFMepgContext* ctx, ID3D11Texture2D* rgbaTexture, void* cudaOutputPtr) {
    if (!ctx || !rgbaTexture || !cudaOutputPtr) {
        std::cerr << "Error: Invalid parameters for to_cuda_input" << std::endl;
        return false;
    }
    
    // 获取纹理信息
    D3D11_TEXTURE2D_DESC textureDesc;
    rgbaTexture->GetDesc(&textureDesc);
    
    if (textureDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        std::cerr << "Error: Expected RGBA8 texture format" << std::endl;
        return false;
    }
    
    int width = textureDesc.Width;
    int height = textureDesc.Height;
    
    // 初始化计算着色器资源
    if (!initializeComputeShader(ctx, width, height)) {
        return false;
    }
    
    HRESULT hr;
    
    // 创建输入纹理的SRV
    if (g_CSResources.inputSRV) {
        g_CSResources.inputSRV->Release();
        g_CSResources.inputSRV = nullptr;
    }
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Texture2D.MipLevels = 1;
    srvDesc.Texture2D.MostDetailedMip = 0;
    
    hr = ctx->d3d_device->CreateShaderResourceView(
        rgbaTexture, &srvDesc, &g_CSResources.inputSRV
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 设置计算着色器资源
    ctx->d3d_context->CSSetShader(g_CSResources.computeShader, nullptr, 0);
    ctx->d3d_context->CSSetConstantBuffers(0, 1, &g_CSResources.constantBuffer);
    ctx->d3d_context->CSSetShaderResources(0, 1, &g_CSResources.inputSRV);
    ctx->d3d_context->CSSetUnorderedAccessViews(0, 1, &g_CSResources.outputUAV, nullptr);
    
    // 执行计算着色器
    UINT groupX = (width + 15) / 16;   // 16x16 线程组
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
    
    // 通过CUDA互操作访问结果
    cudaError_t cudaErr;
    
    // 映射资源到CUDA
    cudaErr = cudaGraphicsMapResources(1, &g_CSResources.cudaResource, 0);
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    // 获取CUDA设备指针
    void* mappedPtr;
    size_t mappedSize;
    cudaErr = cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, g_CSResources.cudaResource);
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(cudaErr) << std::endl;
        cudaGraphicsUnmapResources(1, &g_CSResources.cudaResource, 0);
        return false;
    }
    
    // 复制数据到目标CUDA缓冲区
    size_t expectedSize = 1 * 4 * width * height * sizeof(float);
    if (mappedSize >= expectedSize) {
        cudaErr = cudaMemcpy(cudaOutputPtr, mappedPtr, expectedSize, cudaMemcpyDeviceToDevice);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to copy CUDA data: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnmapResources(1, &g_CSResources.cudaResource, 0);
            return false;
        }
    } else {
        std::cerr << "Mapped size mismatch: " << mappedSize << " vs expected " << expectedSize << std::endl;
        cudaGraphicsUnmapResources(1, &g_CSResources.cudaResource, 0);
        return false;
    }
    
    // 取消映射
    cudaErr = cudaGraphicsUnmapResources(1, &g_CSResources.cudaResource, 0);
    if (cudaErr != cudaSuccess) {
        std::cerr << "Failed to unmap CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
        return false;
    }
    
    std::cout << "Successfully converted D3D11 RGBA8 to CUDA float32 NCHW format" << std::endl;
    std::cout << "Output shape: (1, 4, " << height << ", " << width << ")" << std::endl;
    
    return true;
}

// 清理全局资源的函数
void cleanup_cuda_input_resources() {
    g_CSResources.cleanup();
    g_lastWidth = 0;
    g_lastHeight = 0;
}
