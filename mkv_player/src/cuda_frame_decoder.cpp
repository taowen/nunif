#include "cuda_frame_decoder.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <d3dcompiler.h>
#include <string>
#include <vector>

// FFmpeg 上下文结构
struct FFMepgContext {
    ID3D11Device* d3d_device = nullptr;
    ID3D11DeviceContext* d3d_context = nullptr;
};

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

// ToCudaInputContext 实现
class ToCudaInputContextImpl {
private:
    ComputeShaderResources csResources;
    int lastWidth = 0, lastHeight = 0;
    bool isMapped = false;
    void* mappedPtr = nullptr;
    size_t mappedSize = 0;

    bool initializeComputeShader(const FFMepgContext* ctx, int width, int height) {
        HRESULT hr;
        
        // 如果尺寸没变且资源已初始化，直接返回
        if (csResources.computeShader && lastWidth == width && lastHeight == height) {
            return true;
        }
        
        // 清理旧资源
        csResources.cleanup();
        
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
            nullptr, &csResources.computeShader
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
        
        hr = ctx->d3d_device->CreateBuffer(&cbDesc, &cbData, &csResources.constantBuffer);
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
        
        hr = ctx->d3d_device->CreateBuffer(&bufferDesc, nullptr, &csResources.outputBuffer);
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
            csResources.outputBuffer, &uavDesc, &csResources.outputUAV
        );
        if (FAILED(hr)) {
            std::cerr << "Failed to create output UAV. HRESULT: 0x" << std::hex << hr << std::endl;
            return false;
        }
        
        // 注册CUDA互操作资源之前进行验证
        std::cout << "Attempting to register CUDA resource..." << std::endl;
        std::cout << "Buffer size: " << bufferSize << " bytes" << std::endl;
        std::cout << "D3D11 device: " << ctx->d3d_device << std::endl;
        std::cout << "Output buffer: " << csResources.outputBuffer << std::endl;
        
        // 检查CUDA设备
        int deviceCount;
        cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to get CUDA device count: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        std::cout << "CUDA devices available: " << deviceCount << std::endl;
        
        if (deviceCount == 0) {
            std::cerr << "No CUDA devices available" << std::endl;
            return false;
        }
        
        // 设置 CUDA 设备  
        cudaErr = cudaSetDevice(0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        std::cout << "CUDA device 0 set successfully" << std::endl;
        
        // 检查 D3D11 和 CUDA 互操作性
        cudaDeviceProp prop;
        cudaErr = cudaGetDeviceProperties(&prop, 0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to get CUDA device properties: " << cudaGetErrorString(cudaErr) << std::endl;
            return false;
        }
        std::cout << "CUDA device: " << prop.name << std::endl;
        std::cout << "CUDA compute capability: " << prop.major << "." << prop.minor << std::endl;
        
        // 尝试不同的注册方式
        std::cout << "Registering CUDA interop resource..." << std::endl;
        
        // 先尝试使用写访问标志
        cudaErr = cudaGraphicsD3D11RegisterResource(
            &csResources.cudaResource, csResources.outputBuffer,
            cudaGraphicsRegisterFlagsWriteDiscard
        );
        
        if (cudaErr != cudaSuccess) {
            std::cout << "Failed with WriteDiscard flag, trying with None flag..." << std::endl;
            // 如果失败，尝试使用默认标志
            cudaErr = cudaGraphicsD3D11RegisterResource(
                &csResources.cudaResource, csResources.outputBuffer,
                cudaGraphicsRegisterFlagsNone
            );
        }
        
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to register CUDA resource with both flags: " << cudaGetErrorString(cudaErr) << std::endl;
            std::cerr << "This may indicate a driver or hardware compatibility issue." << std::endl;
            return false;
        }
        
        lastWidth = width;
        lastHeight = height;
        
        std::cout << "Compute shader initialized successfully for " << width << "x" << height << std::endl;
        return true;
    }

public:
    ToCudaInputContextImpl() = default;
    
    ~ToCudaInputContextImpl() {
        cleanup();
    }

    // 禁止拷贝构造和赋值
    ToCudaInputContextImpl(const ToCudaInputContextImpl&) = delete;
    ToCudaInputContextImpl& operator=(const ToCudaInputContextImpl&) = delete;

    void* to_cuda_input(const FFMepgContext* ctx, ID3D11Texture2D* rgbaTexture) {
        if (!ctx || !rgbaTexture) {
            std::cerr << "Error: Invalid parameters for to_cuda_input" << std::endl;
            return nullptr;
        }
        
        // 如果已经有映射的资源，先取消映射
        if (isMapped) {
            unmap_cuda_input();
        }
        
        // 获取纹理信息
        D3D11_TEXTURE2D_DESC textureDesc;
        rgbaTexture->GetDesc(&textureDesc);
        
        std::cout << "Texture format: " << textureDesc.Format << std::endl;
        
        // 支持 RGBA 和 BGRA 格式
        if (textureDesc.Format != DXGI_FORMAT_R8G8B8A8_UNORM && 
            textureDesc.Format != DXGI_FORMAT_B8G8R8A8_UNORM) {
            std::cerr << "Error: Expected RGBA8 or BGRA8 texture format, got: " << textureDesc.Format << std::endl;
            return nullptr;
        }
        
        std::cout << "Texture format is supported: " << 
            (textureDesc.Format == DXGI_FORMAT_R8G8B8A8_UNORM ? "RGBA8" : "BGRA8") << std::endl;
        
        int width = textureDesc.Width;
        int height = textureDesc.Height;
        
        // 初始化计算着色器资源
        if (!initializeComputeShader(ctx, width, height)) {
            return nullptr;
        }
        
        HRESULT hr;
        
        // 创建输入纹理的SRV
        if (csResources.inputSRV) {
            csResources.inputSRV->Release();
            csResources.inputSRV = nullptr;
        }
        
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.Format = textureDesc.Format; // 使用实际的纹理格式
        srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Texture2D.MostDetailedMip = 0;
        
        hr = ctx->d3d_device->CreateShaderResourceView(
            rgbaTexture, &srvDesc, &csResources.inputSRV
        );
        if (FAILED(hr)) {
            std::cerr << "Failed to create input SRV. HRESULT: 0x" << std::hex << hr << std::endl;
            return nullptr;
        }
        
        // 设置计算着色器资源
        ctx->d3d_context->CSSetShader(csResources.computeShader, nullptr, 0);
        ctx->d3d_context->CSSetConstantBuffers(0, 1, &csResources.constantBuffer);
        ctx->d3d_context->CSSetShaderResources(0, 1, &csResources.inputSRV);
        ctx->d3d_context->CSSetUnorderedAccessViews(0, 1, &csResources.outputUAV, nullptr);
        
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
        cudaErr = cudaGraphicsMapResources(1, &csResources.cudaResource, 0);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to map CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
            return nullptr;
        }
        
        // 获取CUDA设备指针
        cudaErr = cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, csResources.cudaResource);
        if (cudaErr != cudaSuccess) {
            std::cerr << "Failed to get mapped pointer: " << cudaGetErrorString(cudaErr) << std::endl;
            cudaGraphicsUnmapResources(1, &csResources.cudaResource, 0);
            return nullptr;
        }
        
        isMapped = true;
        
        size_t expectedSize = 1 * 4 * width * height * sizeof(float);
        if (mappedSize < expectedSize) {
            std::cerr << "Mapped size mismatch: " << mappedSize << " vs expected " << expectedSize << std::endl;
            unmap_cuda_input();
            return nullptr;
        }
        
        std::cout << "Successfully converted D3D11 RGBA8 to CUDA float32 NCHW format" << std::endl;
        std::cout << "Output shape: (1, 4, " << height << ", " << width << ")" << std::endl;
        std::cout << "Mapped CUDA pointer: " << mappedPtr << ", size: " << mappedSize << std::endl;
        
        return mappedPtr;
    }

    void unmap_cuda_input() {
        if (isMapped && csResources.cudaResource) {
            cudaError_t cudaErr = cudaGraphicsUnmapResources(1, &csResources.cudaResource, 0);
            if (cudaErr != cudaSuccess) {
                std::cerr << "Failed to unmap CUDA resource: " << cudaGetErrorString(cudaErr) << std::endl;
            }
            isMapped = false;
            mappedPtr = nullptr;
            mappedSize = 0;
        }
    }

    void cleanup() {
        if (isMapped) {
            unmap_cuda_input();
        }
        csResources.cleanup();
        lastWidth = 0;
        lastHeight = 0;
    }
};

// ToCudaInputContext 包装类
class ToCudaInputContext {
private:
    std::unique_ptr<ToCudaInputContextImpl> pImpl;

public:
    ToCudaInputContext() : pImpl(std::make_unique<ToCudaInputContextImpl>()) {}
    ~ToCudaInputContext() = default;
    
    // 禁止拷贝构造和赋值
    ToCudaInputContext(const ToCudaInputContext&) = delete;
    ToCudaInputContext& operator=(const ToCudaInputContext&) = delete;
    
    void* to_cuda_input(const FFMepgContext* ctx, ID3D11Texture2D* rgbaTexture) {
        return pImpl->to_cuda_input(ctx, rgbaTexture);
    }
    
    void unmap_cuda_input() {
        pImpl->unmap_cuda_input();
    }
    
    void cleanup() {
        pImpl->cleanup();
    }
};

CudaFrameDecoder::CudaFrameDecoder() 
    : is_initialized_(false), last_cuda_ptr_(nullptr) {
    cuda_context_ = std::make_unique<ToCudaInputContext>();
}

CudaFrameDecoder::~CudaFrameDecoder() {
    close();
}

bool CudaFrameDecoder::open(const std::string& filepath, ID3D11Device* external_device) {
    if (!rgb_decoder_.open(filepath, external_device)) {
        return false;
    }
    
    is_initialized_ = true;
    return true;
}

bool CudaFrameDecoder::readNextFrames(DecodedFrames& decoded_frames) {
    if (!is_initialized_) {
        return false;
    }
    
    if (last_cuda_ptr_) {
        cuda_context_->unmap_cuda_input();
        last_cuda_ptr_ = nullptr;
    }
    
    RGBFrameDecoder::DecodedFrames rgb_frames;
    if (!rgb_decoder_.readNextFrames(rgb_frames)) {
        return false;
    }
    
    decoded_frames.audio_frame = rgb_frames.audio_frame;
    
    std::cout << "RGB frame valid: " << rgb_frames.rgb_frame.is_valid << std::endl;
    if (!rgb_frames.rgb_frame.is_valid) {
        std::cout << "RGB frame is invalid, skipping CUDA conversion" << std::endl;
        decoded_frames.cuda_frame.release();
        return true;
    }
    
    FrameDecoder* frame_decoder = rgb_decoder_.getFrameDecoder();
    if (!frame_decoder) {
        decoded_frames.cuda_frame.release();
        return false;
    }
    
    FFMepgContext ctx = {};
    ctx.d3d_device = frame_decoder->getD3D11Device();
    ctx.d3d_context = frame_decoder->getD3D11Context();
    
    std::cout << "Setting up CUDA context..." << std::endl;
    std::cout << "D3D11 device: " << ctx.d3d_device << std::endl;
    std::cout << "D3D11 context: " << ctx.d3d_context << std::endl;
    std::cout << "RGB texture: " << rgb_frames.rgb_frame.rgb_texture << std::endl;
    
    if (!ctx.d3d_device || !ctx.d3d_context) {
        std::cerr << "Error: Missing D3D11 device or context" << std::endl;
        decoded_frames.cuda_frame.release();
        return false;
    }
    
    std::cout << "Calling to_cuda_input..." << std::endl;
    void* cuda_ptr = cuda_context_->to_cuda_input(&ctx, rgb_frames.rgb_frame.rgb_texture);
    if (!cuda_ptr) {
        decoded_frames.cuda_frame.release();
        return false;
    }
    
    last_cuda_ptr_ = cuda_ptr;
    
    decoded_frames.cuda_frame.cuda_ptr = cuda_ptr;
    decoded_frames.cuda_frame.timestamp = rgb_frames.rgb_frame.timestamp;
    decoded_frames.cuda_frame.width = rgb_frames.rgb_frame.width;
    decoded_frames.cuda_frame.height = rgb_frames.rgb_frame.height;
    decoded_frames.cuda_frame.channels = 4;
    decoded_frames.cuda_frame.size = 1 * 4 * rgb_frames.rgb_frame.width * rgb_frames.rgb_frame.height * sizeof(float);
    decoded_frames.cuda_frame.is_valid = true;
    
    return true;
}

void CudaFrameDecoder::flush() {
    if (last_cuda_ptr_) {
        cuda_context_->unmap_cuda_input();
        last_cuda_ptr_ = nullptr;
    }
    rgb_decoder_.flush();
}

void CudaFrameDecoder::close() {
    if (last_cuda_ptr_) {
        cuda_context_->unmap_cuda_input();
        last_cuda_ptr_ = nullptr;
    }
    
    if (cuda_context_) {
        cuda_context_->cleanup();
    }
    
    rgb_decoder_.close();
    is_initialized_ = false;
}

void CudaFrameDecoder::unmapCudaInput() {
    if (last_cuda_ptr_) {
        cuda_context_->unmap_cuda_input();
        last_cuda_ptr_ = nullptr;
    }
}