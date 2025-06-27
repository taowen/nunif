#include "infer_sbs.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <memory>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

// Prevent Windows min/max macros from interfering
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

// TensorRT includes
#include "NvInfer.h"
#include "NvOnnxParser.h"

void checkCudaErrors(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << " (" << static_cast<int>(result) << ")\n";
        throw std::runtime_error("CUDA error occurred");
    }
}

namespace {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

class TensorRTInferenceEngine {
private:
    std::shared_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;
    
    // Input/output dimensions
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;
    
    // Device memory buffers
    float* d_input = nullptr;
    float* d_output = nullptr;
    size_t input_size = 0;
    size_t output_size = 0;
    size_t current_input_size = 0;  // 当前分配的输入缓冲区大小
    size_t current_output_size = 0; // 当前分配的输出缓冲区大小
    
    std::string get_trt_cache_path(const std::string& onnx_file_path) {
        // 将 .onnx 后缀替换为 .trt
        std::string trt_path = onnx_file_path;
        size_t pos = trt_path.find_last_of('.');
        if (pos != std::string::npos) {
            trt_path = trt_path.substr(0, pos) + ".trt";
        } else {
            trt_path += ".trt";
        }
        return trt_path;
    }
    
    bool load_trt_engine(const std::string& trt_file_path) {
        std::ifstream file(trt_file_path, std::ios::binary);
        if (!file.good()) {
            return false;
        }
        
        // 获取文件大小
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // 读取引擎数据
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        // 创建运行时和引擎
        runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), size));
        if (!engine) {
            std::cerr << "Failed to deserialize TensorRT engine from cache" << std::endl;
            return false;
        }
        
        context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Get input/output dimensions
        input_dims = engine->getTensorShape(engine->getIOTensorName(0));
        output_dims = engine->getTensorShape(engine->getIOTensorName(1));
        
        std::cout << "TensorRT engine loaded from cache: " << trt_file_path << std::endl;
        return true;
    }
    
    bool save_trt_engine(const std::string& trt_file_path, nvinfer1::IHostMemory* serialized_engine) {
        std::ofstream file(trt_file_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to create TensorRT cache file: " << trt_file_path << std::endl;
            return false;
        }
        
        file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        file.close();
        
        std::cout << "TensorRT engine saved to cache: " << trt_file_path << std::endl;
        return true;
    }
    
    bool load_onnx_model(const std::string& onnx_file_path) {
        // 检查是否存在 TensorRT 缓存文件
        std::string trt_cache_path = get_trt_cache_path(onnx_file_path);
        
        // 首先尝试从缓存加载
        if (load_trt_engine(trt_cache_path)) {
            std::cout << "Input tensor: " << engine->getIOTensorName(0) << std::endl;
            std::cout << "Output tensor: " << engine->getIOTensorName(1) << std::endl;
            return true;
        }
        
        std::cout << "TensorRT cache not found, building engine from ONNX..." << std::endl;
        
        // 缓存不存在或加载失败，从 ONNX 构建引擎
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            std::cerr << "Failed to create TensorRT builder" << std::endl;
            return false;
        }
        
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        if (!network) {
            std::cerr << "Failed to create network definition" << std::endl;
            return false;
        }
        
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config" << std::endl;
            return false;
        }
        
        auto parser = std::unique_ptr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        // Parse ONNX model
        if (!parser->parseFromFile(onnx_file_path.c_str(), 
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file: " << onnx_file_path << std::endl;
            return false;
        }
        
        // Set optimization profile for dynamic shapes
        auto profile = builder->createOptimizationProfile();
        const char* input_name = network->getInput(0)->getName();
        
        // Set dynamic dimensions: batch=1, channels=4, height=[512,2160], width=[512,4096]  
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 4, 512, 512});
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1, 4, 1080, 1920});
        profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{1, 4, 2160, 4096});
        config->addOptimizationProfile(profile);
        
        // Enable FP16 precision if available
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
            std::cout << "Using FP16 precision" << std::endl;
        }
        
        // Build engine
        std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        // 保存引擎到缓存文件
        save_trt_engine(trt_cache_path, plan.get());
        
        runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Get input/output dimensions
        input_dims = engine->getTensorShape(engine->getIOTensorName(0));
        output_dims = engine->getTensorShape(engine->getIOTensorName(1));
        
        std::cout << "TensorRT engine built and cached successfully" << std::endl;
        std::cout << "Input tensor: " << engine->getIOTensorName(0) << std::endl;
        std::cout << "Output tensor: " << engine->getIOTensorName(1) << std::endl;
        
        return true;
    }
    
public:
    bool initialize(const std::string& onnx_file_path) {
        if (!load_onnx_model(onnx_file_path)) {
            return false;
        }
        return true;
    }
    
    bool infer(float* input_data_device, UINT width, UINT height, cudaStream_t stream) {
        // Set dynamic input dimensions
        nvinfer1::Dims4 input_shape{1, 4, static_cast<int>(height), static_cast<int>(width)};
        if (!context->setInputShape(engine->getIOTensorName(0), input_shape)) {
            std::cerr << "Failed to set input shape" << std::endl;
            return false;
        }
        
        // Calculate sizes
        input_size = 1 * 4 * height * width * sizeof(float);
        
        // Get output dimensions from the context after setting input shape
        nvinfer1::Dims output_dims_actual = context->getTensorShape(engine->getIOTensorName(1));
        
        // 添加详细的输出维度检查
        UINT actual_output_channels = static_cast<UINT>(output_dims_actual.d[1]);
        UINT actual_output_height = static_cast<UINT>(output_dims_actual.d[2]);
        UINT actual_output_width = static_cast<UINT>(output_dims_actual.d[3]);
        
        std::cout << "    → Input dimensions: " << width << "x" << height << std::endl;
        std::cout << "    → Output dimensions: " << actual_output_width << "x" << actual_output_height << "x" << actual_output_channels << std::endl;
        
        // 检查输出尺寸是否与输入匹配
        if (actual_output_height != height || actual_output_width != width) {
            std::cout << "    ⚠ WARNING: Output dimensions don't match input dimensions!" << std::endl;
            std::cout << "    → This might cause the white strip issue" << std::endl;
        }

        size_t output_elements = 1;
        for (int i = 0; i < output_dims_actual.nbDims; ++i) {
            if (output_dims_actual.d[i] < 0) {
                 std::cerr << "    ✗ Invalid output dimension: " << output_dims_actual.d[i] << std::endl;
                 return false;
            }
            output_elements *= output_dims_actual.d[i];
        }
        output_size = output_elements * sizeof(float);

        // Reallocate input device memory if needed
        if (current_input_size < input_size) {
            if (d_input) {
                checkCudaErrors(cudaFree(d_input));
            }
            checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_input), input_size));
            current_input_size = input_size;
            std::cout << "    → Allocated input buffer: " << input_size << " bytes\n";
        }
        
        // Reallocate output device memory if needed
        if (current_output_size < output_size) {
            if (d_output) {
                checkCudaErrors(cudaFree(d_output));
            }
            checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_output), output_size));
            current_output_size = output_size;
            std::cout << "    → Allocated output buffer: " << output_size << " bytes\n";
        }
        
        // Copy input data to device buffer (input_data_device is already on device)
        checkCudaErrors(cudaMemcpyAsync(d_input, input_data_device, input_size, cudaMemcpyDeviceToDevice, stream));
        
        // Prepare bindings for executeV2
        void* bindings[2];
        bindings[0] = d_input;  // input tensor
        bindings[1] = d_output; // output tensor
        
        // Execute synchronous inference
        if (!context->executeV2(bindings)) {
            std::cerr << "TensorRT synchronous inference failed" << std::endl;
            return false;
        }
        
        std::cout << "    ✓ TensorRT synchronous inference completed" << std::endl;
        std::cout << "    → Input format: RGBA (4 channels)" << std::endl;
        std::cout << "    → Output format: RGBA (4 channels)" << std::endl;
        std::cout << "    → Input size: " << input_size << " bytes" << std::endl;
        std::cout << "    → Output size: " << output_size << " bytes" << std::endl;
        
        // Get inference output data
        float* cuda_output_data = d_output;
        
        // 添加调试代码：检查输出数据的一些样本值
        std::cout << "    → Checking output data samples..." << std::endl;
        
        // 创建一个小的主机缓冲区来检查输出数据
        const int sample_size = 16; // 检查前16个像素
        std::vector<float> sample_data(sample_size * actual_output_channels);
        checkCudaErrors(cudaMemcpy(sample_data.data(), cuda_output_data, 
                                 sample_size * actual_output_channels * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << "      - First " << sample_size << " pixels (RGBA):" << std::endl;
        for (int i = 0; i < sample_size && i < 4; ++i) {
            std::cout << "        Pixel " << i << ": ("
                     << sample_data[i*4] << ", " << sample_data[i*4+1] << ", "
                     << sample_data[i*4+2] << ", " << sample_data[i*4+3] << ")" << std::endl;
        }
        
        // 检查中间区域的一些像素（可能的白色区域）
        size_t middle_offset = (actual_output_width * actual_output_height / 2) * actual_output_channels;
        std::vector<float> middle_sample(sample_size * actual_output_channels);
        checkCudaErrors(cudaMemcpy(middle_sample.data(), cuda_output_data + middle_offset/sizeof(float), 
                                 sample_size * actual_output_channels * sizeof(float), cudaMemcpyDeviceToHost));
        
        std::cout << "      - Middle region pixels (RGBA):" << std::endl;
        for (int i = 0; i < sample_size && i < 4; ++i) {
            std::cout << "        Pixel " << i << ": ("
                     << middle_sample[i*4] << ", " << middle_sample[i*4+1] << ", "
                     << middle_sample[i*4+2] << ", " << middle_sample[i*4+3] << ")" << std::endl;
        }
        
        // 检查输出数据中是否有大量的1.0值（白色）
        const int check_stride = actual_output_width * actual_output_height / 100; // 每1%检查一次
        std::vector<float> spot_check(actual_output_channels);
        int white_pixel_count = 0;
        const int total_checks = 20;
        
        for (int i = 0; i < total_checks; ++i) {
            size_t offset = i * check_stride * actual_output_channels;
            if (offset + actual_output_channels <= actual_output_width * actual_output_height * actual_output_channels) {
                checkCudaErrors(cudaMemcpy(spot_check.data(), cuda_output_data + offset, 
                                         actual_output_channels * sizeof(float), cudaMemcpyDeviceToHost));
                
                // 检查是否为白色像素（所有通道都接近1.0）
                bool is_white = true;
                for (int c = 0; c < 3; ++c) { // 只检查RGB，忽略Alpha
                    if (spot_check[c] < 0.9f) {
                        is_white = false;
                        break;
                    }
                }
                if (is_white) white_pixel_count++;
            }
        }
        
        std::cout << "      - White pixel ratio in spot check: " << white_pixel_count << "/" << total_checks 
                 << " (" << (white_pixel_count * 100.0f / total_checks) << "%)" << std::endl;
        
        // 添加 TensorRT 输出维度的详细检查
        std::cout << "    → Detailed TensorRT output analysis:" << std::endl;
        std::cout << "      - Expected output: " << width << "x" << height << "x4" << std::endl;
        std::cout << "      - Actual output dims: ";
        for (int i = 0; i < output_dims_actual.nbDims; ++i) {
            std::cout << output_dims_actual.d[i];
            if (i < output_dims_actual.nbDims - 1) std::cout << "x";
        }
        std::cout << std::endl;

        // 验证输出格式
        if (output_dims_actual.nbDims != 4) {
            std::cerr << "    ✗ Unexpected output dimensions count: " << output_dims_actual.nbDims << " (expected: 4)" << std::endl;
            return false;
        }

        if (output_dims_actual.d[0] != 1) {
            std::cerr << "    ✗ Unexpected batch size: " << output_dims_actual.d[0] << " (expected: 1)" << std::endl;
            return false;
        }
        
        // 检查输出内存布局
        size_t expected_output_size = static_cast<size_t>(output_dims_actual.d[0]) * 
                                     static_cast<size_t>(output_dims_actual.d[1]) * 
                                     static_cast<size_t>(output_dims_actual.d[2]) * 
                                     static_cast<size_t>(output_dims_actual.d[3]) * sizeof(float);
        std::cout << "      - Expected output buffer size: " << expected_output_size << " bytes" << std::endl;
        std::cout << "      - Actual output buffer size: " << output_size << " bytes" << std::endl;
        
        if (expected_output_size != output_size) {
            std::cerr << "    ⚠ Output buffer size mismatch!" << std::endl;
        }
        
        return true;
    }
    
    float* get_output_data() const {
        return d_output;
    }
    
    // Add this method to get output dimensions
    bool get_output_dimensions(UINT& width, UINT& height, UINT& channels) const {
        if (!context || !engine) {
            return false;
        }
        
        nvinfer1::Dims output_dims = context->getTensorShape(engine->getIOTensorName(1));
        if (output_dims.nbDims != 4) {
            return false;
        }
        
        channels = static_cast<UINT>(output_dims.d[1]);
        height = static_cast<UINT>(output_dims.d[2]);
        width = static_cast<UINT>(output_dims.d[3]);
        return true;
    }
    
    ~TensorRTInferenceEngine() {
        if (d_input) {
            cudaFree(d_input);
        }
        if (d_output) {
            cudaFree(d_output);
        }
    }
};

// Global inference engine instance
std::unique_ptr<TensorRTInferenceEngine> g_inference_engine;

// Helper function to create output D3D11 texture
bool create_output_d3d11_texture(UINT width, UINT height, ID3D11Device* d3d11_device, 
                                 ID3D11Texture2D** output_texture) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // Float32 RGBA format
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = 0;
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, output_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output D3D11 texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        
        switch (hr) {
            case E_INVALIDARG:
                std::cerr << "    → Invalid argument in texture creation\n";
                break;
            case E_OUTOFMEMORY:
                std::cerr << "    → Out of memory\n";
                break;
            case DXGI_ERROR_INVALID_CALL:
                std::cerr << "    → Invalid DXGI call\n";
                break;
            default:
                std::cerr << "    → Unknown error\n";
                break;
        }
        return false;
    }
    
    if (!*output_texture) {
        std::cerr << "Output texture pointer is null after creation\n";
        return false;
    }
    
    std::cout << "    ✓ D3D11 output texture created successfully (" << width << "x" << height << ")\n";
    return true;
}

// Helper function to write CUDA output to D3D11 texture (synchronous)
bool write_cuda_output_to_d3d11(float* cuda_output_data, UINT width, UINT height, UINT channels,
                                ID3D11Texture2D* d3d11_output_texture, cudaStream_t stream) {
    if (!cuda_output_data) {
        std::cerr << "CUDA output data is null\n";
        return false;
    }
    
    if (!d3d11_output_texture) {
        std::cerr << "D3D11 output texture is null\n";
        return false;
    }
    
    if (width == 0 || height == 0 || channels == 0) {
        std::cerr << "Invalid dimensions: " << width << "x" << height << "x" << channels << "\n";
        return false;
    }
    
    std::cout << "    → Attempting to register D3D11 texture with CUDA...\n";
    std::cout << "    → Texture dimensions: " << width << "x" << height << "x" << channels << "\n";
    
    int current_device;
    cudaError_t device_result = cudaGetDevice(&current_device);
    if (device_result == cudaSuccess) {
        std::cout << "    → Current CUDA device: " << current_device << "\n";
    }
    
    cudaGraphicsResource_t cuda_resource = nullptr;
    cudaError_t cuda_result = cudaGraphicsD3D11RegisterResource(&cuda_resource, d3d11_output_texture, 
                                                               cudaGraphicsRegisterFlagsNone);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to register D3D11 texture with CUDA: " << cudaGetErrorString(cuda_result) << "\n";
        
        switch (cuda_result) {
            case cudaErrorInvalidValue:
                std::cerr << "    → CUDA error: Invalid value (check texture format/flags)\n";
                break;
            case cudaErrorInvalidResourceHandle:
                std::cerr << "    → CUDA error: Invalid resource handle\n";
                break;
            case cudaErrorInvalidDevice:
                std::cerr << "    → CUDA error: Invalid device (D3D11/CUDA device mismatch)\n";
                break;
            case cudaErrorNotSupported:
                std::cerr << "    → CUDA error: Operation not supported\n";
                break;
            default:
                std::cerr << "    → CUDA error code: " << static_cast<int>(cuda_result) << "\n";
                break;
        }
        
        ID3D11Device* texture_device = nullptr;
        d3d11_output_texture->GetDevice(&texture_device);
        if (texture_device) {
            std::cout << "    → D3D11 texture device obtained for diagnostics\n";
            
            IDXGIDevice* dxgi_device = nullptr;
            HRESULT hr = texture_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
            if (SUCCEEDED(hr)) {
                IDXGIAdapter* dxgi_adapter = nullptr;
                hr = dxgi_device->GetAdapter(&dxgi_adapter);
                dxgi_device->Release();
                
                if (SUCCEEDED(hr)) {
                    int cuda_device_for_adapter;
                    cudaError_t compat_result = cudaD3D11GetDevice(&cuda_device_for_adapter, dxgi_adapter);
                    dxgi_adapter->Release();
                    
                    if (compat_result == cudaSuccess) {
                        std::cout << "    → CUDA device for this D3D11 adapter: " << cuda_device_for_adapter << "\n";
                        if (cuda_device_for_adapter != current_device) {
                            std::cerr << "    ✗ CUDA device mismatch! Current: " << current_device 
                                     << ", Required: " << cuda_device_for_adapter << "\n";
                        }
                    } else {
                        std::cerr << "    ✗ D3D11 adapter not compatible with CUDA: " 
                                 << cudaGetErrorString(compat_result) << "\n";
                    }
                }
            }
            texture_device->Release();
        }
        
        return false;
    }
    
    std::cout << "    ✓ D3D11 texture registered with CUDA successfully\n";
    
    // Map the resource for CUDA access (synchronous)
    cuda_result = cudaGraphicsMapResources(1, &cuda_resource, stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to map D3D11 resource for CUDA: " << cudaGetErrorString(cuda_result) << "\n";
        cudaGraphicsUnregisterResource(cuda_resource);
        return false;
    }
    
    // Get the mapped CUDA array
    cudaArray_t cuda_array;
    cuda_result = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to get mapped CUDA array: " << cudaGetErrorString(cuda_result) << "\n";
        cudaGraphicsUnmapResources(1, &cuda_resource, stream);
        cudaGraphicsUnregisterResource(cuda_resource);
        return false;
    }
    
    // Synchronize the stream before copying (ensure inference is complete)
    cuda_result = cudaStreamSynchronize(stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to synchronize CUDA stream: " << cudaGetErrorString(cuda_result) << "\n";
        cudaGraphicsUnmapResources(1, &cuda_resource, stream);
        cudaGraphicsUnregisterResource(cuda_resource);
        return false;
    }
    
    // 获取实际的输出纹理描述
    D3D11_TEXTURE2D_DESC texture_desc;
    d3d11_output_texture->GetDesc(&texture_desc);
    
    std::cout << "    → CUDA output size: " << width << "x" << height << "x" << channels << std::endl;
    std::cout << "    → D3D11 texture size: " << texture_desc.Width << "x" << texture_desc.Height << std::endl;
    
    // 使用实际的纹理尺寸，而不是传入的参数
    UINT actual_width = std::min(width, texture_desc.Width);
    UINT actual_height = std::min(height, texture_desc.Height);
    
    // 如果尺寸不匹配，先清空纹理
    if (width != texture_desc.Width || height != texture_desc.Height) {
        std::cout << "    ⚠ Size mismatch detected, clearing texture first" << std::endl;
        
        // 创建一个临时的白色数据来填充纹理
        size_t temp_size = texture_desc.Width * texture_desc.Height * channels * sizeof(float);
        float* temp_white_data = nullptr;
        checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&temp_white_data), temp_size));
        
        // 创建主机端白色数据，然后拷贝到设备
        std::vector<float> host_white_data(texture_desc.Width * texture_desc.Height * channels, 1.0f);
        checkCudaErrors(cudaMemcpy(temp_white_data, host_white_data.data(), temp_size, cudaMemcpyHostToDevice));
        
        // 先用白色填充整个纹理
        cuda_result = cudaMemcpy2DToArray(
            cuda_array, 0, 0,
            temp_white_data,
            texture_desc.Width * channels * sizeof(float),
            texture_desc.Width * channels * sizeof(float),
            texture_desc.Height,
            cudaMemcpyDeviceToDevice
        );
        
        checkCudaErrors(cudaFree(temp_white_data));
        
        if (cuda_result != cudaSuccess) {
            std::cerr << "Failed to clear texture: " << cudaGetErrorString(cuda_result) << "\n";
            return false;
        }
    }
    
    // 输出详细的拷贝参数用于调试
    std::cout << "    → Copy parameters:" << std::endl;
    std::cout << "      - Source pitch: " << width * channels * sizeof(float) << " bytes" << std::endl;
    std::cout << "      - Copy width: " << actual_width * channels * sizeof(float) << " bytes" << std::endl;
    std::cout << "      - Copy height: " << actual_height << " rows" << std::endl;
    std::cout << "      - Total copy size: " << (actual_width * channels * sizeof(float) * actual_height) << " bytes" << std::endl;
    
    // 然后拷贝实际的输出数据
    cuda_result = cudaMemcpy2DToArray(
        cuda_array, 0, 0,                              // destination: array, x_offset, y_offset
        cuda_output_data,                              // source: CUDA device memory
        width * channels * sizeof(float),             // source pitch
        actual_width * channels * sizeof(float),      // width in bytes (使用实际宽度)
        actual_height,                                 // height (使用实际高度)
        cudaMemcpyDeviceToDevice                       // copy type
    );
    
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to copy CUDA output to D3D11 texture: " << cudaGetErrorString(cuda_result) << "\n";
        cudaGraphicsUnmapResources(1, &cuda_resource, stream);
        cudaGraphicsUnregisterResource(cuda_resource);
        return false;
    }
    
    // Synchronize again to ensure copy is complete
    cuda_result = cudaStreamSynchronize(stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to synchronize CUDA stream after copy: " << cudaGetErrorString(cuda_result) << "\n";
    }
    
    // Unmap the resource
    cuda_result = cudaGraphicsUnmapResources(1, &cuda_resource, stream);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to unmap D3D11 resource: " << cudaGetErrorString(cuda_result) << "\n";
    }
    
    // Unregister the resource
    cuda_result = cudaGraphicsUnregisterResource(cuda_resource);
    if (cuda_result != cudaSuccess) {
        std::cerr << "Failed to unregister D3D11 resource: " << cudaGetErrorString(cuda_result) << "\n";
    }
    
    std::cout << "    ✓ CUDA output successfully written to D3D11 texture (synchronous)\n";
    return true;
}

} // anonymous namespace

void start_infer_sbs(
    D11FrameQueue& input_frame_queue,
    D11FrameQueue& output_frame_queue,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context) {
    
    cudaStream_t cuda_stream = nullptr;
    checkCudaErrors(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));

    std::cout << "=== Depth Inference Thread Started ===\n";
    std::cout << "Using its own CUDA stream: " << cuda_stream << "\n";
    
    // Initialize TensorRT inference engine
    if (!g_inference_engine) {
        g_inference_engine = std::make_unique<TensorRTInferenceEngine>();
        if (!g_inference_engine->initialize("stereo_module_left_eye.onnx")) {
            std::cerr << "Failed to initialize TensorRT inference engine" << std::endl;
            cudaStreamDestroy(cuda_stream);
            // Send end signal to output queue
            output_frame_queue.push(D11Frame::end_signal());
            return;
        }
    }
    
    int processed_count = 0;
    bool cuda_device_set = false;
    
    while (true) {
        D11Frame converted_frame = input_frame_queue.pop();
        
        // Check for end signal
        if (converted_frame.is_end_signal) {
            std::cout << "=== Depth Inference Thread Received End Signal ===\n";
            // Forward end signal to output queue
            output_frame_queue.push(D11Frame::end_signal());
            break;
        }
        
        if (converted_frame.texture) {
            if (!cuda_device_set) {
                // Use the FFmpeg-provided D3D11 device instead of getting it from texture
                if (d3d11_device) {
                    int cuda_device = -1;
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
                                checkCudaErrors(cudaSetDevice(cuda_device));
                                std::cout << "Depth inference thread set CUDA device to " << cuda_device << " (using FFmpeg D3D11 device)" << std::endl;
                                
                                // 验证设备设置成功
                                int current_device;
                                checkCudaErrors(cudaGetDevice(&current_device));
                                if (current_device != cuda_device) {
                                    std::cerr << "CUDA device setting verification failed. Expected: " 
                                             << cuda_device << ", Actual: " << current_device << "\n";
                                    output_frame_queue.push(D11Frame::end_signal());
                                    break;
                                }
                                
                                cuda_device_set = true;
                            } else {
                                std::cerr << "Failed to get CUDA device for D3D11 adapter: " << cudaGetErrorString(cuda_status) << "\n";
                            }
                        } else {
                             std::cerr << "Failed to get DXGI adapter\n";
                        }
                    } else {
                         std::cerr << "Failed to get DXGI device\n";
                    }
                } else {
                    std::cerr << "FFmpeg D3D11 device is null\n";
                }

                if (!cuda_device_set) {
                    std::cerr << "Failed to set CUDA device for depth inference thread. Aborting thread." << std::endl;
                    output_frame_queue.push(D11Frame::end_signal());
                    break;
                }
            }

            processed_count++;
            std::cout << ">>> Processing frame " << processed_count << " for stereo inference\n";
            std::cout << "    → Frame dimensions: " << converted_frame.width << "x" << converted_frame.height << "\n";
            
            // Validate input data
            if (converted_frame.width == 0 || converted_frame.height == 0) {
                std::cerr << "    ✗ Invalid frame dimensions for frame " << processed_count << "\n";
                continue;
            }
            
            // Perform CUDA-D3D11 interop for input
            cudaGraphicsResource_t cuda_resource = nullptr;
            checkCudaErrors(cudaGraphicsD3D11RegisterResource(&cuda_resource, converted_frame.texture, cudaGraphicsRegisterFlagsNone));

            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource, cuda_stream));
            cudaArray_t cuda_array;
            checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0));

            // Execute TensorRT inference
            size_t input_size = converted_frame.width * converted_frame.height * 4 * sizeof(float);
            float* d_temp_input = nullptr;
            checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&d_temp_input), input_size));
            
            checkCudaErrors(cudaMemcpy2DFromArray(
                d_temp_input, converted_frame.width * 4 * sizeof(float),
                cuda_array, 0, 0,
                converted_frame.width * 4 * sizeof(float), converted_frame.height,
                cudaMemcpyDeviceToDevice
            ));

            if (!g_inference_engine->infer(d_temp_input, converted_frame.width, converted_frame.height, cuda_stream)) {
                std::cerr << "    ✗ TensorRT inference failed for frame " << processed_count << "\n";
                checkCudaErrors(cudaFree(d_temp_input));
                checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
                checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
                continue;
            }

            // Get output dimensions from inference engine first
            UINT output_width, output_height, output_channels;
            if (!g_inference_engine->get_output_dimensions(output_width, output_height, output_channels)) {
                std::cerr << "    ✗ Failed to get output dimensions for frame " << processed_count << "\n";
                checkCudaErrors(cudaFree(d_temp_input));
                checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
                checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
                continue;
            }
            
            // Get inference output data
            float* cuda_output_data = g_inference_engine->get_output_data();
            
            // 添加调试代码：检查输出数据的一些样本值
            std::cout << "    → Checking output data samples..." << std::endl;
            
            // 创建一个小的主机缓冲区来检查输出数据
            const int sample_size = 16; // 检查前16个像素
            std::vector<float> sample_data(sample_size * output_channels);
            checkCudaErrors(cudaMemcpy(sample_data.data(), cuda_output_data, 
                                     sample_size * output_channels * sizeof(float), cudaMemcpyDeviceToHost));
            
            std::cout << "      - First " << sample_size << " pixels (RGBA):" << std::endl;
            for (int i = 0; i < sample_size && i < 4; ++i) {
                std::cout << "        Pixel " << i << ": ("
                         << sample_data[i*4] << ", " << sample_data[i*4+1] << ", "
                         << sample_data[i*4+2] << ", " << sample_data[i*4+3] << ")" << std::endl;
            }
            
            // 检查中间区域的一些像素（可能的白色区域）
            size_t middle_offset = (output_width * output_height / 2) * output_channels;
            std::vector<float> middle_sample(sample_size * output_channels);
            checkCudaErrors(cudaMemcpy(middle_sample.data(), cuda_output_data + middle_offset/sizeof(float), 
                                     sample_size * output_channels * sizeof(float), cudaMemcpyDeviceToHost));
            
            std::cout << "      - Middle region pixels (RGBA):" << std::endl;
            for (int i = 0; i < sample_size && i < 4; ++i) {
                std::cout << "        Pixel " << i << ": ("
                         << middle_sample[i*4] << ", " << middle_sample[i*4+1] << ", "
                         << middle_sample[i*4+2] << ", " << middle_sample[i*4+3] << ")" << std::endl;
            }
            
            // 检查输出数据中是否有大量的1.0值（白色）
            const int check_stride = output_width * output_height / 100; // 每1%检查一次
            std::vector<float> spot_check(output_channels);
            int white_pixel_count = 0;
            const int total_checks = 20;
            
            for (int i = 0; i < total_checks; ++i) {
                size_t offset = i * check_stride * output_channels;
                if (offset + output_channels <= output_width * output_height * output_channels) {
                    checkCudaErrors(cudaMemcpy(spot_check.data(), cuda_output_data + offset, 
                                             output_channels * sizeof(float), cudaMemcpyDeviceToHost));
                    
                    // 检查是否为白色像素（所有通道都接近1.0）
                    bool is_white = true;
                    for (int c = 0; c < 3; ++c) { // 只检查RGB，忽略Alpha
                        if (spot_check[c] < 0.9f) {
                            is_white = false;
                            break;
                        }
                    }
                    if (is_white) white_pixel_count++;
                }
            }
            
            std::cout << "      - White pixel ratio in spot check: " << white_pixel_count << "/" << total_checks 
                     << " (" << (white_pixel_count * 100.0f / total_checks) << "%)" << std::endl;
            
            std::cout << "    → Output dimensions: " << output_width << "x" << output_height << "x" << output_channels << "\n";
            
            // Create output D3D11 texture
            ID3D11Texture2D* d3d11_output_texture = nullptr;
            if (!create_output_d3d11_texture(output_width, output_height, d3d11_device, &d3d11_output_texture)) {
                std::cerr << "    ✗ Failed to create output D3D11 texture for frame " << processed_count << "\n";
                checkCudaErrors(cudaFree(d_temp_input));
                checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
                checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
                continue;
            }
            
            // Write CUDA output to D3D11 texture (synchronous)
            if (!write_cuda_output_to_d3d11(cuda_output_data, output_width, output_height, output_channels, 
                                           d3d11_output_texture, cuda_stream)) {
                std::cerr << "    ✗ Failed to write CUDA output to D3D11 texture for frame " << processed_count << "\n";
                d3d11_output_texture->Release();
                checkCudaErrors(cudaFree(d_temp_input));
                checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
                checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
                continue;
            }

            std::cout << ">>> Stereo inference frame " << processed_count << " completed\n";
            
            // Create output frame with the result texture
            D11Frame output_frame(d3d11_output_texture, output_width, output_height);
            output_frame_queue.push(std::move(output_frame));

            // Clean up
            d3d11_output_texture->Release(); // D11Frame constructor already AddRef'd
            checkCudaErrors(cudaFree(d_temp_input));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream));
            checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
        }
    }
    
    cudaStreamDestroy(cuda_stream);
    std::cout << "=== Depth Inference Thread Finished ===\n";
    std::cout << "Total frames processed for stereo inference: " << processed_count << "\n";
} 