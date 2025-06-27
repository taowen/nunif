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
#include <memory>

// TensorRT includes
#include "NvInfer.h"
#include "NvOnnxParser.h"

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// Add TensorRT logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// Global logger instance
static Logger gLogger;

// CUDA error checking helper
#define checkCudaErrors(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Add TensorRT inference engine class declaration
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
    
    bool infer(cudaArray_t cuda_array, UINT width, UINT height, cudaStream_t stream) {
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
        std::cout << "    → Actual output dimensions from context: [";
        for (int i = 0; i < output_dims_actual.nbDims; ++i) {
            std::cout << output_dims_actual.d[i] << (i == output_dims_actual.nbDims - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;

        size_t output_elements = 1;
        for (int i = 0; i < output_dims_actual.nbDims; ++i) {
            if (output_dims_actual.d[i] < 0) {
                 std::cerr << "    ✗ Invalid output dimension: " << output_dims_actual.d[i] << std::endl;
                 return false;
            }
            output_elements *= output_dims_actual.d[i];
        }
        output_size = output_elements * sizeof(float);

        // Extract actual dimensions from TensorRT output (NCHW format: [1, 4, H, W])
        UINT actual_output_height = static_cast<UINT>(output_dims_actual.d[2]);
        UINT actual_output_width = static_cast<UINT>(output_dims_actual.d[3]);
        UINT actual_output_channels = static_cast<UINT>(output_dims_actual.d[1]);
        
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
        
        // 直接从 CUDA array 复制到 TensorRT 输入缓冲区
        size_t pixel_size = 4 * sizeof(float); // RGBA float32
        checkCudaErrors(cudaMemcpy2DFromArrayAsync(
            d_input, width * pixel_size,
            cuda_array, 0, 0,
            width * pixel_size, height,
            cudaMemcpyDeviceToDevice, stream
        ));
        
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
        
        height = static_cast<UINT>(output_dims.d[2]);
        width = static_cast<UINT>(output_dims.d[3]); 
        channels = static_cast<UINT>(output_dims.d[1]);
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
    // Remove unused variable to fix warning
    // ID3D11DeviceContext* d3d11_context = hw_ctx.d3d_context;
    
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

TEST_CASE("D3D11 texture to TensorRT inference pipeline") {
    const char* input_file = "06 4k.mp4";
    const std::string onnx_model_path = "stereo_module_half_sbs.onnx";
    const int frame_index = 0;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    REQUIRE(hw_frame != nullptr);
    
    // Convert to RGBA texture
    ID3D11Texture2D* rgba_texture = convert_color(&hw_ctx, hw_frame);
    REQUIRE(rgba_texture != nullptr);
    
    // Get texture dimensions
    D3D11_TEXTURE2D_DESC texture_desc;
    rgba_texture->GetDesc(&texture_desc);
    
    // Initialize TensorRT engine
    TensorRTInferenceEngine trt_engine;
    REQUIRE(trt_engine.initialize(onnx_model_path));
    
    // Set up CUDA device for interop
    int cuda_device = -1;
    IDXGIDevice* dxgi_device = nullptr;
    HRESULT hr = hw_ctx.d3d_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
    REQUIRE(SUCCEEDED(hr));
    
    IDXGIAdapter* dxgi_adapter = nullptr;
    hr = dxgi_device->GetAdapter(&dxgi_adapter); 
    dxgi_device->Release();
    REQUIRE(SUCCEEDED(hr));
    
    cudaError_t cuda_status = cudaD3D11GetDevice(&cuda_device, dxgi_adapter);
    dxgi_adapter->Release();
    REQUIRE(cuda_status == cudaSuccess);
    
    cuda_status = cudaSetDevice(cuda_device);
    REQUIRE(cuda_status == cudaSuccess);
    
    // Register texture with CUDA
    cudaGraphicsResource_t cuda_resource = nullptr;
    cuda_status = cudaGraphicsD3D11RegisterResource(
        &cuda_resource, rgba_texture, cudaGraphicsRegisterFlagsNone);
    REQUIRE(cuda_status == cudaSuccess);
    
    // Create CUDA stream for async operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Map the CUDA graphics resource to get access to texture data
    cuda_status = cudaGraphicsMapResources(1, &cuda_resource, stream);
    REQUIRE(cuda_status == cudaSuccess);
    
    // Get mapped array from the texture
    cudaArray_t cuda_array;
    cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
    REQUIRE(cuda_status == cudaSuccess);
    
    // Unmap the graphics resource
    cuda_status = cudaGraphicsUnmapResources(1, &cuda_resource, stream);
    REQUIRE(cuda_status == cudaSuccess);
    
    // 直接使用 cuda_array 进行推理
    bool inference_success = trt_engine.infer(
        cuda_array, texture_desc.Width, texture_desc.Height, stream);
    REQUIRE(inference_success);
    
    // Get inference results
    float* output_data = trt_engine.get_output_data();
    REQUIRE(output_data != nullptr);
    
    // Verify output dimensions
    UINT out_width, out_height, out_channels;
    REQUIRE(trt_engine.get_output_dimensions(out_width, out_height, out_channels));
    
    std::cout << "Inference completed:" << std::endl;
    std::cout << "Input: " << texture_desc.Width << "x" << texture_desc.Height << "x4" << std::endl;
    std::cout << "Output: " << out_width << "x" << out_height << "x" << out_channels << std::endl;
    
    // Optional: Copy first few output values to verify
    std::vector<float> host_output(16);
    cuda_status = cudaMemcpy(host_output.data(), output_data, 
                            host_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    REQUIRE(cuda_status == cudaSuccess);
    
    std::cout << "Sample output values: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << host_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaStreamDestroy(stream);
    cudaGraphicsUnregisterResource(cuda_resource);
    rgba_texture->Release();
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
}