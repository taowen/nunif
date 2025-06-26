#include "infer_depth_thread.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <memory>
#include <fstream>

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

extern void checkCudaErrors(cudaError_t result);

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
    
    bool infer(float* input_data, UINT width, UINT height, cudaStream_t stream) {
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
        
        // Copy input data to device buffer (修复：使用 cudaMemcpy 而不是直接赋值指针)
        checkCudaErrors(cudaMemcpyAsync(d_input, input_data, input_size, cudaMemcpyDeviceToDevice, stream));
        
        // Set tensor addresses
        if (!context->setTensorAddress(engine->getIOTensorName(0), d_input)) {
            std::cerr << "Failed to set input tensor address" << std::endl;
            return false;
        }
        if (!context->setTensorAddress(engine->getIOTensorName(1), d_output)) {
            std::cerr << "Failed to set output tensor address" << std::endl;
            return false;
        }
        
        // Execute inference
        if (!context->enqueueV3(stream)) {
            std::cerr << "TensorRT inference failed" << std::endl;
            return false;
        }
        
        // Wait for completion
        checkCudaErrors(cudaStreamSynchronize(stream));
        
        std::cout << "    ✓ TensorRT inference completed" << std::endl;
        std::cout << "    → Input size: " << input_size << " bytes" << std::endl;
        std::cout << "    → Output size: " << output_size << " bytes" << std::endl;
        
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

} // anonymous namespace

void start_infer_depth_thread(
    ColorConvertedFrameQueue& input_frame_queue,
    cudaStream_t cuda_stream) {
    
    std::cout << "=== Depth Inference Thread Started ===\n";
    std::cout << "Using CUDA stream: " << cuda_stream << "\n";
    
    // 检查 CUDA stream 是否有效
    if (cuda_stream == nullptr) {
        std::cerr << "Invalid CUDA stream provided to depth inference thread" << std::endl;
        return;
    }
    
    // Initialize TensorRT inference engine
    if (!g_inference_engine) {
        g_inference_engine = std::make_unique<TensorRTInferenceEngine>();
        if (!g_inference_engine->initialize("stereo_module_half_sbs.onnx")) {
            std::cerr << "Failed to initialize TensorRT inference engine" << std::endl;
            return;
        }
    }
    
    int processed_count = 0;
    
    while (true) {
        ColorConvertedFrame converted_frame = input_frame_queue.pop();
        
        // Check for end signal
        if (converted_frame.is_end_signal) {
            std::cout << "=== Depth Inference Thread Received End Signal ===\n";
            break;
        }
        
        if (converted_frame.cuda_data) {
            processed_count++;
            std::cout << ">>> Processing frame " << processed_count << " for stereo inference\n";
            std::cout << "    → Frame dimensions: " << converted_frame.width << "x" << converted_frame.height << "\n";
            std::cout << "    → Frame pitch: " << converted_frame.pitch << " bytes\n";
            
            // Validate input data
            if (converted_frame.width == 0 || converted_frame.height == 0) {
                std::cerr << "    ✗ Invalid frame dimensions for frame " << processed_count << "\n";
                continue;
            }
            
            // Execute TensorRT inference
            if (!g_inference_engine->infer(converted_frame.cuda_data, converted_frame.width, 
                                          converted_frame.height, cuda_stream)) {
                std::cerr << "    ✗ TensorRT inference failed for frame " << processed_count << "\n";
                continue;
            }
            
            std::cout << ">>> Stereo inference frame " << processed_count << " completed\n";
        }
    }
    
    std::cout << "=== Depth Inference Thread Finished ===\n";
    std::cout << "Total frames processed for stereo inference: " << processed_count << "\n";
} 