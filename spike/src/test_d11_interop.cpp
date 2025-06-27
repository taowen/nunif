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
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

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

// Add BMP saving function
bool save_rgba_as_bmp(const std::string& filename, const float* rgba_data, 
                      UINT width, UINT height) {
    // BMP header structures
    #pragma pack(push, 1)
    struct BMPFileHeader {
        uint16_t signature = 0x4D42; // "BM"
        uint32_t file_size;
        uint16_t reserved1 = 0;
        uint16_t reserved2 = 0;
        uint32_t data_offset = 54; // Size of headers
    };
    
    struct BMPInfoHeader {
        uint32_t header_size = 40;
        int32_t width;
        int32_t height;
        uint16_t planes = 1;
        uint16_t bits_per_pixel = 24; // RGB (no alpha for BMP)
        uint32_t compression = 0;
        uint32_t image_size = 0;
        int32_t x_pixels_per_meter = 0;
        int32_t y_pixels_per_meter = 0;
        uint32_t colors_used = 0;
        uint32_t colors_important = 0;
    };
    #pragma pack(pop)
    
    // Calculate padding for 4-byte alignment
    UINT bytes_per_row = width * 3; // RGB
    UINT padding = (4 - (bytes_per_row % 4)) % 4;
    UINT padded_row_size = bytes_per_row + padding;
    
    BMPFileHeader file_header;
    BMPInfoHeader info_header;
    
    file_header.file_size = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + 
                           padded_row_size * height;
    info_header.width = static_cast<int32_t>(width);
    info_header.height = static_cast<int32_t>(height);
    info_header.image_size = padded_row_size * height;
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to create BMP file: " << filename << std::endl;
        return false;
    }
    
    // Write headers
    file.write(reinterpret_cast<const char*>(&file_header), sizeof(file_header));
    file.write(reinterpret_cast<const char*>(&info_header), sizeof(info_header));
    
    // Convert RGBA float to RGB bytes and write (BMP is bottom-up)
    std::vector<uint8_t> row_data(padded_row_size, 0);
    
    for (int y = height - 1; y >= 0; --y) { // BMP is stored bottom-up
        for (UINT x = 0; x < width; ++x) {
            const float* pixel = &rgba_data[(y * width + x) * 4];
            
            // Convert float [0,1] to uint8 [0,255] and BGR format for BMP
            row_data[x * 3 + 0] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, pixel[2] * 255.0f))); // B
            row_data[x * 3 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, pixel[1] * 255.0f))); // G
            row_data[x * 3 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, pixel[0] * 255.0f))); // R
        }
        file.write(reinterpret_cast<const char*>(row_data.data()), padded_row_size);
    }
    
    return true;
}

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
        
        // 诊断：初始化输入缓冲区为已知值进行测试
        std::cout << "    → Initializing input buffer with test pattern..." << std::endl;
        checkCudaErrors(cudaMemset(d_input, 0, input_size));
        
        // 诊断：检查 CUDA array 的有效性
        std::cout << "    → Verifying CUDA array validity..." << std::endl;
        cudaArray_t test_array = cuda_array;
        if (test_array == nullptr) {
            std::cerr << "    ✗ CUDA array is null!" << std::endl;
            return false;
        }
        
        // 诊断：获取 CUDA array 的描述信息
        std::cout << "    → Getting CUDA array information..." << std::endl;
        struct cudaChannelFormatDesc format_desc;
        struct cudaExtent extent;
        unsigned int flags;
        
        cudaError_t info_result = cudaArrayGetInfo(&format_desc, &extent, &flags, test_array);
        if (info_result == cudaSuccess) {
            std::cout << "    → CUDA array info:" << std::endl;
            std::cout << "      - Width: " << extent.width << std::endl;
            std::cout << "      - Height: " << extent.height << std::endl; 
            std::cout << "      - Depth: " << extent.depth << std::endl;
            std::cout << "      - Format: x=" << format_desc.x << " y=" << format_desc.y 
                      << " z=" << format_desc.z << " w=" << format_desc.w << std::endl;
            std::cout << "      - Format kind: " << format_desc.f << std::endl;
            std::cout << "      - Flags: 0x" << std::hex << flags << std::dec << std::endl;
            
            // 检查格式是否匹配预期的 RGBA float32
            if (format_desc.x == 32 && format_desc.y == 32 && format_desc.z == 32 && format_desc.w == 32 &&
                format_desc.f == cudaChannelFormatKindFloat) {
                std::cout << "      ✓ Format matches expected RGBA float32" << std::endl;
            } else {
                std::cout << "      ⚠ Format does NOT match expected RGBA float32" << std::endl;
            }
            
            // 检查尺寸是否匹配
            if (extent.width == width && extent.height == height) {
                std::cout << "      ✓ Dimensions match expected " << width << "x" << height << std::endl;
            } else {
                std::cout << "      ⚠ Dimensions mismatch: expected " << width << "x" << height 
                          << ", got " << extent.width << "x" << extent.height << std::endl;
            }
        } else {
            std::cout << "    ⚠ Could not get CUDA array info: " << cudaGetErrorString(info_result) << std::endl;
        }
        
        // 诊断：在复制之前，测试目标缓冲区是否可写
        std::cout << "    → Testing input buffer accessibility..." << std::endl;
        float test_value = 1.234f;
        checkCudaErrors(cudaMemcpy(d_input, &test_value, sizeof(float), cudaMemcpyHostToDevice));
        
        float readback_value = 0.0f;
        checkCudaErrors(cudaMemcpy(&readback_value, d_input, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "    → Buffer test: wrote " << test_value << ", read " << readback_value << std::endl;
        
        // 直接从 CUDA array 复制到 TensorRT 输入缓冲区
        size_t pixel_size = 4 * sizeof(float); // RGBA float32
        std::cout << "    → Starting cudaMemcpy2DFromArray..." << std::endl;
        std::cout << "      - Source: CUDA array" << std::endl;
        std::cout << "      - Destination: d_input (0x" << std::hex << reinterpret_cast<uintptr_t>(d_input) << std::dec << ")" << std::endl;
        std::cout << "      - Width bytes: " << width * pixel_size << std::endl;
        std::cout << "      - Height: " << height << std::endl;
        std::cout << "      - Pitch: " << width * pixel_size << std::endl;
        
        cudaError_t copy_result = cudaMemcpy2DFromArray(
            d_input, width * pixel_size,
            cuda_array, 0, 0,
            width * pixel_size, height,
            cudaMemcpyDeviceToDevice
        );
        
        if (copy_result != cudaSuccess) {
            std::cerr << "    ✗ cudaMemcpy2DFromArray failed: " << cudaGetErrorString(copy_result) << std::endl;
            return false;
        }
        std::cout << "    ✓ cudaMemcpy2DFromArray completed" << std::endl;
        
        // 诊断：立即检查复制后的数据
        std::cout << "    → Verifying copied data..." << std::endl;
        std::vector<float> verify_sample(16); // 前4个像素
        checkCudaErrors(cudaMemcpy(verify_sample.data(), d_input, 
                                   verify_sample.size() * sizeof(float), cudaMemcpyDeviceToHost));
        
        bool has_non_zero = false;
        float min_val = verify_sample[0], max_val = verify_sample[0];
        for (size_t i = 0; i < verify_sample.size(); ++i) {
            if (verify_sample[i] != 0.0f) has_non_zero = true;
            min_val = std::min(min_val, verify_sample[i]);
            max_val = std::max(max_val, verify_sample[i]);
        }
        
        std::cout << "    → Sample verification:" << std::endl;
        std::cout << "      - Has non-zero values: " << (has_non_zero ? "YES" : "NO") << std::endl;
        std::cout << "      - Value range: [" << min_val << ", " << max_val << "]" << std::endl;
        std::cout << "      - First 4 values: " << verify_sample[0] << ", " << verify_sample[1] 
                  << ", " << verify_sample[2] << ", " << verify_sample[3] << std::endl;
        
        if (!has_non_zero) {
            std::cerr << "    ✗ WARNING: All sample values are zero - data may not have been copied correctly!" << std::endl;
        }
        
        // Save TensorRT input as BMP for debugging
        std::vector<float> host_input(width * height * 4);
        checkCudaErrors(cudaMemcpy(host_input.data(), d_input, 
                                   host_input.size() * sizeof(float), 
                                   cudaMemcpyDeviceToHost));
        
        // 诊断：检查完整数据统计
        size_t zero_count = 0;
        size_t total_count = host_input.size();
        float data_min = host_input[0], data_max = host_input[0];
        double data_sum = 0.0;
        
        for (size_t i = 0; i < total_count; ++i) {
            float val = host_input[i];
            if (val == 0.0f) zero_count++;
            data_min = std::min(data_min, val);
            data_max = std::max(data_max, val);
            data_sum += val;
        }
        
        double data_avg = data_sum / total_count;
        std::cout << "    → Complete data statistics:" << std::endl;
        std::cout << "      - Total elements: " << total_count << std::endl;
        std::cout << "      - Zero elements: " << zero_count << " (" << (100.0 * zero_count / total_count) << "%)" << std::endl;
        std::cout << "      - Data range: [" << data_min << ", " << data_max << "]" << std::endl;
        std::cout << "      - Average value: " << data_avg << std::endl;
        
        // Generate filename with timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "tensorrt_input_" << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S") 
           << "_" << width << "x" << height << ".bmp";
        
        if (save_rgba_as_bmp(ss.str(), host_input.data(), width, height)) {
            std::cout << "    ✓ TensorRT input saved as: " << ss.str() << std::endl;
        } else {
            std::cout << "    ✗ Failed to save TensorRT input as BMP" << std::endl;
        }
        
        // 如果数据全是零，不继续推理
        if (zero_count == total_count) {
            std::cerr << "    ✗ ABORT: All input data is zero, skipping inference!" << std::endl;
            return false;
        }
        
        // Prepare bindings for executeV2
        void* bindings[2];
        bindings[0] = d_input;  // input tensor
        bindings[1] = d_output; // output tensor
        
        std::cout << "    → Starting TensorRT inference..." << std::endl;
        
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
        
        // 诊断：保存CUDA array直接读取的数据作为对比
        std::cout << "    → Creating comparison data from CUDA array..." << std::endl;
        std::vector<float> cuda_array_data(width * height * 4);
        checkCudaErrors(cudaMemcpy2DFromArray(
            cuda_array_data.data(), width * pixel_size,
            cuda_array, 0, 0,
            width * pixel_size, height,
            cudaMemcpyDeviceToHost
        ));
        
        // 保存CUDA array数据作为对比
        auto now_ = std::chrono::system_clock::now();
        auto time_t_ = std::chrono::system_clock::to_time_t(now_);
        std::stringstream cuda_ss;
        cuda_ss << "cuda_array_direct_" << std::put_time(std::localtime(&time_t_), "%Y%m%d_%H%M%S") 
                << "_" << width << "x" << height << ".bmp";
        
        if (save_rgba_as_bmp(cuda_ss.str(), cuda_array_data.data(), width, height)) {
            std::cout << "    ✓ CUDA array direct data saved as: " << cuda_ss.str() << std::endl;
        }
        
        // 计算CUDA array数据统计
        size_t cuda_zero_count = 0;
        float cuda_min_ = cuda_array_data[0], cuda_max_ = cuda_array_data[0];
        double cuda_sum_ = 0.0;
        
        for (size_t i = 0; i < cuda_array_data.size(); ++i) {
            float val = cuda_array_data[i];
            if (val == 0.0f) cuda_zero_count++;
            cuda_min_ = std::min(cuda_min_, val);
            cuda_max_ = std::max(cuda_max_, val);
            cuda_sum_ += val;
        }
        
        double cuda_avg_ = cuda_sum_ / cuda_array_data.size();
        std::cout << "    → CUDA array direct statistics:" << std::endl;
        std::cout << "      - Zero elements: " << cuda_zero_count << " (" << (100.0 * cuda_zero_count / cuda_array_data.size()) << "%)" << std::endl;
        std::cout << "      - Data range: [" << cuda_min_ << ", " << cuda_max_ << "]" << std::endl;
        std::cout << "      - Average value: " << cuda_avg_ << std::endl;
        
        // 比较CUDA array和TensorRT输入缓冲区的前几个像素
        std::cout << "    → Comparing first 4 pixels:" << std::endl;
        for (int pixel = 0; pixel < 4; ++pixel) {
            int offset = pixel * 4;
            std::cout << "      Pixel " << pixel << ":" << std::endl;
            std::cout << "        CUDA array:    R=" << cuda_array_data[offset+0] << " G=" << cuda_array_data[offset+1] 
                      << " B=" << cuda_array_data[offset+2] << " A=" << cuda_array_data[offset+3] << std::endl;
            std::cout << "        TensorRT buf:  R=" << host_input[offset+0] << " G=" << host_input[offset+1] 
                      << " B=" << host_input[offset+2] << " A=" << host_input[offset+3] << std::endl;
            
            // 检查是否完全一致
            bool identical = true;
            for (int c = 0; c < 4; ++c) {
                if (std::abs(cuda_array_data[offset+c] - host_input[offset+c]) > 1e-6f) {
                    identical = false;
                    break;
                }
            }
            std::cout << "        Match: " << (identical ? "✓" : "✗") << std::endl;
        }
        
        // 检查整体数据一致性
        size_t mismatch_count = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < cuda_array_data.size(); ++i) {
            float diff = std::abs(cuda_array_data[i] - host_input[i]);
            if (diff > 1e-6f) {
                mismatch_count++;
                max_diff = std::max(max_diff, diff);
            }
        }
        
        std::cout << "    → Overall data comparison:" << std::endl;
        std::cout << "      - Mismatched elements: " << mismatch_count << " (" << (100.0 * mismatch_count / cuda_array_data.size()) << "%)" << std::endl;
        std::cout << "      - Maximum difference: " << max_diff << std::endl;
        
        if (mismatch_count == 0) {
            std::cout << "      ✓ CUDA array and TensorRT buffer data are identical" << std::endl;
        } else {
            std::cout << "      ✗ CUDA array and TensorRT buffer data differ!" << std::endl;
        }
        
        // 诊断：检查TensorRT输入的内存布局是否正确
        std::cout << "    → Analyzing TensorRT input layout..." << std::endl;
        
        // 检查是否是NCHW格式 (TensorRT期望的格式)
        // 对于NCHW: [batch=1][channel=4][height][width]
        // 计算几个关键位置的索引来验证布局
        size_t pixel_stride = 4; // RGBA interleaved
        size_t row_stride = width * pixel_stride;
        
        // 中心像素位置
        size_t center_y = height / 2;
        size_t center_x = width / 2;
        size_t center_offset = center_y * row_stride + center_x * pixel_stride;
        
        if (center_offset + 3 < host_input.size()) {
            std::cout << "      - Center pixel (" << center_x << "," << center_y << "): R=" << host_input[center_offset+0] 
                      << " G=" << host_input[center_offset+1] << " B=" << host_input[center_offset+2] << " A=" << host_input[center_offset+3] << std::endl;
        }
        
        // 检查alpha通道是否合理（大多数应该是1.0）
        size_t alpha_ones = 0;
        for (size_t i = 3; i < host_input.size(); i += 4) {
            if (std::abs(host_input[i] - 1.0f) < 0.01f) {
                alpha_ones++;
            }
        }
        size_t total_pixels = host_input.size() / 4;
        std::cout << "      - Alpha channel analysis: " << alpha_ones << "/" << total_pixels 
                  << " pixels have alpha ≈ 1.0 (" << (100.0 * alpha_ones / total_pixels) << "%)" << std::endl;
        
        // 如果大部分alpha不是1.0，可能数据格式有问题
        if (alpha_ones < total_pixels * 0.8) {
            std::cout << "      ⚠ WARNING: Low alpha=1.0 ratio suggests possible format issue" << std::endl;
        }
        
        // 保存完整数据后，也进行统计分析
        size_t rgba_zero_count = 0;
        size_t rgba_total_count = host_input.size();
        float rgba_data_min = host_input[0], rgba_data_max = host_input[0];
        double rgba_data_sum = 0.0;
        
        for (size_t i = 0; i < rgba_total_count; ++i) {
            float val = host_input[i];
            if (val == 0.0f) rgba_zero_count++;
            rgba_data_min = std::min(rgba_data_min, val);
            rgba_data_max = std::max(rgba_data_max, val);
            rgba_data_sum += val;
        }
        
        double rgba_data_avg = rgba_data_sum / rgba_total_count;
        std::cout << "Original RGBA texture statistics:" << std::endl;
        std::cout << "  - Total elements: " << rgba_total_count << std::endl;
        std::cout << "  - Zero elements: " << rgba_zero_count << " (" << (100.0 * rgba_zero_count / rgba_total_count) << "%)" << std::endl;
        std::cout << "  - Data range: [" << rgba_data_min << ", " << rgba_data_max << "]" << std::endl;
        std::cout << "  - Average value: " << rgba_data_avg << std::endl;
        std::cout << "  - First 4 pixel values:" << std::endl;
        for (int pixel = 0; pixel < 4; ++pixel) {
            int offset = pixel * 4;
            if (offset + 3 < host_input.size()) {
                std::cout << "    P" << pixel << ": R=" << host_input[offset+0] << " G=" << host_input[offset+1] 
                          << " B=" << host_input[offset+2] << " A=" << host_input[offset+3] << std::endl;
            }
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
        
        nvinfer1::Dims actual_output_dims = context->getTensorShape(engine->getIOTensorName(1));
        if (actual_output_dims.nbDims != 4) {
            return false;
        }
        
        height = static_cast<UINT>(actual_output_dims.d[2]);
        width = static_cast<UINT>(actual_output_dims.d[3]); 
        channels = static_cast<UINT>(actual_output_dims.d[1]);
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
    const int frame_index = 650;
    
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
    
    // Initialize TensorRT engine
    std::cout << "\n=== TensorRT Initialization ===" << std::endl;
    TensorRTInferenceEngine trt_engine;
    REQUIRE(trt_engine.initialize("stereo_module_half_sbs.onnx"));

    // Try to register the texture with CUDA
    cudaGraphicsResource_t cuda_resource = nullptr;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &cuda_resource, rgba_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status == cudaSuccess) {
        std::cout << "Successfully registered D3D11 texture with CUDA" << std::endl;
        
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        
        cuda_status = cudaGraphicsMapResources(1, &cuda_resource, stream);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Get mapped array
        cudaArray_t cuda_array;
        cuda_status = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource, 0, 0);
        REQUIRE(cuda_status == cudaSuccess);
        
        // Perform inference using the mapped CUDA array
        std::cout << "\n=== Starting TensorRT Inference ===" << std::endl;
        bool inference_success = trt_engine.infer(
            cuda_array, texture_desc.Width, texture_desc.Height, stream);
        REQUIRE(inference_success);

        // Inference is done, now we can unmap the resource
        cuda_status = cudaGraphicsUnmapResources(1, &cuda_resource, stream);
        REQUIRE(cuda_status == cudaSuccess);

        // Get inference results and verify
        float* output_data = trt_engine.get_output_data();
        REQUIRE(output_data != nullptr);

        UINT out_width, out_height, out_channels;
        REQUIRE(trt_engine.get_output_dimensions(out_width, out_height, out_channels));

        std::cout << "\n=== Inference Results ===" << std::endl;
        std::cout << "Input: " << texture_desc.Width << "x" << texture_desc.Height << "x4" << std::endl;
        std::cout << "Output: " << out_width << "x" << out_height << "x" << out_channels << std::endl;

        // Optional: Copy a sample of output data to host and print
        std::vector<float> host_output_sample(16);
        cudaMemcpy(host_output_sample.data(), output_data,
                   host_output_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "Sample output values: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << host_output_sample[i] << " ";
        }
        std::cout << std::endl;

        // Cleanup
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