#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <cassert>

// TensorRT headers
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

using namespace nvinfer1;

class IW3TensorRTInference {
private:
    std::unique_ptr<IRuntime> mRuntime;
    std::unique_ptr<ICudaEngine> mEngine;
    std::unique_ptr<IExecutionContext> mContext;
    
    // Model input/output info
    std::string mInputName = "input";
    std::string mOutputName = "half_sbs";
    Dims mInputDims;
    Dims mOutputDims;
    
    // CUDA memory
    void* mInputDeviceBuffer = nullptr;
    void* mOutputDeviceBuffer = nullptr;
    
    size_t mInputSize = 0;
    size_t mOutputSize = 0;
    
    class Logger : public ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    } mLogger;

public:
    IW3TensorRTInference() = default;
    
    ~IW3TensorRTInference() {
        cleanup();
    }
    
    bool initialize(const std::string& onnxModelPath, bool useFP16 = true) {
        // Generate TRT cache file path
        std::string trtCachePath = onnxModelPath;
        size_t dotPos = trtCachePath.find_last_of('.');
        if (dotPos != std::string::npos) {
            trtCachePath = trtCachePath.substr(0, dotPos);
        }
        trtCachePath += useFP16 ? "_fp16.trt" : "_fp32.trt";
        
        // Try to load cached TRT engine first
        if (loadCachedEngine(trtCachePath)) {
            std::cout << "Loaded cached TensorRT engine from: " << trtCachePath << std::endl;
            return setupInferenceContext();
        }
        
        std::cout << "Building TensorRT engine from ONNX model..." << std::endl;
        
        // Create builder
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(mLogger));
        if (!builder) {
            std::cerr << "Failed to create TensorRT builder" << std::endl;
            return false;
        }
        
        // Create network
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        if (!network) {
            std::cerr << "Failed to create TensorRT network" << std::endl;
            return false;
        }
        
        // Create ONNX parser
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
        if (!parser) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }
        
        // Parse ONNX model
        if (!parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX model: " << onnxModelPath << std::endl;
            return false;
        }
        
        // Create builder config
        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "Failed to create builder config" << std::endl;
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
        
        // Enable FP16 if requested
        if (useFP16 && builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
            std::cout << "Using FP16 precision" << std::endl;
        }
        
        // Set memory pool size
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1U << 30); // 1GB
        
        // Build engine
        auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            return false;
        }
        
        // Save the engine to cache file
        saveCachedEngine(trtCachePath, serializedEngine->data(), serializedEngine->size());
        
        // Create runtime and deserialize engine
        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(mLogger));
        if (!mRuntime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        mEngine = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(
            serializedEngine->data(), serializedEngine->size()));
        if (!mEngine) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        return setupInferenceContext();
    }
    
    /**
     * Stereo depth inference using direct CUDA tensors
     * 
     * @param inputDevicePtr Input CUDA device pointer
     *        Format: RGBA (4 channels)
     *        Data type: float32
     *        Value range: [0.0, 1.0] 
     *        Layout: NCHW (batch, channel, height, width)
     *        Shape: (1, 4, height, width) - Fixed batch size of 1
     *        Color space: RGB with alpha channel
     * 
     * @param outputDevicePtr Output CUDA device pointer  
     *        Format: RGBA (4 channels)
     *        Data type: float32
     *        Value range: [0.0, 1.0]
     *        Layout: NCHW (batch, channel, height, width)
     *        Shape: (1, 4, height, width/2) - Half side-by-side stereo, batch size 1
     *        Color space: RGB with alpha channel
     *        Content: Left eye view concatenated with right eye view horizontally
     * 
     * @param height Image height in pixels
     * @param width Image width in pixels
     * @return true on success, false on failure
     */
    bool infer(void* inputDevicePtr, void* outputDevicePtr, int height, int width) {
        if (!mEngine || !mContext) {
            std::cerr << "Model not initialized" << std::endl;
            return false;
        }
        
        // Set dynamic input dimensions
        nvinfer1::Dims4 input_shape{1, 4, static_cast<int>(height), static_cast<int>(width)};
        if (!mContext->setInputShape(mEngine->getIOTensorName(0), input_shape)) {
            std::cerr << "Failed to set input shape" << std::endl;
            return false;
        }
        
        // Calculate sizes
        size_t input_size = 1 * 4 * height * width * sizeof(float);
        
        // Get output dimensions from the context after setting input shape
        nvinfer1::Dims output_dims_actual = mContext->getTensorShape(mEngine->getIOTensorName(1));
        
        // Add detailed output dimension checking
        uint32_t actual_output_channels = static_cast<uint32_t>(output_dims_actual.d[1]);
        uint32_t actual_output_height = static_cast<uint32_t>(output_dims_actual.d[2]);
        uint32_t actual_output_width = static_cast<uint32_t>(output_dims_actual.d[3]);
        
        std::cout << "    → Input dimensions: " << width << "x" << height << std::endl;
        std::cout << "    → Output dimensions: " << actual_output_width << "x" << actual_output_height << "x" << actual_output_channels << std::endl;
        
        // Check if output dimensions match input dimensions
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
        size_t output_size = output_elements * sizeof(float);

        // Allocate input device memory if needed
        if (mInputSize < input_size) {
            if (mInputDeviceBuffer) {
                cudaFree(mInputDeviceBuffer);
            }
            cudaError_t err = cudaMalloc(&mInputDeviceBuffer, input_size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate input buffer: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            mInputSize = input_size;
            std::cout << "    → Allocated input buffer: " << input_size << " bytes\n";
        }
        
        // Allocate output device memory if needed
        if (mOutputSize < output_size) {
            if (mOutputDeviceBuffer) {
                cudaFree(mOutputDeviceBuffer);
            }
            cudaError_t err = cudaMalloc(&mOutputDeviceBuffer, output_size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate output buffer: " << cudaGetErrorString(err) << std::endl;
                return false;
            }
            mOutputSize = output_size;
            std::cout << "    → Allocated output buffer: " << output_size << " bytes\n";
        }
        
        // Copy input data to device buffer
        cudaError_t err = cudaMemcpy(mInputDeviceBuffer, inputDevicePtr, input_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy input data: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        void* bindings[2];
        bindings[0] = mInputDeviceBuffer;  // input tensor
        bindings[1] = mOutputDeviceBuffer; // output tensor
        
        // Execute inference
        if (!mContext->executeV2(bindings)) {
            std::cerr << "Inference execution failed" << std::endl;
            return false;
        }
        
        // Copy output data back to the provided output buffer
        err = cudaMemcpy(outputDevicePtr, mOutputDeviceBuffer, output_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy output data: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        return true;
    }

private:
    bool loadCachedEngine(const std::string& trtCachePath) {
        std::ifstream file(trtCachePath, std::ios::binary);
        if (!file.good()) {
            return false;
        }
        
        // Get file size
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        if (size == 0) {
            return false;
        }
        
        // Read engine data
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();
        
        // Create runtime
        mRuntime = std::unique_ptr<IRuntime>(createInferRuntime(mLogger));
        if (!mRuntime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        // Deserialize engine
        mEngine = std::unique_ptr<ICudaEngine>(mRuntime->deserializeCudaEngine(
            engineData.data(), size));
        if (!mEngine) {
            std::cerr << "Failed to deserialize cached TensorRT engine" << std::endl;
            return false;
        }
        
        return true;
    }
    
    void saveCachedEngine(const std::string& trtCachePath, const void* engineData, size_t size) {
        std::ofstream file(trtCachePath, std::ios::binary);
        if (file.good()) {
            file.write(static_cast<const char*>(engineData), size);
            file.close();
            std::cout << "Saved TensorRT engine cache to: " << trtCachePath << std::endl;
        } else {
            std::cerr << "Failed to save TensorRT engine cache to: " << trtCachePath << std::endl;
        }
    }
    
    bool setupInferenceContext() {
        // Create execution context
        mContext = std::unique_ptr<IExecutionContext>(mEngine->createExecutionContext());
        if (!mContext) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Get input/output dimensions
        mInputDims = mEngine->getTensorShape(mInputName.c_str());
        mOutputDims = mEngine->getTensorShape(mOutputName.c_str());
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Input dimensions: ";
        for (int i = 0; i < mInputDims.nbDims; ++i) {
            std::cout << mInputDims.d[i] << " ";
        }
        std::cout << std::endl;
        
        return true;
    }
    
    void cleanup() {
        if (mInputDeviceBuffer) {
            cudaFree(mInputDeviceBuffer);
            mInputDeviceBuffer = nullptr;
        }
        if (mOutputDeviceBuffer) {
            cudaFree(mOutputDeviceBuffer);
            mOutputDeviceBuffer = nullptr;
        }
        mInputSize = 0;
        mOutputSize = 0;
    }
};

/**
 * Main inference function with CUDA tensor interface
 * 
 * Input requirements (matching export_iw3.py ONNX model):
 * - Format: RGBA (4 channels) 
 * - Data type: float32
 * - Value range: [0.0, 1.0]
 * - Layout: NCHW (batch, channel, height, width)
 * - Shape: (1, 4, height, width) - Fixed batch size of 1
 * - Color space: RGB with alpha channel
 * 
 * Output format:
 * - Format: RGBA (4 channels)
 * - Data type: float32  
 * - Value range: [0.0, 1.0]
 * - Layout: NCHW (batch, channel, height, width/2)
 * - Shape: (1, 4, height, width/2) - Fixed batch size of 1
 * - Content: Half side-by-side stereo (left eye | right eye)
 * - Color space: RGB with alpha channel
 */
bool inferIW3(void* inputDevicePtr, void* outputDevicePtr, int height, int width) {
    IW3TensorRTInference inferencer;
    
    // Initialize the model
    if (!inferencer.initialize("stereo_module_half_sbs.onnx", true)) {
        std::cerr << "Failed to initialize IW3 TensorRT inference" << std::endl;
        return false;
    }
    
    // Run inference with direct CUDA tensors (batch size fixed to 1)
    if (!inferencer.infer(inputDevicePtr, outputDevicePtr, height, width)) {
        std::cerr << "Inference failed" << std::endl;
        return false;
    }
    
    std::cout << "Inference completed successfully!" << std::endl;
    
    return true;
}