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
        
        const int batchSize = 1;        // Fixed batch size
        const int inputChannels = 4;   // RGBA input
        const int outputChannels = 4;  // RGBA output
        
        // Set dynamic dimensions
        Dims inputDims{4, {batchSize, inputChannels, height, width}};
        Dims outputDims{4, {batchSize, outputChannels, height, width}};
        
        if (!mContext->setInputShape(mInputName.c_str(), inputDims)) {
            std::cerr << "Failed to set input shape" << std::endl;
            return false;
        }
        
        // Calculate buffer sizes
        mInputSize = batchSize * inputChannels * height * width * sizeof(float);
        mOutputSize = batchSize * outputChannels * height * width * sizeof(float);
        
        // Use provided device pointers directly
        mInputDeviceBuffer = inputDevicePtr;
        mOutputDeviceBuffer = outputDevicePtr;
        
        // Set tensor addresses
        mContext->setTensorAddress(mInputName.c_str(), mInputDeviceBuffer);
        mContext->setTensorAddress(mOutputName.c_str(), mOutputDeviceBuffer);
        
        // Execute inference
        if (!mContext->executeV2(nullptr)) {
            std::cerr << "Inference execution failed" << std::endl;
            return false;
        }
        
        return true;
    }

private:
    void cleanup() {
        // Note: We don't free mInputDeviceBuffer and mOutputDeviceBuffer 
        // as they are managed externally
        mInputDeviceBuffer = nullptr;
        mOutputDeviceBuffer = nullptr;
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
    if (!inferencer.initialize("stereo_module_half_sbs", true)) {
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

// Example usage for testing (can be removed in production)
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <onnx_model_path> <height> <width>" << std::endl;
        std::cout << "Note: This example allocates dummy CUDA tensors for testing (batch size fixed to 1)" << std::endl;
        return -1;
    }
    
    std::string onnxModelPath = argv[1];
    int height = std::stoi(argv[2]);
    int width = std::stoi(argv[3]);
    
    // Allocate CUDA memory for testing (batch size = 1)
    const int batchSize = 1;
    const int inputChannels = 4;
    const int outputChannels = 4;
    const int outputWidth = width / 2;
    
    size_t inputSize = batchSize * inputChannels * height * width * sizeof(float);
    size_t outputSize = batchSize * outputChannels * height * outputWidth * sizeof(float);
    
    void* inputDevicePtr = nullptr;
    void* outputDevicePtr = nullptr;
    
    if (cudaMalloc(&inputDevicePtr, inputSize) != cudaSuccess) {
        std::cerr << "Failed to allocate input CUDA memory" << std::endl;
        return -1;
    }
    
    if (cudaMalloc(&outputDevicePtr, outputSize) != cudaSuccess) {
        std::cerr << "Failed to allocate output CUDA memory" << std::endl;
        cudaFree(inputDevicePtr);
        return -1;
    }
    
    // Initialize input with dummy data (should be replaced with actual image data)
    cudaMemset(inputDevicePtr, 0, inputSize);
    
    // Run inference (batch size now fixed to 1)
    bool success = inferIW3(inputDevicePtr, outputDevicePtr, height, width);
    
    // Cleanup
    cudaFree(inputDevicePtr);
    cudaFree(outputDevicePtr);
    
    return success ? 0 : -1;
}
