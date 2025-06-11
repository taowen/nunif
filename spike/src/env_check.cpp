#include <iostream>

#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
#endif

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <memory>
#endif

#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char* msg) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

#ifdef CUDNN_AVAILABLE
void checkCudnnError(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        std::cerr << "cuDNN Error: " << msg << " - " << cudnnGetErrorString(status) << std::endl;
        exit(1);
    }
}
#endif

#ifdef TENSORRT_AVAILABLE
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only print errors and warnings
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};
#endif

int main() {
    std::cout << "Starting CUDA/cuDNN/TensorRT Test!" << std::endl;
    
    // Test CUDA first
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        checkCudaError(cudaGetDeviceProperties(&prop, 0), "Failed to get device properties");
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    }
    
#ifdef CUDNN_AVAILABLE
    std::cout << "\nTesting cuDNN..." << std::endl;
    
    // Initialize cuDNN
    cudnnHandle_t cudnn;
    checkCudnnError(cudnnCreate(&cudnn), "Failed to create cuDNN handle");
    
    // Get cuDNN version
    size_t version = cudnnGetVersion();
    std::cout << "cuDNN Version: " << version << std::endl;
    
    // Cleanup cuDNN
    checkCudnnError(cudnnDestroy(cudnn), "Failed to destroy cuDNN handle");
    std::cout << "cuDNN test completed successfully!" << std::endl;
#else
    std::cout << "cuDNN not available - skipping cuDNN test" << std::endl;
#endif

#ifdef TENSORRT_AVAILABLE
    std::cout << "\nTesting TensorRT..." << std::endl;
    
    // Create logger
    Logger logger;
    
    // Create builder using unique_ptr (TensorRT 10+ uses smart pointers)
    std::unique_ptr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(logger)};
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return 1;
    }
    
    // Get TensorRT version
    std::cout << "TensorRT Version: " 
              << NV_TENSORRT_MAJOR << "." 
              << NV_TENSORRT_MINOR << "." 
              << NV_TENSORRT_PATCH << "." 
              << NV_TENSORRT_BUILD << std::endl;
    
    // Create network definition
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<nvinfer1::INetworkDefinition> network{builder->createNetworkV2(flag)};
    if (!network) {
        std::cerr << "Failed to create TensorRT network" << std::endl;
        return 1;
    }
    
    // Create builder config
    std::unique_ptr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config) {
        std::cerr << "Failed to create TensorRT builder config" << std::endl;
        return 1;
    }
    
    std::cout << "TensorRT initialization successful!" << std::endl;
    
    // Show platform capabilities
    std::cout << "Platform has fast FP16: " << (builder->platformHasFastFp16() ? "Yes" : "No") << std::endl;
    std::cout << "Platform has fast INT8: " << (builder->platformHasFastInt8() ? "Yes" : "No") << std::endl;
    
    // Show some basic TensorRT info (removed getMaxWorkspaceSize as it's deprecated in TensorRT 10+)
    std::cout << "TensorRT builder and config created successfully!" << std::endl;
    
    // Cleanup is automatic with unique_ptr
    std::cout << "TensorRT test completed successfully!" << std::endl;
#else
    std::cout << "TensorRT not available - skipping TensorRT test" << std::endl;
#endif
    
    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}