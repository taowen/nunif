#include <iostream>

#ifdef CUDNN_AVAILABLE
#include <cudnn.h>
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

int main() {
    std::cout << "Starting CUDA/cuDNN Test!" << std::endl;
    
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
    std::cout << "cuDNN not available - running CUDA-only test" << std::endl;
#endif
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}