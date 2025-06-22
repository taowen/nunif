#pragma once

#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

// TensorRT processing context that holds all state
class ProcessorContext {
public:
    // TensorRT objects
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    // GPU memory buffers
    float* d_input_nv12 = nullptr;
    float* d_output_nv12 = nullptr;
    
    // Buffer sizes
    size_t input_size = 0;
    size_t output_size = 0;
    size_t max_input_size = 0;
    size_t max_output_size = 0;
    
    // Logger instance
    Logger logger;
    
    // Destructor to cleanup resources
    ~ProcessorContext() {
        cleanup();
    }
    
    // Cleanup method
    void cleanup() {
        cudaDeviceSynchronize();
        if (d_input_nv12) {
            cudaFree(d_input_nv12);
            d_input_nv12 = nullptr;
        }
        if (d_output_nv12) {
            cudaFree(d_output_nv12);
            d_output_nv12 = nullptr;
        }
        context.reset();
        engine.reset();
        runtime.reset();
    }
    
    // Set maximum buffer sizes
    void setMaxBufferSize(size_t max_input_mb, size_t max_output_mb) {
        max_input_size = max_input_mb * 1024 * 1024;
        max_output_size = max_output_mb * 1024 * 1024;
    }
};
