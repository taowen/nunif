#include "infer_depth_thread.h"
#include <iostream>
#include <cuda_runtime_api.h>
#include <algorithm>

extern void checkCudaErrors(cudaError_t result);

namespace {

bool verify_cuda_data(float* cuda_data, UINT width, UINT height, size_t pitch, cudaStream_t cuda_stream) {
    // 验证 CUDA 数据的可用性
    std::cout << "  → Verifying CUDA data accessibility...\n";
    
    // 分配主机内存来检查数据
    size_t data_size = height * pitch;
    float* host_data = new float[data_size / sizeof(float)];
    
    // 将 CUDA 数据复制到主机内存进行验证
    cudaError_t result = cudaMemcpyAsync(host_data, cuda_data, data_size, 
                                        cudaMemcpyDeviceToHost, cuda_stream);
    if (result != cudaSuccess) {
        std::cerr << "    ✗ Failed to copy CUDA data to host: " << cudaGetErrorString(result) << "\n";
        delete[] host_data;
        return false;
    }
    
    // 等待异步复制完成
    checkCudaErrors(cudaStreamSynchronize(cuda_stream));
    
    // 检查数据范围（RGB 值应该在 [0, 1] 范围内）
    float min_val = 1.0f, max_val = 0.0f;
    size_t num_pixels = width * height * 4; // RGBA
    
    for (size_t i = 0; i < num_pixels && i < 1000; i += 4) { // 只检查前250个像素以节省时间
        float r = host_data[i];
        float g = host_data[i + 1];
        float b = host_data[i + 2];
        
        min_val = (std::min)({min_val, r, g, b});
        max_val = (std::max)({max_val, r, g, b});
    }
    
    std::cout << "    ✓ CUDA data verification completed\n";
    std::cout << "    → Data range: [" << min_val << ", " << max_val << "]\n";
    std::cout << "    → Data size: " << data_size << " bytes\n";
    std::cout << "    → Dimensions: " << width << "x" << height << "\n";
    std::cout << "    → Pitch: " << pitch << " bytes per row\n";
    
    delete[] host_data;
    
    // 检查数据范围是否合理
    if (min_val < 0.0f || max_val > 1.0f) {
        std::cerr << "    ⚠ Warning: Data values outside expected [0,1] range\n";
    }
    
    return true;
}

bool fake_depth_inference(float* cuda_input, UINT width, UINT height, size_t pitch, 
                         cudaStream_t cuda_stream) {
    std::cout << "  → Running fake depth inference...\n";
    
    // 分配输出 CUDA 内存（深度图，单通道浮点）
    float* depth_output = nullptr;
    size_t depth_size = width * height * sizeof(float);
    
    checkCudaErrors(cudaMalloc((void**)&depth_output, depth_size));
    
    // 假的深度推理：简单地将RGB转换为灰度作为"深度"
    // 在实际应用中，这里会调用深度估计模型
    
    // 启动一个简单的 CUDA kernel 来模拟推理
    // 这里我们用 cudaMemset 来模拟
    checkCudaErrors(cudaMemsetAsync(depth_output, 0x80, depth_size, cuda_stream)); // 填充中等深度值
    
    // 等待 "推理" 完成
    checkCudaErrors(cudaStreamSynchronize(cuda_stream));
    
    std::cout << "    ✓ Fake depth inference completed\n";
    std::cout << "    → Output depth map size: " << width << "x" << height << "\n";
    
    // 验证输出数据
    float* host_depth = new float[width * height];
    checkCudaErrors(cudaMemcpy(host_depth, depth_output, depth_size, cudaMemcpyDeviceToHost));
    
    std::cout << "    → Sample depth values: ";
    for (int i = 0; i < 5 && i < (int)(width * height); i++) {
        std::cout << host_depth[i] << " ";
    }
    std::cout << "\n";
    
    delete[] host_depth;
    cudaFree(depth_output);
    
    return true;
}

} // anonymous namespace

void start_infer_depth_thread(
    ColorConvertedFrameQueue& input_frame_queue,
    cudaStream_t cuda_stream) {
    
    std::cout << "=== Depth Inference Thread Started ===\n";
    
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
            std::cout << ">>> Depth inference frame " << processed_count << "\n";
            
            // 验证 CUDA 输入数据
            if (!verify_cuda_data(converted_frame.cuda_data, converted_frame.width, 
                                 converted_frame.height, converted_frame.pitch, cuda_stream)) {
                std::cerr << "    ✗ CUDA data verification failed for frame " << processed_count << "\n";
                continue;
            }
            
            // 执行假的深度推理
            if (!fake_depth_inference(converted_frame.cuda_data, converted_frame.width, 
                                     converted_frame.height, converted_frame.pitch, cuda_stream)) {
                std::cerr << "    ✗ Depth inference failed for frame " << processed_count << "\n";
                continue;
            }
            
            std::cout << ">>> Depth inference frame " << processed_count << " completed\n";
        }
    }
    
    std::cout << "=== Depth Inference Thread Finished ===\n";
    std::cout << "Total frames processed for depth inference: " << processed_count << "\n";
} 