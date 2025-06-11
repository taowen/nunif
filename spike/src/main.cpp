#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#ifdef TENSORRT_AVAILABLE
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#endif

class DistillAnyDepthTensorRT {
private:
#ifdef TENSORRT_AVAILABLE
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    void* gpu_input_buffer;
    void* gpu_output_buffer;
    
    int input_width, input_height;
    int output_width, output_height;
    size_t input_size, output_size;
#endif
    
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING) {
                std::cout << "TensorRT: " << msg << std::endl;
            }
        }
    } logger;

public:
    DistillAnyDepthTensorRT() : gpu_input_buffer(nullptr), gpu_output_buffer(nullptr) {}
    
    ~DistillAnyDepthTensorRT() {
#ifdef TENSORRT_AVAILABLE
        if (gpu_input_buffer) cudaFree(gpu_input_buffer);
        if (gpu_output_buffer) cudaFree(gpu_output_buffer);
#endif
    }
    
    bool loadEngine(const std::string& engine_path) {
#ifdef TENSORRT_AVAILABLE
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        file.close();
        
        runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        engine = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(engine_data.data(), size)
        );
        if (!engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) {
            std::cerr << "Failed to create execution context" << std::endl;
            return false;
        }
        
        // Get input/output dimensions
        auto input_dims = engine->getBindingDimensions(0);
        auto output_dims = engine->getBindingDimensions(1);
        
        input_height = input_dims.d[2];
        input_width = input_dims.d[3];
        output_height = output_dims.d[2];
        output_width = output_dims.d[3];
        
        input_size = 3 * input_height * input_width * sizeof(float);
        output_size = 1 * output_height * output_width * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&gpu_input_buffer, input_size);
        cudaMalloc(&gpu_output_buffer, output_size);
        
        std::cout << "Engine loaded successfully!" << std::endl;
        std::cout << "Input size: " << input_width << "x" << input_height << std::endl;
        std::cout << "Output size: " << output_width << "x" << output_height << std::endl;
        
        return true;
#else
        std::cerr << "TensorRT not available" << std::endl;
        return false;
#endif
    }
    
    cv::Mat preprocess(const cv::Mat& input_image, int lower_bound = 392) {
        cv::Mat image;
        input_image.convertTo(image, CV_32FC3, 1.0/255.0);
        
        int H = image.rows;
        int W = image.cols;
        
        // Calculate scale factor
        float scale_factor = (W < H) ? (float)lower_bound / W : (float)lower_bound / H;
        int new_h = (int)(H * scale_factor);
        int new_w = (int)(W * scale_factor);
        
        // Limit aspect ratio (max 4:1)
        if (new_h < new_w) {
            new_w = std::min(new_w, 4 * new_h);
        } else {
            new_h = std::min(new_h, 4 * new_w);
        }
        
        // Ensure multiple of 14
        new_h = new_h - (new_h % 14);
        new_w = new_w - (new_w % 14);
        
        if (new_h < lower_bound) new_h = lower_bound;
        if (new_w < lower_bound) new_w = lower_bound;
        
        // Resize
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
        
        // Normalize: ImageNet mean and std
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar std(0.229, 0.224, 0.225);
        
        cv::Mat normalized;
        resized.convertTo(normalized, CV_32FC3);
        normalized = (normalized - mean) / std;
        
        return normalized;
    }
    
    cv::Mat postprocess(const std::vector<float>& output_data, int width, int height) {
        cv::Mat depth_map(height, width, CV_32FC1, (void*)output_data.data());
        
        // Handle NaN values
        cv::Mat mask;
        cv::compare(depth_map, depth_map, mask, cv::CMP_EQ); // NaN != NaN
        depth_map.setTo(0, ~mask);
        
        // Invert for compatibility (DistillAnyDepth outputs inverted depth)
        depth_map = -depth_map;
        
        // Scale to 16-bit range for visualization
        cv::Mat depth_16;
        depth_map.convertTo(depth_16, CV_16UC1, 256.0);
        
        return depth_16;
    }
    
    cv::Mat infer(const cv::Mat& input_image) {
#ifdef TENSORRT_AVAILABLE
        if (!context) {
            std::cerr << "Engine not loaded" << std::endl;
            return cv::Mat();
        }
        
        // Preprocess
        cv::Mat preprocessed = preprocess(input_image);
        
        // Convert to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * preprocessed.rows * preprocessed.cols);
        
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            channel = channel.reshape(1, 1); // Flatten
            std::vector<float> channel_data = channel.isContinuous() ? 
                channel : channel.clone();
            float* ptr = (float*)channel_data.data();
            input_data.insert(input_data.end(), ptr, ptr + channel.total());
        }
        
        // Copy input to GPU
        cudaMemcpy(gpu_input_buffer, input_data.data(), 
                   input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Setup bindings
        void* bindings[] = {gpu_input_buffer, gpu_output_buffer};
        
        // Execute inference
        bool success = context->executeV2(bindings);
        if (!success) {
            std::cerr << "Inference failed" << std::endl;
            return cv::Mat();
        }
        
        // Copy output back to CPU
        std::vector<float> output_data(output_height * output_width);
        cudaMemcpy(output_data.data(), gpu_output_buffer, 
                   output_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Postprocess
        return postprocess(output_data, output_width, output_height);
#else
        std::cerr << "TensorRT not available" << std::endl;
        return cv::Mat();
#endif
    }
};

int main() {
    std::cout << "Distill Any Depth TensorRT Implementation" << std::endl;
    
#ifdef TENSORRT_AVAILABLE
    DistillAnyDepthTensorRT model;
    
    // Load TensorRT engine (you need to convert the model to TensorRT first)
    std::string engine_path = "distill_any_depth.trt";
    if (!model.loadEngine(engine_path)) {
        std::cerr << "Failed to load engine. Please convert the PyTorch model to TensorRT first." << std::endl;
        std::cout << "\nTo convert the model:" << std::endl;
        std::cout << "1. Export PyTorch model to ONNX" << std::endl;
        std::cout << "2. Convert ONNX to TensorRT using trtexec or Python API" << std::endl;
        return 1;
    }
    
    // Test with an image
    std::string image_path = "test_image.jpg";
    cv::Mat input_image = cv::imread(image_path);
    
    if (input_image.empty()) {
        std::cout << "Test image not found. Creating dummy image for demo." << std::endl;
        input_image = cv::Mat::ones(512, 512, CV_8UC3) * 128; // Gray image
    }
    
    std::cout << "Running inference..." << std::endl;
    cv::Mat depth_map = model.infer(input_image);
    
    if (!depth_map.empty()) {
        cv::imwrite("depth_output.png", depth_map);
        std::cout << "Depth map saved to depth_output.png" << std::endl;
    } else {
        std::cerr << "Inference failed" << std::endl;
    }
    
#else
    std::cout << "TensorRT not available. Please compile with TensorRT support." << std::endl;
#endif
    
    return 0;
}