#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

class DistillAnyDepthTensorRT {
private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    void* gpu_input_buffer;
    void* gpu_output_buffer;
    
    int input_width, input_height;
    int output_width, output_height;
    size_t input_size, output_size;
    
    std::string input_tensor_name;
    std::string output_tensor_name;
    
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
        if (gpu_input_buffer) cudaFree(gpu_input_buffer);
        if (gpu_output_buffer) cudaFree(gpu_output_buffer);
    }
    
    bool loadEngine(const std::string& engine_path) {
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
        
        // Debug: Print all tensor information
        int num_bindings = engine->getNbIOTensors();
        std::cout << "Number of IO tensors: " << num_bindings << std::endl;
        
        for (int i = 0; i < num_bindings; ++i) {
            const char* tensor_name = engine->getIOTensorName(i);
            auto tensor_mode = engine->getTensorIOMode(tensor_name);
            auto tensor_dims = engine->getTensorShape(tensor_name);
            
            std::cout << "Tensor " << i << ": " << tensor_name;
            std::cout << " Mode: " << (tensor_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
            std::cout << " Shape: [";
            for (int j = 0; j < tensor_dims.nbDims; ++j) {
                std::cout << tensor_dims.d[j];
                if (j < tensor_dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Get input/output dimensions properly
        const char* input_name = nullptr;
        const char* output_name = nullptr;
        
        for (int i = 0; i < num_bindings; ++i) {
            const char* tensor_name = engine->getIOTensorName(i);
            auto tensor_mode = engine->getTensorIOMode(tensor_name);
            
            if (tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
                input_name = tensor_name;
                input_tensor_name = tensor_name;
            } else {
                output_name = tensor_name;
                output_tensor_name = tensor_name;
            }
        }
        
        if (!input_name || !output_name) {
            std::cerr << "Failed to find input/output tensors" << std::endl;
            return false;
        }
        
        auto input_dims = engine->getTensorShape(input_name);
        auto output_dims = engine->getTensorShape(output_name);
        
        // Handle dynamic shapes - use default values if dimensions are -1
        input_height = input_dims.d[2] > 0 ? input_dims.d[2] : 518;  // Default size
        input_width = input_dims.d[3] > 0 ? input_dims.d[3] : 518;   // Default size
        
        // Output tensor shape depends on number of dimensions
        if (output_dims.nbDims == 4) {
            output_height = output_dims.d[2] > 0 ? output_dims.d[2] : 392;
            output_width = output_dims.d[3] > 0 ? output_dims.d[3] : 392;
        } else if (output_dims.nbDims == 3) {
            // Shape is [1, H, W] 
            output_height = output_dims.d[1] > 0 ? output_dims.d[1] : 392;
            output_width = output_dims.d[2] > 0 ? output_dims.d[2] : 392;
        } else {
            std::cerr << "Unexpected output tensor shape" << std::endl;
            return false;
        }
        
        input_size = 3 * input_height * input_width * sizeof(float);
        output_size = 1 * output_height * output_width * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&gpu_input_buffer, input_size);
        cudaMalloc(&gpu_output_buffer, output_size);
        
        std::cout << "Engine loaded successfully!" << std::endl;
        std::cout << "Input tensor: " << input_tensor_name << " - " << input_width << "x" << input_height << std::endl;
        std::cout << "Output tensor: " << output_tensor_name << " - " << output_width << "x" << output_height << std::endl;
        
        return true;
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
        if (!context) {
            std::cerr << "Engine not loaded" << std::endl;
            return cv::Mat();
        }
        
        // Preprocess
        cv::Mat preprocessed = preprocess(input_image);
        
        // Set dynamic input shape if needed
        int actual_height = preprocessed.rows;
        int actual_width = preprocessed.cols;
        
        // Use stored tensor names
        const char* input_name = input_tensor_name.c_str();
        const char* output_name = output_tensor_name.c_str();
        
        // Set input shape for dynamic models
        nvinfer1::Dims input_shape;
        input_shape.nbDims = 4;
        input_shape.d[0] = 1;  // batch size
        input_shape.d[1] = 3;  // channels
        input_shape.d[2] = actual_height;
        input_shape.d[3] = actual_width;
        
        context->setInputShape(input_name, input_shape);
        
        // Get output shape after setting input shape
        auto output_dims = context->getTensorShape(output_name);
        int actual_output_height, actual_output_width;
        
        if (output_dims.nbDims == 4) {
            actual_output_height = output_dims.d[2];
            actual_output_width = output_dims.d[3];
        } else if (output_dims.nbDims == 3) {
            // Shape is [1, H, W]
            actual_output_height = output_dims.d[1];
            actual_output_width = output_dims.d[2];
        } else {
            std::cerr << "Unexpected output tensor dimensions: " << output_dims.nbDims << std::endl;
            return cv::Mat();
        }
        
        std::cout << "Actual input shape: " << actual_width << "x" << actual_height << std::endl;
        std::cout << "Actual output shape: " << actual_output_width << "x" << actual_output_height << std::endl;
        
        // Reallocate GPU memory if needed
        size_t actual_input_size = 3 * actual_height * actual_width * sizeof(float);
        size_t actual_output_size = actual_output_height * actual_output_width * sizeof(float);
        
        if (actual_input_size > input_size) {
            cudaFree(gpu_input_buffer);
            cudaMalloc(&gpu_input_buffer, actual_input_size);
            input_size = actual_input_size;
        }
        
        if (actual_output_size > output_size) {
            cudaFree(gpu_output_buffer);
            cudaMalloc(&gpu_output_buffer, actual_output_size);
            output_size = actual_output_size;
        }
        
        // Convert to CHW format
        std::vector<cv::Mat> channels(3);
        cv::split(preprocessed, channels);
        
        std::vector<float> input_data;
        input_data.reserve(3 * actual_height * actual_width);
        
        for (int c = 0; c < 3; ++c) {
            cv::Mat channel = channels[c];
            channel = channel.reshape(1, 1); // Flatten
            std::vector<float> channel_data;
            if (channel.isContinuous()) {
                float* ptr = (float*)channel.data;
                channel_data.assign(ptr, ptr + channel.total());
            } else {
                channel = channel.clone();
                float* ptr = (float*)channel.data;
                channel_data.assign(ptr, ptr + channel.total());
            }
            input_data.insert(input_data.end(), channel_data.begin(), channel_data.end());
        }
        
        // Copy input to GPU
        cudaMemcpy(gpu_input_buffer, input_data.data(), 
                   input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Set tensor addresses for the new TensorRT API
        context->setTensorAddress(input_name, gpu_input_buffer);
        context->setTensorAddress(output_name, gpu_output_buffer);
        
        // Execute inference using enqueueV3 (recommended for newer TensorRT)
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        bool success = context->enqueueV3(stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if (!success) {
            std::cerr << "Inference failed" << std::endl;
            return cv::Mat();
        }
        
        // Copy output back to CPU
        std::vector<float> output_data(actual_output_height * actual_output_width);
        cudaMemcpy(output_data.data(), gpu_output_buffer, 
                   output_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Postprocess
        return postprocess(output_data, actual_output_width, actual_output_height);
    }
};

int main() {
    std::cout << "Distill Any Depth TensorRT Implementation" << std::endl;
    
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
    std::string image_path = "miku_128.png";
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
    
    return 0;
}