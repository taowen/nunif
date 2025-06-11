#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>

// Simple struct to hold TensorRT engine data
struct TensorRTEngine {
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    
    void* gpu_input_buffer = nullptr;
    void* gpu_output_buffer = nullptr;
    
    std::string input_tensor_name;
    std::string output_tensor_name;
    
    size_t input_buffer_size = 0;
    size_t output_buffer_size = 0;
};

// Simple logger class
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "TensorRT: " << msg << std::endl;
        }
    }
};

static Logger g_logger;

// Static function to load TensorRT engine
static bool loadEngine(const std::string& engine_path, TensorRTEngine& engine_data) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    
    std::vector<char> engine_buffer(size);
    file.read(engine_buffer.data(), size);
    file.close();
    
    engine_data.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    if (!engine_data.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    engine_data.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        engine_data.runtime->deserializeCudaEngine(engine_buffer.data(), size)
    );
    if (!engine_data.engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    engine_data.context = std::unique_ptr<nvinfer1::IExecutionContext>(
        engine_data.engine->createExecutionContext()
    );
    if (!engine_data.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // Get tensor information
    int num_bindings = engine_data.engine->getNbIOTensors();
    std::cout << "Number of IO tensors: " << num_bindings << std::endl;
    
    for (int i = 0; i < num_bindings; ++i) {
        const char* tensor_name = engine_data.engine->getIOTensorName(i);
        auto tensor_mode = engine_data.engine->getTensorIOMode(tensor_name);
        auto tensor_dims = engine_data.engine->getTensorShape(tensor_name);
        
        std::cout << "Tensor " << i << ": " << tensor_name;
        std::cout << " Mode: " << (tensor_mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT");
        std::cout << " Shape: [";
        for (int j = 0; j < tensor_dims.nbDims; ++j) {
            std::cout << tensor_dims.d[j];
            if (j < tensor_dims.nbDims - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        if (tensor_mode == nvinfer1::TensorIOMode::kINPUT) {
            engine_data.input_tensor_name = tensor_name;
        } else {
            engine_data.output_tensor_name = tensor_name;
        }
    }
    
    if (engine_data.input_tensor_name.empty() || engine_data.output_tensor_name.empty()) {
        std::cerr << "Failed to find input/output tensors" << std::endl;
        return false;
    }
    
    std::cout << "Engine loaded successfully!" << std::endl;
    std::cout << "Input tensor: " << engine_data.input_tensor_name << std::endl;
    std::cout << "Output tensor: " << engine_data.output_tensor_name << std::endl;
    
    return true;
}

// Static function to cleanup GPU memory
static void cleanupEngine(TensorRTEngine& engine_data) {
    if (engine_data.gpu_input_buffer) {
        cudaFree(engine_data.gpu_input_buffer);
        engine_data.gpu_input_buffer = nullptr;
    }
    if (engine_data.gpu_output_buffer) {
        cudaFree(engine_data.gpu_output_buffer);
        engine_data.gpu_output_buffer = nullptr;
    }
}

// Static function to preprocess input image
static cv::Mat preprocessImage(const cv::Mat& input_image, int lower_bound = 392) {
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

// Static function to postprocess output
static cv::Mat postprocessOutput(const std::vector<float>& output_data, int width, int height) {
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

// Static function to convert image to CHW format
static std::vector<float> imageToChwFormat(const cv::Mat& image) {
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);
    
    std::vector<float> chw_data;
    chw_data.reserve(3 * image.rows * image.cols);
    
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel = channels[c];
        channel = channel.reshape(1, 1); // Flatten
        
        if (channel.isContinuous()) {
            float* ptr = (float*)channel.data;
            chw_data.insert(chw_data.end(), ptr, ptr + channel.total());
        } else {
            channel = channel.clone();
            float* ptr = (float*)channel.data;
            chw_data.insert(chw_data.end(), ptr, ptr + channel.total());
        }
    }
    
    return chw_data;
}

// Static function to run inference
static cv::Mat runInference(TensorRTEngine& engine_data, const cv::Mat& input_image) {
    if (!engine_data.context) {
        std::cerr << "Engine not initialized" << std::endl;
        return cv::Mat();
    }
    
    // Preprocess
    cv::Mat preprocessed = preprocessImage(input_image);
    
    int actual_height = preprocessed.rows;
    int actual_width = preprocessed.cols;
    
    const char* input_name = engine_data.input_tensor_name.c_str();
    const char* output_name = engine_data.output_tensor_name.c_str();
    
    // Set input shape for dynamic models
    nvinfer1::Dims input_shape;
    input_shape.nbDims = 4;
    input_shape.d[0] = 1;  // batch size
    input_shape.d[1] = 3;  // channels
    input_shape.d[2] = actual_height;
    input_shape.d[3] = actual_width;
    
    engine_data.context->setInputShape(input_name, input_shape);
    
    // Get output shape after setting input shape
    auto output_dims = engine_data.context->getTensorShape(output_name);
    int actual_output_height, actual_output_width;
    
    if (output_dims.nbDims == 4) {
        actual_output_height = output_dims.d[2];
        actual_output_width = output_dims.d[3];
    } else if (output_dims.nbDims == 3) {
        actual_output_height = output_dims.d[1];
        actual_output_width = output_dims.d[2];
    } else {
        std::cerr << "Unexpected output tensor dimensions: " << output_dims.nbDims << std::endl;
        return cv::Mat();
    }
    
    std::cout << "Actual input shape: " << actual_width << "x" << actual_height << std::endl;
    std::cout << "Actual output shape: " << actual_output_width << "x" << actual_output_height << std::endl;
    
    // Calculate buffer sizes
    size_t actual_input_size = 3 * actual_height * actual_width * sizeof(float);
    size_t actual_output_size = actual_output_height * actual_output_width * sizeof(float);
    
    // Reallocate GPU memory if needed
    if (actual_input_size > engine_data.input_buffer_size) {
        if (engine_data.gpu_input_buffer) cudaFree(engine_data.gpu_input_buffer);
        cudaMalloc(&engine_data.gpu_input_buffer, actual_input_size);
        engine_data.input_buffer_size = actual_input_size;
    }
    
    if (actual_output_size > engine_data.output_buffer_size) {
        if (engine_data.gpu_output_buffer) cudaFree(engine_data.gpu_output_buffer);
        cudaMalloc(&engine_data.gpu_output_buffer, actual_output_size);
        engine_data.output_buffer_size = actual_output_size;
    }
    
    // Convert to CHW format
    std::vector<float> input_data = imageToChwFormat(preprocessed);
    
    // Copy input to GPU
    cudaMemcpy(engine_data.gpu_input_buffer, input_data.data(), 
               input_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // Set tensor addresses
    engine_data.context->setTensorAddress(input_name, engine_data.gpu_input_buffer);
    engine_data.context->setTensorAddress(output_name, engine_data.gpu_output_buffer);
    
    // Execute inference
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    bool success = engine_data.context->enqueueV3(stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    if (!success) {
        std::cerr << "Inference failed" << std::endl;
        return cv::Mat();
    }
    
    // Copy output back to CPU
    std::vector<float> output_data(actual_output_height * actual_output_width);
    cudaMemcpy(output_data.data(), engine_data.gpu_output_buffer, 
               output_data.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Postprocess
    return postprocessOutput(output_data, actual_output_width, actual_output_height);
}

int main() {
    std::cout << "Distill Any Depth TensorRT Implementation" << std::endl;
    
    TensorRTEngine engine_data;
    
    // Load TensorRT engine
    std::string engine_path = "distill_any_depth.trt";
    if (!loadEngine(engine_path, engine_data)) {
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
    cv::Mat depth_map = runInference(engine_data, input_image);
    
    if (!depth_map.empty()) {
        cv::imwrite("depth_output.png", depth_map);
        std::cout << "Depth map saved to depth_output.png" << std::endl;
    } else {
        std::cerr << "Inference failed" << std::endl;
    }
    
    // Cleanup
    cleanupEngine(engine_data);
    
    return 0;
}