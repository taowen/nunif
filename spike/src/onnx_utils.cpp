#include "onnx_utils.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>  // 添加这个头文件用于文件状态检查

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "context.h"

// 缓存相关函数实现
std::string getCacheFilePath(const std::string& onnx_path) {
    // 基于ONNX文件路径生成缓存文件路径
    std::string cache_path = onnx_path;
    size_t pos = cache_path.find_last_of('.');
    if (pos != std::string::npos) {
        cache_path = cache_path.substr(0, pos);
    }
    cache_path += ".trt_cache";
    return cache_path;
}

bool isCacheValid(const std::string& onnx_path, const std::string& cache_path) {
    // 检查缓存文件是否存在
    std::ifstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.good()) {
        return false;
    }
    cache_file.close();
    
    // 检查ONNX文件和缓存文件的修改时间
    struct stat onnx_stat, cache_stat;
    if (stat(onnx_path.c_str(), &onnx_stat) != 0 || stat(cache_path.c_str(), &cache_stat) != 0) {
        return false;
    }
    
    // 如果ONNX文件比缓存文件新，则缓存无效
    return cache_stat.st_mtime >= onnx_stat.st_mtime;
}

bool loadEngineFromCache(ProcessorContext& ctx, const std::string& cache_path) {
    std::cout << "Loading TensorRT engine from cache: " << cache_path << std::endl;
    
    std::ifstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.good()) {
        std::cerr << "Failed to open cache file for reading" << std::endl;
        return false;
    }
    
    // 读取文件大小
    cache_file.seekg(0, std::ios::end);
    size_t engine_size = cache_file.tellg();
    cache_file.seekg(0, std::ios::beg);
    
    if (engine_size == 0) {
        std::cerr << "Cache file is empty" << std::endl;
        return false;
    }
    
    // 读取引擎数据
    std::vector<char> engine_data(engine_size);
    cache_file.read(engine_data.data(), engine_size);
    cache_file.close();
    
    // 创建runtime并反序列化引擎
    ctx.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(ctx.logger));
    if (!ctx.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    ctx.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        ctx.runtime->deserializeCudaEngine(engine_data.data(), engine_size));
    if (!ctx.engine) {
        std::cerr << "Failed to deserialize cached engine" << std::endl;
        return false;
    }
    
    ctx.context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx.engine->createExecutionContext());
    if (!ctx.context) {
        std::cerr << "Failed to create execution context from cached engine" << std::endl;
        return false;
    }
    
    std::cout << "Successfully loaded engine from cache" << std::endl;
    return true;
}

bool saveEngineToCache(const std::string& cache_path, nvinfer1::IHostMemory* serialized_engine) {
    std::cout << "Saving TensorRT engine to cache: " << cache_path << std::endl;
    
    std::ofstream cache_file(cache_path, std::ios::binary);
    if (!cache_file.good()) {
        std::cerr << "Failed to create cache file for writing" << std::endl;
        return false;
    }
    
    cache_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
    cache_file.close();
    
    if (cache_file.good()) {
        std::cout << "Engine cached successfully (" << serialized_engine->size() / 1024 / 1024 << " MB)" << std::endl;
        return true;
    } else {
        std::cerr << "Failed to write engine cache" << std::endl;
        return false;
    }
}

bool buildEngineFromOnnx(ProcessorContext& ctx, const std::string& onnx_path, const std::string& cache_path) {
    // Create builder, network and parser
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(ctx.logger));
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder" << std::endl;
        return false;
    }
    
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        std::cerr << "Failed to create TensorRT network" << std::endl;
        return false;
    }
    
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, ctx.logger));
    if (!parser) {
        std::cerr << "Failed to create ONNX parser" << std::endl;
        return false;
    }
    
    // Parse ONNX file
    std::cout << "Parsing ONNX file: " << onnx_path << std::endl;
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "Parser error: " << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    
    // Build engine
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        std::cerr << "Failed to create builder config" << std::endl;
        return false;
    }
    
    // Get GPU memory info for better memory management
    size_t total_mem, free_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Set memory pool size (use 80% of available memory, leave some for other operations)
    size_t workspace_size = (free_mem * 8) / 10;
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    
    // Enable FP16 precision
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "FP16 mode enabled" << std::endl;
    }
    
    // Set optimization profile for dynamic shapes
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        std::cerr << "Failed to create optimization profile" << std::endl;
        return false;
    }
    
    auto input = network->getInput(0);
    auto inputName = input->getName();
    
    // NV12格式的形状约束：(height + height/2, width)
    // Min: 392 + 196 = 588, Opt: 392 + 196 = 588, Max: 2160 + 1080 = 3240
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{588, 392});
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{588, 392});
    profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{3240, 3840});
    
    config->addOptimizationProfile(profile);
    
    std::cout << "Building TensorRT engine (this may take several minutes)..." << std::endl;
    auto start_build = std::chrono::high_resolution_clock::now();
    
    auto serialized_engine = std::unique_ptr<nvinfer1::IHostMemory>(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized_engine) {
        std::cerr << "Failed to build TensorRT engine" << std::endl;
        return false;
    }
    
    auto end_build = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::seconds>(end_build - start_build);
    std::cout << "Engine built successfully in " << build_time.count() << " seconds" << std::endl;
    
    // Save engine to cache
    saveEngineToCache(cache_path, serialized_engine.get());
    
    // Create runtime and deserialize engine
    ctx.runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(ctx.logger));
    if (!ctx.runtime) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }
    
    ctx.engine = std::unique_ptr<nvinfer1::ICudaEngine>(
        ctx.runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
    if (!ctx.engine) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }
    
    ctx.context = std::unique_ptr<nvinfer1::IExecutionContext>(ctx.engine->createExecutionContext());
    if (!ctx.context) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }
    
    // Print memory usage after engine creation
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t engine_required = ctx.engine->getDeviceMemorySize();
    
    std::cout << "Total GPU Memory: " << total_mem / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Available GPU Memory: " << free_mem / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Engine Required Memory: " << engine_required / 1024 / 1024 << " MB" << std::endl;

    return true;
}

bool loadOnnxModel(ProcessorContext& ctx, const std::string& onnx_path) {
    std::string cache_path = getCacheFilePath(onnx_path);
    
    // 首先尝试从缓存加载
    if (isCacheValid(onnx_path, cache_path)) {
        if (loadEngineFromCache(ctx, cache_path)) {
            // 打印内存使用情况
            size_t total_mem, free_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            size_t engine_required = ctx.engine->getDeviceMemorySize();
            
            std::cout << "Engine loaded from cache - GPU Memory:" << std::endl;
            std::cout << "  Total: " << total_mem / 1024 / 1024 << " MB" << std::endl;
            std::cout << "  Available: " << free_mem / 1024 / 1024 << " MB" << std::endl;
            std::cout << "  Engine Required: " << engine_required / 1024 / 1024 << " MB" << std::endl;
            
            return true;
        } else {
            std::cout << "Failed to load from cache, will rebuild engine" << std::endl;
        }
    } else {
        std::cout << "No valid cache found, building new engine" << std::endl;
    }
    
    // 缓存无效或加载失败，重新构建引擎
    return buildEngineFromOnnx(ctx, onnx_path, cache_path);
}
