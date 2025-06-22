#pragma once

#include <string>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
}

#include "context.h"

// 缓存相关函数
std::string getCacheFilePath(const std::string& onnx_path);
bool isCacheValid(const std::string& onnx_path, const std::string& cache_path);
bool loadEngineFromCache(ProcessorContext& ctx, const std::string& cache_path);
bool saveEngineToCache(const std::string& cache_path, nvinfer1::IHostMemory* serialized_engine);

// TensorRT引擎构建相关函数
bool buildEngineFromOnnx(ProcessorContext& ctx, const std::string& onnx_path, const std::string& cache_path);
bool loadOnnxModel(ProcessorContext& ctx, const std::string& onnx_path);
