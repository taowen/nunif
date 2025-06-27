#pragma once
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>
}
// Include D3D11 hardware context header outside of extern "C" block
#include <libavutil/hwcontext_d3d11va.h>
#include <iostream>
#include <string>
#include <d3d11.h>

struct FFMepgContext {
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVBufferRef* hw_device_ctx = nullptr;
    int video_stream_idx = -1;
    AVStream* video_stream = nullptr;
    std::string current_file;
    bool initialized = false;
    
    // D3D11 resources
    ID3D11Device* d3d_device = nullptr;
    ID3D11DeviceContext* d3d_context = nullptr;
};

/**
 * @brief 清理FFMepgContext中的所有资源
 * 
 * @param ctx 要清理的上下文
 *            - 函数负责：释放所有内部FFmpeg资源
 *            - 调用者负责：释放FFMepgContext结构体本身(如果是动态分配的)
 */
void cleanup_context(FFMepgContext* ctx);

/**
 * @brief 使用D3D11硬件加速解码指定帧
 * 
 * @param ctx FFMepg上下文，用于缓存解码器状态
 *            - 调用者负责：分配FFMepgContext结构体
 *            - 函数负责：管理ctx内部的FFmpeg资源(fmt_ctx, codec_ctx, hw_device_ctx等)
 *            - 调用者负责：在使用完毕后调用清理函数释放ctx内部资源
 * 
 * @param inputFile 输入视频文件路径
 *                  - 调用者负责：确保文件路径有效且可访问
 *                  - 函数负责：打开和关闭文件资源
 * 
 * @param theIndex 目标帧索引(从0开始)
 *                 - 调用者负责：确保索引在有效范围内
 * 
 * @return AVFrame* 解码后的帧数据
 *         - 成功：返回有效的AVFrame指针，已分配内存并包含解码数据
 *         - 失败：返回nullptr
 *         - 调用者负责：使用av_frame_free()释放返回的AVFrame
 *         - 注意：返回的帧已从GPU内存传输到系统内存，格式通常为YUV420P或NV12
 * 
 * @note 资源管理责任：
 *       1. FFMepgContext管理：调用者分配结构体，函数管理内部FFmpeg资源
 *       2. 返回的AVFrame：调用者必须调用av_frame_free()释放
 *       3. 上下文可重复使用：相同文件的多次调用会复用已初始化的解码器
 *       4. 不同文件：函数会自动清理旧资源并重新初始化
 * 
 * @example
 *       FFMepgContext ctx = {};
 *       AVFrame* frame = d11_decode(&ctx, "video.mp4", 100);
 *       if (frame) {
 *           // 使用frame数据...
 *           av_frame_free(&frame);
 *       }
 *       // 清理上下文(需要实现cleanup函数)
 *       cleanup_context(&ctx);
 */
AVFrame* d11_decode(FFMepgContext* ctx, const std::string& inputFile, int theIndex);

/**
 * @brief 将NV12 BT709格式的AVFrame转换为RGBA CUDA内存，支持GPU加速处理
 * 
 * @param ctx FFMepg上下文，包含解码相关信息
 *            - 调用者负责：确保ctx有效
 *            - 函数负责：管理CUDA资源进行颜色转换
 * 
 * @param frame 输入的AVFrame，格式应为NV12，色彩空间为BT709
 *              - 调用者负责：确保frame有效且格式正确
 *              - 函数负责：读取frame数据进行颜色转换
 * 
 * @return void* 转换后的RGBA数据在CUDA设备内存中的指针
 *         - 成功：返回有效的CUDA设备内存指针，数据格式为RGBA float32
 *         - 失败：返回nullptr
 *         - 调用者负责：调用cudaFree()释放返回的内存
 *         - 注意：返回的内存直接在GPU显存中，可直接用于CUDA kernel处理
 * 
 * @note 功能特性：
 *       1. 颜色转换：NV12 BT709 -> RGBA sRGB
 *       2. GPU加速：使用CUDA kernel进行颜色空间转换
 *       3. 内存效率：直接在GPU显存中完成转换，避免CPU-GPU数据传输
 *       4. 资源管理：内部管理CUDA上下文和流
 *       5. 数据布局：RGBA数据按行优先顺序存储
 * 
 * @example
 *       void* cuda_rgba = convert_color(&ctx, frame);
 *       if (cuda_rgba) {
 *           // 可以直接在CUDA kernel中使用此数据
 *           // ... 处理 cuda_rgba ...
 *           cudaFree(cuda_rgba);
 *       }
 */
void* convert_color(const FFMepgContext* ctx, AVFrame* frame);

// 添加保存RGBA数据为BMP文件的函数声明
void save_rgba_as_bmp(const char* filename, const uint8_t* rgba_data, uint32_t width, uint32_t height);