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
 * @brief 将NV12 BT709格式的AVFrame转换为RGBA D3D11纹理，支持CUDA互操作
 * 
 * @param ctx FFMepg上下文，包含解码相关信息和D3D11设备资源
 *            - 调用者负责：确保ctx->d3d_device和ctx->d3d_context有效
 *            - 函数负责：使用ctx中的D3D11资源进行颜色转换
 * 
 * @param frame 输入的AVFrame，格式应为NV12，色彩空间为BT709
 *              - 调用者负责：确保frame有效且格式正确
 *              - 函数负责：读取frame数据进行颜色转换
 * 
 * @return ID3D11Texture2D* 转换后的RGBA纹理
 *         - 成功：返回有效的D3D11纹理指针，格式为DXGI_FORMAT_R8G8B8A8_UNORM
 *         - 失败：返回nullptr
 *         - 调用者负责：调用Release()释放返回的纹理
 *         - 注意：返回的纹理创建时启用D3D11_RESOURCE_MISC_SHARED，支持CUDA互操作
 * 
 * @note 功能特性：
 *       1. 颜色转换：NV12 BT709 -> RGBA sRGB
 *       2. GPU加速：使用D3D11 Compute Shader或Video Processor
 *       3. CUDA互操作：纹理支持与CUDA的互操作访问
 *       4. 内存效率：直接在GPU显存中完成转换，避免CPU-GPU数据传输
 *       5. 资源管理：使用ctx中的D3D11设备和上下文，无需额外传参
 * 
 * @example
 *       // 先设置D3D11资源到context中
 *       ctx.d3d_device = my_d3d_device;
 *       ctx.d3d_context = my_d3d_context;
 *       
 *       ID3D11Texture2D* rgba_texture = convert_color(&ctx, frame);
 *       if (rgba_texture) {
 *           // 可以将此纹理用于CUDA处理或D3D11渲染
 *           // ... 使用纹理 ...
 *           rgba_texture->Release();
 *       }
 */
ID3D11Texture2D* convert_color(const FFMepgContext* ctx, AVFrame* frame);