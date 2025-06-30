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
#include <vector>
#include <memory>

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

// 前向声明
class ToCudaInputContextImpl;

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
 * @brief 将NV12格式的D3D11 AVFrame转换为RGBA格式的D3D11纹理
 * 
 * @param ctx FFMepg上下文，包含D3D11设备和上下文
 *            - 调用者负责：确保ctx有效且包含有效的D3D11设备
 *            - 函数负责：使用D3D11视频处理器进行颜色转换
 * 
 * @param frame 输入的AVFrame，格式应为AV_PIX_FMT_D3D11，底层纹理格式为DXGI_FORMAT_NV12
 *              - 调用者负责：确保frame有效且为D3D11格式
 *              - 函数负责：读取frame中的D3D11纹理数据进行转换
 * 
 * @return ID3D11Texture2D* 转换后的RGBA纹理指针
 *         - 成功：返回有效的ID3D11Texture2D指针，格式为DXGI_FORMAT_R8G8B8A8_UNORM
 *         - 失败：返回nullptr
 *         - 调用者负责：调用Release()方法释放返回的纹理
 *         - 注意：返回的纹理在GPU显存中，可用于进一步的D3D11处理
 * 
 * @note 功能特性：
 *       1. 颜色转换：NV12 -> RGBA (8位无符号归一化)
 *       2. GPU加速：使用D3D11视频处理器进行硬件加速转换
 *       3. 内存效率：直接在GPU显存中完成转换，避免CPU-GPU数据传输
 *       4. 资源管理：内部管理D3D11视频设备、上下文和处理器资源
 *       5. 色彩空间：支持BT709到sRGB的色彩空间转换
 * 
 * @example
 *       ID3D11Texture2D* rgba_texture = convert_color(&ctx, frame);
 *       if (rgba_texture) {
 *           // 可以直接用于D3D11渲染管线
 *           // ... 使用 rgba_texture ...
 *           rgba_texture->Release();
 *       }
 */
ID3D11Texture2D* convert_color(const FFMepgContext* ctx, AVFrame* frame);

/**
 * @brief CUDA输入转换上下文类
 * 
 * 封装D3D11到CUDA的纹理转换功能，管理相关资源和状态
 */
class ToCudaInputContext {
public:
    ToCudaInputContext();
    ~ToCudaInputContext();
    
    // 禁止拷贝构造和赋值
    ToCudaInputContext(const ToCudaInputContext&) = delete;
    ToCudaInputContext& operator=(const ToCudaInputContext&) = delete;
    
    /**
     * @brief 将D3D11 RGBA8纹理转换为CUDA float32 NCHW格式，返回映射的CUDA指针
     * 
     * @param ctx FFMepg上下文，包含D3D11设备和上下文
     * @param rgbaTexture convert_color输出的RGBA8纹理
     * @return void* 映射的CUDA设备指针，格式：float32, NCHW布局, shape=(1,4,H,W)
     *               失败时返回nullptr
     * 
     * @note 返回的指针在使用完毕后必须调用unmap_cuda_input()取消映射
     */
    void* to_cuda_input(const FFMepgContext* ctx, ID3D11Texture2D* rgbaTexture);
    
    /**
     * @brief 取消CUDA资源映射
     * 
     * @note 在使用完to_cuda_input返回的指针后必须调用此函数
     */
    void unmap_cuda_input();
    
    /**
     * @brief 清理所有资源
     * 
     * @note 释放DirectCompute着色器和相关资源
     */
    void cleanup();

private:
    std::unique_ptr<ToCudaInputContextImpl> pImpl;
};

// 兼容性接口 - 不再使用全局实例
/**
 * @brief 将D3D11 RGBA8纹理转换为CUDA float32 NCHW格式，返回映射的CUDA指针
 * 
 * @param context CUDA输入转换上下文
 * @param ctx FFMepg上下文，包含D3D11设备和上下文
 * @param rgbaTexture convert_color输出的RGBA8纹理
 * @return void* 映射的CUDA设备指针，格式：float32, NCHW布局, shape=(1,4,H,W)
 *               失败时返回nullptr
 * 
 * @note 返回的指针在使用完毕后必须调用context->unmap_cuda_input()取消映射
 */
void* to_cuda_input(ToCudaInputContext* context, const FFMepgContext* ctx, ID3D11Texture2D* rgbaTexture);


/**
 * Main inference function with CUDA tensor interface
 * 
 * Input requirements (matching export_iw3.py ONNX model):
 * - Format: RGBA (4 channels) 
 * - Data type: float32
 * - Value range: [0.0, 1.0]
 * - Layout: NCHW (batch, channel, height, width)
 * - Shape: (1, 4, height, width) - Fixed batch size of 1
 * - Color space: RGB with alpha channel
 * 
 * Output format:
 * - Format: RGBA (4 channels)
 * - Data type: float32  
 * - Value range: [0.0, 1.0]
 * - Layout: NCHW (batch, channel, height, width/2)
 * - Shape: (1, 4, height, width/2) - Fixed batch size of 1
 * - Content: Half side-by-side stereo (left eye | right eye)
 * - Color space: RGB with alpha channel
 * 
 * @param inputDevicePtr 输入CUDA设备指针（void*类型）
 *        - 类型说明：使用void*提供泛型接口，避免强类型约束
 *        - 实际数据：指向float32格式的NCHW张量数据
 *        - 内存位置：必须是有效的CUDA设备内存
 *        - 调用者职责：确保指针有效且数据格式正确
 *        - 函数职责：只读取数据，不修改指针值
 * 
 * @param height 输入图像高度（像素）
 * 
 * @param width 输入图像宽度（像素）
 * 
 * @param out_size_bytes 输出缓冲区大小（size_t*类型）
 *        - 类型说明：使用指针类型以便函数修改调用者的变量值
 *        - 输出参数：函数会写入分配的输出内存字节数
 *        - 可选参数：可以传入nullptr如果不需要获取大小信息
 *        - 用途：帮助调用者了解返回的CUDA内存块大小，便于后续操作
 * 
 * @return void* 指向推理结果的CUDA设备指针
 *         - 成功：返回新分配的CUDA设备内存指针，包含推理结果
 *         - 失败：返回nullptr
 *         - 调用者职责：使用cudaFree()释放返回的内存
 *         - 数据格式：与输出格式规范一致的float32 NCHW张量
 * 
 * @note 参数设计原理：
 *       - inputDevicePtr使用void*：输入参数，提供泛型接口，只需读取数据
 *       - out_size_bytes使用size_t*：输出参数，需要修改调用者变量值
 *       
 * @note 内存管理：
 *       - 输入内存：调用者分配和释放
 *       - 输出内存：函数分配，调用者释放
 *       - 失败处理：函数内部清理，不会泄露内存
 */
void* inferIW3(void* inputDevicePtr, int height, int width, size_t* out_size_bytes);
