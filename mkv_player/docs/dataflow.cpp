/**
 * MKV Player 数据流文档
 * 
 * 功能要求：
 * - 音画同步播放
 * - FFmpeg + DirectX11 硬件解码（无软件解码fallback）
 * - DirectX11 渲染输出
 * - 单线程实现（文档目的）
 */

#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
    #include <libswresample/swresample.h>
}

class MKVPlayerDataFlow {
private:
    // DirectX11 组件
    ID3D11Device* d3d11_device;
    ID3D11DeviceContext* d3d11_context;
    IDXGISwapChain* swap_chain;
    ID3D11RenderTargetView* render_target_view;
    
    // FFmpeg 解码组件
    AVFormatContext* format_context;
    AVCodecContext* video_codec_context;
    AVCodecContext* audio_codec_context;
    AVBufferRef* hw_device_ctx;
    
    // 流索引
    int video_stream_index;
    int audio_stream_index;
    
    // 音频重采样
    SwrContext* audio_resampler;
    
    // 同步相关
    double video_clock;
    double audio_clock;
    double frame_timer;

public:
    /**
     * 数据流处理主循环
     * 这是整个播放器的核心数据流程
     */
    void ProcessDataFlow() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* video_frame = av_frame_alloc();
        AVFrame* audio_frame = av_frame_alloc();
        AVFrame* hw_frame = av_frame_alloc();
        
        // 主数据流循环
        while (av_read_frame(format_context, packet) >= 0) {
            
            if (packet->stream_index == video_stream_index) {
                // ========== 视频数据流 ==========
                ProcessVideoPacket(packet, video_frame, hw_frame);
                
            } else if (packet->stream_index == audio_stream_index) {
                // ========== 音频数据流 ==========
                ProcessAudioPacket(packet, audio_frame);
            }
            
            av_packet_unref(packet);
        }
        
        av_frame_free(&video_frame);
        av_frame_free(&audio_frame);
        av_frame_free(&hw_frame);
        av_packet_free(&packet);
    }

private:
    /**
     * 视频数据流处理
     * 路径: MKV文件 -> FFmpeg解封装 -> D3D11硬件解码 -> DirectX11渲染
     */
    void ProcessVideoPacket(AVPacket* packet, AVFrame* cpu_frame, AVFrame* hw_frame) {
        // 1. 发送数据包到硬件解码器
        if (avcodec_send_packet(video_codec_context, packet) < 0) {
            return; // 硬件解码失败，不fallback到软件
        }
        
        // 2. 从硬件解码器接收帧
        while (avcodec_receive_frame(video_codec_context, hw_frame) >= 0) {
            
            // 3. 计算视频时间戳
            double pts = hw_frame->pts * av_q2d(format_context->streams[video_stream_index]->time_base);
            video_clock = pts;
            
            // 4. 音画同步检查
            double sync_delay = CalculateSyncDelay(pts);
            if (sync_delay > 0) {
                // 需要延迟显示这一帧
                Sleep((DWORD)(sync_delay * 1000));
            }
            
            // 5. DirectX11硬件纹理渲染
            RenderVideoFrame(hw_frame);
        }
    }
    
    /**
     * 音频数据流处理  
     * 路径: MKV文件 -> FFmpeg解封装 -> 软件解码 -> 重采样 -> 音频输出
     */
    void ProcessAudioPacket(AVPacket* packet, AVFrame* audio_frame) {
        // 1. 发送数据包到音频解码器
        if (avcodec_send_packet(audio_codec_context, packet) < 0) {
            return;
        }
        
        // 2. 从解码器接收音频帧
        while (avcodec_receive_frame(audio_codec_context, audio_frame) >= 0) {
            
            // 3. 计算音频时间戳
            double pts = audio_frame->pts * av_q2d(format_context->streams[audio_stream_index]->time_base);
            audio_clock = pts;
            
            // 4. 音频重采样（转换为播放设备要求的格式）
            uint8_t* output_buffer;
            int output_samples = swr_convert(audio_resampler, 
                                           &output_buffer, audio_frame->nb_samples,
                                           (const uint8_t**)audio_frame->data, audio_frame->nb_samples);
            
            // 5. 发送到音频输出设备
            OutputAudioSamples(output_buffer, output_samples);
        }
    }
    
    /**
     * 音画同步计算
     * 使用音频时钟作为主时钟，视频同步到音频
     */
    double CalculateSyncDelay(double video_pts) {
        double diff = video_pts - audio_clock;
        
        // 如果视频超前音频太多，需要延迟
        if (diff > 0.04) { // 40ms容忍度
            return diff;
        }
        
        // 如果视频落后音频太多，跳过这一帧
        if (diff < -0.04) {
            return -1; // 表示应该跳过这一帧
        }
        
        return 0; // 同步良好，立即显示
    }
    
    /**
     * DirectX11视频帧渲染
     * 将硬件解码的D3D11纹理直接渲染到屏幕
     */
    void RenderVideoFrame(AVFrame* hw_frame) {
        // 1. 从FFmpeg硬件帧获取D3D11纹理
        ID3D11Texture2D* decoded_texture = (ID3D11Texture2D*)hw_frame->data[0];
        int texture_index = (intptr_t)hw_frame->data[1];
        
        // 2. 创建着色器资源视图
        ID3D11ShaderResourceView* srv;
        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
        srv_desc.Format = DXGI_FORMAT_NV12; // 或其他硬件解码格式
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        
        d3d11_device->CreateShaderResourceView(decoded_texture, &srv_desc, &srv);
        
        // 3. 设置渲染管线
        d3d11_context->PSSetShaderResources(0, 1, &srv);
        
        // 4. 渲染全屏四边形
        DrawFullscreenQuad();
        
        // 5. 呈现到屏幕
        swap_chain->Present(1, 0); // 垂直同步
        
        srv->Release();
    }
    
    /**
     * 初始化硬件解码器
     * 配置FFmpeg使用DirectX11硬件加速
     */
    bool InitializeHardwareDecoder() {
        // 1. 创建D3D11硬件设备上下文
        av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        
        // 2. 查找支持D3D11硬件解码的解码器
        const AVCodec* decoder = avcodec_find_decoder_by_name("h264_d3d11va"); // 或其他硬件解码器
        if (!decoder) {
            return false; // 不支持硬件解码
        }
        
        // 3. 配置解码器上下文
        video_codec_context = avcodec_alloc_context3(decoder);
        video_codec_context->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        
        // 4. 打开解码器
        if (avcodec_open2(video_codec_context, decoder, nullptr) < 0) {
            return false; // 硬件解码器初始化失败
        }
        
        return true;
    }
    
    /**
     * 数据流架构总结：
     * 
     * [MKV文件] 
     *     ↓ (av_read_frame)
     * [解封装器] 
     *     ↓ (分离音视频包)
     * [视频包] ────────────────────── [音频包]
     *     ↓                           ↓
     * [D3D11硬件解码]                [软件音频解码]
     *     ↓                           ↓
     * [硬件纹理帧]                   [PCM音频帧]
     *     ↓                           ↓
     * [时间戳同步] ←──────────────── [音频重采样]
     *     ↓                           ↓
     * [DirectX11渲染]                [音频输出设备]
     *     ↓                           ↓
     * [屏幕显示]                     [扬声器播放]
     * 
     * 关键同步点：
     * - 音频时钟作为主时钟
     * - 视频帧根据音频时钟调整显示时机
     * - 硬件解码失败时不fallback，直接跳过
     */
};