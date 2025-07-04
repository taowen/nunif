#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "frame_decoder.h"
#include <memory>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
}

class RGBFrameDecoder {
public:
    struct RGBFrame {
        ID3D11Texture2D* rgb_texture = nullptr;
        ID3D11ShaderResourceView* rgb_srv = nullptr;
        double timestamp = 0.0;
        bool is_valid = false;
        int width = 0;
        int height = 0;
        
        // 资源清理方法
        void release() {
            if (rgb_srv) {
                rgb_srv->Release();
                rgb_srv = nullptr;
            }
            if (rgb_texture) {
                rgb_texture->Release();
                rgb_texture = nullptr;
            }
            is_valid = false;
        }
    };
    
    struct DecodedFrames {
        FrameDecoder::DecodedFrame audio_frame;
        RGBFrame rgb_frame;
    };

    RGBFrameDecoder();
    ~RGBFrameDecoder();

    // 使用外部提供的D3D11设备进行初始化（可选，如果为nullptr则使用内部设备）
    bool open(const std::string& filepath, ID3D11Device* external_device = nullptr);
    
    // Pull-style解码接口 - 返回RGB转换后的帧和音频帧
    bool readNextFrames(DecodedFrames& decoded_frames);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isHardwareAccelerated() const { return frame_decoder_.isHardwareAccelerated(); }
    const char* getVideoCodecName() const { return frame_decoder_.getVideoCodecName(); }
    const char* getAudioCodecName() const { return frame_decoder_.getAudioCodecName(); }
    
    // 获取视频信息
    int getVideoWidth() const { return video_width_; }
    int getVideoHeight() const { return video_height_; }
    
    // 获取内部FrameDecoder的访问（用于测试）
    FrameDecoder* getFrameDecoder() { return &frame_decoder_; }
    
    // 资源管理
    void flush();
    void close();

private:
    // 内部FrameDecoder实例
    FrameDecoder frame_decoder_;
    
    // D3D11设备（不拥有，只使用）
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // 视频信息
    int video_width_;
    int video_height_;
    bool is_initialized_;
    
    // 纹理池 - 3帧循环复用
    struct TextureSlot {
        ID3D11Texture2D* texture = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        int width = 0;
        int height = 0;
        bool is_created = false;
    };
    static const int TEXTURE_POOL_SIZE = 3;
    TextureSlot texture_pool_[TEXTURE_POOL_SIZE];
    int current_slot_index_;
    
    // Video Processor复用
    ID3D11VideoDevice* video_device_;
    ID3D11VideoContext* video_context_;
    ID3D11VideoProcessorEnumerator* video_enum_;
    ID3D11VideoProcessor* video_processor_;
    bool video_processor_initialized_;
    
    // 内部方法
    bool convertNV12ToRGB(const FrameDecoder::DecodedFrame& nv12_frame, TextureSlot* slot);
    bool ensureVideoProcessor();
    bool createTextureSlot(TextureSlot* slot, int width, int height);
    void releaseTextureSlot(TextureSlot* slot);
    void releaseVideoProcessor();
    void releaseResources();
};