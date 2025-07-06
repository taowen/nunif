#pragma once

#include <d3d11.h>
#include <dxgi.h>
#include "hw_frame_decoder.h"
#include <memory>
#include <wrl/client.h>
extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/hwcontext_d3d11va.h>
}

// DirectX智能指针类型别名
using D3D11TexturePtr = Microsoft::WRL::ComPtr<ID3D11Texture2D>;
using D3D11SRVPtr = Microsoft::WRL::ComPtr<ID3D11ShaderResourceView>;
using D3D11DevicePtr = Microsoft::WRL::ComPtr<ID3D11Device>;
using D3D11ContextPtr = Microsoft::WRL::ComPtr<ID3D11DeviceContext>;
using D3D11VideoDevicePtr = Microsoft::WRL::ComPtr<ID3D11VideoDevice>;
using D3D11VideoContextPtr = Microsoft::WRL::ComPtr<ID3D11VideoContext>;
using D3D11VideoProcessorPtr = Microsoft::WRL::ComPtr<ID3D11VideoProcessor>;
using D3D11VideoEnumPtr = Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator>;
using D3D11VideoInputViewPtr = Microsoft::WRL::ComPtr<ID3D11VideoProcessorInputView>;
using D3D11VideoOutputViewPtr = Microsoft::WRL::ComPtr<ID3D11VideoProcessorOutputView>;

class RGBFrameDecoder {
public:
    struct RGBFrame {
        D3D11TexturePtr rgb_texture;
        D3D11SRVPtr rgb_srv;
        double timestamp = 0.0;
        bool is_valid = false;
        int width = 0;
        int height = 0;
        
        // 智能指针自动清理，无需手动release
        void reset() {
            rgb_texture.Reset();
            rgb_srv.Reset();
            is_valid = false;
            width = 0;
            height = 0;
        }
        
        // 检查资源是否有效
        bool hasValidResources() const {
            return rgb_texture && rgb_srv && is_valid;
        }
    };
    
    struct RGBFramePair {
        HwFrameDecoder::HwFrame audio_frame;
        RGBFrame rgb_frame;
        bool is_valid = false;
    };

    RGBFrameDecoder();
    ~RGBFrameDecoder();

    // 使用外部提供的D3D11设备进行初始化（必须提供有效设备）
    bool open(const std::string& filepath, ID3D11Device* external_device);
    
    // Pull-style解码接口 - 返回RGB转换后的帧和音频帧
    bool readNextRGBFramePair(RGBFramePair& rgb_pair);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isHardwareAccelerated() const { return frame_decoder_.isHardwareAccelerated(); }
    const char* getVideoCodecName() const { return frame_decoder_.getVideoCodecName(); }
    const char* getAudioCodecName() const { return frame_decoder_.getAudioCodecName(); }
    
    // 获取视频信息
    int getVideoWidth() const { return video_width_; }
    int getVideoHeight() const { return video_height_; }
    
    // 获取内部HwFrameDecoder的访问（用于测试）
    HwFrameDecoder* getFrameDecoder() { return &frame_decoder_; }
    
    // 获取D3D11设备（用于设备共享）
    ID3D11Device* getD3D11Device() const { return d3d11_device_; }
    ID3D11DeviceContext* getD3D11Context() const { return d3d11_context_; }
    
    // 资源管理
    void flush();
    void close();

private:
    // 内部HwFrameDecoder实例
    HwFrameDecoder frame_decoder_;
    
    // D3D11设备（不拥有，只使用）
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // 视频信息
    int video_width_;
    int video_height_;
    bool is_initialized_;
    
    // 纹理池 - 改为智能指针管理，避免生命周期问题
    struct TextureSlot {
        D3D11TexturePtr texture;
        D3D11SRVPtr srv;
        int width = 0;
        int height = 0;
        bool is_created = false;
        
        void reset() {
            texture.Reset();
            srv.Reset();
            width = 0;
            height = 0;
            is_created = false;
        }
    };
    static const int TEXTURE_POOL_SIZE = 3;
    TextureSlot texture_pool_[TEXTURE_POOL_SIZE];
    int current_slot_index_;
    
    // Video Processor使用智能指针管理
    D3D11VideoDevicePtr video_device_;
    D3D11VideoContextPtr video_context_;
    D3D11VideoEnumPtr video_enum_;
    D3D11VideoProcessorPtr video_processor_;
    bool video_processor_initialized_;
    
    // 内部方法
    bool convertNV12ToRGB(const HwFrameDecoder::HwFrame& nv12_frame, TextureSlot* slot);
    bool ensureVideoProcessor();
    bool createTextureSlot(TextureSlot* slot, int width, int height);
    void releaseTextureSlot(TextureSlot* slot);
    void releaseVideoProcessor();
    void releaseResources();
};