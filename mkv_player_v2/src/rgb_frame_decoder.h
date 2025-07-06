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
        // 缓存视频处理器输出视图以避免每帧重新创建
        D3D11VideoOutputViewPtr cached_output_view;
        double timestamp = 0.0;
        bool is_valid = false;
        int width = 0;
        int height = 0;
        
        // 智能指针自动清理，无需手动release
        void reset() {
            rgb_texture.Reset();
            rgb_srv.Reset();
            cached_output_view.Reset();
            is_valid = false;
            width = 0;
            height = 0;
        }
        
        // 检查纹理资源是否存在（不检查帧有效性）
        bool hasValidResources() const {
            return rgb_texture && rgb_srv;
        }
        
        // 检查输出视图是否存在
        bool hasValidOutputView() const {
            return cached_output_view.Get() != nullptr;
        }
    };
    
    struct RGBFramePair {
        HwFrameDecoder::HwFrame audio_frame;
        RGBFrame rgb_frame;
        bool is_valid = false;
    };

    RGBFrameDecoder();
    ~RGBFrameDecoder();

    // 初始化 - 将从内部HwFrameDecoder获取D3D11设备
    bool open(const std::string& filepath);
    
    // Pull-style解码接口 - 返回RGB转换后的帧和音频帧
    bool readNextRGBFramePair(RGBFramePair& rgb_pair);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    
    // 获取内部组件
    HwFrameDecoder* getFrameDecoder() { return &frame_decoder_; }
    
    // 资源管理
    void close();

private:
    // 内部HwFrameDecoder实例
    HwFrameDecoder frame_decoder_;
    
    // D3D11设备（从HwFrameDecoder获取，不拥有）
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // 状态信息
    bool is_initialized_;
    
    // 双缓冲机制 - 支持两个RGBFramePair同时存在
    RGBFramePair borrowed_pairs_[2];
    int current_pair_index_;
    
    // Video Processor使用智能指针管理
    D3D11VideoDevicePtr video_device_;
    D3D11VideoContextPtr video_context_;
    D3D11VideoEnumPtr video_enum_;
    D3D11VideoProcessorPtr video_processor_;
    bool video_processor_initialized_;
    
    // 内部方法
    bool convertNV12ToRGB(const HwFrameDecoder::HwFrame& nv12_frame, RGBFrame& rgb_frame);
    bool ensureVideoProcessor();
    bool createRGBTexture(RGBFrame& rgb_frame, int width, int height);
    void releaseVideoProcessor();
    void releaseResources();
    void initializeBorrowedPairs();
};