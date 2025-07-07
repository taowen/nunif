#pragma once

#include <d3d11.h>
#include <d3d11_1.h>
#include <wrl/client.h>
#include <string>
#include <memory>
#include "hw_video_decoder.h"

using Microsoft::WRL::ComPtr;

class RgbVideoDecoder {
public:
    struct RgbFrame {
        ComPtr<ID3D11Texture2D> rgb_texture;
        ComPtr<ID3D11ShaderResourceView> rgb_srv;
        ComPtr<ID3D11VideoProcessorOutputView> cached_output_view;
        int width;
        int height;
        double timestamp;
        bool is_valid;
        
        RgbFrame() : width(0), height(0), timestamp(0.0), is_valid(false) {}
        
        void reset() {
            rgb_texture.Reset();
            rgb_srv.Reset();
            cached_output_view.Reset();
            width = 0;
            height = 0;
            timestamp = 0.0;
            is_valid = false;
        }
        
        bool hasValidOutputView() const {
            return cached_output_view != nullptr;
        }
    };
    
    struct DecodedFrame {
        AVFrame* hw_frame;
        RgbFrame rgb_frame;
        bool is_valid;
        
        DecodedFrame() : hw_frame(nullptr), is_valid(false) {}
    };

    RgbVideoDecoder();
    ~RgbVideoDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    bool seekToTime(double seconds);
    bool seekToFrame(int64_t frame_number);
    
    // D3D11 resource getters for testing
    ID3D11Device* getD3D11Device() const { return d3d11_device_; }
    ID3D11DeviceContext* getD3D11Context() const { return d3d11_context_; }

private:
    std::unique_ptr<HwVideoDecoder> hw_decoder_;
    
    // D3D11 资源 (从hw_decoder_获取，不拥有)
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // Video Processor 资源
    ComPtr<ID3D11VideoDevice> video_device_;
    ComPtr<ID3D11VideoContext> video_context_;
    ComPtr<ID3D11VideoProcessorEnumerator> video_enum_;
    ComPtr<ID3D11VideoProcessor> video_processor_;
    bool video_processor_initialized_;
    
    // 双缓冲 RGB 帧
    RgbFrame rgb_frames_[2];
    int current_rgb_frame_index_;
    
    bool initializeVideoProcessor();
    bool convertNV12ToRGB(AVFrame* nv12_frame, RgbFrame& rgb_frame);
    bool createRGBTexture(RgbFrame& rgb_frame, int width, int height);
    void cleanup();
};