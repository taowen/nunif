#pragma once

#include <memory>
#include <string>
#include "rgb_frame_decoder.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

class ToCudaInputContext;

class CudaFrameDecoder {
public:
    struct CudaFrame {
        void* cuda_ptr = nullptr;
        size_t size = 0;
        double timestamp = 0.0;
        bool is_valid = false;
        int width = 0;
        int height = 0;
        int channels = 4;
        
        void release() {
            cuda_ptr = nullptr;
            size = 0;
            is_valid = false;
        }
    };
    
    struct DecodedFrames {
        FrameDecoder::DecodedFrame audio_frame;
        CudaFrame cuda_frame;
    };

    CudaFrameDecoder();
    ~CudaFrameDecoder();

    bool open(const std::string& filepath, ID3D11Device* external_device = nullptr);
    
    bool readNextFrames(DecodedFrames& decoded_frames);
    
    bool isInitialized() const { return rgb_decoder_.isInitialized(); }
    bool isHardwareAccelerated() const { return rgb_decoder_.isHardwareAccelerated(); }
    const char* getVideoCodecName() const { return rgb_decoder_.getVideoCodecName(); }
    const char* getAudioCodecName() const { return rgb_decoder_.getAudioCodecName(); }
    
    int getVideoWidth() const { return rgb_decoder_.getVideoWidth(); }
    int getVideoHeight() const { return rgb_decoder_.getVideoHeight(); }
    
    RGBFrameDecoder* getRGBFrameDecoder() { return &rgb_decoder_; }
    
    void flush();
    void close();
    
    void unmapCudaInput();

private:
    RGBFrameDecoder rgb_decoder_;
    std::unique_ptr<ToCudaInputContext> cuda_context_;
    
    bool is_initialized_;
    void* last_cuda_ptr_;
};