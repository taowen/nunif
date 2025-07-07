#include "hw_video_decoder.h"
#include <iostream>

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , hw_frame_(nullptr)
    , sw_frame_(nullptr)
    , hw_device_ctx_(nullptr)
    , hw_frames_ctx_(nullptr)
    , is_open_(false)
    , is_eof_(false)
    , width_(0)
    , height_(0)
    , fps_(0.0)
    , duration_(0.0) {
}

HwVideoDecoder::~HwVideoDecoder() {
    close();
}

bool HwVideoDecoder::open(const std::string& filepath) {
    if (is_open_) {
        close();
    }
    
    if (!stream_reader_->open(filepath)) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    auto stream_info = stream_reader_->getStreamInfo();
    width_ = stream_info.width;
    height_ = stream_info.height;
    fps_ = stream_info.fps;
    duration_ = stream_info.duration;
    
    if (!initializeDirectX()) {
        std::cerr << "Failed to initialize DirectX 11" << std::endl;
        close();
        return false;
    }
    
    if (!initializeFFmpegHWDecoder()) {
        std::cerr << "Failed to initialize FFmpeg hardware decoder" << std::endl;
        close();
        return false;
    }
    
    if (!createHWFramesContext()) {
        std::cerr << "Failed to create hardware frames context" << std::endl;
        close();
        return false;
    }
    
    hw_frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    if (!hw_frame_ || !sw_frame_) {
        std::cerr << "Failed to allocate frames" << std::endl;
        close();
        return false;
    }
    
    is_open_ = true;
    is_eof_ = false;
    
    return true;
}

bool HwVideoDecoder::readNextFrame(DecodedFrame& frame) {
    if (!is_open_ || is_eof_) {
        return false;
    }
    
    frame.is_valid = false;
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        return false;
    }
    
    while (stream_reader_->readNextPacket(packet)) {
        if (stream_reader_->isVideoPacket(packet)) {
            bool success = processPacket(packet, frame);
            av_packet_unref(packet);
            
            if (success && frame.is_valid) {
                av_packet_free(&packet);
                return true;
            }
        } else {
            av_packet_unref(packet);
        }
    }
    
    av_packet_free(&packet);
    is_eof_ = stream_reader_->isEOF();
    return false;
}

bool HwVideoDecoder::isOpen() const {
    return is_open_;
}

bool HwVideoDecoder::isEOF() const {
    return is_eof_;
}

void HwVideoDecoder::close() {
    cleanup();
    
    if (stream_reader_) {
        stream_reader_->close();
    }
    
    is_open_ = false;
    is_eof_ = false;
}

int HwVideoDecoder::getWidth() const {
    return width_;
}

int HwVideoDecoder::getHeight() const {
    return height_;
}

double HwVideoDecoder::getFPS() const {
    return fps_;
}

double HwVideoDecoder::getDuration() const {
    return duration_;
}

bool HwVideoDecoder::seekToTime(double seconds) {
    if (!is_open_) {
        return false;
    }
    
    bool success = stream_reader_->seekToTime(seconds);
    if (success) {
        is_eof_ = false;
        avcodec_flush_buffers(codec_context_);
    }
    return success;
}

bool HwVideoDecoder::seekToFrame(int64_t frame_number) {
    if (!is_open_) {
        return false;
    }
    
    bool success = stream_reader_->seekToFrame(frame_number);
    if (success) {
        is_eof_ = false;
        avcodec_flush_buffers(codec_context_);
    }
    return success;
}

bool HwVideoDecoder::initializeDirectX() {
    HRESULT hr;
    
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    
    D3D_FEATURE_LEVEL featureLevels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    
    hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        createDeviceFlags,
        featureLevels,
        ARRAYSIZE(featureLevels),
        D3D11_SDK_VERSION,
        &d3d11_device_,
        nullptr,
        &d3d11_context_
    );
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create D3D11 device: " << std::hex << hr << std::endl;
        return false;
    }
    
    return true;
}

bool HwVideoDecoder::initializeFFmpegHWDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getVideoCodecParameters();
    if (!codecpar) {
        std::cerr << "No video codec parameters" << std::endl;
        return false;
    }
    
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "Codec not found for codec_id: " << codecpar->codec_id << std::endl;
        return false;
    }
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate codec context" << std::endl;
        return false;
    }
    
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy codec parameters to context" << std::endl;
        return false;
    }
    
    hw_device_ctx_ = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
    if (!hw_device_ctx_) {
        std::cerr << "Failed to allocate hardware device context" << std::endl;
        return false;
    }
    
    AVHWDeviceContext* hw_device_context = (AVHWDeviceContext*)hw_device_ctx_->data;
    AVD3D11VADeviceContext* d3d11_device_context = (AVD3D11VADeviceContext*)hw_device_context->hwctx;
    
    d3d11_device_context->device = d3d11_device_.Get();
    d3d11_device_context->device_context = d3d11_context_.Get();
    
    d3d11_device_.Get()->AddRef();
    d3d11_context_.Get()->AddRef();
    
    if (av_hwdevice_ctx_init(hw_device_ctx_) < 0) {
        std::cerr << "Failed to initialize hardware device context" << std::endl;
        return false;
    }
    
    codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        return false;
    }
    
    return true;
}

bool HwVideoDecoder::createHWFramesContext() {
    hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ctx_) {
        std::cerr << "Failed to allocate hardware frames context" << std::endl;
        return false;
    }
    
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx_->data;
    frames_ctx->format = AV_PIX_FMT_D3D11;
    frames_ctx->sw_format = codec_context_->sw_pix_fmt;
    frames_ctx->width = codec_context_->width;
    frames_ctx->height = codec_context_->height;
    frames_ctx->initial_pool_size = 10;
    
    if (av_hwframe_ctx_init(hw_frames_ctx_) < 0) {
        std::cerr << "Failed to initialize hardware frames context" << std::endl;
        return false;
    }
    
    return true;
}

bool HwVideoDecoder::processPacket(AVPacket* packet, DecodedFrame& frame) {
    int ret = avcodec_send_packet(codec_context_, packet);
    if (ret < 0) {
        std::cerr << "Error sending packet to decoder: " << ret << std::endl;
        return false;
    }
    
    ret = avcodec_receive_frame(codec_context_, hw_frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving frame from decoder: " << ret << std::endl;
        return false;
    }
    
    return convertFrameToTexture(hw_frame_, frame);
}

bool HwVideoDecoder::convertFrameToTexture(AVFrame* frame, DecodedFrame& decoded_frame) {
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Frame is not in D3D11 format" << std::endl;
        return false;
    }
    
    ID3D11Texture2D* texture = (ID3D11Texture2D*)frame->data[0];
    int texture_index = (intptr_t)frame->data[1];
    
    D3D11_TEXTURE2D_DESC texture_desc;
    texture->GetDesc(&texture_desc);
    
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = 0;
    texture_desc.ArraySize = 1;
    
    ComPtr<ID3D11Texture2D> output_texture;
    HRESULT hr = d3d11_device_->CreateTexture2D(&texture_desc, nullptr, &output_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output texture" << std::endl;
        return false;
    }
    
    d3d11_context_->CopySubresourceRegion(
        output_texture.Get(), 0, 0, 0, 0,
        texture, texture_index, nullptr
    );
    
    decoded_frame.texture = output_texture;
    decoded_frame.pts = frame->pts;
    decoded_frame.duration = frame->duration;
    decoded_frame.width = frame->width;
    decoded_frame.height = frame->height;
    decoded_frame.is_valid = true;
    
    return true;
}

void HwVideoDecoder::cleanup() {
    if (hw_frame_) {
        av_frame_free(&hw_frame_);
        hw_frame_ = nullptr;
    }
    
    if (sw_frame_) {
        av_frame_free(&sw_frame_);
        sw_frame_ = nullptr;
    }
    
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
    
    if (hw_frames_ctx_) {
        av_buffer_unref(&hw_frames_ctx_);
        hw_frames_ctx_ = nullptr;
    }
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
    
    d3d11_context_.Reset();
    d3d11_device_.Reset();
}