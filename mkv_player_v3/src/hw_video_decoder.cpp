#include "hw_video_decoder.h"
#include <iostream>

HwVideoDecoder::HwVideoDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , hw_frame_(nullptr)
    , sw_frame_(nullptr)
    , hw_device_ctx_(nullptr)
    , is_open_(false)
    , is_eof_(false) {
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
    
    if (!initializeFFmpegHWDecoder()) {
        std::cerr << "Failed to initialize FFmpeg hardware decoder" << std::endl;
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

MKVStreamReader* HwVideoDecoder::getStreamReader() const {
    return stream_reader_.get();
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
    
    // 让FFmpeg自动创建和管理D3D11VA设备
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
    if (ret < 0) {
        std::cerr << "Failed to create D3D11VA device context: " << ret << std::endl;
        return false;
    }
    
    codec_context_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
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
    
    return fillDecodedFrame(hw_frame_, frame);
}

bool HwVideoDecoder::fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame) {
    decoded_frame.frame = frame;
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
    
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
        hw_device_ctx_ = nullptr;
    }
}