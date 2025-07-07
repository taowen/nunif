#include "audio_decoder.h"
#include <iostream>

AudioDecoder::AudioDecoder()
    : stream_reader_(std::make_unique<MKVStreamReader>())
    , codec_context_(nullptr)
    , current_frame_index_(0) {
    audio_frames_[0] = nullptr;
    audio_frames_[1] = nullptr;
}

AudioDecoder::~AudioDecoder() {
    close();
}

bool AudioDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    if (!stream_reader_->open(filepath)) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }
    
    if (!initializeFFmpegAudioDecoder()) {
        std::cerr << "Failed to initialize FFmpeg audio decoder" << std::endl;
        close();
        return false;
    }
    
    // 分配双缓冲frames
    audio_frames_[0] = av_frame_alloc();
    audio_frames_[1] = av_frame_alloc();
    if (!audio_frames_[0] || !audio_frames_[1]) {
        std::cerr << "Failed to allocate audio frames" << std::endl;
        close();
        return false;
    }
    
    current_frame_index_ = 0;
    
    return true;
}

bool AudioDecoder::readNextFrame(DecodedFrame& frame) {
    if (!isOpen() || isEOF()) {
        frame.is_eof = isEOF();
        return false;
    }
    
    frame.is_valid = false;
    frame.is_eof = false;
    
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        return false;
    }
    
    while (stream_reader_->readNextPacket(packet)) {
        if (stream_reader_->isAudioPacket(packet)) {
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
    frame.is_eof = stream_reader_->isEOF();
    return false;
}

bool AudioDecoder::isOpen() const {
    return stream_reader_ && stream_reader_->isOpen();
}

bool AudioDecoder::isEOF() const {
    return stream_reader_ && stream_reader_->isEOF();
}

void AudioDecoder::close() {
    cleanup();
    
    if (stream_reader_) {
        stream_reader_->close();
    }
}

MKVStreamReader* AudioDecoder::getStreamReader() const {
    return stream_reader_.get();
}

bool AudioDecoder::seekToTime(double seconds) {
    if (!isOpen()) {
        return false;
    }
    
    bool success = stream_reader_->seekToTime(seconds);
    if (success) {
        avcodec_flush_buffers(codec_context_);
    }
    return success;
}

bool AudioDecoder::seekToFrame(int64_t frame_number) {
    if (!isOpen()) {
        return false;
    }
    
    bool success = stream_reader_->seekToFrame(frame_number);
    if (success) {
        avcodec_flush_buffers(codec_context_);
    }
    return success;
}

bool AudioDecoder::initializeFFmpegAudioDecoder() {
    AVCodecParameters* codecpar = stream_reader_->getAudioCodecParameters();
    if (!codecpar) {
        std::cerr << "No audio codec parameters" << std::endl;
        return false;
    }
    
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "Audio codec not found for codec_id: " << codecpar->codec_id << std::endl;
        return false;
    }
    
    codec_context_ = avcodec_alloc_context3(codec);
    if (!codec_context_) {
        std::cerr << "Failed to allocate audio codec context" << std::endl;
        return false;
    }
    
    if (avcodec_parameters_to_context(codec_context_, codecpar) < 0) {
        std::cerr << "Failed to copy audio codec parameters to context" << std::endl;
        return false;
    }
    
    if (avcodec_open2(codec_context_, codec, nullptr) < 0) {
        std::cerr << "Failed to open audio codec" << std::endl;
        return false;
    }
    
    return true;
}

bool AudioDecoder::processPacket(AVPacket* packet, DecodedFrame& frame) {
    int ret = avcodec_send_packet(codec_context_, packet);
    if (ret < 0) {
        std::cerr << "Error sending audio packet to decoder: " << ret << std::endl;
        return false;
    }
    
    ret = avcodec_receive_frame(codec_context_, audio_frames_[current_frame_index_]);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        std::cerr << "Error receiving audio frame from decoder: " << ret << std::endl;
        return false;
    }
    
    return fillDecodedFrame(audio_frames_[current_frame_index_], frame);
}

bool AudioDecoder::fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame) {
    decoded_frame.frame = frame;
    decoded_frame.is_valid = true;
    decoded_frame.is_eof = false;
    
    // 轮换到下一个frame - 实现双缓冲
    current_frame_index_ = (current_frame_index_ + 1) % 2;
    
    return true;
}

void AudioDecoder::cleanup() {
    // 清理双缓冲frames
    for (int i = 0; i < 2; i++) {
        if (audio_frames_[i]) {
            av_frame_free(&audio_frames_[i]);
            audio_frames_[i] = nullptr;
        }
    }
    
    if (codec_context_) {
        avcodec_free_context(&codec_context_);
        codec_context_ = nullptr;
    }
    
    current_frame_index_ = 0;
}