#pragma once

#include <string>
#include <memory>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
}

#include "mkv_stream_reader.h"

class AudioDecoder {
public:
    struct DecodedFrame {
        AVFrame* frame;
        bool is_valid;
        bool is_eof;
        
        DecodedFrame() : frame(nullptr), is_valid(false), is_eof(false) {}
    };

    AudioDecoder();
    ~AudioDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    MKVStreamReader* getStreamReader() const;
    
    bool seekToTime(double seconds);
    bool seekToFrame(int64_t frame_number);

private:
    std::unique_ptr<MKVStreamReader> stream_reader_;
    
    AVCodecContext* codec_context_;
    AVFrame* audio_frames_[2];
    int current_frame_index_;
    
    bool initializeFFmpegAudioDecoder();
    bool processPacket(AVPacket* packet, DecodedFrame& frame);
    bool fillDecodedFrame(AVFrame* frame, DecodedFrame& decoded_frame);
    void cleanup();
};