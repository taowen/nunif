#pragma once

#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include "audio_decoder.h"

class AsyncAudioDecoder {
public:
    using DecodedFrame = AudioDecoder::DecodedFrame;

    AsyncAudioDecoder();
    ~AsyncAudioDecoder();

    bool open(const std::string& filepath);
    bool readNextFrame(DecodedFrame& frame);
    
    bool isOpen() const;
    bool isEOF() const;
    void close();
    
    MKVStreamReader* getStreamReader() const;
    
    bool seekToTime(double seconds);
    bool seekToFrame(int64_t frame_number);

private:
    std::unique_ptr<AudioDecoder> audio_decoder_;
    
    // 异步缓冲机制
    DecodedFrame next_frame_;
    DecodedFrame current_frame_;
    
    // 线程同步
    std::thread worker_thread_;
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    
    // 状态管理
    std::atomic<bool> next_frame_ready_;
    std::atomic<bool> should_stop_;
    std::atomic<bool> seek_requested_;
    
    void workerThreadFunc();
    void startWorkerThread();
    void stopWorkerThread();
    bool prepareNextFrame();
};