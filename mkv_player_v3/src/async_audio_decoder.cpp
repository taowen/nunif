#include "async_audio_decoder.h"
#include <iostream>

AsyncAudioDecoder::AsyncAudioDecoder()
    : audio_decoder_(std::make_unique<AudioDecoder>())
    , next_frame_ready_(false)
    , should_stop_(false)
    , seek_requested_(false) {
}

AsyncAudioDecoder::~AsyncAudioDecoder() {
    close();
}

bool AsyncAudioDecoder::open(const std::string& filepath) {
    MKVStreamReader* stream_reader = getStreamReader();
    if (stream_reader && stream_reader->isOpen()) {
        close();
    }
    
    if (!audio_decoder_->open(filepath)) {
        return false;
    }
    
    startWorkerThread();
    return true;
}

bool AsyncAudioDecoder::readNextFrame(DecodedFrame& frame) {
    MKVStreamReader* stream_reader = getStreamReader();
    if (!stream_reader || !stream_reader->isOpen()) {
        frame.is_eof = true;
        return false;
    }
    
    // 等待下一帧准备好
    std::unique_lock<std::mutex> lock(frame_mutex_);
    frame_cv_.wait(lock, [this] { 
        MKVStreamReader* sr = getStreamReader();
        return next_frame_ready_.load() || (sr && sr->isEOF()) || should_stop_.load(); 
    });
    
    if (should_stop_.load()) {
        frame.is_eof = true;
        return false;
    }
    
    stream_reader = getStreamReader();
    if (stream_reader && stream_reader->isEOF() && !next_frame_ready_.load()) {
        frame.is_eof = true;
        return false;
    }
    
    // 交换帧数据
    frame = next_frame_;
    current_frame_ = next_frame_;
    next_frame_ready_ = false;
    
    // 通知工作线程准备下一帧
    frame_cv_.notify_all();
    
    return frame.is_valid;
}


void AsyncAudioDecoder::close() {
    stopWorkerThread();
    
    if (audio_decoder_) {
        audio_decoder_->close();
    }
}

MKVStreamReader* AsyncAudioDecoder::getStreamReader() const {
    return audio_decoder_ ? audio_decoder_->getStreamReader() : nullptr;
}

bool AsyncAudioDecoder::seekToTime(double seconds) {
    MKVStreamReader* stream_reader = getStreamReader();
    if (!stream_reader || !stream_reader->isOpen()) {
        return false;
    }
    
    // 停止工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = true;
        next_frame_ready_ = false;
    }
    frame_cv_.notify_all();
    
    // 执行seek操作
    bool success = audio_decoder_->seekToTime(seconds);
    
    // 重启工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = false;
    }
    frame_cv_.notify_all();
    
    return success;
}

bool AsyncAudioDecoder::seekToFrame(int64_t frame_number) {
    MKVStreamReader* stream_reader = getStreamReader();
    if (!stream_reader || !stream_reader->isOpen()) {
        return false;
    }
    
    // 停止工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = true;
        next_frame_ready_ = false;
    }
    frame_cv_.notify_all();
    
    // 执行seek操作
    bool success = audio_decoder_->seekToFrame(frame_number);
    
    // 重启工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = false;
    }
    frame_cv_.notify_all();
    
    return success;
}

void AsyncAudioDecoder::workerThreadFunc() {
    while (!should_stop_.load()) {
        std::unique_lock<std::mutex> lock(frame_mutex_);
        
        // 等待需要准备下一帧或停止信号
        frame_cv_.wait(lock, [this] { 
            return !next_frame_ready_.load() || should_stop_.load() || seek_requested_.load();
        });
        
        if (should_stop_.load()) {
            break;
        }
        
        if (seek_requested_.load()) {
            // 等待seek完成
            frame_cv_.wait(lock, [this] { 
                return !seek_requested_.load() || should_stop_.load(); 
            });
            continue;
        }
        
        // 释放锁来执行解码
        lock.unlock();
        
        // 准备下一帧
        if (prepareNextFrame()) {
            std::lock_guard<std::mutex> frame_lock(frame_mutex_);
            next_frame_ready_ = true;
            frame_cv_.notify_all();
        } else {
            // 解码失败或EOF，通知主线程
            frame_cv_.notify_all();
        }
    }
}

void AsyncAudioDecoder::startWorkerThread() {
    should_stop_ = false;
    next_frame_ready_ = false;
    seek_requested_ = false;
    
    worker_thread_ = std::thread(&AsyncAudioDecoder::workerThreadFunc, this);
}

void AsyncAudioDecoder::stopWorkerThread() {
    if (worker_thread_.joinable()) {
        should_stop_ = true;
        frame_cv_.notify_all();
        worker_thread_.join();
    }
}

bool AsyncAudioDecoder::prepareNextFrame() {
    if (!audio_decoder_) {
        return false;
    }
    
    MKVStreamReader* stream_reader = getStreamReader();
    if (!stream_reader || !stream_reader->isOpen()) {
        return false;
    }
    
    return audio_decoder_->readNextFrame(next_frame_);
}