#include "async_rgb_video_decoder.h"
#include <iostream>

AsyncRgbVideoDecoder::AsyncRgbVideoDecoder()
    : rgb_decoder_(std::make_unique<RgbVideoDecoder>())
    , next_frame_ready_(false)
    , should_stop_(false)
    , seek_requested_(false) {
}

AsyncRgbVideoDecoder::~AsyncRgbVideoDecoder() {
    close();
}

bool AsyncRgbVideoDecoder::open(const std::string& filepath) {
    if (isOpen()) {
        close();
    }
    
    if (!rgb_decoder_->open(filepath)) {
        return false;
    }
    
    startWorkerThread();
    return true;
}

bool AsyncRgbVideoDecoder::readNextFrame(DecodedFrame& frame) {
    if (!isOpen()) {
        frame.is_valid = false;
        return false;
    }
    
    // 等待下一帧准备好
    std::unique_lock<std::mutex> lock(frame_mutex_);
    frame_cv_.wait(lock, [this] { 
        return next_frame_ready_.load() || isEOF() || should_stop_.load(); 
    });
    
    if (should_stop_.load()) {
        frame.is_valid = false;
        return false;
    }
    
    if (isEOF() && !next_frame_ready_.load()) {
        frame.is_valid = false;
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

void AsyncRgbVideoDecoder::close() {
    stopWorkerThread();
    
    if (rgb_decoder_) {
        rgb_decoder_->close();
    }
}

bool AsyncRgbVideoDecoder::isOpen() const {
    return rgb_decoder_ && rgb_decoder_->isOpen();
}

bool AsyncRgbVideoDecoder::isEOF() const {
    return rgb_decoder_ && rgb_decoder_->isEOF();
}

bool AsyncRgbVideoDecoder::seekToTime(double seconds) {
    if (!isOpen()) {
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
    bool success = rgb_decoder_->seekToTime(seconds);
    
    // 重启工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = false;
    }
    frame_cv_.notify_all();
    
    return success;
}

bool AsyncRgbVideoDecoder::seekToFrame(int64_t frame_number) {
    if (!isOpen()) {
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
    bool success = rgb_decoder_->seekToFrame(frame_number);
    
    // 重启工作线程
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        seek_requested_ = false;
    }
    frame_cv_.notify_all();
    
    return success;
}

ID3D11Device* AsyncRgbVideoDecoder::getD3D11Device() const {
    return rgb_decoder_ ? rgb_decoder_->getD3D11Device() : nullptr;
}

ID3D11DeviceContext* AsyncRgbVideoDecoder::getD3D11Context() const {
    return rgb_decoder_ ? rgb_decoder_->getD3D11Context() : nullptr;
}

void AsyncRgbVideoDecoder::workerThreadFunc() {
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

void AsyncRgbVideoDecoder::startWorkerThread() {
    should_stop_ = false;
    next_frame_ready_ = false;
    seek_requested_ = false;
    
    worker_thread_ = std::thread(&AsyncRgbVideoDecoder::workerThreadFunc, this);
}

void AsyncRgbVideoDecoder::stopWorkerThread() {
    if (worker_thread_.joinable()) {
        should_stop_ = true;
        frame_cv_.notify_all();
        worker_thread_.join();
    }
}

bool AsyncRgbVideoDecoder::prepareNextFrame() {
    if (!rgb_decoder_) {
        return false;
    }
    
    if (!isOpen()) {
        return false;
    }
    
    return rgb_decoder_->readNextFrame(next_frame_);
}