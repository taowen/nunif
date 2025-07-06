#include "async_rgb_frame_decoder.h"
#include <iostream>
#include <chrono>

AsyncRGBFrameDecoder::AsyncRGBFrameDecoder()
    : is_initialized_(false)
    , first_frame_loaded_(false)
{
}

AsyncRGBFrameDecoder::~AsyncRGBFrameDecoder() {
    close();
}

bool AsyncRGBFrameDecoder::open(const std::string& filepath) {
    if (is_initialized_) {
        close();
    }
    
    // 初始化底层RGB解码器
    if (!rgb_decoder_.open(filepath)) {
        return false;
    }
    
    // 重置状态
    is_initialized_ = true;
    first_frame_loaded_ = false;
    next_pair_ready_.store(false);
    worker_should_stop_.store(false);
    worker_running_.store(false);
    
    // 启动worker线程
    worker_thread_ = std::thread(&AsyncRGBFrameDecoder::workerThreadFunc, this);
    
    return true;
}

bool AsyncRGBFrameDecoder::readNextRGBFramePair(RGBFrameDecoder::RGBFramePair& pair) {
    if (!is_initialized_) {
        return false;
    }
    
    // 第一次读取时，直接从底层解码器获取
    if (!first_frame_loaded_) {
        RGBFrameDecoder::RGBFramePair rgb_pair;
        if (!rgb_decoder_.readNextRGBFramePair(rgb_pair)) {
            return false;
        }
        
        current_pair_ = rgb_pair;
        first_frame_loaded_ = true;
        
        // 通知worker线程开始预取下一帧
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            buffer_cv_.notify_one();
        }
    }
    
    // 等待worker线程准备好下一帧
    {
        std::unique_lock<std::mutex> lock(buffer_mutex_);
        buffer_cv_.wait(lock, [this] { 
            return next_pair_ready_.load() || worker_should_stop_.load(); 
        });
        
        if (worker_should_stop_.load()) {
            return false;
        }
        
        // 交换缓冲区
        swapBuffers();
        next_pair_ready_.store(false);
        
        // 通知worker线程继续预取
        buffer_cv_.notify_one();
    }
    
    // 返回当前帧
    pair = current_pair_;
    return pair.is_valid;
}

void AsyncRGBFrameDecoder::close() {
    if (is_initialized_) {
        stopWorker();
        rgb_decoder_.close();
        releaseResources();
        is_initialized_ = false;
    }
}

void AsyncRGBFrameDecoder::workerThreadFunc() {
    worker_running_.store(true);
    
    while (!worker_should_stop_.load()) {
        // 等待主线程的信号
        {
            std::unique_lock<std::mutex> lock(buffer_mutex_);
            buffer_cv_.wait(lock, [this] { 
                return worker_should_stop_.load() || 
                       (first_frame_loaded_ && !next_pair_ready_.load()); 
            });
            
            if (worker_should_stop_.load()) {
                break;
            }
        }
        
        // 预取下一帧
        RGBFrameDecoder::RGBFramePair rgb_pair;
        if (rgb_decoder_.readNextRGBFramePair(next_pair_)) {
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                next_pair_ready_.store(true);
            }
            buffer_cv_.notify_one();
        } else {
            // 读取失败，可能是EOF
            {
                std::lock_guard<std::mutex> lock(buffer_mutex_);
                next_pair_.is_valid = false;
                next_pair_ready_.store(true);
            }
            buffer_cv_.notify_one();
            break;
        }
    }
    
    worker_running_.store(false);
}

void AsyncRGBFrameDecoder::swapBuffers() {
    // 交换当前帧和下一帧
    std::swap(current_pair_, next_pair_);
}


void AsyncRGBFrameDecoder::releaseResources() {
    // 清理缓冲区
    current_pair_.rgb_frame.reset();
    current_pair_.is_valid = false;
    
    next_pair_.rgb_frame.reset();
    next_pair_.is_valid = false;
    
    first_frame_loaded_ = false;
}

void AsyncRGBFrameDecoder::stopWorker() {
    if (worker_thread_.joinable()) {
        // 通知worker线程停止
        worker_should_stop_.store(true);
        buffer_cv_.notify_all();
        
        // 等待worker线程结束
        worker_thread_.join();
    }
}