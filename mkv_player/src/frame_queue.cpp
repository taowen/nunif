#include "frame_queue.h"
#include <iostream>

FrameQueue::FrameQueue() : stopped_(false) {}

FrameQueue::~FrameQueue() {
    stop();
}

bool FrameQueue::push(const VideoFrame& frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 等待队列有空间
    auto timeout = std::chrono::milliseconds(timeout_ms);
    if (!not_full_.wait_for(lock, timeout, [this] { return queue_.size() < MAX_QUEUE_SIZE || stopped_; })) {
        std::cout << "Frame queue push timeout, dropping frame at " << frame.timestamp << "s" << std::endl;
        return false; // 超时，丢弃帧
    }
    
    if (stopped_) {
        return false;
    }
    
    queue_.push(frame);
    not_empty_.notify_one();
    
    static int push_count = 0;
    if (++push_count % 30 == 0) {
        std::cout << "Frame queue: pushed " << push_count << " frames, current size: " << queue_.size() << std::endl;
    }
    
    return true;
}

bool FrameQueue::pop(VideoFrame& frame, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 等待队列有数据
    auto timeout = std::chrono::milliseconds(timeout_ms);
    if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty() || stopped_; })) {
        return false; // 超时，没有帧可用
    }
    
    if (stopped_ && queue_.empty()) {
        return false;
    }
    
    frame = queue_.front();
    queue_.pop();
    not_full_.notify_one();
    
    return true;
}

size_t FrameQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool FrameQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

bool FrameQueue::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size() >= MAX_QUEUE_SIZE;
}

void FrameQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
    not_full_.notify_all();
}

void FrameQueue::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    not_full_.notify_all();
    not_empty_.notify_all();
}