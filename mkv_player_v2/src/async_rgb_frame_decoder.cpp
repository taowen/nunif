#include "async_rgb_frame_decoder.h"
#include <iostream>
#include <chrono>

AsyncRGBFrameDecoder::AsyncRGBFrameDecoder()
    : is_initialized_(false)
    , first_frame_loaded_(false)
    , last_frame_timestamp_(AV_NOPTS_VALUE)
    , frame_sequence_number_(0)
    , last_returned_sequence_(0)
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
    
    // 重置时间戳和序号
    last_frame_timestamp_ = AV_NOPTS_VALUE;
    frame_sequence_number_ = 0;
    last_returned_sequence_ = 0;
    
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
            std::cerr << "AsyncRGBFrameDecoder: Failed to read first RGB frame pair" << std::endl;
            return false;
        }
        
        current_pair_ = rgb_pair;
        first_frame_loaded_ = true;
        
        // 记录第一帧的时间戳和序号
        if (current_pair_.rgb_frame.is_valid) {
            last_frame_timestamp_ = current_pair_.rgb_frame.timestamp;
            frame_sequence_number_ = 1;
            last_returned_sequence_ = 1;
        }
        
        // 立即返回第一帧，同时通知worker线程开始预取下一帧
        pair = current_pair_;
        
        // 通知worker线程开始预取下一帧
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            buffer_cv_.notify_one();
        }
        
        return pair.is_valid;
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
            // 验证是否为新帧（防止worker线程重复读取相同帧）
            bool is_valid_new_frame = false;
            
            if (next_pair_.rgb_frame.is_valid) {
                // 检查时间戳是否比已返回的最新帧更新
                if (next_pair_.rgb_frame.timestamp > last_frame_timestamp_) {
                    is_valid_new_frame = true;
                } else {
                    std::cerr << "Worker thread: Skipping frame with old timestamp " 
                              << next_pair_.rgb_frame.timestamp << " (last: " << last_frame_timestamp_ << ")" << std::endl;
                }
            }
            
            if (is_valid_new_frame) {
                {
                    std::lock_guard<std::mutex> lock(buffer_mutex_);
                    next_pair_ready_.store(true);
                }
                buffer_cv_.notify_one();
            } else {
                // 跳过这个旧帧，继续尝试读取下一帧
                continue;
            }
        } else {
            // 读取失败，可能是EOF或暂时错误
            std::cerr << "AsyncRGBFrameDecoder: Worker thread failed to read next RGB frame pair" << std::endl;
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
    // 验证下一帧是否真的是新帧
    bool is_new_frame = false;
    
    if (next_pair_.rgb_frame.is_valid) {
        // 检查时间戳是否前进
        if (next_pair_.rgb_frame.timestamp > last_frame_timestamp_) {
            is_new_frame = true;
            last_frame_timestamp_ = next_pair_.rgb_frame.timestamp;
            frame_sequence_number_++;
        } else {
            std::cerr << "WARNING: Next frame timestamp (" << next_pair_.rgb_frame.timestamp 
                      << ") is not newer than last (" << last_frame_timestamp_ 
                      << "), skipping frame to avoid flashback" << std::endl;
        }
    }
    
    // 只有确实是新帧时才交换
    if (is_new_frame) {
        std::swap(current_pair_, next_pair_);
        last_returned_sequence_ = frame_sequence_number_;
    } else {
        // 保持当前帧不变，标记next_pair为无效以便worker线程重新获取
        next_pair_.is_valid = false;
        next_pair_.rgb_frame.is_valid = false;
        next_pair_.audio_frame.is_valid = false;
    }
}


void AsyncRGBFrameDecoder::releaseResources() {
    // 清理缓冲区
    current_pair_.rgb_frame.reset();
    current_pair_.is_valid = false;
    
    next_pair_.rgb_frame.reset();
    next_pair_.is_valid = false;
    
    first_frame_loaded_ = false;
}

ID3D11Device* AsyncRGBFrameDecoder::getD3D11Device() {
    if (!is_initialized_) {
        return nullptr;
    }
    
    // 通过内部RGBFrameDecoder获取D3D11设备
    HwFrameDecoder* hw_decoder = rgb_decoder_.getFrameDecoder();
    if (!hw_decoder) {
        return nullptr;
    }
    
    // 尝试解码第一帧以获取设备
    AVFrame* temp_frame = av_frame_alloc();
    if (!temp_frame) {
        return nullptr;
    }
    
    if (!hw_decoder->tryDecodeFirstVideoFrame(temp_frame)) {
        av_frame_free(&temp_frame);
        return nullptr;
    }
    
    ID3D11Device* device = HwFrameDecoder::getD3D11DeviceFromFrame(temp_frame);
    av_frame_free(&temp_frame);
    
    return device;
}

double AsyncRGBFrameDecoder::getVideoFPS() {
    if (!is_initialized_) {
        return 0.0;
    }
    
    // 通过内部RGBFrameDecoder获取帧率
    HwFrameDecoder* hw_decoder = rgb_decoder_.getFrameDecoder();
    if (!hw_decoder) {
        return 0.0;
    }
    
    // 获取底层的MKVStreamReader
    MKVStreamReader* reader = hw_decoder->getReader();
    if (!reader) {
        return 0.0;
    }
    
    // 获取流信息并返回帧率
    MKVStreamReader::StreamInfo info = reader->getStreamInfo();
    return info.fps;
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