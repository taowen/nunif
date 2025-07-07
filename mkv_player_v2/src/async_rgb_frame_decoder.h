#pragma once

#include "rgb_frame_decoder.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

class AsyncRGBFrameDecoder {
public:

    AsyncRGBFrameDecoder();
    ~AsyncRGBFrameDecoder();

    // 初始化异步解码器
    bool open(const std::string& filepath);
    
    // 异步读取接口 - 返回当前帧，同时在后台预取下一帧
    bool readNextRGBFramePair(RGBFrameDecoder::RGBFramePair& pair);
    
    // 状态查询
    bool isInitialized() const { return is_initialized_; }
    bool isWorkerRunning() const { return worker_running_; }
    
    // 获取内部组件（用于测试）
    RGBFrameDecoder* getRGBDecoder() { return &rgb_decoder_; }
    
    // 获取D3D11设备（用于GUI渲染，设备可以跨线程共享）
    ID3D11Device* getD3D11Device();
    
    // 获取视频帧率
    double getVideoFPS();
    
    // 资源管理
    void close();

private:
    // 底层RGB解码器
    RGBFrameDecoder rgb_decoder_;
    
    // 异步双缓冲机制
    RGBFrameDecoder::RGBFramePair current_pair_;   // 当前可用的帧对
    RGBFrameDecoder::RGBFramePair next_pair_;      // 后台预取的帧对
    
    // 线程同步
    std::mutex buffer_mutex_;
    std::condition_variable buffer_cv_;
    std::atomic<bool> next_pair_ready_{false};
    std::atomic<bool> worker_should_stop_{false};
    std::atomic<bool> worker_running_{false};
    
    // Worker线程
    std::thread worker_thread_;
    
    // 状态
    bool is_initialized_;
    bool first_frame_loaded_;
    
    // 时间戳验证，避免重复帧
    int64_t last_frame_timestamp_;
    uint64_t frame_sequence_number_;
    uint64_t last_returned_sequence_;
    
    // 内部方法
    void workerThreadFunc();
    void swapBuffers();
    void releaseResources();
    void stopWorker();
};