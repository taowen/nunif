#pragma once

#include <atomic>
#include <memory>
#include <semaphore>
#include <d3d11.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// 简单的帧数据
struct SimpleVideoFrame {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    double timestamp = 0.0;
    int width = 0;
    int height = 0;
    bool is_valid = false;
    
    void reset() {
        texture.Reset();
        srv.Reset();
        is_valid = false;
    }
};

/**
 * 线程安全的帧同步器 - 支持背压控制
 * 
 * 特点：
 * 1. 双缓冲机制，零拷贝交换
 * 2. 信号量控制背压，生产者会被阻塞
 * 3. 原子操作保证线程安全
 * 4. 简单高效，无复杂队列逻辑
 */
class SimpleFrameSync {
public:
    SimpleFrameSync() 
        : write_index_(0)
        , read_index_(0)
        , render_ready_(0)  // 初始没有帧可消费
        , decode_ready_(1)  // 初始可以生产1帧
    {}
    
    ~SimpleFrameSync() = default;
    
    // 生产者接口 - 解码线程调用
    // 返回false表示应该停止解码（背压生效）
    bool submitFrame(ComPtr<ID3D11Texture2D> texture, 
                    ComPtr<ID3D11ShaderResourceView> srv,
                    double timestamp, int width, int height, 
                    int timeout_ms = 100) {
        
        // 等待渲染线程处理完上一帧（背压控制）
        if (!decode_ready_.try_acquire_for(std::chrono::milliseconds(timeout_ms))) {
            return false; // 超时，渲染线程太慢
        }
        
        // 写入新帧到写缓冲区
        int write_idx = write_index_.load();
        frames_[write_idx].texture = texture;
        frames_[write_idx].srv = srv;
        frames_[write_idx].timestamp = timestamp;
        frames_[write_idx].width = width;
        frames_[write_idx].height = height;
        frames_[write_idx].is_valid = true;
        
        // 原子交换读写索引
        write_index_.store(1 - write_idx);
        read_index_.store(write_idx);
        
        // 通知渲染线程有新帧
        render_ready_.release();
        
        return true;
    }
    
    // 消费者接口 - 渲染线程调用
    // 返回false表示没有新帧
    bool consumeFrame(SimpleVideoFrame& frame, int timeout_ms = 0) {
        
        // 检查是否有帧可用
        if (!render_ready_.try_acquire_for(std::chrono::milliseconds(timeout_ms))) {
            return false; // 没有新帧
        }
        
        // 读取帧数据
        int read_idx = read_index_.load();
        frame = std::move(frames_[read_idx]);
        frames_[read_idx].reset();
        
        // 通知解码线程可以继续
        decode_ready_.release();
        
        return true;
    }
    
    // 停止同步器
    void stop() {
        // 释放所有等待的线程
        render_ready_.release();
        decode_ready_.release();
    }
    
private:
    // 双缓冲帧存储
    SimpleVideoFrame frames_[2];
    
    // 原子索引
    std::atomic<int> write_index_;  // 解码线程写入索引
    std::atomic<int> read_index_;   // 渲染线程读取索引
    
    // 信号量 - 实现背压控制
    std::counting_semaphore<1> render_ready_;  // 渲染线程信号
    std::counting_semaphore<1> decode_ready_;  // 解码线程信号
};