#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <d3d11.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// 视频帧数据结构
struct VideoFrame {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11ShaderResourceView> srv;
    double timestamp;
    int width;
    int height;
    bool is_valid;
    
    VideoFrame() : timestamp(0.0), 
                  width(0), height(0), is_valid(false) {}
    
    VideoFrame(ID3D11Texture2D* tex, ID3D11ShaderResourceView* shader_view, 
               double ts, int w, int h) 
        : texture(tex), srv(shader_view), timestamp(ts), width(w), height(h), is_valid(true) {}
};

// 线程安全的帧队列
class FrameQueue {
public:
    static const size_t MAX_QUEUE_SIZE = 30; // 最多缓冲30帧
    
    FrameQueue();
    ~FrameQueue();
    
    // 生产者接口（解码线程使用）
    bool push(const VideoFrame& frame, int timeout_ms = 100);
    
    // 消费者接口（渲染线程使用）
    bool pop(VideoFrame& frame, int timeout_ms = 100);
    
    // 队列状态
    size_t size() const;
    bool empty() const;
    bool full() const;
    
    // 清空队列
    void clear();
    
    // 停止队列（用于线程退出）
    void stop();
    
private:
    mutable std::mutex mutex_;
    std::condition_variable not_full_;
    std::condition_variable not_empty_;
    std::queue<VideoFrame> queue_;
    bool stopped_;
};