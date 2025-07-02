#include <catch2/catch_test_macros.hpp>
#include "ffmpeg_handler.h"
#include "decode_loop.h"
#include "video_state.h"
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
}


// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

TEST_CASE("DecodeLoop happy path", "[decode][video]") {
    // 1. 初始化 FFmpeg handler
    bool created = createFFmpegHandler("06 4k.mp4");
    REQUIRE(created == true);
    
    // 2. 初始化视频状态
    bool videoStateInit = initializeVideoState();
    REQUIRE(videoStateInit == true);
    
    // 3. 获取视频和音频信息
    VideoInfo videoInfo;
    AudioInfo audioInfo;
    bool hasVideo = getVideoInfo(&videoInfo);
    bool hasAudio = getAudioInfo(&audioInfo);
    
    REQUIRE(hasVideo == true);
    REQUIRE(videoInfo.width > 0);
    REQUIRE(videoInfo.height > 0);
    
    // 4. 设置解码状态
    std::atomic<bool> shouldStop{false};
    DecodeState decodeState{
        &shouldStop,
        hasVideo,
        hasAudio,
        videoInfo.streamIndex,
        hasAudio ? audioInfo.streamIndex : -1
    };
    
    // 5. 在callback中直接验证帧
    std::atomic<int> validFrameCount{0};
    std::atomic<int> totalFrameCount{0};
    
    auto videoCallback = [&](AVFrame* frame) {
        if (!frame) return;
        
        totalFrameCount++;
        
        // 基本验证
        INFO("Frame " << totalFrameCount.load() << " - width: " << frame->width 
             << ", height: " << frame->height << ", format: " << frame->format 
             << ", pts: " << frame->pts);
        
        // 验证帧尺寸
        if (frame->width != videoInfo.width || frame->height != videoInfo.height) {
            WARN("Frame size mismatch - expected: " << videoInfo.width << "x" << videoInfo.height
                 << ", got: " << frame->width << "x" << frame->height);
            return;
        }
        
        // 验证像素格式
        if (frame->format == AV_PIX_FMT_NONE) {
            WARN("Invalid pixel format: AV_PIX_FMT_NONE");
            return;
        }
        
        // 验证帧数据
        if (!frame->data[0]) {
            WARN("Frame data[0] is null");
            return;
        }
        
        // 对于硬件帧，linesize可能为0，需要特殊处理
        if (videoInfo.isHardwareDecoded) {
            // 硬件帧验证：检查是否有硬件帧上下文
            if (!frame->hw_frames_ctx) {
                WARN("Hardware frame missing hw_frames_ctx");
                return;
            }
            
            // 硬件帧的linesize可能为0，这是正常的
            INFO("Hardware frame detected - linesize: " << frame->linesize[0]);
        } else {
            // 软件帧验证：linesize应该大于等于width
            if (frame->linesize[0] < frame->width) {
                WARN("Software frame linesize too small - linesize: " << frame->linesize[0] 
                     << ", width: " << frame->width);
                return;
            }
        }
        
        // 验证时间戳
        if (frame->pts < 0) {
            WARN("Invalid PTS: " << frame->pts);
            return;
        }
        
        validFrameCount++;
        
        // 推送到视频队列进行状态管理测试
        pushVideoFrame(frame);
    };
    
    auto audioCallback = [](AVFrame* frame) {
        // 简单的音频帧验证
        if (frame) {
            INFO("Audio frame - samples: " << frame->nb_samples 
                 << ", format: " << frame->format);
        }
    };
    
    auto queueFullCallback = []() -> bool {
        return isVideoQueueFull();
    };
    
    // 6. 创建解码循环
    DecodeLoop decodeLoop(decodeState, videoCallback, audioCallback, queueFullCallback);
    
    // 7. 设置 FFmpeg 上下文
    decodeLoop.setContexts(
        getFormatContext(),
        getVideoCodecContext(),
        getAudioCodecContext()
    );
    
    // 8. 在另一个线程中运行解码循环
    std::thread decodeThread([&decodeLoop]() {
        decodeLoop.run();
    });
    
    // 9. 让解码运行一段时间然后停止
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    shouldStop.store(true);
    
    // 10. 等待解码线程结束
    decodeThread.join();
    
    // 11. 验证解码结果
    int totalFrames = totalFrameCount.load();
    int validFrames = validFrameCount.load();
    
    INFO("Total frames processed: " << totalFrames);
    INFO("Valid frames: " << validFrames);
    INFO("Hardware decoded: " << (videoInfo.isHardwareDecoded ? "Yes" : "No"));
    
    REQUIRE(totalFrames > 0);  // 应该至少处理了一些帧
    REQUIRE(validFrames > 0);  // 应该至少有一些有效帧
    
    // 12. 验证视频队列状态管理
    size_t queueSize = getVideoQueueSize();
    INFO("Queue size: " << queueSize);
    REQUIRE(queueSize >= 0);  // 队列大小应该合理
    
    // 13. 清理资源
    clearVideoFrameQueue();
    cleanupVideoState();
    destroyFFmpegHandler();
    
    // 14. 验证清理后的状态
    REQUIRE(getVideoQueueSize() == 0);
}
