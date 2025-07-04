#pragma once

#include "media_sink.h"
#include "rgb_frame_decoder.h"
#include <memory>
#include <string>
#include <chrono>

/**
 * 媒体播放器类
 * 连接RGB解码器和媒体输出Sink
 */
class MediaPlayer {
public:
    MediaPlayer();
    ~MediaPlayer();

    /**
     * 初始化播放器
     * @param sink 媒体输出sink（CLI或GUI实现）
     * @return 是否成功初始化
     */
    bool initialize(std::unique_ptr<IMediaSink> sink);

    /**
     * 打开媒体文件
     * @param filepath 媒体文件路径
     * @return 是否成功打开
     */
    bool openFile(const std::string& filepath);

    /**
     * 播放指定数量的帧
     * @param frame_count 要播放的帧数（0表示播放到结束）
     * @return 实际播放的帧数
     */
    int playFrames(int frame_count = 0);

    /**
     * 播放整个文件
     * @return 总共播放的帧数
     */
    int playToEnd();

    /**
     * 播放单帧（用于GUI循环）
     * @return 是否成功播放了一帧
     */
    bool playOneFrame();

    /**
     * 暂停/恢复播放
     */
    void pause();
    void resume();
    bool isPaused() const;

    /**
     * 获取播放信息
     */
    int getVideoWidth() const;
    int getVideoHeight() const;
    const char* getVideoCodecName() const;
    const char* getAudioCodecName() const;

    /**
     * 获取媒体sink（用于GUI访问）
     */
    IMediaSink* getMediaSink() const { return sink_.get(); }

    /**
     * 关闭播放器
     */
    void close();

private:
    std::unique_ptr<IMediaSink> sink_;
    std::unique_ptr<RGBFrameDecoder> decoder_;
    
    bool is_initialized_;
    bool is_file_open_;
    
    // 统计信息
    int total_frames_played_;
    int video_frames_played_;
    int audio_frames_played_;
    
    // 同步相关
    double last_video_timestamp_;
    double last_audio_timestamp_;
    std::chrono::high_resolution_clock::time_point playback_start_time_;
    
    // 内部方法
    bool initializeSink();
    bool processNextFrame();
    void updateStatistics(const RGBFrameDecoder::DecodedFrames& frames);
    void printPlaybackSummary();
};