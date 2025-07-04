#pragma once

#include <d3d11.h>
#include <cstdint>

/**
 * 媒体播放Sink接口
 * 被动接收解码后的音视频数据进行播放
 */
class IMediaSink {
public:
    virtual ~IMediaSink() = default;

    /**
     * 初始化sink
     * @param video_width 视频宽度
     * @param video_height 视频高度
     * @param audio_sample_rate 音频采样率
     * @param audio_channels 音频声道数
     * @return 是否成功初始化
     */
    virtual bool initialize(int video_width, int video_height, 
                           int audio_sample_rate, int audio_channels) = 0;

    /**
     * 接收RGB视频帧
     * @param rgb_texture D3D11 RGB纹理（RGBA格式）
     * @param rgb_srv 对应的着色器资源视图
     * @param timestamp 时间戳（秒）
     * @param width 帧宽度
     * @param height 帧高度
     */
    virtual void onVideoFrame(ID3D11Texture2D* rgb_texture, 
                             ID3D11ShaderResourceView* rgb_srv,
                             double timestamp, int width, int height) = 0;

    /**
     * 接收PCM音频数据
     * @param samples 音频样本数据（交错格式）
     * @param sample_count 样本数量
     * @param timestamp 时间戳（秒）
     * @param sample_rate 采样率
     * @param channels 声道数
     */
    virtual void onAudioFrame(const int16_t* samples, int sample_count,
                             double timestamp, int sample_rate, int channels) = 0;

    /**
     * 获取当前播放时间（用于同步）
     * @return 当前播放时间戳（秒）
     */
    virtual double getCurrentTime() const = 0;

    /**
     * 检查是否应该跳过当前帧（用于同步控制）
     * @param timestamp 帧时间戳
     * @return true表示应该跳过此帧
     */
    virtual bool shouldSkipFrame(double timestamp) const = 0;

    /**
     * 暂停/恢复播放
     */
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual bool isPaused() const = 0;

    /**
     * 关闭并清理资源
     */
    virtual void close() = 0;
};