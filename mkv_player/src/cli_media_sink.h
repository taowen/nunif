#pragma once

#include "media_sink.h"
#include <chrono>
#include <fstream>
#include <string>

/**
 * CLI媒体播放Sink实现
 * 将视频帧保存为BMP文件，音频丢弃或保存为WAV
 * 用于无头环境下的测试和验证
 */
class CLIMediaSink : public IMediaSink {
public:
    CLIMediaSink();
    ~CLIMediaSink() override;

    // IMediaSink接口实现
    bool initialize(int video_width, int video_height, 
                   int audio_sample_rate, int audio_channels) override;
    
    void onVideoFrame(ID3D11Texture2D* rgb_texture, 
                     ID3D11ShaderResourceView* rgb_srv,
                     double timestamp, int width, int height) override;
    
    void onAudioFrame(const int16_t* samples, int sample_count,
                     double timestamp, int sample_rate, int channels) override;
    
    void pause() override;
    void resume() override;
    bool isPaused() const override;
    
    void close() override;

    // CLI特有配置
    void setOutputDirectory(const std::string& dir) { output_dir_ = dir; }
    void setSaveVideoFrames(bool save) { save_video_frames_ = save; }
    void setSaveAudio(bool save) { save_audio_ = save; }
    void setFrameInterval(int interval) { frame_save_interval_ = interval; } // 每N帧保存一次
    void setMaxFramesToSave(int max_frames) { max_frames_to_save_ = max_frames; }
    void setPlaybackSpeed(double speed) { playback_speed_ = speed; } // 1.0 = 正常速度

private:
    // D3D11设备（用于纹理读取）
    ID3D11Device* d3d11_device_;
    ID3D11DeviceContext* d3d11_context_;
    
    // 时间控制
    std::chrono::high_resolution_clock::time_point start_time_;
    double current_time_;
    bool is_paused_;
    double playback_speed_;
    
    // 输出配置
    std::string output_dir_;
    bool save_video_frames_;
    bool save_audio_;
    int frame_save_interval_;
    int max_frames_to_save_;
    
    // 计数器
    int frame_count_;
    int saved_frame_count_;
    
    // 音频输出
    std::ofstream audio_file_;
    bool audio_file_header_written_;
    
    // 视频信息
    int video_width_;
    int video_height_;
    int audio_sample_rate_;
    int audio_channels_;
    
    // 内部方法
    bool saveFrameAsBMP(ID3D11Texture2D* texture, const std::string& filename);
    bool initializeAudioFile();
    void writeWAVHeader();
    void updateAudioFileHeader();
    std::string generateFrameFilename(int frame_number, double timestamp);
    void printStatus(double timestamp, const std::string& frame_info = "");
    
    // 需要引用rgb_verification的纹理读取功能
    bool readTextureData(ID3D11Texture2D* texture, 
                        std::vector<uint8_t>& pixel_data, 
                        int& width, int& height);
};