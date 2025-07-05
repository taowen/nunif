#include "media_player.h"
#include "gui_media_sink.h"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

MediaPlayer::MediaPlayer()
    : is_initialized_(false)
    , is_file_open_(false)
    , total_frames_played_(0)
    , video_frames_played_(0)
    , audio_frames_played_(0)
    , last_video_timestamp_(0.0)
    , last_audio_timestamp_(0.0) {
    
    decoder_ = std::make_unique<RGBFrameDecoder>();
}

MediaPlayer::~MediaPlayer() {
    close();
}

bool MediaPlayer::initialize(std::unique_ptr<IMediaSink> sink) {
    if (!sink) {
        std::cerr << "Error: No media sink provided" << std::endl;
        return false;
    }
    
    sink_ = std::move(sink);
    is_initialized_ = true;
    
    std::cout << "MediaPlayer initialized successfully" << std::endl;
    return true;
}

bool MediaPlayer::initializeWithDecoder(std::unique_ptr<IMediaSink> sink, std::unique_ptr<RGBFrameDecoder> decoder) {
    if (!sink || !decoder) {
        std::cerr << "Error: No media sink or decoder provided" << std::endl;
        return false;
    }
    
    sink_ = std::move(sink);
    decoder_ = std::move(decoder);
    is_initialized_ = true;
    is_file_open_ = true; // 解码器已经打开文件
    
    std::cout << "MediaPlayer initialized successfully with existing decoder" << std::endl;
    return true;
}

bool MediaPlayer::openFile(const std::string& filepath) {
    if (!is_initialized_) {
        std::cerr << "Error: MediaPlayer not initialized" << std::endl;
        return false;
    }
    
    if (is_file_open_) {
        std::cerr << "Error: File is already open. Use initializeWithDecoder() for pre-opened files." << std::endl;
        return false;
    }
    
    // 打开解码器（内部设备模式，用于CLI）
    if (!decoder_->open(filepath)) {
        std::cerr << "Error: Failed to open file with decoder: " << filepath << std::endl;
        return false;
    }
    
    is_file_open_ = true;
    
    std::cout << "\n=== Media File Opened ===" << std::endl;
    std::cout << "File: " << filepath << std::endl;
    std::cout << "Video: " << getVideoWidth() << "x" << getVideoHeight() 
              << " (" << getVideoCodecName() << ")" << std::endl;
    std::cout << "Audio: " << getAudioCodecName() << std::endl;
    std::cout << "=========================" << std::endl;
    
    return true;
}

int MediaPlayer::playFrames(int frame_count) {
    if (!is_file_open_) {
        std::cerr << "Error: No file is open" << std::endl;
        return 0;
    }
    
    int frames_played = 0;
    int target_frames = frame_count;
    
    std::cout << "\nStarting playback";
    if (target_frames > 0) {
        std::cout << " (" << target_frames << " frames)";
    }
    std::cout << "..." << std::endl;
    
    while (target_frames == 0 || frames_played < target_frames) {
        if (sink_->isPaused()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 暂停时短暂休眠
            continue;
        }
        
        if (!processNextFrame()) {
            std::cout << "\nEnd of file reached." << std::endl;
            break;
        }
        
        frames_played++;
        
        // 每100帧打印一次进度
        if (frames_played % 100 == 0) {
            std::cout << "\nProcessed " << frames_played << " frame groups..." << std::endl;
        }
    }
    
    std::cout << "\nPlayback completed." << std::endl;
    printPlaybackSummary();
    
    return frames_played;
}

int MediaPlayer::playToEnd() {
    return playFrames(0);
}

bool MediaPlayer::playOneFrame() {
    if (!is_file_open_) {
        return false;
    }
    
    if (sink_->isPaused()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        return true; // 暂停时返回true以继续循环
    }
    
    bool success = processNextFrame();
    if (success) {
        total_frames_played_++;
    }
    
    return success;
}

void MediaPlayer::pause() {
    if (sink_) {
        sink_->pause();
    }
}

void MediaPlayer::resume() {
    if (sink_) {
        sink_->resume();
    }
}

bool MediaPlayer::isPaused() const {
    return sink_ ? sink_->isPaused() : false;
}

int MediaPlayer::getVideoWidth() const {
    return decoder_ ? decoder_->getVideoWidth() : 0;
}

int MediaPlayer::getVideoHeight() const {
    return decoder_ ? decoder_->getVideoHeight() : 0;
}

const char* MediaPlayer::getVideoCodecName() const {
    return decoder_ ? decoder_->getVideoCodecName() : "Unknown";
}

const char* MediaPlayer::getAudioCodecName() const {
    return decoder_ ? decoder_->getAudioCodecName() : "Unknown";
}

void MediaPlayer::close() {
    if (sink_) {
        sink_->close();
    }
    
    if (decoder_) {
        decoder_->close();
    }
    
    is_file_open_ = false;
    is_initialized_ = false;
    
    std::cout << "\nMediaPlayer closed." << std::endl;
}

bool MediaPlayer::processNextFrame() {
    RGBFrameDecoder::DecodedFrames frames;
    
    // 从解码器读取下一帧
    bool success = decoder_->readNextFrames(frames);
    if (!success) {
        return false; // 文件结束或错误
    }
    
    // 处理视频帧
    if (frames.rgb_frame.is_valid) {
        sink_->onVideoFrame(
            frames.rgb_frame.rgb_texture.Get(),
            frames.rgb_frame.rgb_srv.Get(),
            frames.rgb_frame.timestamp,
            frames.rgb_frame.width,
            frames.rgb_frame.height
        );
        video_frames_played_++;
        last_video_timestamp_ = frames.rgb_frame.timestamp;
    }
    
    // 处理音频帧
    if (frames.audio_frame.is_valid) {
        // TODO: 转换AVFrame音频数据为int16_t格式
        // 这里暂时跳过音频处理，专注于视频
        // 实际实现需要从frames.audio_frame.frame中提取PCM数据
        
        audio_frames_played_++;
        last_audio_timestamp_ = frames.audio_frame.timestamp;
    }
    
    updateStatistics(frames);
    total_frames_played_++;
    
    return true;
}

void MediaPlayer::updateStatistics(const RGBFrameDecoder::DecodedFrames& frames) {
    // 更新统计信息
    // 这里可以添加更多的性能监控逻辑
}

void MediaPlayer::printPlaybackSummary() {
    std::cout << "\n=== Playback Summary ===" << std::endl;
    std::cout << "Total frame groups processed: " << total_frames_played_ << std::endl;
    std::cout << "Video frames played: " << video_frames_played_ << std::endl;
    std::cout << "Audio frames played: " << audio_frames_played_ << std::endl;
    std::cout << "Last video timestamp: " << std::fixed << std::setprecision(3) 
              << last_video_timestamp_ << "s" << std::endl;
    std::cout << "Last audio timestamp: " << std::fixed << std::setprecision(3) 
              << last_audio_timestamp_ << "s" << std::endl;
    std::cout << "========================" << std::endl;
}