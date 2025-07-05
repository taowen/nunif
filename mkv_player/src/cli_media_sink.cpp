#include "cli_media_sink.h"
#include "rgb_verification.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <cmath>
#include <thread>
#include <chrono>
#include <algorithm>

CLIMediaSink::CLIMediaSink()
    : d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , current_time_(0.0)
    , is_paused_(false)
    , playback_speed_(1.0)
    , output_dir_("output")
    , save_video_frames_(true)
    , save_audio_(false)
    , frame_save_interval_(1)  // 保存每一帧
    , max_frames_to_save_(100) // 最多保存100帧
    , frame_count_(0)
    , saved_frame_count_(0)
    , audio_file_header_written_(false)
    , video_width_(0)
    , video_height_(0)
    , audio_sample_rate_(0)
    , audio_channels_(0) {
}

CLIMediaSink::~CLIMediaSink() {
    close();
}

bool CLIMediaSink::initialize(int video_width, int video_height, 
                             int audio_sample_rate, int audio_channels) {
    video_width_ = video_width;
    video_height_ = video_height;
    audio_sample_rate_ = audio_sample_rate;
    audio_channels_ = audio_channels;
    
    // 创建输出目录
    try {
        std::filesystem::create_directories(output_dir_);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create output directory: " << e.what() << std::endl;
        return false;
    }
    
    // 初始化音频文件（如果需要）
    if (save_audio_) {
        if (!initializeAudioFile()) {
            std::cerr << "Failed to initialize audio output file" << std::endl;
            return false;
        }
    }
    
    start_time_ = std::chrono::high_resolution_clock::now();
    
    std::cout << "=== CLI Media Sink Initialized ===" << std::endl;
    std::cout << "Video: " << video_width_ << "x" << video_height_ << std::endl;
    std::cout << "Audio: " << audio_sample_rate_ << "Hz, " << audio_channels_ << " channels" << std::endl;
    std::cout << "Output directory: " << output_dir_ << std::endl;
    std::cout << "Save video frames: " << (save_video_frames_ ? "Yes" : "No") << std::endl;
    std::cout << "Save audio: " << (save_audio_ ? "Yes" : "No") << std::endl;
    std::cout << "Frame save interval: " << frame_save_interval_ << std::endl;
    std::cout << "Max frames to save: " << max_frames_to_save_ << std::endl;
    std::cout << "Playback speed: " << playback_speed_ << "x" << std::endl;
    std::cout << "=================================" << std::endl;
    
    return true;
}

void CLIMediaSink::onVideoFrame(ID3D11Texture2D* rgb_texture, 
                               ID3D11ShaderResourceView* rgb_srv,
                               double timestamp, int width, int height) {
    frame_count_++;
    
    // 更新当前时间
    current_time_ = timestamp;
    
    // 检查是否需要保存这一帧
    bool should_save = save_video_frames_ && 
                      (frame_count_ % frame_save_interval_ == 0) &&
                      (saved_frame_count_ < max_frames_to_save_);
    
    if (should_save) {
        std::string filename = generateFrameFilename(saved_frame_count_, timestamp);
        std::string filepath = output_dir_ + "/" + filename;
        
        // 保存帧为BMP
        if (saveFrameAsBMP(rgb_texture, filepath)) {
            saved_frame_count_++;
            printStatus(timestamp, "Saved: " + filename);
        } else {
            printStatus(timestamp, "Failed to save frame");
        }
    } else {
        printStatus(timestamp);
    }
    
    // 模拟播放速度（简单的帧率控制）
    if (playback_speed_ > 0) {
        // 计算应该等待的时间
        static double last_frame_time = 0.0;
        double frame_duration = timestamp - last_frame_time;
        if (frame_duration > 0 && last_frame_time > 0) {
            double sleep_time = frame_duration / playback_speed_;
            if (sleep_time > 0.001) { // 最少1ms
                std::this_thread::sleep_for(std::chrono::milliseconds((int)(sleep_time * 1000)));
            }
        }
        last_frame_time = timestamp;
    }
}

void CLIMediaSink::onAudioFrame(const int16_t* samples, int sample_count,
                               double timestamp, int sample_rate, int channels) {
    if (save_audio_ && audio_file_.is_open()) {
        // 写入音频数据到WAV文件
        audio_file_.write(reinterpret_cast<const char*>(samples), 
                         sample_count * channels * sizeof(int16_t));
    }
    
    // 更新当前时间（如果音频时间戳更新）
    if (timestamp > current_time_) {
        current_time_ = timestamp;
    }
}

void CLIMediaSink::pause() {
    is_paused_ = true;
    std::cout << "\\n[PAUSED at " << std::fixed << std::setprecision(3) 
              << current_time_ << "s]" << std::endl;
}

void CLIMediaSink::resume() {
    is_paused_ = false;
    start_time_ = std::chrono::high_resolution_clock::now();
    std::cout << "\\n[RESUMED from " << std::fixed << std::setprecision(3) 
              << current_time_ << "s]" << std::endl;
}

bool CLIMediaSink::isPaused() const {
    return is_paused_;
}

void CLIMediaSink::close() {
    if (audio_file_.is_open()) {
        updateAudioFileHeader();
        audio_file_.close();
    }
    
    // 释放缓存的D3D11设备
    if (d3d11_context_) {
        d3d11_context_->Release();
        d3d11_context_ = nullptr;
    }
    
    if (d3d11_device_) {
        d3d11_device_->Release();
        d3d11_device_ = nullptr;
    }
    
    std::cout << "\\n=== CLI Media Sink Summary ===" << std::endl;
    std::cout << "Total frames processed: " << frame_count_ << std::endl;
    std::cout << "Frames saved: " << saved_frame_count_ << std::endl;
    std::cout << "Final timestamp: " << std::fixed << std::setprecision(3) 
              << current_time_ << "s" << std::endl;
    std::cout << "=============================" << std::endl;
}

bool CLIMediaSink::saveFrameAsBMP(ID3D11Texture2D* texture, const std::string& filename) {
    if (!texture) {
        return false;
    }
    
    // 如果没有缓存设备，从纹理获取（只在第一次）
    if (!d3d11_device_ || !d3d11_context_) {
        texture->GetDevice(&d3d11_device_);
        if (!d3d11_device_) {
            return false;
        }
        
        d3d11_device_->GetImmediateContext(&d3d11_context_);
        if (!d3d11_context_) {
            d3d11_device_->Release();
            d3d11_device_ = nullptr;
            return false;
        }
    }
    
    // 使用缓存的设备保存纹理
    bool success = RGBVerification::saveTextureAsBMP(d3d11_device_, d3d11_context_, texture, filename);
    
    return success;
}

bool CLIMediaSink::initializeAudioFile() {
    std::string audio_filename = output_dir_ + "/audio_output.wav";
    audio_file_.open(audio_filename, std::ios::binary);
    
    if (!audio_file_.is_open()) {
        return false;
    }
    
    writeWAVHeader();
    return true;
}

void CLIMediaSink::writeWAVHeader() {
    // WAV文件头（44字节）
    // 先写入占位符，稍后更新文件大小信息
    
    // RIFF header
    audio_file_.write("RIFF", 4);
    uint32_t file_size = 0; // 稍后更新
    audio_file_.write(reinterpret_cast<const char*>(&file_size), 4);
    audio_file_.write("WAVE", 4);
    
    // fmt chunk
    audio_file_.write("fmt ", 4);
    uint32_t fmt_chunk_size = 16;
    audio_file_.write(reinterpret_cast<const char*>(&fmt_chunk_size), 4);
    uint16_t audio_format = 1; // PCM
    audio_file_.write(reinterpret_cast<const char*>(&audio_format), 2);
    uint16_t num_channels = audio_channels_;
    audio_file_.write(reinterpret_cast<const char*>(&num_channels), 2);
    uint32_t sample_rate = audio_sample_rate_;
    audio_file_.write(reinterpret_cast<const char*>(&sample_rate), 4);
    uint32_t byte_rate = sample_rate * num_channels * 2; // 16-bit samples
    audio_file_.write(reinterpret_cast<const char*>(&byte_rate), 4);
    uint16_t block_align = num_channels * 2;
    audio_file_.write(reinterpret_cast<const char*>(&block_align), 2);
    uint16_t bits_per_sample = 16;
    audio_file_.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    
    // data chunk header
    audio_file_.write("data", 4);
    uint32_t data_size = 0; // 稍后更新
    audio_file_.write(reinterpret_cast<const char*>(&data_size), 4);
    
    audio_file_header_written_ = true;
}

void CLIMediaSink::updateAudioFileHeader() {
    if (!audio_file_.is_open() || !audio_file_header_written_) {
        return;
    }
    
    // 获取当前文件位置（数据大小）
    auto current_pos = audio_file_.tellp();
    uint32_t data_size = static_cast<uint32_t>(current_pos) - 44; // 减去头部44字节
    uint32_t file_size = static_cast<uint32_t>(current_pos) - 8;  // 减去前8字节
    
    // 更新RIFF大小
    audio_file_.seekp(4);
    audio_file_.write(reinterpret_cast<const char*>(&file_size), 4);
    
    // 更新data大小
    audio_file_.seekp(40);
    audio_file_.write(reinterpret_cast<const char*>(&data_size), 4);
    
    // 回到文件末尾
    audio_file_.seekp(0, std::ios::end);
}

std::string CLIMediaSink::generateFrameFilename(int frame_number, double timestamp) {
    std::ostringstream oss;
    oss << "frame_" << std::setfill('0') << std::setw(6) << frame_number
        << "_t" << std::fixed << std::setprecision(3) << timestamp
        << "s.bmp";
    return oss.str();
}

void CLIMediaSink::printStatus(double timestamp, const std::string& frame_info) {
    // 打印简洁的进度信息
    std::cout << "\\rFrame " << std::setw(6) << frame_count_ 
              << " | Time: " << std::fixed << std::setprecision(3) << timestamp << "s";
    
    if (!frame_info.empty()) {
        std::cout << " | " << frame_info;
    }
    
    std::cout << std::flush;
}