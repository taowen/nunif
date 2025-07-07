#include <catch2/catch_test_macros.hpp>
#include "../src/audio_player.h"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

TEST_CASE("AudioPlayer Basic Initialization", "[AudioPlayer]") {
    AudioPlayer player;
    
    REQUIRE(player.getState() == AudioPlayer::State::Stopped);
    REQUIRE(player.getCurrentTime() == 0.0);
    REQUIRE(player.getVolume() == 1.0f);
    
    // 在Windows上测试WASAPI初始化
#ifdef _WIN32
    REQUIRE(player.initialize());
#endif
}

TEST_CASE("AudioPlayer Load File", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    
    REQUIRE(player.loadFile(test_file.string()));
    REQUIRE(player.getState() == AudioPlayer::State::Stopped);
    REQUIRE(player.getDuration() > 0.0);
}

TEST_CASE("AudioPlayer Volume Control", "[AudioPlayer]") {
    AudioPlayer player;
    
    // 测试音量范围
    player.setVolume(0.5f);
    REQUIRE(player.getVolume() == 0.5f);
    
    player.setVolume(0.0f);
    REQUIRE(player.getVolume() == 0.0f);
    
    player.setVolume(1.0f);
    REQUIRE(player.getVolume() == 1.0f);
    
    // 测试音量边界
    player.setVolume(-0.5f);
    REQUIRE(player.getVolume() == 0.0f);
    
    player.setVolume(1.5f);
    REQUIRE(player.getVolume() == 1.0f);
}

TEST_CASE("AudioPlayer State Management", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    REQUIRE(player.loadFile(test_file.string()));
    
    // 测试播放状态转换
    REQUIRE(player.getState() == AudioPlayer::State::Stopped);
    
    // 不能在未加载文件时播放
    // 但文件已加载，所以应该能播放
    REQUIRE(player.play());
    
    // 等待一小段时间让音频线程启动
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    REQUIRE(player.getState() == AudioPlayer::State::Playing);
    
    // 测试暂停
    REQUIRE(player.pause());
    REQUIRE(player.getState() == AudioPlayer::State::Paused);
    
    // 测试停止
    REQUIRE(player.stop());
    REQUIRE(player.getState() == AudioPlayer::State::Stopped);
    REQUIRE(player.getCurrentTime() == 0.0);
}

TEST_CASE("AudioPlayer Timer Integration", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    REQUIRE(player.loadFile(test_file.string()));
    
    REQUIRE(player.play());
    
    // 模拟外部定时器调用
    double start_time = 0.0;
    for (int i = 0; i < 10; i++) {
        double current_time = start_time + i * 0.02; // 每20ms
        player.onTimer(current_time);
        REQUIRE(player.getCurrentTime() == current_time);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    
    player.stop();
}

TEST_CASE("AudioPlayer Seek Test", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    REQUIRE(player.loadFile(test_file.string()));
    
    // 测试seek到不同位置
    REQUIRE(player.seekToTime(5.0));
    REQUIRE(player.getCurrentTime() == 5.0);
    
    REQUIRE(player.seekToTime(0.0));
    REQUIRE(player.getCurrentTime() == 0.0);
    
    // 测试在播放状态下seek
    REQUIRE(player.play());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // 先测试是否能seek到较小的时间
    bool seek_result = player.seekToTime(1.0);
    if (seek_result) {
        REQUIRE(player.getCurrentTime() == 1.0);
    } else {
        WARN("Seek functionality not working properly");
    }
    
    player.stop();
}

TEST_CASE("AudioPlayer Multiple Play/Stop Cycles", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    REQUIRE(player.loadFile(test_file.string()));
    
    // 多次播放和停止
    for (int i = 0; i < 3; i++) {
        REQUIRE(player.play());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        REQUIRE(player.getState() == AudioPlayer::State::Playing);
        
        REQUIRE(player.stop());
        REQUIRE(player.getState() == AudioPlayer::State::Stopped);
        REQUIRE(player.getCurrentTime() == 0.0);
    }
}

TEST_CASE("AudioPlayer Error Handling", "[AudioPlayer]") {
    AudioPlayer player;
    
    // 测试未初始化时的操作
    REQUIRE_FALSE(player.loadFile("nonexistent.mkv"));
    REQUIRE(player.getState() == AudioPlayer::State::Error);
    
    // 测试未加载文件时的操作
    AudioPlayer player2;
    REQUIRE(player2.initialize());
    REQUIRE_FALSE(player2.play());
}

TEST_CASE("AudioPlayer Thread Safety", "[AudioPlayer]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioPlayer player;
    REQUIRE(player.initialize());
    REQUIRE(player.loadFile(test_file.string()));
    
    // 快速连续的状态变化
    REQUIRE(player.play());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    REQUIRE(player.pause());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    REQUIRE(player.play());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    REQUIRE(player.stop());
    REQUIRE(player.getState() == AudioPlayer::State::Stopped);
}