#include "media_player.h"
#include "cli_media_sink.h"
#include <iostream>
#include <memory>

int main() {
    std::cout << "=== CLI Media Player Test ===" << std::endl;
    
    // 创建CLI sink
    auto cli_sink = std::make_unique<CLIMediaSink>();
    
    // 配置CLI sink
    cli_sink->setOutputDirectory("cli_output");
    cli_sink->setSaveVideoFrames(true);
    cli_sink->setSaveAudio(false);
    cli_sink->setFrameInterval(5);  // 每5帧保存一次
    cli_sink->setMaxFramesToSave(20); // 最多保存20帧
    cli_sink->setPlaybackSpeed(2.0); // 2倍速播放
    
    // 创建媒体播放器
    MediaPlayer player;
    if (!player.initialize(std::move(cli_sink))) {
        std::cerr << "Failed to initialize media player" << std::endl;
        return 1;
    }
    
    // 打开测试文件
    std::string test_file = "test_data/sample_hw.mkv";
    if (!player.openFile(test_file)) {
        std::cerr << "Failed to open test file: " << test_file << std::endl;
        return 1;
    }
    
    // 播放前50帧
    std::cout << "\\nPlaying first 50 frame groups..." << std::endl;
    int frames_played = player.playFrames(50);
    
    std::cout << "\\nTest completed successfully!" << std::endl;
    std::cout << "Played " << frames_played << " frame groups." << std::endl;
    std::cout << "Check 'cli_output' directory for saved frames." << std::endl;
    
    return 0;
}