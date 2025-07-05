#include "media_player.h"
#include "cli_media_sink.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    std::cout << "=== CLI Media Player ===\n";
    
    // 检查命令行参数
    std::string media_file = "test_data/sample_hw.mkv"; // 默认文件
    if (argc > 1) {
        media_file = argv[1];
        std::cout << "Using file: " << media_file << "\n";
    } else {
        std::cout << "No file specified, using default: " << media_file << "\n";
        std::cout << "Usage: " << argv[0] << " <path_to_media_file>\n";
    }
    
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
        std::cerr << "Failed to initialize media player\n";
        return 1;
    }
    
    // 打开媒体文件
    if (!player.openFile(media_file)) {
        std::cerr << "Failed to open file: " << media_file << "\n";
        return 1;
    }
    
    // 播放前50帧
    std::cout << "\nPlaying first 50 frame groups...\n";
    int frames_played = player.playFrames(50);
    
    std::cout << "\nPlayback completed!\n";
    std::cout << "Played " << frames_played << " frame groups.\n";
    std::cout << "Check 'cli_output' directory for saved frames.\n";
    
    return 0;
}