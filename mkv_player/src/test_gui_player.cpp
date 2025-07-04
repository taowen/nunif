#include "media_player.h"
#include "gui_media_sink.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== GUI Media Player Test ===" << std::endl;
    
    // 创建GUI sink
    auto gui_sink = std::make_unique<GUIMediaSink>();
    
    // 创建媒体播放器
    MediaPlayer player;
    if (!player.initialize(std::move(gui_sink))) {
        std::cerr << "Failed to initialize media player" << std::endl;
        return 1;
    }
    
    // 打开测试文件
    std::string test_file = "test_data/sample_hw.mkv";
    if (!player.openFile(test_file)) {
        std::cerr << "Failed to open test file: " << test_file << std::endl;
        return 1;
    }
    
    std::cout << "\\nStarting GUI playback..." << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  SPACE - Pause/Resume" << std::endl;
    std::cout << "  ESC   - Exit" << std::endl;
    std::cout << "\\nPress any key to start...";
    std::cin.get();
    
    // 获取GUI sink的窗口句柄
    GUIMediaSink* gui_sink_ptr = static_cast<GUIMediaSink*>(player.getMediaSink());
    if (!gui_sink_ptr) {
        std::cerr << "Failed to get GUI sink pointer" << std::endl;
        return 1;
    }
    
    // 主播放循环
    int frames_played = 0;
    const int max_frames = 1000; // 播放最多1000帧，防止无限循环
    
    while (frames_played < max_frames) {
        // 处理窗口消息
        if (!gui_sink_ptr->processWindowMessages()) {
            std::cout << "\\nUser requested exit." << std::endl;
            break;
        }
        
        // 播放一帧
        if (!player.playOneFrame()) {
            std::cout << "\\nEnd of file reached." << std::endl;
            break;
        }
        
        frames_played++;
        
        // 短暂休眠以避免过度占用CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    std::cout << "\\nGUI playback completed." << std::endl;
    std::cout << "Total frames played: " << frames_played << std::endl;
    
    return 0;
}