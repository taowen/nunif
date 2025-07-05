#include "media_player.h"
#include "gui_media_sink.h"
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    std::cout << "=== GUI Media Player ===" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_mkv_file>" << std::endl;
        return 1;
    }

    // 创建GUI sink
    auto gui_sink = std::make_unique<GUIMediaSink>();
    
    // 创建媒体播放器
    MediaPlayer player;
    if (!player.initialize(std::move(gui_sink))) {
        std::cerr << "Failed to initialize media player" << std::endl;
        return 1;
    }
    
    // 打开文件
    std::string media_file = argv[1];
    if (!player.openFile(media_file)) {
        std::cerr << "Failed to open file: " << media_file << std::endl;
        return 1;
    }
    
    std::cout << "\\nPlaying: " << media_file << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  SPACE - Pause/Resume" << std::endl;
    std::cout << "  ESC   - Exit" << std::endl;
    
    // 获取GUI sink的窗口句柄
    GUIMediaSink* gui_sink_ptr = static_cast<GUIMediaSink*>(player.getMediaSink());
    if (!gui_sink_ptr) {
        std::cerr << "Failed to get GUI sink pointer" << std::endl;
        return 1;
    }
    
    // 显示窗口
    gui_sink_ptr->showWindow();
    
    // 主播放循环
    int frames_played = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    while (true) {
        // 处理窗口消息
        if (!gui_sink_ptr->processMessages()) {
            std::cout << "\\nUser requested exit." << std::endl;
            break;
        }
        
        // 持续渲染（测试DirectX11渲染管道）
        gui_sink_ptr->renderFrame();
        gui_sink_ptr->present();
        
        // 播放一帧（如果可能）
        if (player.playOneFrame()) {
            frames_played++;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (frames_played % 100 == 0) {  // 减少打印频率
                std::cout << "\\rFrames played: " << frames_played << " | Elapsed time: " << elapsed_ms << "ms" << std::flush;
            }
        }

        // 短暂休眠以避免过度占用CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(16));  // ~60fps
    }
    
    std::cout << "\\nGUI playback completed." << std::endl;
    std::cout << "Total frames played: " << frames_played << std::endl;
    
    return 0;
}