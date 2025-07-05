#include "media_player.h"
#include "gui_media_sink.h"
#include "rgb_frame_decoder.h"
#include <iostream>
#include <string>
#include <windows.h>

int main(int argc, char* argv[]) {
    std::cout << "=== GUI Media Player ===" << std::endl;
    
    // 检查命令行参数
    std::string filepath;
    if (argc >= 2) {
        filepath = argv[1];
    } else {
        std::cout << "No video file specified." << std::endl;
        std::cout << "Usage: " << argv[0] << " <video_file>" << std::endl;
        std::cout << "Supported formats: .mp4, .mkv, .avi, .mov" << std::endl;
        return -1;
    }
    
    try {
        // 创建临时解码器获取视频信息
        RGBFrameDecoder temp_decoder;
        if (!temp_decoder.open(filepath)) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return -1;
        }
        
        // 获取视频信息
        int video_width = temp_decoder.getVideoWidth();
        int video_height = temp_decoder.getVideoHeight();
        temp_decoder.close();
        
        std::cout << "Video dimensions: " << video_width << "x" << video_height << std::endl;
        
        if (video_width <= 0 || video_height <= 0) {
            std::cerr << "Invalid video dimensions: " << video_width << "x" << video_height << std::endl;
            return -1;
        }
        
        // 创建媒体播放器和GUI sink
        MediaPlayer player;
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        // 设置测试模式（可选）
        // gui_sink->setTestMode(true, 5000); // 5秒后自动关闭
        
        // 初始化播放器
        if (!player.initialize(std::move(gui_sink))) {
            std::cerr << "Failed to initialize media player" << std::endl;
            return -1;
        }
        
        // 获取GUI sink的引用
        auto* gui_sink_ptr = static_cast<GUIMediaSink*>(player.getMediaSink());
        
        // 设置视频尺寸
        gui_sink_ptr->setVideoDimensions(video_width, video_height);
        
        // 创建窗口
        if (!gui_sink_ptr->createWindow("Multi-threaded Video Player")) {
            std::cerr << "Failed to create window" << std::endl;
            return -1;
        }
        
        // 显示窗口
        gui_sink_ptr->showWindow();
        
        // 启动多线程播放
        std::cout << "Starting multi-threaded playback..." << std::endl;
        if (!gui_sink_ptr->startPlayback(&player, filepath)) {
            std::cerr << "Failed to start playback" << std::endl;
            return -1;
        }
        
        std::cout << "Multi-threaded playback started successfully!" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  SPACE - Pause/Resume" << std::endl;
        std::cout << "  ESC   - Exit" << std::endl;
        
        // 主渲染循环（运行在主线程）
        while (gui_sink_ptr->processMessages()) {
            // 渲染帧
            gui_sink_ptr->renderFrame();
            gui_sink_ptr->present();
            
            // 稍微休眠以避免100%CPU使用
            Sleep(1);
        }
        
        // 停止播放
        gui_sink_ptr->stopPlayback();
        
        std::cout << "GUI Media Player closed successfully." << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return -1;
    }
}