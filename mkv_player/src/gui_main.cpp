#include "media_player.h"
#include "gui_media_sink.h"
#include "rgb_frame_decoder.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
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
        // 第一步：创建解码器建立FFmpeg的D3D11设备
        auto main_decoder = std::make_unique<RGBFrameDecoder>();
        if (!main_decoder->open(filepath)) {
            std::cerr << "Failed to open file: " << filepath << std::endl;
            return -1;
        }
        
        // 获取视频信息
        int video_width = main_decoder->getVideoWidth();
        int video_height = main_decoder->getVideoHeight();
        
        // 验证视频信息有效性
        if (video_width <= 0 || video_height <= 0) {
            std::cerr << "Invalid video dimensions from decoder: " << video_width << "x" << video_height << std::endl;
            main_decoder->close();
            return -1;
        }
        
        std::cout << "Video dimensions: " << video_width << "x" << video_height << std::endl;
        
        // 第二步：获取FFmpeg创建的D3D11设备
        ID3D11Device* ffmpeg_device = main_decoder->getD3D11Device();
        ID3D11DeviceContext* ffmpeg_context = main_decoder->getD3D11Context();
        
        if (!ffmpeg_device || !ffmpeg_context) {
            std::cerr << "Failed to get FFmpeg D3D11 device" << std::endl;
            main_decoder->close();
            return -1;
        }
        
        std::cout << "FFmpeg D3D11 device established successfully" << std::endl;
        
        // 第三步：创建GUI sink并使用FFmpeg的设备
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        // 设置测试模式（可选）
        // gui_sink->setTestMode(true, 5000); // 5秒后自动关闭
        
        // 初始化GUI sink（使用FFmpeg的设备）
        if (!gui_sink->initializeWithDevice(video_width, video_height, 44100, 2, ffmpeg_device, ffmpeg_context)) {
            std::cerr << "Failed to initialize GUI sink with FFmpeg device" << std::endl;
            main_decoder->close();
            return -1;
        }
        
        // 获取GUI sink的引用
        auto* gui_sink_ptr = gui_sink.get();
        
        // 第四步：创建MediaPlayer并传递现有的解码器
        MediaPlayer player;
        if (!player.initializeWithDecoder(std::move(gui_sink), std::move(main_decoder))) {
            std::cerr << "Failed to initialize media player" << std::endl;
            return -1;
        }
        
        // 显示窗口
        gui_sink_ptr->showWindow();
        
        // 等待窗口和DirectX11资源完全初始化
        std::cout << "Waiting for window to be ready..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        // 确保所有消息都已处理
        if (!gui_sink_ptr->processMessages()) {
            std::cerr << "Window closed during initialization" << std::endl;
            return -1;
        }
        
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
        
        // 主渲染循环（运行在主线程）- 使用最佳实践
        while (gui_sink_ptr->processMessages()) {
            // 使用主动渲染循环，包含智能帧率控制
            if (!gui_sink_ptr->renderLoop()) {
                std::cerr << "Render loop failed" << std::endl;
                break;
            }
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