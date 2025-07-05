#include "src/gui_media_sink.h"
#include <iostream>
#include <thread>
#include <chrono>

int main() {
    std::cout << "=== Test Auto Close ===" << std::endl;
    
    auto gui_sink = std::make_unique<GUIMediaSink>();
    
    // 启用测试模式，500ms自动关闭
    gui_sink->setTestMode(true, 500);
    
    // 初始化（这会创建窗口）
    if (!gui_sink->initialize(800, 600, 44100, 2)) {
        std::cerr << "Failed to initialize" << std::endl;
        return 1;
    }
    
    // 显示窗口
    gui_sink->showWindow();
    
    std::cout << "Starting render loop..." << std::endl;
    
    // 测试渲染管道
    int render_count = 0;
    while (gui_sink->processMessages() && render_count < 20) {
        gui_sink->renderFrame();
        gui_sink->present();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        render_count++;
        std::cout << "Rendered frame " << render_count << std::endl;
    }
    
    std::cout << "Test completed. Rendered " << render_count << " frames" << std::endl;
    
    return 0;
}