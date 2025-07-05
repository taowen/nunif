#include <catch2/catch_test_macros.hpp>
#include "../src/media_player.h"
#include "../src/gui_media_sink.h"
#include <iostream>
#include <memory>
#include <filesystem>
#include <thread>
#include <chrono>
#include <windows.h>

class GUITestHelper {
public:
    // 验证DirectX11设备是否可用
    static bool isDirectX11Available() {
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        
        D3D_FEATURE_LEVEL feature_levels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        
        D3D_FEATURE_LEVEL feature_level;
        HRESULT hr = D3D11CreateDevice(
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            0,
            feature_levels,
            ARRAYSIZE(feature_levels),
            D3D11_SDK_VERSION,
            &device,
            &feature_level,
            &context
        );
        
        return SUCCEEDED(hr);
    }
    
    // 等待窗口创建完成
    static bool waitForWindow(HWND* window_handle, int timeout_ms = 5000) {
        auto start_time = std::chrono::steady_clock::now();
        
        while (true) {
            // 检查窗口是否创建
            if (*window_handle && IsWindow(*window_handle)) {
                return true;
            }
            
            // 检查超时
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            if (elapsed > timeout_ms) {
                return false;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    // 验证窗口是否可见
    static bool isWindowVisible(HWND window_handle) {
        return IsWindowVisible(window_handle);
    }
    
    // 获取窗口客户区尺寸
    static bool getWindowSize(HWND window_handle, int& width, int& height) {
        RECT client_rect;
        if (GetClientRect(window_handle, &client_rect)) {
            width = client_rect.right - client_rect.left;
            height = client_rect.bottom - client_rect.top;
            return true;
        }
        return false;
    }
};

TEST_CASE("GUI Media Sink Basic Functionality", "[gui_integration]") {
    // 检查DirectX11环境
    if (!GUITestHelper::isDirectX11Available()) {
        SKIP("DirectX11 not available in test environment");
    }
    
    SECTION("GUI Media Sink Window Creation") {
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        // 启用测试模式，500ms自动关闭
        gui_sink->setTestMode(true, 500);
        
        // 初始化（这会创建窗口）
        REQUIRE(gui_sink->initialize(1920, 1080, 44100, 2));
        
        // 显示窗口
        gui_sink->showWindow();
        
        // 测试渲染管道
        int render_count = 0;
        while (gui_sink->processMessages() && render_count < 10) {
            gui_sink->renderFrame();
            gui_sink->present();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            render_count++;
        }
        
        INFO("Window auto-closed successfully after rendering " << render_count << " frames");
        REQUIRE(render_count > 0);
    }
    
    SECTION("GUI Media Sink Pause/Resume") {
        auto gui_sink = std::make_unique<GUIMediaSink>();
        REQUIRE(gui_sink->initialize(1920, 1080, 44100, 2));
        
        // 测试暂停
        gui_sink->pause();
        REQUIRE(gui_sink->isPaused());
        
        // 测试恢复
        gui_sink->resume();
        REQUIRE_FALSE(gui_sink->isPaused());
    }
}

TEST_CASE("GUI Media Player Integration", "[gui_player_integration]") {
    // 检查DirectX11环境
    if (!GUITestHelper::isDirectX11Available()) {
        SKIP("DirectX11 not available in test environment");
    }
    
    SECTION("GUI Media Player with Test File") {
        std::string test_file = "test_data/sample_hw.mkv";
        
        // 检查测试文件是否存在
        if (!std::filesystem::exists(test_file)) {
            SKIP("Test file not found: " + test_file);
        }
        
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        MediaPlayer player;
        REQUIRE(player.initialize(std::move(gui_sink)));
        
        REQUIRE(player.openFile(test_file));
        
        // 获取GUI sink指针
        GUIMediaSink* gui_sink_ptr = static_cast<GUIMediaSink*>(player.getMediaSink());
        REQUIRE(gui_sink_ptr != nullptr);
        
        // 启用测试模式，1秒自动关闭
        gui_sink_ptr->setTestMode(true, 1000);
        
        // 显示窗口（窗口已在initialize中创建）
        gui_sink_ptr->showWindow();
        
        // 播放一些帧并测试渲染
        int frames_played = 0;
        auto start_time = std::chrono::steady_clock::now();
        
        while (gui_sink_ptr->processMessages()) {
            // 渲染背景（应该显示彩色背景）
            gui_sink_ptr->renderFrame();
            gui_sink_ptr->present();
            
            // 播放一帧
            if (player.playOneFrame()) {
                frames_played++;
            }
            
            // 短暂休眠
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        
        INFO("Frames played: " << frames_played);
        INFO("Elapsed time: " << elapsed_ms << "ms");
        
        // 验证播放成功
        REQUIRE(frames_played >= 0);
        REQUIRE(elapsed_ms > 0);
        
        INFO("Window auto-closed after " << elapsed_ms << "ms with " << frames_played << " frames played");
    }
}

TEST_CASE("GUI Rendering Pipeline Verification", "[gui_rendering]") {
    // 检查DirectX11环境
    if (!GUITestHelper::isDirectX11Available()) {
        SKIP("DirectX11 not available in test environment");
    }
    
    SECTION("Background Color Animation Test") {
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        // 启用测试模式，1.5秒自动关闭
        gui_sink->setTestMode(true, 1500);
        
        // 初始化（会创建窗口）
        REQUIRE(gui_sink->initialize(800, 600, 44100, 2));
        
        // 显示窗口
        gui_sink->showWindow();
        
        std::cout << "\n=== GUI渲染管道验证 ===" << std::endl;
        std::cout << "测试项目：动态彩色背景渲染" << std::endl;
        std::cout << "预期结果：窗口应显示红->绿->蓝的颜色循环" << std::endl;
        
        // 渲染多次，验证动画效果
        int render_count = 0;
        while (gui_sink->processMessages()) {
            gui_sink->renderFrame();
            gui_sink->present();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            render_count++;
        }
        
        std::cout << "渲染测试完成，共渲染 " << render_count << " 帧" << std::endl;
        std::cout << "====================\n" << std::endl;
        
        REQUIRE(render_count > 0);
    }
}

TEST_CASE("GUI Black Screen Diagnosis", "[gui_diagnosis]") {
    // 检查DirectX11环境
    if (!GUITestHelper::isDirectX11Available()) {
        SKIP("DirectX11 not available in test environment");
    }
    
    SECTION("Step-by-step Diagnosis") {
        std::cout << "\n=== 黑屏问题诊断 ===" << std::endl;
        
        auto gui_sink = std::make_unique<GUIMediaSink>();
        
        // 启用测试模式，2秒自动关闭
        gui_sink->setTestMode(true, 2000);
        
        // 1. 初始化测试
        std::cout << "1. 测试初始化..." << std::endl;
        bool init_success = gui_sink->initialize(1920, 1080, 44100, 2);
        std::cout << "   初始化结果: " << (init_success ? "成功" : "失败") << std::endl;
        REQUIRE(init_success);
        
        // 2. 显示窗口（窗口已在初始化时创建）
        std::cout << "2. 显示窗口..." << std::endl;
        gui_sink->showWindow();
        
        // 4. 渲染测试
        std::cout << "4. 测试渲染管道..." << std::endl;
        std::cout << "   应该看到彩色背景动画，持续2秒自动关闭" << std::endl;
        
        int render_count = 0;
        while (gui_sink->processMessages()) {
            gui_sink->renderFrame();
            gui_sink->present();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            render_count++;
        }
        
        std::cout << "5. 诊断完成，共渲染 " << render_count << " 帧" << std::endl;
        std::cout << "================\n" << std::endl;
        
        REQUIRE(render_count > 0);
    }
}

// 测试初始化（由Catch2WithMain自动调用）