#include <catch2/catch_test_macros.hpp>
#include "main.h"
// 添加FFmpeg软件缩放库
extern "C" {
#include <libswscale/swscale.h>
}
#include <vector>
#include <cstring>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

// 辅助函数：使用FFmpeg将D3D11硬件帧转换为YUV420P CPU数据
YUVData ffmpeg_d3d11_to_yuv(AVFrame* d3d11_frame) {
    YUVData result;
    
    // 创建软件帧来接收下载的数据
    AVFrame* sw_frame = av_frame_alloc();
    if (!sw_frame) {
        return result;
    }
    
    // 从硬件帧传输数据到软件帧
    int ret = av_hwframe_transfer_data(sw_frame, d3d11_frame, 0);
    if (ret < 0) {
        av_frame_free(&sw_frame);
        return result;
    }
    
    std::cout << "[ffmpeg_d3d11_to_yuv] hw-transferred sw_frame format: "
        << av_get_pix_fmt_name((AVPixelFormat)sw_frame->format) << std::endl;
    
    // 确保格式正确，如果不是YUV420P则转换
    AVFrame* yuv_frame = nullptr;
    SwsContext* sws_ctx = nullptr;
    
    if (sw_frame->format == AV_PIX_FMT_YUV420P) {
        yuv_frame = sw_frame;
    } else {
        // 需要格式转换到YUV420P
        yuv_frame = av_frame_alloc();
        if (!yuv_frame) {
            av_frame_free(&sw_frame);
            return result;
        }
        
        yuv_frame->format = AV_PIX_FMT_YUV420P;
        yuv_frame->width = sw_frame->width;
        yuv_frame->height = sw_frame->height;
        
        ret = av_frame_get_buffer(yuv_frame, 32);
        if (ret < 0) {
            av_frame_free(&sw_frame);
            av_frame_free(&yuv_frame);
            return result;
        }
        
        // 创建转换上下文
        sws_ctx = sws_getContext(
            sw_frame->width, sw_frame->height, (AVPixelFormat)sw_frame->format,
            yuv_frame->width, yuv_frame->height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        
        if (!sws_ctx) {
            av_frame_free(&sw_frame);
            av_frame_free(&yuv_frame);
            return result;
        }
        
        // 执行格式转换
        sws_scale(sws_ctx, sw_frame->data, sw_frame->linesize, 0, sw_frame->height,
                  yuv_frame->data, yuv_frame->linesize);
    }
    
    // 提取YUV数据
    result.width = yuv_frame->width;
    result.height = yuv_frame->height;
    
    // Y平面 (全分辨率)
    int y_size = result.width * result.height;
    result.y_plane.resize(y_size);
    for (int i = 0; i < result.height; i++) {
        memcpy(result.y_plane.data() + i * result.width,
               yuv_frame->data[0] + i * yuv_frame->linesize[0],
               result.width);
    }
    
    // U和V平面 (1/4分辨率)
    int uv_width = result.width / 2;
    int uv_height = result.height / 2;
    int uv_size = uv_width * uv_height;
    
    result.u_plane.resize(uv_size);
    result.v_plane.resize(uv_size);
    
    for (int i = 0; i < uv_height; i++) {
        memcpy(result.u_plane.data() + i * uv_width,
               yuv_frame->data[1] + i * yuv_frame->linesize[1],
               uv_width);
        memcpy(result.v_plane.data() + i * uv_width,
               yuv_frame->data[2] + i * yuv_frame->linesize[2],
               uv_width);
    }
    
    result.valid = true;
    
    // 清理资源
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
    }
    if (yuv_frame != sw_frame) {
        av_frame_free(&yuv_frame);
    }
    av_frame_free(&sw_frame);
    
    return result;
}

// 辅助函数：比较两个YUV数据的差异
struct CompareResult {
    double y_mse;      // Y平面均方误差
    double u_mse;      // U平面均方误差  
    double v_mse;      // V平面均方误差
    double y_psnr;     // Y平面PSNR
    double u_psnr;     // U平面PSNR
    double v_psnr;     // V平面PSNR
    int max_y_diff;    // Y平面最大差异
    int max_u_diff;    // U平面最大差异
    int max_v_diff;    // V平面最大差异
};

void log_yuv_data(const YUVData& data, const std::string& name) {
    if (!data.valid) {
        std::cout << "YUVData (" << name << ") is invalid." << std::endl;
        return;
    }
    std::cout << "\n--- YUV Data Details: " << name << " ---" << std::endl;
    std::cout << "Dimensions: " << data.width << "x" << data.height << std::endl;
    auto print_plane_info = [](const std::vector<uint8_t>& plane, const std::string& plane_name) {
        if (plane.empty()) {
            std::cout << "  " << plane_name << " is empty." << std::endl;
            return;
        }
        std::cout << "  " << plane_name << " size: " << plane.size() << ". First 8 bytes: ";
        std::cout << std::hex;
        for (size_t i = 0; i < std::min((size_t)8, plane.size()); ++i) {
            std::cout << (int)plane[i] << " ";
        }
        std::cout << std::dec << std::endl;
    };
    print_plane_info(data.y_plane, "Y plane");
    print_plane_info(data.u_plane, "U plane");
    print_plane_info(data.v_plane, "V plane");
}

CompareResult compare_yuv_data(const YUVData& data1, const YUVData& data2) {
    CompareResult result = {};
    
    if (!data1.valid || !data2.valid || 
        data1.width != data2.width || data1.height != data2.height ||
        data1.y_plane.size() != data2.y_plane.size() ||
        data1.u_plane.size() != data2.u_plane.size() ||
        data1.v_plane.size() != data2.v_plane.size()) {
        return result;
    }
    
    auto calc_mse_and_max = [](const std::vector<uint8_t>& plane1, 
                               const std::vector<uint8_t>& plane2) -> std::pair<double, int> {
        if (plane1.empty()) {
            return { 0.0, 0 };
        }
        double mse = 0.0;
        int max_diff = 0;
        for (size_t i = 0; i < plane1.size(); i++) {
            int diff = abs((int)plane1[i] - (int)plane2[i]);
            mse += diff * diff;
            max_diff = std::max(max_diff, diff);
        }
        mse /= plane1.size();
        return {mse, max_diff};
    };
    
    auto calc_psnr = [](double mse) -> double {
        if (mse == 0.0) return 100.0; // 完全相同
        return 20.0 * log10(255.0 / sqrt(mse));
    };
    
    // 计算Y平面差异
    auto [y_mse, max_y] = calc_mse_and_max(data1.y_plane, data2.y_plane);
    result.y_mse = y_mse;
    result.y_psnr = calc_psnr(y_mse);
    result.max_y_diff = max_y;
    
    // 计算U平面差异
    auto [u_mse, max_u] = calc_mse_and_max(data1.u_plane, data2.u_plane);
    result.u_mse = u_mse;
    result.u_psnr = calc_psnr(u_mse);
    result.max_u_diff = max_u;
    
    // 计算V平面差异
    auto [v_mse, max_v] = calc_mse_and_max(data1.v_plane, data2.v_plane);
    result.v_mse = v_mse;
    result.v_psnr = calc_psnr(v_mse);
    result.max_v_diff = max_v;
    
    return result;
}

TEST_CASE("Try dump d3d11 avframe") {
    // Input file
    const char* input_file = "06 4k.mp4";
    const int frame_index = 200;
    
    FFMepgContext hw_ctx;
    AVFrame* hw_frame = d11_decode(&hw_ctx, input_file, frame_index);
    
    // 验证解码是否成功
    REQUIRE(hw_frame != nullptr);
    
    // 验证帧格式（应该是D3D11硬件格式）
    REQUIRE(hw_frame->format == AV_PIX_FMT_D3D11);
    
    // 验证D3D11设备是否可用
    REQUIRE(hw_ctx.d3d_device != nullptr);
    REQUIRE(hw_ctx.d3d_context != nullptr);
    
    // 方法1：使用dump_d3d11_avframe获取YUV数据
    YUVData dump_result = dump_d3d11_avframe(&hw_ctx, hw_frame, true);
    REQUIRE(dump_result.valid);
    
    // 方法2：使用FFmpeg标准方法获取YUV数据
    YUVData ffmpeg_result = ffmpeg_d3d11_to_yuv(hw_frame);
    REQUIRE(ffmpeg_result.valid);
    
    // 验证基本信息一致
    REQUIRE(dump_result.width == ffmpeg_result.width);
    REQUIRE(dump_result.height == ffmpeg_result.height);
    
    // 打印YUV数据样本以供调试
    log_yuv_data(dump_result, "dump_d3d11_avframe");
    log_yuv_data(ffmpeg_result, "ffmpeg_d3d11_to_yuv");

    // 比较两种方法的结果
    CompareResult comparison = compare_yuv_data(dump_result, ffmpeg_result);
    
    // 输出比较结果
    std::cout << "\n=== YUV数据比较结果 ===" << std::endl;
    std::cout << "分辨率: " << dump_result.width << "x" << dump_result.height << std::endl;
    std::cout << "Y平面 - MSE: " << comparison.y_mse << ", PSNR: " << comparison.y_psnr 
              << " dB, 最大差异: " << comparison.max_y_diff << std::endl;
    std::cout << "U平面 - MSE: " << comparison.u_mse << ", PSNR: " << comparison.u_psnr 
              << " dB, 最大差异: " << comparison.max_u_diff << std::endl;
    std::cout << "V平面 - MSE: " << comparison.v_mse << ", PSNR: " << comparison.v_psnr 
              << " dB, 最大差异: " << comparison.max_v_diff << std::endl;
    
    // 设置合理的容差阈值
    const double min_acceptable_psnr = 30.0; // PSNR应该大于30dB
    const int max_acceptable_diff = 5;        // 单像素最大差异不超过5
    
    // 验证两种方法的结果足够接近
    REQUIRE(comparison.y_psnr > min_acceptable_psnr);
    REQUIRE(comparison.u_psnr > min_acceptable_psnr);
    REQUIRE(comparison.v_psnr > min_acceptable_psnr);
    
    REQUIRE(comparison.max_y_diff <= max_acceptable_diff);
    REQUIRE(comparison.max_u_diff <= max_acceptable_diff);
    REQUIRE(comparison.max_v_diff <= max_acceptable_diff);
    
    // 清理资源
    av_frame_free(&hw_frame);
    cleanup_context(&hw_ctx);
    
    std::cout << "✓ dump_d3d11_avframe的结果与FFmpeg标准方法基本一致!" << std::endl;
}