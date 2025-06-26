#pragma once

#include "main.h"

// 启动颜色转换线程的函数
void start_convert_color_thread(
    DecodedFrameQueue& input_frame_queue,
    ColorConvertedFrameQueue& output_frame_queue,
    const ColorSpaceInfo& color_info,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context
); 