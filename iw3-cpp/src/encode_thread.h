#pragma once

#include "main.h"
#include <string>


// 启动编码线程的函数
void start_encode_thread(
    StereoInferredFrameQueue& input_frame_queue,
    const std::string& output_filename,
    const ColorSpaceInfo& color_info,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context
); 