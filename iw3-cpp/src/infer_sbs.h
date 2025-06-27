#pragma once

#include "main.h"

// 启动深度推理线程的函数
void start_infer_sbs(
    D11FrameQueue& input_frame_queue,
    D11FrameQueue& output_frame_queue,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context
); 