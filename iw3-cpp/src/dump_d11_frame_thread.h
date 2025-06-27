#pragma once

#include "main.h"
#include <d3d11.h>

void start_dump_d11_frame_thread(
    D11FrameQueue& input_queue,
    D11FrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context,
    const std::string& prefix = "frame"
); 