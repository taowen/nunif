#pragma once

#include "main.h"
#include <d3d11.h>

void start_diagnose_convert_color_thread(
    D11FrameQueue& input_queue,
    D11FrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context
); 