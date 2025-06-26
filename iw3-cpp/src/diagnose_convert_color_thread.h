#pragma once

#include "main.h"
#include <d3d11.h>

void start_diagnose_convert_color_thread(
    ColorConvertedFrameQueue& input_queue,
    ColorConvertedFrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context
); 