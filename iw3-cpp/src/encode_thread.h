#pragma once

#include "main.h"
#include <string>

// 启动编码线程的函数
void start_encode_thread(
    StereoInferredFrameQueue& input_frame_queue,
    const std::string& output_filename
); 