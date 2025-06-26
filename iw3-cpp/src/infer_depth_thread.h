#pragma once

#include "main.h"

// 启动深度推理线程的函数
void start_infer_depth_thread(
    ColorConvertedFrameQueue& input_frame_queue
); 