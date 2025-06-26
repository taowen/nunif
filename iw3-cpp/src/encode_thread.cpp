#include "encode_thread.h"
#include <iostream>

void start_encode_thread(
    StereoInferredFrameQueue& input_frame_queue,
    const std::string& output_filename
) {
    std::cout << "Starting encode thread, output: " << output_filename << std::endl;
    
    // TODO: 实现编码逻辑
    // 1. 初始化编码器 (FFmpeg)
    // 2. 配置输出格式和编码参数
    // 3. 从输入队列获取帧数据
    // 4. 编码并写入输出文件
    // 5. 处理结束信号
    
    while (true) {
        StereoInferredFrame frame = input_frame_queue.pop();
        
        if (frame.is_end_signal) {
            std::cout << "Encode thread received end signal" << std::endl;
            break;
        }
        
        // TODO: 编码当前帧
        std::cout << "Encoding frame: " << frame.width << "x" << frame.height << std::endl;
    }
    
    std::cout << "Encode thread finished" << std::endl;
} 