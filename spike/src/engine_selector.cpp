// 在你的 C++ 代码中添加引擎选择逻辑
std::string selectEngineForDevice() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int major = prop.major;
    int minor = prop.minor;
    
    // 根据计算能力选择引擎
    if (major == 7 && minor == 5) {
        return "stereo_module_half_sbs_sm_75.trt";
    } else if (major == 8 && minor == 0) {
        return "stereo_module_half_sbs_sm_80.trt";
    } else if (major == 8 && minor == 6) {
        return "stereo_module_half_sbs_sm_86.trt";
    } else if (major == 8 && minor == 9) {
        return "stereo_module_half_sbs_sm_89.trt";
    } else {
        // 降级到 ONNX 运行时
        return "";
    }
} 