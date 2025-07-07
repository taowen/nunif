#include "rgb_video_decoder.h"
#include <iostream>

RgbVideoDecoder::RgbVideoDecoder() 
    : hw_decoder_(std::make_unique<HwVideoDecoder>())
    , d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , video_processor_initialized_(false)
    , current_rgb_frame_index_(0) {
}

RgbVideoDecoder::~RgbVideoDecoder() {
    close();
}

bool RgbVideoDecoder::open(const std::string& filepath) {
    close();
    
    // 1. 打开硬件解码器
    if (!hw_decoder_->open(filepath)) {
        std::cerr << "Failed to open file with HwVideoDecoder: " << filepath << std::endl;
        return false;
    }
    
    // 2. 获取D3D11设备和上下文
    d3d11_device_ = hw_decoder_->getD3D11Device();
    d3d11_context_ = hw_decoder_->getD3D11Context();
    
    if (!d3d11_device_ || !d3d11_context_) {
        std::cerr << "Failed to get D3D11 device from HwVideoDecoder" << std::endl;
        hw_decoder_->close();
        return false;
    }
    
    // 3. 初始化Video Processor
    if (!initializeVideoProcessor()) {
        std::cerr << "Failed to initialize Video Processor" << std::endl;
        hw_decoder_->close();
        return false;
    }
    
    return true;
}

bool RgbVideoDecoder::readNextFrame(DecodedFrame& frame) {
    if (!isOpen()) {
        frame.is_valid = false;
        return false;
    }
    
    // 1. 从硬件解码器读取帧
    HwVideoDecoder::DecodedFrame hw_frame;
    if (!hw_decoder_->readNextFrame(hw_frame)) {
        frame.is_valid = false;
        return false;
    }
    
    // 2. 获取当前双缓冲RGB帧
    RgbFrame& current_rgb_frame = rgb_frames_[current_rgb_frame_index_];
    
    // 3. 转换为RGB
    if (!convertNV12ToRGB(hw_frame.frame, current_rgb_frame)) {
        frame.is_valid = false;
        return false;
    }
    
    // 4. 填充返回结果
    frame.hw_frame = hw_frame.frame;
    frame.rgb_frame = current_rgb_frame;
    frame.is_valid = true;
    
    // 5. 切换到下一个RGB缓冲区
    current_rgb_frame_index_ = (current_rgb_frame_index_ + 1) % 2;
    
    return true;
}

bool RgbVideoDecoder::convertNV12ToRGB(AVFrame* nv12_frame, RgbFrame& rgb_frame) {
    if (!nv12_frame || !d3d11_device_ || !d3d11_context_) {
        std::cerr << "Invalid frame or D3D11 resources" << std::endl;
        return false;
    }
    
    if (nv12_frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Expected D3D11 format, got " << nv12_frame->format << std::endl;
        return false;
    }
    
    if (!nv12_frame->data[0]) {
        std::cerr << "D3D11 texture pointer is null" << std::endl;
        return false;
    }
    
    // 获取输入纹理
    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(nv12_frame->data[0]);
    int texture_index = (int)(intptr_t)nv12_frame->data[1];
    
    // 检查输入纹理格式
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Expected NV12 format, got " << input_desc.Format << std::endl;
        return false;
    }
    
    // 创建或更新RGB纹理
    if (!rgb_frame.is_valid || rgb_frame.width != nv12_frame->width || rgb_frame.height != nv12_frame->height) {
        if (!createRGBTexture(rgb_frame, nv12_frame->width, nv12_frame->height)) {
            return false;
        }
    }
    
    // 创建输入视图
    ComPtr<ID3D11VideoProcessorInputView> input_view;
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
    input_view_desc.FourCC = 0;
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    input_view_desc.Texture2D.ArraySlice = texture_index;
    
    HRESULT hr = video_device_->CreateVideoProcessorInputView(
        input_texture, 
        video_enum_.Get(), 
        &input_view_desc, 
        input_view.GetAddressOf()
    );
    if (FAILED(hr)) {
        std::cerr << "Failed to create input view: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建或复用输出视图
    if (!rgb_frame.hasValidOutputView()) {
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
        output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        output_view_desc.Texture2D.MipSlice = 0;
        
        hr = video_device_->CreateVideoProcessorOutputView(
            rgb_frame.rgb_texture.Get(), 
            video_enum_.Get(), 
            &output_view_desc, 
            rgb_frame.cached_output_view.GetAddressOf()
        );
        if (FAILED(hr)) {
            std::cerr << "Failed to create cached output view: 0x" << std::hex << hr << std::endl;
            return false;
        }
    }
    
    // 执行颜色空间转换
    D3D11_VIDEO_PROCESSOR_STREAM stream_data = {};
    stream_data.Enable = TRUE;
    stream_data.OutputIndex = 0;
    stream_data.InputFrameOrField = 0;
    stream_data.PastFrames = 0;
    stream_data.FutureFrames = 0;
    stream_data.ppPastSurfaces = nullptr;
    stream_data.ppFutureSurfaces = nullptr;
    stream_data.pInputSurface = input_view.Get();
    stream_data.ppPastSurfacesRight = nullptr;
    stream_data.ppFutureSurfacesRight = nullptr;
    
    hr = video_context_->VideoProcessorBlt(
        video_processor_.Get(), 
        rgb_frame.cached_output_view.Get(), 
        0, 
        1, 
        &stream_data
    );
    if (FAILED(hr)) {
        std::cerr << "VideoProcessorBlt failed: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 更新RGB帧信息
    if (nv12_frame->pts != AV_NOPTS_VALUE && nv12_frame->time_base.num > 0) {
        rgb_frame.timestamp = nv12_frame->pts * av_q2d(nv12_frame->time_base);
    } else {
        // 如果没有时间戳，使用帧序号作为简单的时间戳
        static int frame_counter = 0;
        rgb_frame.timestamp = frame_counter * (1.0 / 30.0); // 假设30fps
        frame_counter++;
    }
    rgb_frame.is_valid = true;
    
    return true;
}

bool RgbVideoDecoder::initializeVideoProcessor() {
    if (video_processor_initialized_) {
        return true;
    }
    
    // 获取Video Device
    HRESULT hr = d3d11_device_->QueryInterface(IID_PPV_ARGS(video_device_.GetAddressOf()));
    if (FAILED(hr)) {
        std::cerr << "Failed to query video device: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 获取Video Context
    hr = d3d11_context_->QueryInterface(IID_PPV_ARGS(video_context_.GetAddressOf()));
    if (FAILED(hr)) {
        std::cerr << "Failed to query video context: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建Video Processor Enumerator
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
    content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    content_desc.InputFrameRate.Numerator = 30;
    content_desc.InputFrameRate.Denominator = 1;
    content_desc.InputWidth = 1920;   // 临时默认值，实际会根据视频调整
    content_desc.InputHeight = 1080;
    content_desc.OutputWidth = 1920;
    content_desc.OutputHeight = 1080;
    content_desc.OutputFrameRate.Numerator = 30;
    content_desc.OutputFrameRate.Denominator = 1;
    content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
    
    hr = video_device_->CreateVideoProcessorEnumerator(&content_desc, video_enum_.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor enumerator: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建Video Processor
    hr = video_device_->CreateVideoProcessor(video_enum_.Get(), 0, video_processor_.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create video processor: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 设置色彩空间
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE input_color_space = {};
    input_color_space.RGB_Range = 0;
    input_color_space.YCbCr_Matrix = 1;
    input_color_space.YCbCr_xvYCC = 0;
    input_color_space.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235;
    video_context_->VideoProcessorSetStreamColorSpace(video_processor_.Get(), 0, &input_color_space);
    
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE output_color_space = {};
    output_color_space.RGB_Range = 0;
    output_color_space.YCbCr_Matrix = 1;
    output_color_space.YCbCr_xvYCC = 0;
    output_color_space.Nominal_Range = D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255;
    video_context_->VideoProcessorSetOutputColorSpace(video_processor_.Get(), &output_color_space);
    
    video_processor_initialized_ = true;
    return true;
}

bool RgbVideoDecoder::createRGBTexture(RgbFrame& rgb_frame, int width, int height) {
    if (!d3d11_device_) {
        return false;
    }
    
    // 清理旧纹理
    rgb_frame.reset();
    
    // 创建RGB纹理
    D3D11_TEXTURE2D_DESC texture_desc = {};
    texture_desc.Width = width;
    texture_desc.Height = height;
    texture_desc.MipLevels = 1;
    texture_desc.ArraySize = 1;
    texture_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    texture_desc.SampleDesc.Count = 1;
    texture_desc.SampleDesc.Quality = 0;
    texture_desc.Usage = D3D11_USAGE_DEFAULT;
    texture_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    texture_desc.CPUAccessFlags = 0;
    texture_desc.MiscFlags = 0;
    
    HRESULT hr = d3d11_device_->CreateTexture2D(&texture_desc, nullptr, rgb_frame.rgb_texture.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create RGB texture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // 创建Shader Resource View
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    hr = d3d11_device_->CreateShaderResourceView(rgb_frame.rgb_texture.Get(), &srv_desc, rgb_frame.rgb_srv.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Failed to create shader resource view: 0x" << std::hex << hr << std::endl;
        rgb_frame.rgb_texture.Reset();
        return false;
    }
    
    rgb_frame.width = width;
    rgb_frame.height = height;
    rgb_frame.is_valid = true;
    
    return true;
}

bool RgbVideoDecoder::isOpen() const {
    return hw_decoder_ && hw_decoder_->isOpen();
}

bool RgbVideoDecoder::isEOF() const {
    return hw_decoder_ && hw_decoder_->isEOF();
}

bool RgbVideoDecoder::seekToTime(double seconds) {
    return hw_decoder_ && hw_decoder_->seekToTime(seconds);
}

bool RgbVideoDecoder::seekToFrame(int64_t frame_number) {
    return hw_decoder_ && hw_decoder_->seekToFrame(frame_number);
}

void RgbVideoDecoder::close() {
    cleanup();
    
    if (hw_decoder_) {
        hw_decoder_->close();
    }
    
    d3d11_device_ = nullptr;
    d3d11_context_ = nullptr;
}

void RgbVideoDecoder::cleanup() {
    // 清理双缓冲RGB帧
    for (int i = 0; i < 2; i++) {
        rgb_frames_[i].reset();
    }
    
    video_processor_.Reset();
    video_enum_.Reset();
    video_context_.Reset();
    video_device_.Reset();
    video_processor_initialized_ = false;
    current_rgb_frame_index_ = 0;
}