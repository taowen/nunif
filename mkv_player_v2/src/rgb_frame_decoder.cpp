#include "rgb_frame_decoder.h"
#include <iostream>

RGBFrameDecoder::RGBFrameDecoder() 
    : d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , is_initialized_(false)
    , current_pair_index_(0)
    , video_device_(nullptr)
    , video_context_(nullptr)
    , video_enum_(nullptr)
    , video_processor_(nullptr)
    , video_processor_initialized_(false) {
    
    // 初始化双缓冲RGBFramePair
    for (int i = 0; i < 2; i++) {
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].rgb_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
    }
}

RGBFrameDecoder::~RGBFrameDecoder() {
    close();
}

bool RGBFrameDecoder::open(const std::string& filepath) {
    // 清理已有资源
    close();
    
    // 1. 首先初始化内部HwFrameDecoder
    if (!frame_decoder_.open(filepath)) {
        std::cerr << "Failed to open file with HwFrameDecoder" << std::endl;
        return false;
    }
    
    // 2. 通过解码第一帧获取FFmpeg的D3D11设备（确保设备一致性）
    AVFrame* temp_frame = av_frame_alloc();
    if (!temp_frame) {
        std::cerr << "Failed to allocate temporary frame" << std::endl;
        frame_decoder_.close();
        return false;
    }
    
    if (!frame_decoder_.tryDecodeFirstVideoFrame(temp_frame)) {
        std::cerr << "Failed to decode first video frame" << std::endl;
        av_frame_free(&temp_frame);
        frame_decoder_.close();
        return false;
    }
    
    d3d11_device_ = HwFrameDecoder::getD3D11DeviceFromFrame(temp_frame);
    d3d11_context_ = HwFrameDecoder::getD3D11ContextFromFrame(temp_frame);
    
    av_frame_free(&temp_frame);
    
    if (!d3d11_device_ || !d3d11_context_) {
        std::cerr << "Failed to get D3D11 device from decoded frame" << std::endl;
        frame_decoder_.close();
        return false;
    }
    
    std::cout << "Using HwFrameDecoder's D3D11 device for RGB conversion" << std::endl;
    
    // 3. 初始化双缓冲
    initializeBorrowedPairs();
    
    is_initialized_ = true;
    return true;
}

bool RGBFrameDecoder::readNextRGBFramePair(RGBFramePair& rgb_pair) {
    if (!is_initialized_) {
        rgb_pair.audio_frame.is_valid = false;
        rgb_pair.rgb_frame.is_valid = false;
        rgb_pair.is_valid = false;
        return false;
    }
    
    // 1. 从内部HwFrameDecoder获取原始帧的描述信息（非指针）
    HwFrameDecoder::HwFramePair raw_frames;
    if (!frame_decoder_.readNextHwFramePair(raw_frames)) {
        rgb_pair.audio_frame.is_valid = false;
        rgb_pair.rgb_frame.is_valid = false;
        rgb_pair.is_valid = false;
        return false;
    }
    
    // 获取当前缓冲pair的引用
    RGBFramePair& current_pair = borrowed_pairs_[current_pair_index_];
    
    // 清理之前的数据，但保持纹理资源不变（池化复用）
    current_pair.audio_frame.is_valid = false;
    current_pair.rgb_frame.is_valid = false;
    // 注意：不清理 rgb_frame 的纹理资源，只清理有效性标志
    
    // 直接传递音频帧描述符
    current_pair.audio_frame = raw_frames.audio_frame;
    
    // 3. 转换视频帧为RGB（使用双缓冲）
    if (raw_frames.video_frame.is_valid) {
        
        // 获取视频尺寸
        AVFrame* video_frame = raw_frames.video_frame.get();
        int video_width = video_frame->width;
        int video_height = video_frame->height;
        
        // 如果尺寸不匹配，重新创建纹理
        if (!current_pair.rgb_frame.hasValidResources() || 
            current_pair.rgb_frame.width != video_width || 
            current_pair.rgb_frame.height != video_height) {
            if (!createRGBTexture(current_pair.rgb_frame, video_width, video_height)) {
                current_pair.rgb_frame.is_valid = false;
                return false;
            }
        }
        
        // 转换到RGB纹理
        if (convertNV12ToRGB(raw_frames.video_frame, current_pair.rgb_frame)) {
            current_pair.rgb_frame.timestamp = raw_frames.video_frame.timestamp;
            current_pair.rgb_frame.is_valid = true;
        } else {
            current_pair.rgb_frame.is_valid = false;
        }
    } else {
        current_pair.rgb_frame.is_valid = false;
    }
    
    // 4. 注意：不再需要释放任何AVFrame，因为所有权从未转移
    
    // 设置整体有效性
    current_pair.is_valid = current_pair.audio_frame.is_valid || current_pair.rgb_frame.is_valid;
    
    // 将当前pair返回给调用者
    if (current_pair.is_valid) {
        rgb_pair = current_pair;
        // 切换到下一个pair
        current_pair_index_ = (current_pair_index_ + 1) % 2;
    }
    
    return current_pair.is_valid;
}

bool RGBFrameDecoder::convertNV12ToRGB(const HwFrameDecoder::HwFrame& nv12_frame_desc, RGBFrame& rgb_frame) {
    AVFrame* nv12_frame = nv12_frame_desc.get();
    if (!nv12_frame || !d3d11_device_ || !d3d11_context_) {
        std::cerr << "Error: Invalid frame, device, or rgb_frame provided." << std::endl;
        return false;
    }
    
    if (nv12_frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << nv12_frame->format << std::endl;
        return false;
    }
    
    if (!nv12_frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return false;
    }

    // 确保Video Processor已初始化
    if (!ensureVideoProcessor()) {
        std::cerr << "Error: Failed to initialize Video Processor." << std::endl;
        return false;
    }

    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(nv12_frame->data[0]);
    int texture_index = (int)(intptr_t)nv12_frame->data[1];
    
    // 检查输入纹理格式
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return false;
    }

    // 创建输入视图 - 确保参数正确
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
    input_view_desc.FourCC = 0;
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    
    // 检查ArraySlice是否超出范围
    if (texture_index >= 0 && texture_index < input_desc.ArraySize) {
        input_view_desc.Texture2D.ArraySlice = texture_index;
    } else {
        std::cerr << "Warning: Invalid texture index " << texture_index << ", using 0. Array size: " << input_desc.ArraySize << std::endl;
        input_view_desc.Texture2D.ArraySlice = 0;
    }

    ID3D11VideoProcessorInputView* input_view = nullptr;
    HRESULT hr = video_device_->CreateVideoProcessorInputView(input_texture, video_enum_.Get(), &input_view_desc, &input_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create input view. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // 创建输出视图
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
    output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    output_view_desc.Texture2D.MipSlice = 0;

    ID3D11VideoProcessorOutputView* output_view = nullptr;
    hr = video_device_->CreateVideoProcessorOutputView(rgb_frame.rgb_texture.Get(), video_enum_.Get(), &output_view_desc, &output_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create output view. HRESULT: 0x" << std::hex << hr << std::endl;
        input_view->Release();
        return false;
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
    stream_data.pInputSurface = input_view;
    stream_data.ppPastSurfacesRight = nullptr;
    stream_data.ppFutureSurfacesRight = nullptr;

    hr = video_context_->VideoProcessorBlt(video_processor_.Get(), output_view, 0, 1, &stream_data);
    
    // 清理临时资源
    output_view->Release();
    input_view->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Error: VideoProcessorBlt failed. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    return true;
}

void RGBFrameDecoder::close() {
    frame_decoder_.close();
    releaseResources();
    
    // 外部设备，只清空指针
    d3d11_context_ = nullptr;
    d3d11_device_ = nullptr;
    
    is_initialized_ = false;
}

bool RGBFrameDecoder::ensureVideoProcessor() {
    if (video_processor_initialized_) {
        return true;
    }
    
    // 获取Video Device
    HRESULT hr = d3d11_device_->QueryInterface(IID_PPV_ARGS(&video_device_));
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to query video device interface. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // 获取Video Context
    hr = d3d11_context_->QueryInterface(IID_PPV_ARGS(&video_context_));
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to query video context interface. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // 创建Video Processor Enumerator
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
    content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    content_desc.InputFrameRate.Numerator = 30;
    content_desc.InputFrameRate.Denominator = 1;
    // 获取视频尺寸信息
    AVFrame* temp_frame = av_frame_alloc();
    if (!temp_frame || !frame_decoder_.tryDecodeFirstVideoFrame(temp_frame)) {
        std::cerr << "Error: Failed to get video dimensions for video processor" << std::endl;
        if (temp_frame) av_frame_free(&temp_frame);
        return false;
    }
    
    int video_width = temp_frame->width;
    int video_height = temp_frame->height;
    av_frame_free(&temp_frame);
    
    content_desc.InputWidth = video_width;
    content_desc.InputHeight = video_height;
    content_desc.OutputWidth = video_width;
    content_desc.OutputHeight = video_height;
    content_desc.OutputFrameRate.Numerator = 30;
    content_desc.OutputFrameRate.Denominator = 1;
    content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
    
    hr = video_device_->CreateVideoProcessorEnumerator(&content_desc, video_enum_.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor enumerator. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // 创建Video Processor
    hr = video_device_->CreateVideoProcessor(video_enum_.Get(), 0, video_processor_.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor. HRESULT: 0x" << std::hex << hr << std::endl;
        return false;
    }

    // 配置色彩空间（一次性设置）
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

bool RGBFrameDecoder::createRGBTexture(RGBFrame& rgb_frame, int width, int height) {
    if (!d3d11_device_) {
        return false;
    }
    
    // 只在尺寸变化时才释放旧纹理
    if (rgb_frame.width != width || rgb_frame.height != height) {
        rgb_frame.reset();
    }
    
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
        std::cerr << "Error: Failed to create RGB texture. HRESULT: 0x" << std::hex << hr << std::endl;
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
        std::cerr << "Error: Failed to create shader resource view. HRESULT: 0x" << std::hex << hr << std::endl;
        rgb_frame.rgb_texture.Reset();
        return false;
    }

    rgb_frame.width = width;
    rgb_frame.height = height;
    rgb_frame.is_valid = true;
    
    return true;
}

void RGBFrameDecoder::releaseVideoProcessor() {
    video_processor_.Reset();
    video_enum_.Reset();
    video_context_.Reset();
    video_device_.Reset();
    video_processor_initialized_ = false;
}

void RGBFrameDecoder::releaseResources() {
    // 释放双缓冲纹理
    for (int i = 0; i < 2; i++) {
        borrowed_pairs_[i].rgb_frame.reset();
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
    }
    
    // 释放Video Processor
    releaseVideoProcessor();
    
    // 重置索引
    current_pair_index_ = 0;
    
    // 外部设备，只清空指针
    d3d11_context_ = nullptr;
    d3d11_device_ = nullptr;
}

void RGBFrameDecoder::initializeBorrowedPairs() {
    // 初始化双缓冲对
    for (int i = 0; i < 2; i++) {
        borrowed_pairs_[i].audio_frame.is_valid = false;
        borrowed_pairs_[i].rgb_frame.is_valid = false;
        borrowed_pairs_[i].is_valid = false;
    }
}