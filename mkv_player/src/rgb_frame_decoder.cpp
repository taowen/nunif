#include "rgb_frame_decoder.h"
#include <iostream>

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3) \
    ((DWORD)(BYTE)(ch0) | ((DWORD)(BYTE)(ch1) << 8) | \
     ((DWORD)(BYTE)(ch2) << 16) | ((DWORD)(BYTE)(ch3) << 24))
#endif

RGBFrameDecoder::RGBFrameDecoder() 
    : d3d11_device_(nullptr)
    , d3d11_context_(nullptr)
    , video_width_(0)
    , video_height_(0)
    , is_initialized_(false) {
}

RGBFrameDecoder::~RGBFrameDecoder() {
    close();
}

bool RGBFrameDecoder::open(const std::string& filepath, ID3D11Device* external_device) {
    // 清理已有资源
    close();
    
    // 1. 首先初始化内部FrameDecoder
    if (!frame_decoder_.open(filepath)) {
        std::cerr << "Failed to open file with FrameDecoder" << std::endl;
        return false;
    }
    
    // 2. 决定使用哪个D3D11设备
    if (external_device) {
        // 使用外部设备
        d3d11_device_ = external_device;
        d3d11_device_->GetImmediateContext(&d3d11_context_);
        std::cout << "Using external D3D11 device for RGB conversion" << std::endl;
    } else {
        // 使用FrameDecoder内部设备
        d3d11_device_ = frame_decoder_.getD3D11Device();
        d3d11_context_ = frame_decoder_.getD3D11Context();
        
        if (!d3d11_device_ || !d3d11_context_) {
            std::cerr << "Failed to get D3D11 device from FrameDecoder" << std::endl;
            frame_decoder_.close();
            return false;
        }
        std::cout << "Using internal D3D11 device for RGB conversion" << std::endl;
    }
    
    // 2. 获取视频尺寸信息 - 从demuxer获取而不是解码帧
    auto* demuxer = frame_decoder_.getDemuxer();
    auto* video_params = demuxer->getVideoCodecParameters();
    
    if (!video_params) {
        std::cerr << "Failed to get video codec parameters" << std::endl;
        frame_decoder_.close();
        return false;
    }
    
    video_width_ = video_params->width;
    video_height_ = video_params->height;
    
    if (video_width_ <= 0 || video_height_ <= 0) {
        std::cerr << "Invalid video dimensions: " << video_width_ << "x" << video_height_ << std::endl;
        frame_decoder_.close();
        return false;
    }
    
    // 3. 新实现不需要预先初始化Video Processor和纹理池
    // 每次转换时会动态创建所需资源
    
    is_initialized_ = true;
    return true;
}

bool RGBFrameDecoder::readNextFrames(DecodedFrames& decoded_frames) {
    if (!is_initialized_) {
        decoded_frames.audio_frame.is_valid = false;
        decoded_frames.rgb_frame.is_valid = false;
        return false;
    }
    
    // 1. 从内部FrameDecoder获取原始帧
    FrameDecoder::DecodedFrames raw_frames;
    raw_frames.video_frame.frame = av_frame_alloc();
    raw_frames.audio_frame.frame = av_frame_alloc();
    
    if (!frame_decoder_.readNextFrames(raw_frames)) {
        av_frame_free(&raw_frames.video_frame.frame);
        av_frame_free(&raw_frames.audio_frame.frame);
        decoded_frames.audio_frame.is_valid = false;
        decoded_frames.rgb_frame.is_valid = false;
        return false;
    }
    
    // 2. 直接传递音频帧
    decoded_frames.audio_frame = raw_frames.audio_frame;
    
    // 3. 转换视频帧为RGB
    if (raw_frames.video_frame.is_valid) {
        decoded_frames.rgb_frame.timestamp = raw_frames.video_frame.timestamp;
        decoded_frames.rgb_frame.is_valid = convertNV12ToRGB(raw_frames.video_frame.frame, decoded_frames.rgb_frame);
    } else {
        decoded_frames.rgb_frame.is_valid = false;
    }
    
    // 4. 释放原始视频帧（音频帧已转移）
    av_frame_free(&raw_frames.video_frame.frame);
    
    return decoded_frames.audio_frame.is_valid || decoded_frames.rgb_frame.is_valid;
}


bool RGBFrameDecoder::convertNV12ToRGB(AVFrame* nv12_frame, RGBFrame& rgb_frame) {
    // 基于convert_color.cpp参考代码的完全重写实现
    if (!nv12_frame || !d3d11_device_ || !d3d11_context_) {
        std::cerr << "Error: Invalid frame or device provided." << std::endl;
        return false;
    }
    
    if (nv12_frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << nv12_frame->format << std::endl;
        return false;
    }
    
    if (nv12_frame->width <= 0 || nv12_frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return false;
    }
    
    if (!nv12_frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return false;
    }

    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(nv12_frame->data[0]);
    int texture_index = (int)(intptr_t)nv12_frame->data[1];
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return false;
    }

    // 声明所有变量以避免goto问题
    HRESULT hr;
    ID3D11VideoDevice* video_device = nullptr;
    ID3D11VideoContext* video_context = nullptr;
    ID3D11VideoProcessorEnumerator* video_enum = nullptr;
    ID3D11VideoProcessor* video_processor = nullptr;
    ID3D11Texture2D* output_texture = nullptr;
    ID3D11VideoProcessorInputView* input_view = nullptr;
    ID3D11VideoProcessorOutputView* output_view = nullptr;
    ID3D11ShaderResourceView* output_srv = nullptr;
    bool color_space_valid = true;
    
    // 初始化结构体
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
    D3D11_TEXTURE2D_DESC output_desc = {};
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
    D3D11_VIDEO_PROCESSOR_STREAM stream_data = {};
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE input_color_space = {};
    D3D11_VIDEO_PROCESSOR_COLOR_SPACE output_color_space = {};
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};

    // 获取video device和context
    hr = d3d11_device_->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&video_device);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to get ID3D11VideoDevice. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    hr = d3d11_context_->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&video_context);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to get ID3D11VideoContext. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建video processor enumerator
    content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
    content_desc.InputWidth = nv12_frame->width;
    content_desc.InputHeight = nv12_frame->height;
    content_desc.OutputWidth = nv12_frame->width;
    content_desc.OutputHeight = nv12_frame->height;
    content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    hr = video_device->CreateVideoProcessorEnumerator(&content_desc, &video_enum);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor enumerator. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建video processor
    hr = video_device->CreateVideoProcessor(video_enum, 0, &video_processor);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建输出纹理 (RGBA)
    output_desc.Width = nv12_frame->width;
    output_desc.Height = nv12_frame->height;
    output_desc.MipLevels = 1;
    output_desc.ArraySize = 1;
    output_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    output_desc.SampleDesc.Count = 1;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    output_desc.CPUAccessFlags = 0;

    hr = d3d11_device_->CreateTexture2D(&output_desc, nullptr, &output_texture);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create output texture. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建input view
    input_view_desc.FourCC = 0;
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    input_view_desc.Texture2D.ArraySlice = texture_index;

    hr = video_device->CreateVideoProcessorInputView(input_texture, video_enum, &input_view_desc, &input_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create input view. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建output view
    output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    output_view_desc.Texture2D.MipSlice = 0;

    hr = video_device->CreateVideoProcessorOutputView(output_texture, video_enum, &output_view_desc, &output_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create output view. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 创建Shader Resource View
    srv_desc.Format = output_desc.Format;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    
    hr = d3d11_device_->CreateShaderResourceView(output_texture, &srv_desc, &output_srv);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create shader resource view. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 验证颜色空间
    color_space_valid = true;
    if (nv12_frame->colorspace != AVCOL_SPC_BT709 && nv12_frame->colorspace != AVCOL_SPC_UNSPECIFIED) {
        std::cerr << "Warning: Unexpected colorspace " << nv12_frame->colorspace << ", expected BT.709 (" << AVCOL_SPC_BT709 << ")" << std::endl;
        color_space_valid = false;
    }
    
    if (nv12_frame->color_range != AVCOL_RANGE_MPEG && nv12_frame->color_range != AVCOL_RANGE_JPEG && nv12_frame->color_range != AVCOL_RANGE_UNSPECIFIED) {
        std::cerr << "Warning: Unexpected color_range " << nv12_frame->color_range << std::endl;
        color_space_valid = false;
    }
    
    std::cout << "Color space validation: " << (color_space_valid ? "PASSED" : "WARNING") << std::endl;
    
    // 配置基于帧属性的输入颜色空间
    input_color_space.Usage = 0; // Video processing
    input_color_space.RGB_Range = 0; // Not RGB input
    // 如果指定或未指定则使用BT.709，否则记录偏差
    input_color_space.YCbCr_Matrix = (nv12_frame->colorspace == AVCOL_SPC_BT709 || nv12_frame->colorspace == AVCOL_SPC_UNSPECIFIED) ? 1 : 1; // 默认为BT.709
    input_color_space.YCbCr_xvYCC = 0; // 标准YCbCr
    // 处理颜色范围:
    // D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_16_235 = 1 (Limited/TV range)
    // D3D11_VIDEO_PROCESSOR_NOMINAL_RANGE_0_255  = 2 (Full/PC range)
    if (nv12_frame->color_range == AVCOL_RANGE_JPEG) {
        input_color_space.Nominal_Range = 2; // Full range
    } else if (nv12_frame->color_range == AVCOL_RANGE_MPEG) {
        input_color_space.Nominal_Range = 1; // Limited range
    } else {
        // AVCOL_RANGE_UNSPECIFIED - 假设视频内容为limited range
        input_color_space.Nominal_Range = 1;
        std::cout << "Note: Unspecified color range, assuming limited range" << std::endl;
    }
    
    // 为RGB配置输出颜色空间
    output_color_space.Usage = 0; // Video processing  
    output_color_space.RGB_Range = 1; // Full range RGB (0-255). 0=Limited, 1=Full.
    output_color_space.YCbCr_Matrix = 1; // BT.709 (用于RGB转换矩阵)
    output_color_space.YCbCr_xvYCC = 0; // 不适用于RGB
    output_color_space.Nominal_Range = 2; // RGB输出的full range (0-255)
    
    // 明确设置颜色空间
    video_context->VideoProcessorSetStreamColorSpace(video_processor, 0, &input_color_space);
    video_context->VideoProcessorSetOutputColorSpace(video_processor, &output_color_space);
    
    // 执行转换
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

    hr = video_context->VideoProcessorBlt(video_processor, output_view, 0, 1, &stream_data);
    if (FAILED(hr)) {
        std::cerr << "Error: VideoProcessorBlt failed. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // 成功 - 设置输出RGB帧信息
    rgb_frame.rgb_texture = output_texture;
    rgb_frame.rgb_srv = output_srv;
    rgb_frame.width = nv12_frame->width;
    rgb_frame.height = nv12_frame->height;
    // timestamp已经在readNextFrames中设置，这里不需要覆盖
    rgb_frame.is_valid = true;
    
    // 成功时不释放output_texture和output_srv，它们将被返回
    goto cleanup_keep_output;

cleanup:
    if (output_srv) {
        output_srv->Release();
        output_srv = nullptr;
    }
    if (output_texture) {
        output_texture->Release();
        output_texture = nullptr;
    }

cleanup_keep_output:
    if (output_view) output_view->Release();
    if (input_view) input_view->Release();
    if (video_processor) video_processor->Release();
    if (video_enum) video_enum->Release();
    if (video_context) video_context->Release();
    if (video_device) video_device->Release();

    return rgb_frame.is_valid;
}

void RGBFrameDecoder::flush() {
    frame_decoder_.flush();
}

void RGBFrameDecoder::close() {
    frame_decoder_.close();
    releaseResources();
    is_initialized_ = false;
}

void RGBFrameDecoder::releaseResources() {
    // 新实现不需要清理预分配的资源
    // 所有Video Processor和纹理资源都在每次转换后立即释放
    
    // 只在使用外部设备时释放context引用（内部设备不需要释放）
    if (d3d11_context_ && d3d11_device_ && frame_decoder_.isInitialized() && 
        d3d11_device_ != frame_decoder_.getD3D11Device()) {
        d3d11_context_->Release();
    }
    d3d11_context_ = nullptr;
    d3d11_device_ = nullptr;
}