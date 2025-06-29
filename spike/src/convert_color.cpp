#include "main.h"
// Add CUDA headers for interoperability
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <algorithm>  // for std::min, std::max
#include <iomanip>    // for std::setw, std::setprecision, std::fixed

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

ID3D11Texture2D* convert_color(const FFMepgContext* ctx, AVFrame* frame) {
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return nullptr;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return nullptr;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return nullptr;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return nullptr;
    }

    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return nullptr;
    }

    // Declare all variables at the beginning to avoid goto issues
    HRESULT hr;
    ID3D11VideoDevice* video_device = nullptr;
    ID3D11VideoContext* video_context = nullptr;
    ID3D11VideoProcessorEnumerator* video_enum = nullptr;
    ID3D11VideoProcessor* video_processor = nullptr;
    ID3D11Texture2D* output_texture = nullptr;
    ID3D11VideoProcessorInputView* input_view = nullptr;
    ID3D11VideoProcessorOutputView* output_view = nullptr;
    
    // Initialize structs
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
    D3D11_TEXTURE2D_DESC output_desc = {};
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
    D3D11_VIDEO_PROCESSOR_STREAM stream_data = {};

    // Get video device and context
    hr = ctx->d3d_device->QueryInterface(__uuidof(ID3D11VideoDevice), (void**)&video_device);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to get ID3D11VideoDevice. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    hr = ctx->d3d_context->QueryInterface(__uuidof(ID3D11VideoContext), (void**)&video_context);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to get ID3D11VideoContext. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Create video processor enumerator
    content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_INTERLACED_TOP_FIELD_FIRST;
    content_desc.InputWidth = frame->width;
    content_desc.InputHeight = frame->height;
    content_desc.OutputWidth = frame->width;
    content_desc.OutputHeight = frame->height;
    content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;

    hr = video_device->CreateVideoProcessorEnumerator(&content_desc, &video_enum);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor enumerator. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Create video processor
    hr = video_device->CreateVideoProcessor(video_enum, 0, &video_processor);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create video processor. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Create output texture (RGBA)
    output_desc.Width = frame->width;
    output_desc.Height = frame->height;
    output_desc.MipLevels = 1;
    output_desc.ArraySize = 1;
    output_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    output_desc.SampleDesc.Count = 1;
    output_desc.Usage = D3D11_USAGE_DEFAULT;
    output_desc.BindFlags = D3D11_BIND_RENDER_TARGET;
    output_desc.CPUAccessFlags = 0;

    hr = ctx->d3d_device->CreateTexture2D(&output_desc, nullptr, &output_texture);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create output texture. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Create input view
    input_view_desc.FourCC = 0;
    input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    input_view_desc.Texture2D.MipSlice = 0;
    input_view_desc.Texture2D.ArraySlice = texture_index;

    hr = video_device->CreateVideoProcessorInputView(input_texture, video_enum, &input_view_desc, &input_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create input view. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Create output view
    output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
    output_view_desc.Texture2D.MipSlice = 0;

    hr = video_device->CreateVideoProcessorOutputView(output_texture, video_enum, &output_view_desc, &output_view);
    if (FAILED(hr)) {
        std::cerr << "Error: Failed to create output view. HRESULT: 0x" << std::hex << hr << std::endl;
        goto cleanup;
    }

    // Set default color space
    video_context->VideoProcessorSetStreamColorSpace(video_processor, 0, nullptr);
    video_context->VideoProcessorSetOutputColorSpace(video_processor, nullptr);

    // Perform the conversion
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

    // Success - don't release output_texture, it will be returned
    goto cleanup_keep_output;

cleanup:
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

    return output_texture;
}