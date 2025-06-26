#include "encode_thread.h"
#include <iostream>
#include <sstream>
#include <d3dcompiler.h>

extern "C" {
#include <libavutil/opt.h>
}

// Helper function for av_err2str to work with MSVC
static std::string av_error_to_string(int error_code) {
    char error_buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(error_code, error_buf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(error_buf);
}

// Encoder state structure
struct EncoderState {
    // FFmpeg encoder resources
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVStream* video_stream = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* yuv_frame = nullptr;
    
    // DirectX shader resources for color conversion
    ID3D11ComputeShader* rgb_to_yuv_shader = nullptr;
    ID3D11Buffer* conversion_constants_buffer = nullptr;
    ID3D11Texture2D* yuv_output_texture = nullptr;
    ID3D11UnorderedAccessView* yuv_output_uav = nullptr;
    ID3D11Texture2D* staging_texture = nullptr;
    
    // Encoder parameters
    int width = 0;
    int height = 0;
    int frame_count = 0;
    
    void cleanup() {
        if (yuv_frame) {
            av_frame_free(&yuv_frame);
        }
        if (packet) {
            av_packet_free(&packet);
        }
        if (codec_ctx) {
            avcodec_free_context(&codec_ctx);
        }
        if (format_ctx && format_ctx->pb) {
            avio_closep(&format_ctx->pb);
        }
        if (format_ctx) {
            avformat_free_context(format_ctx);
        }
        
        if (staging_texture) {
            staging_texture->Release();
            staging_texture = nullptr;
        }
        if (yuv_output_uav) {
            yuv_output_uav->Release();
            yuv_output_uav = nullptr;
        }
        if (yuv_output_texture) {
            yuv_output_texture->Release();
            yuv_output_texture = nullptr;
        }
        if (conversion_constants_buffer) {
            conversion_constants_buffer->Release();
            conversion_constants_buffer = nullptr;
        }
        if (rgb_to_yuv_shader) {
            rgb_to_yuv_shader->Release();
            rgb_to_yuv_shader = nullptr;
        }
    }
};

namespace {

std::string generate_rgb_to_yuv_shader() {
    return R"(
cbuffer ConversionConstants : register(b0)
{
    float4x4 ColorMatrix;
    float4 LumaCoeffs;
    float4 ChromaCoeffs;
    float4 Offset;
    int InputFormat;
    int ColorSpace;
    int BitDepth;
    int IsHDR;
};

Texture2D<float4> InputTexture : register(t0);
RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    OutputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height)
        return;
    
    // Sample RGB input
    float3 rgb = InputTexture.Load(int3(id.xy, 0)).rgb;
    
    // Convert RGB to YUV using matrix transformation
    float3 yuv = mul(ColorMatrix, float4(rgb, 1.0)).xyz;
    
    // Apply color range compression (full range to limited range)
    if (ColorSpace != 0) { // Not full range
        yuv.x = yuv.x * 219.0/255.0 + 16.0/255.0;  // Y
        yuv.yz = yuv.yz * 224.0/255.0 + 128.0/255.0; // UV
    }
    
    // Clamp values
    yuv = saturate(yuv);
    
    // Output in YUV format
    OutputTexture[id.xy] = float4(yuv, 1.0);
}
)";
}

ConversionConstants generate_rgb_to_yuv_constants(const ColorSpaceInfo& color_info) {
    ConversionConstants constants = {};
    
    constants.color_space = static_cast<int>(color_info.color_space);
    constants.bit_depth = color_info.bit_depth;
    constants.is_hdr = color_info.is_hdr ? 1 : 0;
    
    // RGB to YUV conversion matrix
    float matrix[16] = {0};
    
    switch (color_info.color_space) {
        case AVCOL_SPC_BT709:
            // BT.709 RGB to YUV matrix
            matrix[0] = 0.2126f;   matrix[1] = 0.7152f;   matrix[2] = 0.0722f;   matrix[3] = 0.0f;
            matrix[4] = -0.1146f;  matrix[5] = -0.3854f;  matrix[6] = 0.5f;      matrix[7] = 0.0f;
            matrix[8] = 0.5f;      matrix[9] = -0.4542f;  matrix[10] = -0.0458f; matrix[11] = 0.0f;
            matrix[12] = 0.0f;     matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            // BT.2020 RGB to YUV matrix
            matrix[0] = 0.2627f;   matrix[1] = 0.6780f;   matrix[2] = 0.0593f;   matrix[3] = 0.0f;
            matrix[4] = -0.1396f;  matrix[5] = -0.3604f;  matrix[6] = 0.5f;      matrix[7] = 0.0f;
            matrix[8] = 0.5f;      matrix[9] = -0.4598f;  matrix[10] = -0.0402f; matrix[11] = 0.0f;
            matrix[12] = 0.0f;     matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
        default:
            // Default to BT.601 RGB to YUV matrix
            matrix[0] = 0.299f;    matrix[1] = 0.587f;    matrix[2] = 0.114f;    matrix[3] = 0.0f;
            matrix[4] = -0.1687f;  matrix[5] = -0.3313f;  matrix[6] = 0.5f;      matrix[7] = 0.0f;
            matrix[8] = 0.5f;      matrix[9] = -0.4187f;  matrix[10] = -0.0813f; matrix[11] = 0.0f;
            matrix[12] = 0.0f;     matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
    }
    
    memcpy(constants.matrix, matrix, sizeof(matrix));
    return constants;
}

bool create_rgb_to_yuv_shader(EncoderState& encoder_state, ID3D11Device* d3d11_device) {
    std::string shader_source = generate_rgb_to_yuv_shader();
    
    ID3DBlob* shader_blob = nullptr;
    ID3DBlob* error_blob = nullptr;
    
    HRESULT hr = D3DCompile(
        shader_source.c_str(),
        shader_source.length(),
        nullptr, nullptr, nullptr,
        "CSMain", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS, 0,
        &shader_blob, &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "RGB to YUV shader compilation error: " << (char*)error_blob->GetBufferPointer() << "\n";
            error_blob->Release();
        }
        return false;
    }
    
    hr = d3d11_device->CreateComputeShader(
        shader_blob->GetBufferPointer(),
        shader_blob->GetBufferSize(),
        nullptr, &encoder_state.rgb_to_yuv_shader
    );
    
    shader_blob->Release();
    return SUCCEEDED(hr);
}

bool create_yuv_output_resources(int width, int height, EncoderState& encoder_state, ID3D11Device* d3d11_device) {
    // Create YUV output texture
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &encoder_state.yuv_output_texture);
    if (FAILED(hr)) return false;
    
    // Create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = desc.Format;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    hr = d3d11_device->CreateUnorderedAccessView(encoder_state.yuv_output_texture, &uav_desc, &encoder_state.yuv_output_uav);
    if (FAILED(hr)) return false;
    
    // Create staging texture for CPU access
    desc.Usage = D3D11_USAGE_STAGING;
    desc.BindFlags = 0;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    
    hr = d3d11_device->CreateTexture2D(&desc, nullptr, &encoder_state.staging_texture);
    return SUCCEEDED(hr);
}

bool initialize_ffmpeg_encoder(const std::string& output_filename, const ColorSpaceInfo& color_info, 
                              int width, int height, EncoderState& encoder_state) {
    // Set FFmpeg log level to reduce x265 verbosity
    av_log_set_level(AV_LOG_ERROR);
    
    // Initialize format context
    int ret = avformat_alloc_output_context2(&encoder_state.format_ctx, nullptr, nullptr, output_filename.c_str());
    if (ret < 0) {
        std::cerr << "Could not create output context: " << av_error_to_string(ret) << "\n";
        return false;
    }
    
    // Find H.265 encoder
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_HEVC);
    if (!codec) {
        std::cerr << "H.265 encoder not found\n";
        return false;
    }
    
    // Create video stream
    encoder_state.video_stream = avformat_new_stream(encoder_state.format_ctx, nullptr);
    if (!encoder_state.video_stream) {
        std::cerr << "Could not create video stream\n";
        return false;
    }
    
    // Create codec context
    encoder_state.codec_ctx = avcodec_alloc_context3(codec);
    if (!encoder_state.codec_ctx) {
        std::cerr << "Could not allocate codec context\n";
        return false;
    }
    
    // Set encoder parameters
    encoder_state.codec_ctx->codec_id = AV_CODEC_ID_HEVC;
    encoder_state.codec_ctx->bit_rate = 8000000; // 8Mbps
    encoder_state.codec_ctx->width = width;
    encoder_state.codec_ctx->height = height;
    encoder_state.codec_ctx->time_base = {1, 30}; // 30 FPS
    encoder_state.codec_ctx->framerate = {30, 1};
    encoder_state.codec_ctx->gop_size = 30;
    encoder_state.codec_ctx->max_b_frames = 2;
    encoder_state.codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    
    // Set color space properties
    encoder_state.codec_ctx->colorspace = color_info.color_space;
    encoder_state.codec_ctx->color_primaries = color_info.color_primaries;
    encoder_state.codec_ctx->color_trc = color_info.color_trc;
    encoder_state.codec_ctx->color_range = color_info.color_range;
    
    // Set encoding quality
    av_opt_set(encoder_state.codec_ctx->priv_data, "preset", "medium", 0);
    av_opt_set(encoder_state.codec_ctx->priv_data, "crf", "23", 0);
    
    // Disable x265 verbose logging
    av_opt_set(encoder_state.codec_ctx->priv_data, "log-level", "error", 0);
    // Alternative: use x265-params to set log level
    av_opt_set(encoder_state.codec_ctx->priv_data, "x265-params", "log-level=error", 0);
    
    if (encoder_state.format_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
        encoder_state.codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    
    // Open codec
    ret = avcodec_open2(encoder_state.codec_ctx, codec, nullptr);
    if (ret < 0) {
        std::cerr << "Could not open codec: " << av_error_to_string(ret) << "\n";
        return false;
    }
    
    // Copy codec parameters to stream
    ret = avcodec_parameters_from_context(encoder_state.video_stream->codecpar, encoder_state.codec_ctx);
    if (ret < 0) {
        std::cerr << "Could not copy codec parameters: " << av_error_to_string(ret) << "\n";
        return false;
    }
    
    // Open output file
    if (!(encoder_state.format_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&encoder_state.format_ctx->pb, output_filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            std::cerr << "Could not open output file: " << av_error_to_string(ret) << "\n";
            return false;
        }
    }
    
    // Write header
    ret = avformat_write_header(encoder_state.format_ctx, nullptr);
    if (ret < 0) {
        std::cerr << "Error writing header: " << av_error_to_string(ret) << "\n";
        return false;
    }
    
    // Allocate frame and packet
    encoder_state.yuv_frame = av_frame_alloc();
    encoder_state.yuv_frame->format = encoder_state.codec_ctx->pix_fmt;
    encoder_state.yuv_frame->width = width;
    encoder_state.yuv_frame->height = height;
    
    ret = av_frame_get_buffer(encoder_state.yuv_frame, 0);
    if (ret < 0) {
        std::cerr << "Could not allocate frame buffer: " << av_error_to_string(ret) << "\n";
        return false;
    }
    
    encoder_state.packet = av_packet_alloc();
    encoder_state.width = width;
    encoder_state.height = height;
    
    return true;
}

bool convert_rgb_to_yuv_and_encode(const StereoInferredFrame& stereo_frame, 
                                  const ColorSpaceInfo& color_info,
                                  EncoderState& encoder_state,
                                  ID3D11Device* d3d11_device,
                                  ID3D11DeviceContext* d3d11_context) {
    // Create input SRV
    ID3D11ShaderResourceView* input_srv = nullptr;
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    HRESULT hr = d3d11_device->CreateShaderResourceView(stereo_frame.stereo_texture, &srv_desc, &input_srv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV for encoding\n";
        return false;
    }
    
    // Update constants buffer
    ConversionConstants constants = generate_rgb_to_yuv_constants(color_info);
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    hr = d3d11_context->Map(encoder_state.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    if (SUCCEEDED(hr)) {
        memcpy(mapped_resource.pData, &constants, sizeof(constants));
        d3d11_context->Unmap(encoder_state.conversion_constants_buffer, 0);
    }
    
    // Execute RGB to YUV conversion shader
    d3d11_context->CSSetShader(encoder_state.rgb_to_yuv_shader, nullptr, 0);
    d3d11_context->CSSetShaderResources(0, 1, &input_srv);
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &encoder_state.yuv_output_uav, nullptr);
    d3d11_context->CSSetConstantBuffers(0, 1, &encoder_state.conversion_constants_buffer);
    
    UINT dispatch_x = (stereo_frame.width + 7) / 8;
    UINT dispatch_y = (stereo_frame.height + 7) / 8;
    d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);
    
    // Copy to staging texture
    d3d11_context->CopyResource(encoder_state.staging_texture, encoder_state.yuv_output_texture);
    d3d11_context->Flush();
    
    // Map staging texture and copy to AVFrame
    D3D11_MAPPED_SUBRESOURCE mapped_staging;
    hr = d3d11_context->Map(encoder_state.staging_texture, 0, D3D11_MAP_READ, 0, &mapped_staging);
    if (FAILED(hr)) {
        input_srv->Release();
        return false;
    }
    
    // Convert float RGBA to YUV420P
    float* rgba_data = (float*)mapped_staging.pData;
    int stride = mapped_staging.RowPitch / sizeof(float) / 4; // 4 components per pixel
    
    // Fill Y plane
    for (int y = 0; y < encoder_state.height; y++) {
        for (int x = 0; x < encoder_state.width; x++) {
            float y_val = rgba_data[(y * stride + x) * 4 + 0]; // Y component
            encoder_state.yuv_frame->data[0][y * encoder_state.yuv_frame->linesize[0] + x] = 
                (uint8_t)(y_val * 255.0f);
        }
    }
    
    // Fill U and V planes (4:2:0 subsampling)
    for (int y = 0; y < encoder_state.height / 2; y++) {
        for (int x = 0; x < encoder_state.width / 2; x++) {
            float u_val = rgba_data[((y * 2) * stride + (x * 2)) * 4 + 1]; // U component
            float v_val = rgba_data[((y * 2) * stride + (x * 2)) * 4 + 2]; // V component
            
            encoder_state.yuv_frame->data[1][y * encoder_state.yuv_frame->linesize[1] + x] = 
                (uint8_t)(u_val * 255.0f);
            encoder_state.yuv_frame->data[2][y * encoder_state.yuv_frame->linesize[2] + x] = 
                (uint8_t)(v_val * 255.0f);
        }
    }
    
    d3d11_context->Unmap(encoder_state.staging_texture, 0);
    
    // Set frame parameters
    encoder_state.yuv_frame->pts = encoder_state.frame_count++;
    
    // Encode frame
    int ret = avcodec_send_frame(encoder_state.codec_ctx, encoder_state.yuv_frame);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        std::cerr << "Error sending frame to encoder: " << av_error_to_string(ret) << "\n";
        input_srv->Release();
        return false;
    }
    
    // Receive packets
    while (ret >= 0) {
        ret = avcodec_receive_packet(encoder_state.codec_ctx, encoder_state.packet);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        } else if (ret < 0) {
            std::cerr << "Error receiving packet from encoder: " << av_error_to_string(ret) << "\n";
            break;
        }
        
        // Scale packet timestamps
        av_packet_rescale_ts(encoder_state.packet, encoder_state.codec_ctx->time_base, 
                            encoder_state.video_stream->time_base);
        encoder_state.packet->stream_index = encoder_state.video_stream->index;
        
        // Write packet
        ret = av_interleaved_write_frame(encoder_state.format_ctx, encoder_state.packet);
        if (ret < 0) {
            std::cerr << "Error writing packet: " << av_error_to_string(ret) << "\n";
        }
        
        av_packet_unref(encoder_state.packet);
    }
    
    // Cleanup
    input_srv->Release();
    
    // Unbind resources
    ID3D11ShaderResourceView* null_srv = nullptr;
    ID3D11UnorderedAccessView* null_uav = nullptr;
    d3d11_context->CSSetShaderResources(0, 1, &null_srv);
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    
    return true;
}

} // anonymous namespace

void start_encode_thread(
    StereoInferredFrameQueue& input_frame_queue,
    const std::string& output_filename,
    const ColorSpaceInfo& color_info,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context
) {
    std::cout << "Starting encode thread for: " << output_filename << std::endl;
    
    EncoderState encoder_state;
    bool encoder_initialized = false;
    
    // Create constants buffer
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.ByteWidth = sizeof(ConversionConstants);
    buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    while (true) {
        StereoInferredFrame frame = input_frame_queue.pop();
        
        if (frame.is_end_signal) {
            std::cout << "Encode thread received end signal" << std::endl;
            break;
        }
        
        // Initialize encoder on first frame
        if (!encoder_initialized && d3d11_device) {
            if (!create_rgb_to_yuv_shader(encoder_state, d3d11_device) ||
                !create_yuv_output_resources(frame.width, frame.height, encoder_state, d3d11_device) ||
                !initialize_ffmpeg_encoder(output_filename, color_info, frame.width, frame.height, encoder_state)) {
                std::cerr << "Failed to initialize encoder\n";
                break;
            }
            
            d3d11_device->CreateBuffer(&buffer_desc, nullptr, &encoder_state.conversion_constants_buffer);
            encoder_initialized = true;
        }
        
        if (encoder_initialized) {
            convert_rgb_to_yuv_and_encode(frame, color_info, encoder_state, d3d11_device, d3d11_context);
        }
    }
    
    // Flush encoder
    if (encoder_initialized) {
        avcodec_send_frame(encoder_state.codec_ctx, nullptr);
        
        int ret;
        while ((ret = avcodec_receive_packet(encoder_state.codec_ctx, encoder_state.packet)) >= 0) {
            av_packet_rescale_ts(encoder_state.packet, encoder_state.codec_ctx->time_base, 
                                encoder_state.video_stream->time_base);
            encoder_state.packet->stream_index = encoder_state.video_stream->index;
            av_interleaved_write_frame(encoder_state.format_ctx, encoder_state.packet);
            av_packet_unref(encoder_state.packet);
        }
        
        av_write_trailer(encoder_state.format_ctx);
        std::cout << "H.265 encoding completed: " << encoder_state.frame_count << " frames" << std::endl;
    }
    
    encoder_state.cleanup();
    std::cout << "Encode thread finished" << std::endl;
} 