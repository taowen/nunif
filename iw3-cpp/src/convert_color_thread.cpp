#include "convert_color_thread.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <d3dcompiler.h>

namespace {

// Validation function for supported input format
bool validate_input_format(const ColorSpaceInfo& color_info) {
    // Output actual format information for debugging
    std::cout << "Input format info:\n";
    std::cout << "  DXGI Format: " << color_info.dxgi_format << " (expected: " << DXGI_FORMAT_NV12 << " for NV12)\n";
    std::cout << "  Color Space: " << color_info.color_space << " (expected: " << AVCOL_SPC_BT709 << " for BT709)\n";
    std::cout << "  Bit Depth: " << color_info.bit_depth << "\n";
    std::cout << "  Is HDR: " << (color_info.is_hdr ? "true" : "false") << "\n";
    
    // Only support yuv420p (NV12 in D3D11) with BT709 colorspace
    if (color_info.dxgi_format != DXGI_FORMAT_NV12) {
        std::ostringstream oss;
        oss << "Unsupported pixel format. Received DXGI format: " << color_info.dxgi_format 
            << ", expected: " << DXGI_FORMAT_NV12 << " (NV12/yuv420p)";
        throw std::runtime_error(oss.str());
    }
    
    if (color_info.color_space != AVCOL_SPC_BT709) {
        std::ostringstream oss;
        oss << "Unsupported colorspace. Received: " << color_info.color_space 
            << ", expected: " << AVCOL_SPC_BT709 << " (BT709)";
        throw std::runtime_error(oss.str());
    }
    
    if (color_info.is_hdr) {
        throw std::runtime_error("HDR content is not supported. Only SDR content is supported.");
    }
    
    // Expect MPEG/TV range (limited range)
    if (color_info.bit_depth != 8) {
        std::ostringstream oss;
        oss << "Unsupported bit depth. Received: " << color_info.bit_depth 
            << ", expected: 8-bit";
        throw std::runtime_error(oss.str());
    }
    
    std::cout << "✓ Input format validation passed: yuv420p + BT709 + 8-bit + SDR\n";
    return true;
}

bool create_intermediate_texture(UINT width, UINT height, DXGI_FORMAT format,
                                 ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED; // For CUDA interop in the next thread
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_state.intermediate_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create intermediate texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    return true;
}

std::string generate_shader_source_bt709_yuv420p() {
    // Simplified shader source for yuv420p + BT709 only
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

Texture2D<float> LumaTexture : register(t0);
Texture2D<float2> ChromaTexture : register(t1);
RWTexture2D<float4> OutputTexture : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID)
{
    uint width, height;
    OutputTexture.GetDimensions(width, height);
    
    if (id.x >= width || id.y >= height)
        return;
    
    // NV12 format handling (yuv420p)
    float3 yuv;
    yuv.x = LumaTexture.Load(int3(id.xy, 0)); // Y
    float2 uv = ChromaTexture.Load(int3(id.xy / 2, 0)); // UV
    yuv.y = uv.x; // U
    yuv.z = uv.y; // V
    
    // Apply MPEG/TV range expansion (limited range to full range)
    // Y: [16/255, 235/255] -> [0, 1]
    // UV: [16/255, 240/255] -> [-0.5, 0.5]
    yuv.x = (yuv.x - 16.0/255.0) * 255.0/219.0;
    yuv.yz = (yuv.yz - 128.0/255.0) * 255.0/224.0;
    
    // BT.709 YUV to RGB conversion matrix
    float3 rgb;
    rgb.r = yuv.x + 1.5748 * yuv.z;
    rgb.g = yuv.x - 0.1873 * yuv.y - 0.4681 * yuv.z;
    rgb.b = yuv.x + 1.8556 * yuv.y;
    
    // Clamp to [0, 1] range for model input
    rgb = saturate(rgb);
    
    // Output in RGBA format for model compatibility
    OutputTexture[id.xy] = float4(rgb, 1.0);
}
)";
}

ConversionConstants generate_conversion_constants_bt709() {
    ConversionConstants constants = {};
    
    // Set basic info for yuv420p + BT709
    constants.input_format = static_cast<int>(DXGI_FORMAT_NV12);
    constants.color_space = static_cast<int>(AVCOL_SPC_BT709);
    constants.bit_depth = 8;
    constants.is_hdr = 0;
    
    // BT.709 YUV to RGB matrix (hardcoded for performance)
    float matrix[16] = {
        1.0f,    0.0f,      1.5748f,   0.0f,
        1.0f,    -0.1873f,  -0.4681f,  0.0f,
        1.0f,    1.8556f,   0.0f,      0.0f,
        0.0f,    0.0f,      0.0f,      1.0f
    };
    
    memcpy(constants.matrix, matrix, sizeof(matrix));
    
    return constants;
}

bool create_color_conversion_shader(ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    // Use simplified shader source for yuv420p + BT709
    std::string shader_source = generate_shader_source_bt709_yuv420p();
    
    ID3DBlob* shader_blob = nullptr;
    ID3DBlob* error_blob = nullptr;
    
    HRESULT hr = D3DCompile(
        shader_source.c_str(),
        shader_source.length(),
        nullptr,
        nullptr,
        nullptr,
        "CSMain",
        "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS,
        0,
        &shader_blob,
        &error_blob
    );
    
    if (FAILED(hr)) {
        if (error_blob) {
            std::cerr << "Shader compilation error: " << (char*)error_blob->GetBufferPointer() << "\n";
            error_blob->Release();
        }
        return false;
    }
    
    hr = d3d11_device->CreateComputeShader(
        shader_blob->GetBufferPointer(),
        shader_blob->GetBufferSize(),
        nullptr,
        &color_state.color_conversion_shader
    );
    
    shader_blob->Release();
    
    if (FAILED(hr)) {
        std::cerr << "Failed to create compute shader: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    // Create constants buffer
    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.ByteWidth = sizeof(ConversionConstants);
    buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    hr = d3d11_device->CreateBuffer(&buffer_desc, nullptr, &color_state.conversion_constants_buffer);
    if (FAILED(hr)) {
        std::cerr << "Failed to create constants buffer: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    std::cout << "✓ BT709 yuv420p color conversion shader created successfully\n";
    return true;
}

bool create_output_texture_and_uav(UINT width, UINT height, ID3D11Device* d3d11_device,
                                   ID3D11Texture2D** out_texture, ID3D11UnorderedAccessView** out_uav) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT; // Float32 RGBA for model input
    desc.SampleDesc.Count = 1;
    desc.SampleDesc.Quality = 0;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED; // For CUDA interop in the next thread
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, out_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    // Create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = desc.Format;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    hr = d3d11_device->CreateUnorderedAccessView(*out_texture, &uav_desc, out_uav);
    if (FAILED(hr)) {
        (*out_texture)->Release();
        std::cerr << "Failed to create output UAV: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    return true;
}

bool create_input_srv_once(ID3D11Texture2D* input_texture, ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    // Only create SRV if not already created
    if (color_state.input_srv_y) {
        return true; // Already created
    }

    D3D11_TEXTURE2D_DESC desc;
    input_texture->GetDesc(&desc);
    
    // Validate that this is NV12 format (yuv420p)
    if (desc.Format != DXGI_FORMAT_NV12) {
        throw std::runtime_error("Expected NV12 format for yuv420p input");
    }
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    // Create Y plane view (luminance)
    srv_desc.Format = DXGI_FORMAT_R8_UNORM;
    HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_y);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV for Y plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }

    // Create UV plane view (chrominance)
    srv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
    hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_uv);
    if (FAILED(hr)) {
        std::cerr << "Failed to create input SRV for UV plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    return true;
}

bool process_d3d11_frame(AVFrame* d3d11_frame, const ColorSpaceInfo& color_info,
                                  ColorConversionState& color_state, ID3D11Device* d3d11_device,
                                  ID3D11DeviceContext* d3d11_context, ColorConvertedFrameQueue& output_queue) {
    
    // Validate input format first - throw error if not supported
    validate_input_format(color_info);
    
    ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)d3d11_frame->data[0];
    int texture_index = (int)(intptr_t)d3d11_frame->data[1];
    
    D3D11_TEXTURE2D_DESC texture_desc;
    d3d11_texture->GetDesc(&texture_desc);
    
    if (!color_state.intermediate_texture) {
        if (!create_intermediate_texture(texture_desc.Width, texture_desc.Height, texture_desc.Format, color_state, d3d11_device)) {
            return false;
        }
    }
    
    // Create shader if not already created (simplified for BT709 only)
    if (!color_state.color_conversion_shader) {
        if (!create_color_conversion_shader(color_state, d3d11_device)) {
            return false;
        }
    }
    
    // Create a new output texture and UAV for each frame
    ID3D11Texture2D* output_texture = nullptr;
    ID3D11UnorderedAccessView* output_uav = nullptr;
    if (!create_output_texture_and_uav(texture_desc.Width, texture_desc.Height, d3d11_device, &output_texture, &output_uav)) {
        return false;
    }
    
    // Copy from original texture to intermediate texture
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, 1);
    UINT dst_subresource = D3D11CalcSubresource(0, 0, 1);
    
    d3d11_context->CopySubresourceRegion(
        color_state.intermediate_texture, dst_subresource, 0, 0, 0,
        d3d11_texture, src_subresource, nullptr);
    
    std::cout << "  ✓ Texture copied to intermediate buffer\n";
    
    // === DirectX Shader Color Conversion (BT709 yuv420p only) ===
    
    // Create input SRV for shader (only once)
    if (!create_input_srv_once(color_state.intermediate_texture, color_state, d3d11_device)) {
        output_texture->Release();
        output_uav->Release();
        return false;
    }
    
    // Update constants buffer with BT709 constants
    ConversionConstants constants = generate_conversion_constants_bt709();
    
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    HRESULT hr = d3d11_context->Map(color_state.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    if (SUCCEEDED(hr)) {
        memcpy(mapped_resource.pData, &constants, sizeof(constants));
        d3d11_context->Unmap(color_state.conversion_constants_buffer, 0);
    }
    
    // Set shader resources (NV12 requires both Y and UV textures)
    d3d11_context->CSSetShader(color_state.color_conversion_shader, nullptr, 0);
    ID3D11ShaderResourceView* srvs[] = { color_state.input_srv_y, color_state.input_srv_uv };
    d3d11_context->CSSetShaderResources(0, 2, srvs);
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &output_uav, nullptr);
    d3d11_context->CSSetConstantBuffers(0, 1, &color_state.conversion_constants_buffer);
    
    // Dispatch shader
    UINT dispatch_x = (texture_desc.Width + 7) / 8;
    UINT dispatch_y = (texture_desc.Height + 7) / 8;
    d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);
    
    // Unbind resources
    ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
    ID3D11UnorderedAccessView* null_uav_ptr = nullptr;
    d3d11_context->CSSetShaderResources(0, 2, null_srvs);
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav_ptr, nullptr);
    
    d3d11_context->Flush();
    
    std::cout << "  ✓ BT709 yuv420p color conversion completed successfully\n";
    
    // === Add converted frame to output queue ===
    ColorConvertedFrame converted_frame(output_texture, texture_desc.Width, texture_desc.Height);
    output_queue.push(std::move(converted_frame));

    // Release local handles, the object in the queue now owns the reference
    output_texture->Release();
    output_uav->Release();
    
    std::cout << "  ✓ Converted frame with D3D11 texture added to output queue\n";
    
    return true;
}

} // anonymous namespace

void start_convert_color_thread(
    DecodedFrameQueue& input_frame_queue,
    ColorConvertedFrameQueue& output_frame_queue,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context) {
    
    std::cout << "=== Process Thread Started (yuv420p + BT709 only) ===\n";
    
    // Create ColorConversionState inside the thread
    ColorConversionState color_state;
    
    int processed_count = 0;
    
    while (true) {
        DecodedFrame decoded_frame = input_frame_queue.pop();
        
        // Check for end signal
        if (decoded_frame.is_end_signal) {
            std::cout << "=== Process Thread Received End Signal ===\n";
            // Send end signal to output queue
            output_frame_queue.push(ColorConvertedFrame::end_signal());
            break;
        }
        
        if (decoded_frame.frame) {
            processed_count++;
            std::cout << ">>> Processing frame " << processed_count << " (yuv420p + BT709)\n";
            
            try {
                // Use color info from the frame instead of the parameter
                process_d3d11_frame(decoded_frame.frame, decoded_frame.color_info, color_state, 
                                            d3d11_device, d3d11_context, output_frame_queue);
            } catch (const std::exception& e) {
                std::cerr << "Frame processing failed: " << e.what() << "\n";
                av_frame_free(&decoded_frame.frame);
                output_frame_queue.push(ColorConvertedFrame::end_signal());
                break;
            }
            
            // Clean up the frame
            av_frame_free(&decoded_frame.frame);
            
            std::cout << ">>> Frame " << processed_count << " processing completed\n";
        }
    }
    
    // Clean up ColorConversionState before thread exits
    color_state.cleanup();
    color_state.process_finished_ = true;
    std::cout << "=== Process Thread Finished ===\n";
    std::cout << "Total frames processed: " << processed_count << "\n";
} 