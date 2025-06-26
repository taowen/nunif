#include "convert_color_thread.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <d3dcompiler.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

extern void checkCudaErrors(cudaError_t result);

namespace {

bool create_cuda_interop_texture(UINT width, UINT height, DXGI_FORMAT format, 
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
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED; // For CUDA interop
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_state.cuda_interop_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create CUDA interop texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &color_state.cuda_resource, color_state.cuda_interop_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status != cudaSuccess) {
        std::cerr << "Failed to register interop texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
        return false;
    }
    
    return true;
}

std::string generate_shader_source(const ColorSpaceInfo& color_info) {
    std::ostringstream shader;
    
    shader << R"(
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
    
    float3 yuv;
    
)";

    // Handle different input formats
    switch (color_info.dxgi_format) {
        case DXGI_FORMAT_NV12:
            shader << R"(
    // NV12 format handling
    yuv.x = LumaTexture.Load(int3(id.xy, 0)); // Y
    float2 uv = ChromaTexture.Load(int3(id.xy / 2, 0)); // UV
    yuv.y = uv.x; // U
    yuv.z = uv.y; // V
)";
            break;
        case DXGI_FORMAT_P010:
        case DXGI_FORMAT_P016:
            shader << R"(
    // P010/P016 format handling (10/16-bit)
    yuv.x = yuv_sample.r;
    yuv.y = yuv_sample.g;
    yuv.z = yuv_sample.b;
    
    // Scale from 10/16-bit to full range
    float scale_factor = )";
            shader << (color_info.bit_depth == 10 ? "1023.0" : "65535.0");
            shader << R"( / 255.0;
    yuv *= scale_factor;
)";
            break;
        default:
            shader << R"(
    // Default YUV handling
    yuv.x = yuv_sample.r;  // Y
    yuv.y = yuv_sample.g;  // U
    yuv.z = yuv_sample.b;  // V
)";
            break;
    }
    
    // Color space conversion
    shader << R"(
    
    // Apply color range expansion
    if (ColorSpace != 0) { // Not full range
        yuv.x = (yuv.x - 16.0/255.0) * 255.0/219.0;
        yuv.yz = (yuv.yz - 128.0/255.0) * 255.0/224.0;
    }
    
    // YUV to RGB conversion using color matrix
    float3 rgb = mul(ColorMatrix, float4(yuv, 1.0)).rgb;
    
)";

    // HDR tone mapping if needed
    if (color_info.is_hdr) {
        shader << R"(
    // HDR tone mapping
    if (IsHDR != 0) {
        // Simple tone mapping - you may want to implement more sophisticated methods
        rgb = rgb / (rgb + 1.0); // Reinhard tone mapping
    }
)";
    }
    
    shader << R"(
    
    // Clamp to [0, 1] range for model input
    rgb = saturate(rgb);
    
    // Output in RGBA format for model compatibility
    OutputTexture[id.xy] = float4(rgb, 1.0);
}
)";
    
    return shader.str();
}

ConversionConstants generate_conversion_constants(const ColorSpaceInfo& color_info) {
    ConversionConstants constants = {};
    
    // Set basic info
    constants.input_format = static_cast<int>(color_info.dxgi_format);
    constants.color_space = static_cast<int>(color_info.color_space);
    constants.bit_depth = color_info.bit_depth;
    constants.is_hdr = color_info.is_hdr ? 1 : 0;
    
    // Generate color conversion matrix based on color space
    float matrix[16] = {0};
    
    switch (color_info.color_space) {
        case AVCOL_SPC_BT709:
            // BT.709 YUV to RGB matrix
            matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.5748f;   matrix[3] = 0.0f;
            matrix[4] = 1.0f;    matrix[5] = -0.1873f;  matrix[6] = -0.4681f;  matrix[7] = 0.0f;
            matrix[8] = 1.0f;    matrix[9] = 1.8556f;   matrix[10] = 0.0f;     matrix[11] = 0.0f;
            matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
        case AVCOL_SPC_BT2020_NCL:
        case AVCOL_SPC_BT2020_CL:
            // BT.2020 YUV to RGB matrix
            matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.7166f;   matrix[3] = 0.0f;
            matrix[4] = 1.0f;    matrix[5] = -0.1916f;  matrix[6] = -0.6657f;  matrix[7] = 0.0f;
            matrix[8] = 1.0f;    matrix[9] = 2.1415f;   matrix[10] = 0.0f;     matrix[11] = 0.0f;
            matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
        default:
            // Default to BT.601
            matrix[0] = 1.0f;    matrix[1] = 0.0f;      matrix[2] = 1.402f;    matrix[3] = 0.0f;
            matrix[4] = 1.0f;    matrix[5] = -0.344f;   matrix[6] = -0.714f;   matrix[7] = 0.0f;
            matrix[8] = 1.0f;    matrix[9] = 1.772f;    matrix[10] = 0.0f;     matrix[11] = 0.0f;
            matrix[12] = 0.0f;   matrix[13] = 0.0f;     matrix[14] = 0.0f;     matrix[15] = 1.0f;
            break;
    }
    
    memcpy(constants.matrix, matrix, sizeof(matrix));
    
    return constants;
}

bool create_color_conversion_shader(const ColorSpaceInfo& color_info, 
                                   ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    // Generate shader source based on color space info
    std::string shader_source = generate_shader_source(color_info);
    
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
    
    std::cout << "✓ Color conversion shader created successfully\n";
    return true;
}

bool create_output_texture(UINT width, UINT height, ColorConversionState& color_state, ID3D11Device* d3d11_device) {
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
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_state.output_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create output texture: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
        return false;
    }
    
    // Create UAV
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = desc.Format;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    uav_desc.Texture2D.MipSlice = 0;
    
    hr = d3d11_device->CreateUnorderedAccessView(color_state.output_texture, &uav_desc, &color_state.output_uav);
    if (FAILED(hr)) {
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
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    if (desc.Format == DXGI_FORMAT_NV12) {
        // Create Y plane view
        srv_desc.Format = DXGI_FORMAT_R8_UNORM;
        HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_y);
        if (FAILED(hr)) {
            std::cerr << "Failed to create input SRV for Y plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }

        // Create UV plane view
        srv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
        hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_uv);
        if (FAILED(hr)) {
            std::cerr << "Failed to create input SRV for UV plane: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
    } else {
        srv_desc.Format = desc.Format;
        HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_y);
        if (FAILED(hr)) {
            std::cerr << "Failed to create input SRV: 0x" << std::hex << static_cast<unsigned int>(hr) << std::dec << "\n";
            return false;
        }
    }
    
    return true;
}

bool process_d3d11_frame_with_cuda(AVFrame* d3d11_frame, const ColorSpaceInfo& color_info,
                                  ColorConversionState& color_state, ID3D11Device* d3d11_device,
                                  ID3D11DeviceContext* d3d11_context, cudaStream_t cuda_stream) {
    ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)d3d11_frame->data[0];
    int texture_index = (int)(intptr_t)d3d11_frame->data[1];
    
    D3D11_TEXTURE2D_DESC texture_desc;
    d3d11_texture->GetDesc(&texture_desc);
    
    if (!color_state.cuda_interop_texture) {
        if (!create_cuda_interop_texture(texture_desc.Width, texture_desc.Height, texture_desc.Format, color_state, d3d11_device)) {
            return false;
        }
    }
    
    // Create shader and output resources if not already created
    if (!color_state.color_conversion_shader) {
        if (!create_color_conversion_shader(color_info, color_state, d3d11_device)) {
            return false;
        }
    }
    
    if (!color_state.output_texture) {
        if (!create_output_texture(texture_desc.Width, texture_desc.Height, color_state, d3d11_device)) {
            return false;
        }
    }
    
    // Copy from original texture to intermediate texture
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, 1);
    UINT dst_subresource = D3D11CalcSubresource(0, 0, 1);
    
    d3d11_context->CopySubresourceRegion(
        color_state.cuda_interop_texture, dst_subresource, 0, 0, 0,
        d3d11_texture, src_subresource, nullptr);
    
    std::cout << "  ✓ Texture copied to intermediate buffer\n";
    
    // === DirectX Shader Color Conversion ===
    
    // Create input SRV for shader (only once)
    if (!create_input_srv_once(color_state.cuda_interop_texture, color_state, d3d11_device)) {
        return false;
    }
    
    // Update constants buffer
    ConversionConstants constants = generate_conversion_constants(color_info);
    
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    HRESULT hr = d3d11_context->Map(color_state.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
    if (SUCCEEDED(hr)) {
        memcpy(mapped_resource.pData, &constants, sizeof(constants));
        d3d11_context->Unmap(color_state.conversion_constants_buffer, 0);
    }
    
    // Set shader resources
    d3d11_context->CSSetShader(color_state.color_conversion_shader, nullptr, 0);
    if (color_state.input_srv_uv) {
        ID3D11ShaderResourceView* srvs[] = { color_state.input_srv_y, color_state.input_srv_uv };
        d3d11_context->CSSetShaderResources(0, 2, srvs);
    } else {
        d3d11_context->CSSetShaderResources(0, 1, &color_state.input_srv_y);
    }
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &color_state.output_uav, nullptr);
    d3d11_context->CSSetConstantBuffers(0, 1, &color_state.conversion_constants_buffer);
    
    // Dispatch shader
    UINT dispatch_x = (texture_desc.Width + 7) / 8;
    UINT dispatch_y = (texture_desc.Height + 7) / 8;
    d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);
    
    // Unbind resources
    ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
    ID3D11UnorderedAccessView* null_uav = nullptr;
    d3d11_context->CSSetShaderResources(0, 2, null_srvs);
    d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
    
    d3d11_context->Flush();
    
    std::cout << "  ✓ Color conversion shader executed successfully\n";
    
    // === CUDA Processing on Converted Data ===
    
    // Now register the output texture with CUDA for further processing
    cudaGraphicsResource* output_cuda_resource = nullptr;
    cudaError_t cuda_status = cudaGraphicsD3D11RegisterResource(
        &output_cuda_resource, color_state.output_texture, cudaGraphicsRegisterFlagsNone);
    
    if (cuda_status == cudaSuccess) {
        checkCudaErrors(cudaGraphicsMapResources(1, &output_cuda_resource, cuda_stream));
        
        try {
            cudaArray_t cuda_array;
            checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuda_array, output_cuda_resource, 0, 0));
            
            std::cout << "  ✓ Converted texture mapped to CUDA successfully\n";
            
            // Here you can process the converted data with CUDA kernels
            
            checkCudaErrors(cudaStreamSynchronize(cuda_stream));
            
        } catch (const std::exception& e) {
            std::cerr << "Error processing converted CUDA data: " << e.what() << "\n";
        }
        
        checkCudaErrors(cudaGraphicsUnmapResources(1, &output_cuda_resource, cuda_stream));
        cudaGraphicsUnregisterResource(output_cuda_resource);
        
        std::cout << "  ✓ Converted texture processing completed\n";
    } else {
        std::cerr << "Failed to register output texture with CUDA: " << cudaGetErrorString(cuda_status) << "\n";
    }
    
    return true;
}

} // anonymous namespace

void start_convert_color_thread(
    FrameQueue& frame_queue,
    ColorConversionState& color_state, 
    const ColorSpaceInfo& color_info,
    ID3D11Device* d3d11_device,
    ID3D11DeviceContext* d3d11_context,
    cudaStream_t cuda_stream) {
    
    std::cout << "=== Process Thread Started ===\n";
    
    int processed_count = 0;
    
    while (true) {
        DecodedFrame decoded_frame = frame_queue.pop();
        
        // Check for end signal
        if (decoded_frame.is_end_signal) {
            std::cout << "=== Process Thread Received End Signal ===\n";
            break;
        }
        
        if (decoded_frame.frame) {
            processed_count++;
            std::cout << ">>> Processing frame " << processed_count << "\n";
            
            // Process the D3D11 frame
            process_d3d11_frame_with_cuda(decoded_frame.frame, color_info, color_state, 
                                        d3d11_device, d3d11_context, cuda_stream);
            
            // Clean up the frame
            av_frame_free(&decoded_frame.frame);
            
            std::cout << ">>> Frame " << processed_count << " processing completed\n";
        }
    }
    
    color_state.process_finished_ = true;
    std::cout << "=== Process Thread Finished ===\n";
    std::cout << "Total frames processed: " << processed_count << "\n";
} 