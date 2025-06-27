#include "main.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct ConversionConstants {
    float matrix[16];
    float luma_coeffs[4];
    float chroma_coeffs[4];
    float offset[4];
    int input_format;
    int color_space;
    int bit_depth;
    int is_hdr;
};

struct ColorConversionState {
    ID3D11ComputeShader* color_conversion_shader = nullptr;
    ID3D11Buffer* conversion_constants_buffer = nullptr;
    ID3D11Texture2D* intermediate_texture = nullptr;
    ID3D11ShaderResourceView* input_srv_y = nullptr;
    ID3D11ShaderResourceView* input_srv_uv = nullptr;

    ~ColorConversionState() {
        cleanup();
    }

    void cleanup() {
        if (color_conversion_shader) { color_conversion_shader->Release(); color_conversion_shader = nullptr; }
        if (conversion_constants_buffer) { conversion_constants_buffer->Release(); conversion_constants_buffer = nullptr; }
        if (intermediate_texture) { intermediate_texture->Release(); intermediate_texture = nullptr; }
        if (input_srv_y) { input_srv_y->Release(); input_srv_y = nullptr; }
        if (input_srv_uv) { input_srv_uv->Release(); input_srv_uv = nullptr; }
    }
};

std::string generate_shader_source_bt709_yuv420p() {
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
    constants.input_format = static_cast<int>(DXGI_FORMAT_NV12);
    constants.color_space = static_cast<int>(AVCOL_SPC_BT709);
    constants.bit_depth = 8;
    constants.is_hdr = 0;
    return constants;
}

void create_color_conversion_shader(ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    std::string shader_source = generate_shader_source_bt709_yuv420p();
    
    ID3DBlob* shader_blob = nullptr;
    ID3DBlob* error_blob = nullptr;
    
    HRESULT hr = D3DCompile(
        shader_source.c_str(), shader_source.length(), nullptr, nullptr, nullptr,
        "CSMain", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0, &shader_blob, &error_blob);
    
    if (FAILED(hr)) {
        std::string error_msg = "Shader compilation error";
        if (error_blob) {
            error_msg += ": ";
            error_msg += (char*)error_blob->GetBufferPointer();
            error_blob->Release();
        }
        if (shader_blob) shader_blob->Release();
        throw std::runtime_error(error_msg);
    }
    if (error_blob) error_blob->Release();

    hr = d3d11_device->CreateComputeShader(
        shader_blob->GetBufferPointer(), shader_blob->GetBufferSize(), nullptr, &color_state.color_conversion_shader);
    shader_blob->Release();
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create compute shader");
    }

    D3D11_BUFFER_DESC buffer_desc = {};
    buffer_desc.ByteWidth = sizeof(ConversionConstants);
    buffer_desc.Usage = D3D11_USAGE_DYNAMIC;
    buffer_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    buffer_desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    
    hr = d3d11_device->CreateBuffer(&buffer_desc, nullptr, &color_state.conversion_constants_buffer);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create constants buffer");
    }
}

void create_input_srv(ID3D11Texture2D* input_texture, ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    D3D11_TEXTURE2D_DESC desc;
    input_texture->GetDesc(&desc);
    if (desc.Format != DXGI_FORMAT_NV12) {
        throw std::runtime_error("Expected NV12 format for input");
    }
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc = {};
    srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MostDetailedMip = 0;
    srv_desc.Texture2D.MipLevels = 1;
    
    srv_desc.Format = DXGI_FORMAT_R8_UNORM;
    HRESULT hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_y);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create SRV for Y plane");
    }

    srv_desc.Format = DXGI_FORMAT_R8G8_UNORM;
    hr = d3d11_device->CreateShaderResourceView(input_texture, &srv_desc, &color_state.input_srv_uv);
    if (FAILED(hr)) {
        if(color_state.input_srv_y) { color_state.input_srv_y->Release(); color_state.input_srv_y = nullptr; }
        throw std::runtime_error("Failed to create SRV for UV plane");
    }
}

void create_intermediate_texture(UINT width, UINT height, DXGI_FORMAT format,
                                 ColorConversionState& color_state, ID3D11Device* d3d11_device) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &color_state.intermediate_texture);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create intermediate texture");
    }
}

ID3D11Texture2D* create_output_texture_and_uav(UINT width, UINT height, ID3D11Device* d3d11_device,
                                                ID3D11UnorderedAccessView** out_uav) {
    ID3D11Texture2D* out_texture = nullptr;
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
    
    HRESULT hr = d3d11_device->CreateTexture2D(&desc, nullptr, &out_texture);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create output texture");
    }
    
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav_desc = {};
    uav_desc.Format = desc.Format;
    uav_desc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
    
    hr = d3d11_device->CreateUnorderedAccessView(out_texture, &uav_desc, out_uav);
    if (FAILED(hr)) {
        out_texture->Release();
        throw std::runtime_error("Failed to create output UAV");
    }
    
    return out_texture;
}

} // namespace

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
        std::cerr << "Error: Expected NV12 format, got DXGI format " << input_desc.Format << std::endl;
        return nullptr;
    }
    
    if (frame->colorspace != AVCOL_SPC_BT709) {
        std::cerr << "Error: Expected BT.709 colorspace, got " << frame->colorspace << std::endl;
        return nullptr;
    }

    static ColorConversionState color_state;
    ID3D11Device* d3d11_device = ctx->d3d_device;
    ID3D11DeviceContext* d3d11_context = ctx->d3d_context;

    try {
        if (!color_state.color_conversion_shader) {
            create_color_conversion_shader(color_state, d3d11_device);
        }

        if (!color_state.intermediate_texture) {
            create_intermediate_texture(input_desc.Width, input_desc.Height, input_desc.Format, color_state, d3d11_device);
            create_input_srv(color_state.intermediate_texture, color_state, d3d11_device);
        }
        
        ID3D11UnorderedAccessView* output_uav = nullptr;
        ID3D11Texture2D* output_texture = create_output_texture_and_uav(frame->width, frame->height, d3d11_device, &output_uav);

        UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
        d3d11_context->CopySubresourceRegion(color_state.intermediate_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);

        ConversionConstants constants = generate_conversion_constants_bt709();
        D3D11_MAPPED_SUBRESOURCE mapped_resource;
        HRESULT hr = d3d11_context->Map(color_state.conversion_constants_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped_resource);
        if (SUCCEEDED(hr)) {
            memcpy(mapped_resource.pData, &constants, sizeof(constants));
            d3d11_context->Unmap(color_state.conversion_constants_buffer, 0);
        }

        d3d11_context->CSSetShader(color_state.color_conversion_shader, nullptr, 0);
        ID3D11ShaderResourceView* srvs[] = { color_state.input_srv_y, color_state.input_srv_uv };
        d3d11_context->CSSetShaderResources(0, 2, srvs);
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &output_uav, nullptr);
        d3d11_context->CSSetConstantBuffers(0, 1, &color_state.conversion_constants_buffer);
        
        UINT dispatch_x = (frame->width + 7) / 8;
        UINT dispatch_y = (frame->height + 7) / 8;
        d3d11_context->Dispatch(dispatch_x, dispatch_y, 1);

        ID3D11ShaderResourceView* null_srvs[] = { nullptr, nullptr };
        ID3D11UnorderedAccessView* null_uav = nullptr;
        d3d11_context->CSSetShaderResources(0, 2, null_srvs);
        d3d11_context->CSSetUnorderedAccessViews(0, 1, &null_uav, nullptr);
        d3d11_context->CSSetShader(nullptr, nullptr, 0);

        output_uav->Release();
        return output_texture;

    } catch (const std::exception& e) {
        std::cerr << "Error during color conversion: " << e.what() << std::endl;
        color_state.cleanup();
        return nullptr;
    }
}