#include "shader_utils.h"
#include <d3dcompiler.h>
#include <cstring>

#pragma comment(lib, "d3dcompiler.lib")

// 简单的顶点着色器
const char* vertexShaderSource = R"(
struct VSInput {
    float2 position : POSITION;
    float2 texCoord : TEXCOORD;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0f, 1.0f);
    output.texCoord = input.texCoord;
    return output;
}
)";

// 简单的像素着色器
const char* pixelShaderSource = R"(
Texture2D videoTexture : register(t0);
SamplerState textureSampler : register(s0);

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

float4 main(PSInput input) : SV_TARGET {
    return videoTexture.Sample(textureSampler, input.texCoord);
}
)";

bool createShaders(ID3D11Device* device, 
                  ID3D11VertexShader** vertexShader,
                  ID3D11PixelShader** pixelShader,
                  ID3D11InputLayout** inputLayout) {
    ID3DBlob* vsBlob = nullptr;
    ID3DBlob* psBlob = nullptr;
    ID3DBlob* errorBlob = nullptr;
    
    // 编译顶点着色器
    HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr,
        nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) errorBlob->Release();
        return false;
    }
    
    // 编译像素着色器
    hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr,
        nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, &errorBlob);
    if (FAILED(hr)) {
        if (errorBlob) errorBlob->Release();
        vsBlob->Release();
        return false;
    }
    
    // 创建着色器
    device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, vertexShader);
    device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, pixelShader);
    
    // 创建输入布局
    D3D11_INPUT_ELEMENT_DESC inputElements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
    };
    
    device->CreateInputLayout(inputElements, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), inputLayout);
    
    vsBlob->Release();
    psBlob->Release();
    return true;
} 