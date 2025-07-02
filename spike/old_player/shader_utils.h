#pragma once

#include <d3d11.h>

// 着色器源码
extern const char* vertexShaderSource;
extern const char* pixelShaderSource;

// 函数声明
bool createShaders(ID3D11Device* device, 
                  ID3D11VertexShader** vertexShader,
                  ID3D11PixelShader** pixelShader,
                  ID3D11InputLayout** inputLayout); 