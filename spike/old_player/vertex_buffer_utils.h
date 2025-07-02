#pragma once

#include <d3d11.h>

struct Vertex {
    float position[2];
    float texCoord[2];
};

// 创建用于全屏渲染的顶点缓冲区
bool createVertexBuffer(ID3D11Device* device, ID3D11Buffer** vertexBuffer); 