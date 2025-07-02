#include "vertex_buffer_utils.h"

bool createVertexBuffer(ID3D11Device* device, ID3D11Buffer** vertexBuffer) {
    Vertex vertices[] = {
        {{-1.0f, -1.0f}, {0.0f, 1.0f}},  // 左下
        {{-1.0f,  1.0f}, {0.0f, 0.0f}},  // 左上
        {{ 1.0f, -1.0f}, {1.0f, 1.0f}},  // 右下
        {{ 1.0f,  1.0f}, {1.0f, 0.0f}}   // 右上
    };
    
    D3D11_BUFFER_DESC bufferDesc = {};
    bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth = sizeof(vertices);
    bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    
    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = vertices;
    
    return SUCCEEDED(device->CreateBuffer(&bufferDesc, &initData, vertexBuffer));
} 