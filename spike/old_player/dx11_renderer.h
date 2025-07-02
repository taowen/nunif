#pragma once

#include <windows.h>
#include <d3d11.h>

extern "C" {
#include <libavformat/avformat.h>
}

// 函数声明 - 移除handle参数，内部管理状态
bool createDX11Renderer(HWND hwnd, ID3D11Device* externalDevice = nullptr, ID3D11DeviceContext* externalContext = nullptr);
void destroyDX11Renderer();
void createVideoTexture(int width, int height);
void updateVideoTexture(AVFrame* frame);
void renderFrame();

// 新增：支持硬件解码帧的更新函数
void updateVideoTextureFromHardwareFrame(AVFrame* frame); 