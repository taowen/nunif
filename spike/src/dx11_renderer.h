#pragma once

#include <windows.h>

extern "C" {
#include <libavformat/avformat.h>
}

// 函数声明 - 移除handle参数，内部管理状态
bool createDX11Renderer(HWND hwnd);
void destroyDX11Renderer();
void createVideoTexture(int width, int height);
void updateVideoTexture(AVFrame* frame);
void renderFrame(); 