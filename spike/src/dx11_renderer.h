#pragma once

#include <windows.h>

extern "C" {
#include <libavformat/avformat.h>
}

// 不透明句柄类型，隐藏内部实现
typedef void* DX11RendererHandle;

// 函数声明 - 简化接口，移除外部依赖
DX11RendererHandle createDX11Renderer(HWND hwnd);
void destroyDX11Renderer(DX11RendererHandle handle);
void createVideoTexture(DX11RendererHandle handle, int width, int height);
void updateVideoTexture(DX11RendererHandle handle, AVFrame* frame);  // 简化参数
void renderFrame(DX11RendererHandle handle); 