#pragma once

#include <windows.h>

// 播放引擎句柄
typedef void* PlayerEngineHandle;

// 播放引擎接口
PlayerEngineHandle createPlayerEngine(const char* filename, HWND hwnd);

// 获取媒体信息
bool getMediaDimensions(PlayerEngineHandle handle, int* width, int* height);

// 全局播放引擎管理接口
void destroyCurrentPlayerEngine();

// 便利接口 - 操作当前播放引擎
bool startCurrentPlayback();
void stopCurrentPlayback();
bool isCurrentPlaying(); 