#pragma once

#include <windows.h>

// 播放引擎句柄
typedef void* PlayerEngineHandle;

// 播放引擎接口
PlayerEngineHandle createPlayerEngine(const char* filename, HWND hwnd);
void destroyPlayerEngine(PlayerEngineHandle handle);

bool startPlayback(PlayerEngineHandle handle);
void stopPlayback(PlayerEngineHandle handle);
bool isPlaying(PlayerEngineHandle handle);

// 获取媒体信息
bool getMediaDimensions(PlayerEngineHandle handle, int* width, int* height); 