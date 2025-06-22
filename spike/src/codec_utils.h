#pragma once
#include "context.h"

int calculateBitrate(int width, int height);
bool setupInputDecoder(VideoContext& ctx);
bool setupOutputEncoder(VideoContext& ctx);
bool initializeEncoder(VideoContext& ctx, AVFrame* first_frame);