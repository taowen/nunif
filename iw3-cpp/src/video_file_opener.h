#pragma once

#include "main.h"
#include <string>

// Function to open video file, setup basic stream info, and setup decoder
bool open_video_file(const std::string& filename, DecoderState& decoder_state);

// Helper function to convert FFmpeg error to string
std::string av_err_to_string(int errnum); 