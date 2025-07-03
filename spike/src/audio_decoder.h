#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct AVFrame;

// Audio buffer structure
typedef struct {
    float* data;
    int sample_count;
    int channels;
    int sample_rate;
} AudioBuffer;

// Initialize audio decoder
// Returns 0 on success, negative on error
int audio_decoder_init();

// Transform AVFrame to audio buffer
// Returns pointer to AudioBuffer on success, NULL on error
AudioBuffer* audio_decoder_frame_to_buffer(struct AVFrame* frame);

// Release audio buffer
void audio_decoder_release_buffer(AudioBuffer* buffer);

// Get audio format information
void audio_decoder_get_format(int* sample_rate, int* channels);

// Cleanup and release all resources
void audio_decoder_cleanup();

#ifdef __cplusplus
}
#endif