#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ID3D11Texture2D;
struct AudioBuffer;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct HWND__;
typedef struct HWND__* HWND;

// Initialize audio/video sink with window handle
// Returns 0 on success, negative on error
int audio_video_sink_init(HWND window, struct ID3D11Device* device, struct ID3D11DeviceContext* context);

// Render one frame with video texture and audio buffer
// Both parameters can be NULL if only one type of data is available
// video_pts and audio_pts are presentation timestamps in milliseconds
// Returns 0 on success, negative on error
int audio_video_sink_render_frame(struct ID3D11Texture2D* video_texture, int64_t video_pts,
                                   AudioBuffer* audio_buffer, int64_t audio_pts);

// Get current audio playback time (master clock)
// Returns current playback position in milliseconds
int64_t audio_video_sink_get_audio_clock();

// Check if video frame should be displayed based on audio clock
// Returns: 1 = display now, 0 = too early (wait), -1 = too late (drop frame)
int audio_video_sink_should_display_video(int64_t video_pts);

// Set presentation parameters
void audio_video_sink_set_volume(float volume);
void audio_video_sink_set_fullscreen(int fullscreen);

// Synchronization control
void audio_video_sink_set_sync_threshold(int64_t threshold_ms); // default: 40ms
void audio_video_sink_reset_sync(); // reset sync state

// Get current playback status
int audio_video_sink_is_playing();
int64_t audio_video_sink_get_position(); // in milliseconds

// Cleanup and release all resources
void audio_video_sink_cleanup();

#ifdef __cplusplus
}
#endif