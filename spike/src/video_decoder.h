#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct AVFrame;
struct ID3D11Texture2D;
struct ID3D11Device;
struct ID3D11DeviceContext;

// Initialize video decoder with D3D11 device
// Returns 0 on success, negative on error
int video_decoder_init(struct ID3D11Device* device, struct ID3D11DeviceContext* context);

// Transform AVFrame to D3D11 renderable texture
// Returns pointer to ID3D11Texture2D on success, NULL on error
struct ID3D11Texture2D* video_decoder_frame_to_texture(struct AVFrame* frame);

// Release D3D11 renderable texture
void video_decoder_release_texture(struct ID3D11Texture2D* texture);

// Get video frame dimensions
void video_decoder_get_dimensions(int* width, int* height);

// Get video frame format information
int video_decoder_get_format();

// Cleanup and release all resources
void video_decoder_cleanup();

#ifdef __cplusplus
}
#endif