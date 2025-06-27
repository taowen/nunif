#include "main.h"
// Add CUDA headers for interoperability
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <iostream>
#include <algorithm>  // for std::min, std::max
#include <iomanip>    // for std::setw, std::setprecision, std::fixed

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

void* convert_color(const FFMepgContext* ctx, AVFrame* frame) {
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return nullptr;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return nullptr;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return nullptr;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return nullptr;
    }

    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

}