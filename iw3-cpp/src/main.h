#pragma once

#include <d3d11.h>

extern "C" {
#include <libavutil/pixfmt.h>
}

struct ColorSpaceInfo {
    AVColorSpace color_space = AVCOL_SPC_UNSPECIFIED;
    AVColorPrimaries color_primaries = AVCOL_PRI_UNSPECIFIED;
    AVColorTransferCharacteristic color_trc = AVCOL_TRC_UNSPECIFIED;
    AVColorRange color_range = AVCOL_RANGE_UNSPECIFIED;
    int bit_depth = 8;
    bool is_hdr = false;
    DXGI_FORMAT dxgi_format = DXGI_FORMAT_UNKNOWN;
}; 