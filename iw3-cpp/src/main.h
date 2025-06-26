#pragma once

#include <d3d11.h>

extern "C" {
#include <libavutil/pixfmt.h>
#include <libavcodec/avcodec.h>
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

// Frame data structure for queue communication
struct DecodedFrame {
    AVFrame* frame;
    bool is_end_signal;
    
    DecodedFrame() : frame(nullptr), is_end_signal(false) {}
    DecodedFrame(AVFrame* f) : frame(f), is_end_signal(false) {}
    static DecodedFrame end_signal() {
        DecodedFrame data;
        data.is_end_signal = true;
        return data;
    }
}; 