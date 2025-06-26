#include "decode_thread.h"
#include <iostream>

std::string av_err_to_string_decode(int errnum) {
    char errbuf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(errnum, errbuf, AV_ERROR_MAX_STRING_SIZE);
    return std::string(errbuf);
}

ColorSpaceInfo detect_color_info_decode(AVFrame* frame, AVCodecContext* codec_ctx) {
    ColorSpaceInfo info;
    
    // Get DXGI format from D3D11 texture
    if (frame->format == AV_PIX_FMT_D3D11) {
        ID3D11Texture2D* d3d11_texture = (ID3D11Texture2D*)frame->data[0];
        D3D11_TEXTURE2D_DESC texture_desc;
        d3d11_texture->GetDesc(&texture_desc);
        info.dxgi_format = texture_desc.Format;
        
        std::cout << "=== Texture Format Detection ===\n";
        std::cout << "DXGI Format: " << static_cast<int>(texture_desc.Format) << " (";
        switch (texture_desc.Format) {
            case DXGI_FORMAT_NV12: std::cout << "NV12"; break;
            case DXGI_FORMAT_P010: std::cout << "P010"; info.bit_depth = 10; break;
            case DXGI_FORMAT_P016: std::cout << "P016"; info.bit_depth = 16; break;
            case DXGI_FORMAT_YUY2: std::cout << "YUY2"; break;
            case DXGI_FORMAT_AYUV: std::cout << "AYUV"; break;
            default: std::cout << "Unknown"; break;
        }
        std::cout << ")\n";
    }
    
    // Get color space information from codec context and frame
    info.color_space = codec_ctx->colorspace != AVCOL_SPC_UNSPECIFIED ? 
                      codec_ctx->colorspace : frame->colorspace;
    info.color_primaries = codec_ctx->color_primaries != AVCOL_PRI_UNSPECIFIED ? 
                          codec_ctx->color_primaries : frame->color_primaries;
    info.color_trc = codec_ctx->color_trc != AVCOL_TRC_UNSPECIFIED ? 
                    codec_ctx->color_trc : frame->color_trc;
    info.color_range = codec_ctx->color_range != AVCOL_RANGE_UNSPECIFIED ? 
                      codec_ctx->color_range : frame->color_range;
    
    // Detect HDR content
    info.is_hdr = (info.color_trc == AVCOL_TRC_SMPTE2084 ||  // PQ
                   info.color_trc == AVCOL_TRC_ARIB_STD_B67 || // HLG
                   info.color_primaries == AVCOL_PRI_BT2020);
    
    // Detect bit depth from DXGI format if not already set
    if (info.bit_depth == 8) {
        switch (info.dxgi_format) {
            case DXGI_FORMAT_P010:
                info.bit_depth = 10;
                break;
            case DXGI_FORMAT_P016:
                info.bit_depth = 16;
                break;
            default:
                info.bit_depth = 8;
                break;
        }
    }
    
    std::cout << "=== Video Color Space Information (Detected Once) ===\n";
    std::cout << "Color Space: " << av_color_space_name(info.color_space) << " (" << static_cast<int>(info.color_space) << ")\n";
    std::cout << "Color Primaries: " << av_color_primaries_name(info.color_primaries) << " (" << static_cast<int>(info.color_primaries) << ")\n";
    std::cout << "Transfer Characteristics: " << av_color_transfer_name(info.color_trc) << " (" << static_cast<int>(info.color_trc) << ")\n";
    std::cout << "Color Range: " << av_color_range_name(info.color_range) << " (" << static_cast<int>(info.color_range) << ")\n";
    std::cout << "Bit Depth: " << info.bit_depth << "\n";
    std::cout << "Is HDR: " << (info.is_hdr ? "Yes" : "No") << "\n";
    std::cout << "====================================================\n";
    
    return info;
}

void start_decode_thread(DecoderState& decoder_state, DecodedFrameQueue& frame_queue) {
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    if (!packet || !frame) {
        std::cerr << "Failed to allocate packet or frame\n";
        frame_queue.push(DecodedFrame::end_signal());
        return;
    }
    
    int frame_count = 0;
    
    std::cout << "=== Decode Thread Started ===\n";
    
    // Read and decode frames
    while (av_read_frame(decoder_state.format_ctx, packet) >= 0) {
        if (packet->stream_index == decoder_state.video_stream_index) {
            int ret = avcodec_send_packet(decoder_state.codec_ctx, packet);
            if (ret < 0) {
                std::cerr << "Error sending packet: " << av_err_to_string_decode(ret) << "\n";
                break;
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(decoder_state.codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                } else if (ret < 0) {
                    std::cerr << "Error receiving frame: " << av_err_to_string_decode(ret) << "\n";
                    break;
                }
                
                frame_count++;
                
                if (frame->format == AV_PIX_FMT_D3D11) {
                    // Detect color space info only for the first frame
                    if (!decoder_state.color_info_detected) {
                        decoder_state.video_color_info = detect_color_info_decode(frame, decoder_state.codec_ctx);
                        decoder_state.color_info_detected = true;
                    }
                    
                    // Create a copy of the frame for the queue
                    AVFrame* frame_copy = av_frame_alloc();
                    if (av_frame_ref(frame_copy, frame) < 0) {
                        std::cerr << "Failed to reference frame\n";
                        av_frame_free(&frame_copy);
                        continue;
                    }
                    
                    // Push to queue (no color info needed, using shared one)
                    DecodedFrame decoded_frame(frame_copy);
                    frame_queue.push(std::move(decoded_frame));
                    
                    std::cout << "✓ Frame " << frame_count << " decoded and queued\n";
                    
                } else {
                    std::cerr << "Unexpected frame format: " << av_get_pix_fmt_name(static_cast<AVPixelFormat>(frame->format)) << " - hardware decoding may have failed\n";
                    continue;
                }
                
                if (frame_count >= 5) {
                    goto decode_cleanup;
                }
            }
        }
        av_packet_unref(packet);
    }
    
decode_cleanup:
    av_frame_free(&frame);
    av_packet_free(&packet);
    
    // Signal end of decoding
    frame_queue.push(DecodedFrame::end_signal());
    decoder_state.decode_finished_ = true;
    
    std::cout << "=== Decode Thread Finished ===\n";
    std::cout << "Total frames decoded: " << frame_count << "\n";
} 