#include "decode_loop.h"
#include <thread>
#include <chrono>

DecodeLoop::DecodeLoop(DecodeState state,
                       VideoFrameCallback videoCallback,
                       AudioFrameCallback audioCallback,
                       QueueFullCheckCallback queueFullCallback)
    : m_state(state)
    , m_videoCallback(videoCallback)
    , m_audioCallback(audioCallback)
    , m_queueFullCallback(queueFullCallback)
{
}

void DecodeLoop::setContexts(AVFormatContext* formatContext,
                            AVCodecContext* videoCodecContext,
                            AVCodecContext* audioCodecContext)
{
    m_formatContext = formatContext;
    m_videoCodecContext = videoCodecContext;
    m_audioCodecContext = audioCodecContext;
}

void DecodeLoop::run() {
    if (!m_formatContext) {
        return;
    }
    
    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    while (!m_state.shouldStop->load() && av_read_frame(m_formatContext, packet) >= 0) {
        if (m_state.hasVideo && packet->stream_index == m_state.videoStreamIndex) {
            processVideoPacket(packet, frame);
        } else if (m_state.hasAudio && packet->stream_index == m_state.audioStreamIndex) {
            processAudioPacket(packet, frame);
        }
        
        av_packet_unref(packet);
        
        // 队列大小控制
        if (m_queueFullCallback && m_queueFullCallback()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    av_packet_free(&packet);
    av_frame_free(&frame);
}

void DecodeLoop::processVideoPacket(AVPacket* packet, AVFrame* frame) {
    if (!m_videoCodecContext || !m_videoCallback) {
        return;
    }
    
    if (avcodec_send_packet(m_videoCodecContext, packet) == 0) {
        while (avcodec_receive_frame(m_videoCodecContext, frame) == 0) {
            m_videoCallback(frame);
        }
    }
}

void DecodeLoop::processAudioPacket(AVPacket* packet, AVFrame* frame) {
    if (!m_audioCodecContext || !m_audioCallback) {
        return;
    }
    
    if (avcodec_send_packet(m_audioCodecContext, packet) == 0) {
        while (avcodec_receive_frame(m_audioCodecContext, frame) == 0) {
            m_audioCallback(frame);
        }
    }
} 