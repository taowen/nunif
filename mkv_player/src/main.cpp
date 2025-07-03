#include "mkv_stream_reader.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mkv_file>" << std::endl;
        return -1;
    }
    
    MKVStreamReader reader;
    
    std::cout << "Opening file: " << argv[1] << std::endl;
    if (!reader.open(argv[1])) {
        std::cout << "Failed to open file!" << std::endl;
        return -1;
    }
    
    auto info = reader.getStreamInfo();
    std::cout << "\n=== Stream Information ===" << std::endl;
    std::cout << "Video stream index: " << info.video_stream_index << std::endl;
    std::cout << "Audio stream index: " << info.audio_stream_index << std::endl;
    std::cout << "Video codec: " << info.video_codec << std::endl;
    std::cout << "Audio codec: " << info.audio_codec << std::endl;
    std::cout << "Resolution: " << info.width << "x" << info.height << std::endl;
    std::cout << "Duration: " << info.duration << " seconds" << std::endl;
    std::cout << "FPS: " << info.fps << std::endl;
    std::cout << "Audio sample rate: " << info.audio_sample_rate << " Hz" << std::endl;
    std::cout << "Audio channels: " << info.audio_channels << std::endl;
    
    // 测试包读取
    std::cout << "\n=== Packet Reading Test ===" << std::endl;
    AVPacket* packet = av_packet_alloc();
    int video_packets = 0;
    int audio_packets = 0;
    int total_packets = 0;
    
    while (reader.readNextPacket(packet) && total_packets < 100) {
        total_packets++;
        
        if (reader.isVideoPacket(packet)) {
            video_packets++;
        }
        
        if (reader.isAudioPacket(packet)) {
            audio_packets++;
        }
        
        av_packet_unref(packet);
    }
    
    std::cout << "Read " << total_packets << " packets total" << std::endl;
    std::cout << "Video packets: " << video_packets << std::endl;
    std::cout << "Audio packets: " << audio_packets << std::endl;
    std::cout << "EOF reached: " << (reader.isEOF() ? "Yes" : "No") << std::endl;
    
    av_packet_free(&packet);
    reader.close();
    
    std::cout << "\nMKVStreamReader test completed successfully!" << std::endl;
    return 0;
}