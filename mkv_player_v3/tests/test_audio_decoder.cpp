#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

static const fs::path TEST_FILE = fs::current_path() / "test_data" / "sample_hw.mkv";

static bool skipIfNoTestFile() {
    if (!fs::exists(TEST_FILE)) {
        WARN("Test file not found: " << TEST_FILE.string() << ". Skipping tests.");
        return true;
    }
    return false;
}

TEST_CASE("AudioDecoder Basic Operations", "[AudioDecoder]") {
    if (skipIfNoTestFile()) return;
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(TEST_FILE.string()));
    
    // 验证可以访问内部的 MKVStreamReader
    MKVStreamReader* stream_reader = decoder.getStreamReader();
    REQUIRE(stream_reader != nullptr);
    REQUIRE(stream_reader->isOpen());
    REQUIRE_FALSE(stream_reader->isEOF());
    
    // 测试关闭和重新打开
    decoder.close();
    REQUIRE_FALSE(stream_reader->isOpen());
    
    REQUIRE(decoder.open(TEST_FILE.string()));
    stream_reader = decoder.getStreamReader();
    REQUIRE(stream_reader != nullptr);
    REQUIRE(stream_reader->isOpen());
    REQUIRE_FALSE(stream_reader->isEOF());
}

TEST_CASE("AudioDecoder Frame Decoding", "[AudioDecoder]") {
    if (skipIfNoTestFile()) return;
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(TEST_FILE.string()));
    
    AudioDecoder::DecodedFrame frame;
    bool success = decoder.readNextFrame(frame);
    
    if (success) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE_FALSE(frame.is_eof);
        
        // 验证音频帧属性
        AVFrame* audio_frame = frame.frame;
        REQUIRE(audio_frame->nb_samples > 0);
        REQUIRE(audio_frame->sample_rate > 0);
        REQUIRE(audio_frame->ch_layout.nb_channels > 0);
        
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AudioDecoder Double Buffer Memory Reuse", "[AudioDecoder]") {
    if (skipIfNoTestFile()) return;
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(TEST_FILE.string()));
    
    std::vector<AVFrame*> frame_addresses;
    AudioDecoder::DecodedFrame frame;
    
    // 解码足够的帧来验证双缓冲模式
    for (int i = 0; i < 4 && decoder.readNextFrame(frame); i++) {
        if (frame.is_valid) {
            REQUIRE(frame.frame != nullptr);
            frame_addresses.push_back(frame.frame);
        }
    }
    
    // 验证双缓冲内存复用：frame[0]==frame[2], frame[1]==frame[3]
    if (frame_addresses.size() >= 3) {
        REQUIRE(frame_addresses[2] == frame_addresses[0]);
        std::cout << "✓ 双缓冲验证：frame[2] == frame[0] (" << frame_addresses[0] << ")\n";
    }
    if (frame_addresses.size() >= 4) {
        REQUIRE(frame_addresses[3] == frame_addresses[1]);
        std::cout << "✓ 双缓冲验证：frame[3] == frame[1] (" << frame_addresses[1] << ")\n";
    }
    
    if (frame_addresses.size() < 3) {
        WARN("Not enough frames to verify double buffering");
    }
}

TEST_CASE("AudioDecoder EOF Detection", "[AudioDecoder]") {
    if (skipIfNoTestFile()) return;
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(TEST_FILE.string()));
    
    AudioDecoder::DecodedFrame frame;
    int frame_count = 0;
    
    // 读取所有音频帧直到EOF
    while (decoder.readNextFrame(frame) && frame_count < 1000) {
        if (frame.is_valid) {
            frame_count++;
        }
        if (frame.is_eof) {
            break;
        }
    }
    
    if (frame_count > 0) {
        MKVStreamReader* stream_reader = decoder.getStreamReader();
        REQUIRE(stream_reader->isEOF());
    } else {
        WARN("No audio frames found in test file");
    }
}

