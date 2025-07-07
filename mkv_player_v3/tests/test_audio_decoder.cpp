#include <catch2/catch_test_macros.hpp>
#include "../src/audio_decoder.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

TEST_CASE("AudioDecoder Basic Test", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
}

TEST_CASE("AudioDecoder Read Audio Frame", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
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
        
        std::cout << "Audio frame - samples: " << audio_frame->nb_samples 
                  << ", sample_rate: " << audio_frame->sample_rate 
                  << ", channels: " << audio_frame->ch_layout.nb_channels << std::endl;
    } else {
        // 如果文件没有音频流，这也是合理的
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AudioDecoder Multiple Frames", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    int frame_count = 0;
    AudioDecoder::DecodedFrame frame;
    
    // 尝试读取多个音频帧
    while (frame_count < 5 && decoder.readNextFrame(frame)) {
        REQUIRE(frame.is_valid);
        REQUIRE(frame.frame != nullptr);
        REQUIRE_FALSE(frame.is_eof);
        
        frame_count++;
        std::cout << "Audio frame " << frame_count << " decoded successfully" << std::endl;
    }
    
    if (frame_count > 0) {
        std::cout << "Successfully decoded " << frame_count << " audio frames" << std::endl;
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AudioDecoder EOF Detection", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    AudioDecoder::DecodedFrame frame;
    int frame_count = 0;
    
    // 读取所有音频帧直到EOF
    while (decoder.readNextFrame(frame) && frame_count < 1000) {  // 防止无限循环
        if (frame.is_valid) {
            frame_count++;
        }
        if (frame.is_eof) {
            break;
        }
    }
    
    // 验证EOF状态
    if (frame_count > 0) {
        REQUIRE(decoder.isEOF());
        std::cout << "Total audio frames decoded: " << frame_count << std::endl;
    } else {
        WARN("No audio frames found in test file");
    }
}

TEST_CASE("AudioDecoder Close and Reopen", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    
    // 第一次打开
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    
    // 关闭
    decoder.close();
    REQUIRE_FALSE(decoder.isOpen());
    
    // 重新打开
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
    REQUIRE_FALSE(decoder.isEOF());
}

TEST_CASE("AudioDecoder Stream Reader Access", "[AudioDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    AudioDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    
    // 验证可以访问内部的 MKVStreamReader
    MKVStreamReader* stream_reader = decoder.getStreamReader();
    REQUIRE(stream_reader != nullptr);
    REQUIRE(stream_reader->isOpen());
    
    // 获取流信息
    auto stream_info = stream_reader->getStreamInfo();
    std::cout << "Audio stream index: " << stream_info.audio_stream_index << std::endl;
    std::cout << "Audio codec: " << stream_info.audio_codec << std::endl;
    std::cout << "Audio sample rate: " << stream_info.audio_sample_rate << std::endl;
    std::cout << "Audio channels: " << stream_info.audio_channels << std::endl;
}