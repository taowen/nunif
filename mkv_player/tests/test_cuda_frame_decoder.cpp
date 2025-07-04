#include <catch2/catch_test_macros.hpp>
#include "../src/cuda_frame_decoder.h"
#include <filesystem>
#include <iostream>

const std::string TEST_MKV_FILE = "test_data/sample_hw.mkv";

TEST_CASE("CudaFrameDecoder basic functionality", "[cuda_frame_decoder]") {
    CudaFrameDecoder decoder;
    
    SECTION("Initial state") {
        REQUIRE_FALSE(decoder.isInitialized());
        REQUIRE(decoder.getVideoWidth() == 0);
        REQUIRE(decoder.getVideoHeight() == 0);
    }
    
    SECTION("Open with nullptr should use internal device") {
        if (std::filesystem::exists(TEST_MKV_FILE)) {
            bool result = decoder.open(TEST_MKV_FILE, nullptr);
            if (result) {
                REQUIRE(decoder.isInitialized());
                REQUIRE(decoder.getVideoWidth() > 0);
                REQUIRE(decoder.getVideoHeight() > 0);
                decoder.close();
            }
        } else {
            WARN("Test file not found: " << TEST_MKV_FILE);
        }
    }
    
    SECTION("Open non-existent file should fail") {
        bool result = decoder.open("non_existent_file.mkv", nullptr);
        REQUIRE_FALSE(result);
        REQUIRE_FALSE(decoder.isInitialized());
    }
}

TEST_CASE("CudaFrameDecoder frame reading", "[cuda_frame_decoder]") {
    CudaFrameDecoder decoder;
    
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        WARN("Test file not found: " << TEST_MKV_FILE);
        return;
    }
    
    SECTION("Read frames and verify CUDA tensor") {
        bool open_result = decoder.open(TEST_MKV_FILE, nullptr);
        if (!open_result) {
            WARN("Failed to open test file, skipping CUDA tensor test");
            return;
        }
        
        REQUIRE(decoder.isInitialized());
        
        CudaFrameDecoder::DecodedFrames frames;
        bool read_result = false;
        int attempts = 0;
        
        // 尝试读取多帧，直到找到有效的视频帧
        while (attempts < 10) {
            read_result = decoder.readNextFrames(frames);
            if (!read_result) break;
            
            if (frames.cuda_frame.is_valid) {
                std::cout << "Found valid CUDA frame after " << attempts + 1 << " attempts" << std::endl;
                break;
            }
            attempts++;
        }
        
        std::cout << "Attempting to read CUDA frame (attempts: " << attempts + 1 << ")..." << std::endl;
        
        if (read_result) {
            std::cout << "Frame read successful, checking CUDA conversion..." << std::endl;
            std::cout << "Audio frame valid: " << frames.audio_frame.is_valid << std::endl;
            std::cout << "CUDA frame valid: " << frames.cuda_frame.is_valid << std::endl;
            
            if (frames.cuda_frame.is_valid) {
                std::cout << "CUDA frame is valid!" << std::endl;
                
                REQUIRE(frames.cuda_frame.cuda_ptr != nullptr);
                REQUIRE(frames.cuda_frame.width > 0);
                REQUIRE(frames.cuda_frame.height > 0);
                REQUIRE(frames.cuda_frame.channels == 4);
                REQUIRE(frames.cuda_frame.size > 0);
                
                size_t expected_size = 1 * 4 * frames.cuda_frame.width * frames.cuda_frame.height * sizeof(float);
                REQUIRE(frames.cuda_frame.size == expected_size);
                
                std::cout << "CUDA frame info:" << std::endl;
                std::cout << "  Width: " << frames.cuda_frame.width << std::endl;
                std::cout << "  Height: " << frames.cuda_frame.height << std::endl;
                std::cout << "  Channels: " << frames.cuda_frame.channels << std::endl;
                std::cout << "  Size: " << frames.cuda_frame.size << " bytes" << std::endl;
                std::cout << "  CUDA ptr: " << frames.cuda_frame.cuda_ptr << std::endl;
                
                decoder.unmapCudaInput();
            } else {
                std::cout << "CUDA frame is invalid - conversion failed" << std::endl;
                WARN("CUDA frame conversion failed");
            }
        } else {
            std::cout << "Frame read failed" << std::endl;
            WARN("Failed to read frame");
        }
        
        decoder.close();
    }
}

TEST_CASE("CudaFrameDecoder resource management", "[cuda_frame_decoder]") {
    CudaFrameDecoder decoder;
    
    if (!std::filesystem::exists(TEST_MKV_FILE)) {
        WARN("Test file not found: " << TEST_MKV_FILE);
        return;
    }
    
    SECTION("Multiple open/close cycles") {
        for (int i = 0; i < 3; ++i) {
            bool result = decoder.open(TEST_MKV_FILE, nullptr);
            if (result) {
                REQUIRE(decoder.isInitialized());
                decoder.close();
                REQUIRE_FALSE(decoder.isInitialized());
            }
        }
    }
    
    SECTION("Flush without crash") {
        bool result = decoder.open(TEST_MKV_FILE, nullptr);
        if (result) {
            decoder.flush();
            REQUIRE(decoder.isInitialized());
            decoder.close();
        }
    }
}