#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <filesystem>
#include <chrono>

namespace fs = std::filesystem;

class HwVideoDecoderTestFixture {
public:
    HwVideoDecoderTestFixture() {
        test_file_path_ = fs::current_path() / "test_data" / "sample_hw.mkv";
    }
    
    bool testFileExists() const {
        return fs::exists(test_file_path_);
    }
    
    std::string getTestFilePath() const {
        return test_file_path_.string();
    }

private:
    fs::path test_file_path_;
};

TEST_CASE("HwVideoDecoder Basic Functionality", "[HwVideoDecoder]") {
    HwVideoDecoderTestFixture fixture;
    
    SECTION("Constructor and Destructor") {
        HwVideoDecoder decoder;
        REQUIRE_FALSE(decoder.isOpen());
        REQUIRE_FALSE(decoder.isEOF());
        REQUIRE(decoder.getWidth() == 0);
        REQUIRE(decoder.getHeight() == 0);
    }
    
    SECTION("Open non-existent file") {
        HwVideoDecoder decoder;
        REQUIRE_FALSE(decoder.open("nonexistent.mkv"));
        REQUIRE_FALSE(decoder.isOpen());
    }
    
    if (fixture.testFileExists()) {
        SECTION("Open valid MKV file") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            REQUIRE(decoder.isOpen());
            REQUIRE_FALSE(decoder.isEOF());
            REQUIRE(decoder.getWidth() > 0);
            REQUIRE(decoder.getHeight() > 0);
            REQUIRE(decoder.getFPS() > 0.0);
            REQUIRE(decoder.getDuration() > 0.0);
            
            decoder.close();
            REQUIRE_FALSE(decoder.isOpen());
        }
        
        SECTION("Read frames from video") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            HwVideoDecoder::DecodedFrame frame;
            bool frame_read = false;
            int frame_count = 0;
            const int max_frames_to_test = 10;
            
            while (frame_count < max_frames_to_test && decoder.readNextFrame(frame)) {
                REQUIRE(frame.is_valid);
                REQUIRE(frame.texture != nullptr);
                REQUIRE(frame.width == decoder.getWidth());
                REQUIRE(frame.height == decoder.getHeight());
                REQUIRE(frame.pts >= 0);
                frame_read = true;
                frame_count++;
            }
            
            REQUIRE(frame_read);
            REQUIRE(frame_count > 0);
        }
        
        SECTION("Seek functionality") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            double seek_time = decoder.getDuration() * 0.1;
            REQUIRE(decoder.seekToTime(seek_time));
            REQUIRE_FALSE(decoder.isEOF());
            
            HwVideoDecoder::DecodedFrame frame;
            REQUIRE(decoder.readNextFrame(frame));
            REQUIRE(frame.is_valid);
        }
        
        SECTION("Frame seek functionality") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            REQUIRE(decoder.seekToFrame(30));
            REQUIRE_FALSE(decoder.isEOF());
            
            HwVideoDecoder::DecodedFrame frame;
            REQUIRE(decoder.readNextFrame(frame));
            REQUIRE(frame.is_valid);
        }
        
        SECTION("Multiple open/close cycles") {
            HwVideoDecoder decoder;
            
            for (int i = 0; i < 3; i++) {
                REQUIRE(decoder.open(fixture.getTestFilePath()));
                REQUIRE(decoder.isOpen());
                
                HwVideoDecoder::DecodedFrame frame;
                REQUIRE(decoder.readNextFrame(frame));
                REQUIRE(frame.is_valid);
                
                decoder.close();
                REQUIRE_FALSE(decoder.isOpen());
            }
        }
        
        SECTION("EOF detection") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            double end_time = decoder.getDuration() - 0.1;
            REQUIRE(decoder.seekToTime(end_time));
            
            HwVideoDecoder::DecodedFrame frame;
            int frames_near_end = 0;
            
            while (decoder.readNextFrame(frame) && frames_near_end < 100) {
                frames_near_end++;
            }
            
            REQUIRE(decoder.isEOF());
        }
        
        SECTION("Invalid seek operations") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            REQUIRE_FALSE(decoder.seekToTime(-1.0));
            REQUIRE_FALSE(decoder.seekToTime(decoder.getDuration() + 100.0));
            REQUIRE_FALSE(decoder.seekToFrame(-1));
        }
        
        SECTION("Texture properties validation") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            HwVideoDecoder::DecodedFrame frame;
            REQUIRE(decoder.readNextFrame(frame));
            REQUIRE(frame.is_valid);
            
            D3D11_TEXTURE2D_DESC desc;
            frame.texture->GetDesc(&desc);
            
            REQUIRE(desc.Width == static_cast<UINT>(decoder.getWidth()));
            REQUIRE(desc.Height == static_cast<UINT>(decoder.getHeight()));
            REQUIRE(desc.Format == DXGI_FORMAT_NV12);
            REQUIRE(desc.Usage == D3D11_USAGE_DEFAULT);
            REQUIRE(desc.BindFlags & D3D11_BIND_SHADER_RESOURCE);
        }
    } else {
        WARN("Test file not found: " << fixture.getTestFilePath() << ". Skipping integration tests.");
    }
}

TEST_CASE("HwVideoDecoder Error Handling", "[HwVideoDecoder][error]") {
    SECTION("Read frame without opening") {
        HwVideoDecoder decoder;
        HwVideoDecoder::DecodedFrame frame;
        REQUIRE_FALSE(decoder.readNextFrame(frame));
        REQUIRE_FALSE(frame.is_valid);
    }
    
    SECTION("Seek without opening") {
        HwVideoDecoder decoder;
        REQUIRE_FALSE(decoder.seekToTime(10.0));
        REQUIRE_FALSE(decoder.seekToFrame(100));
    }
    
    SECTION("Double close") {
        HwVideoDecoder decoder;
        decoder.close();
        decoder.close();
        REQUIRE_FALSE(decoder.isOpen());
    }
    
    SECTION("Operations after close") {
        HwVideoDecoderTestFixture fixture;
        
        if (fixture.testFileExists()) {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            decoder.close();
            
            HwVideoDecoder::DecodedFrame frame;
            REQUIRE_FALSE(decoder.readNextFrame(frame));
            REQUIRE_FALSE(decoder.seekToTime(5.0));
            REQUIRE_FALSE(decoder.seekToFrame(50));
        }
    }
}

TEST_CASE("HwVideoDecoder Performance", "[HwVideoDecoder][performance]") {
    HwVideoDecoderTestFixture fixture;
    
    if (fixture.testFileExists()) {
        SECTION("Frame decode performance") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            HwVideoDecoder::DecodedFrame frame;
            int decoded_frames = 0;
            const int frames_to_decode = 100;
            
            while (decoded_frames < frames_to_decode && decoder.readNextFrame(frame)) {
                REQUIRE(frame.is_valid);
                decoded_frames++;
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            if (decoded_frames > 0) {
                double fps = (double)decoded_frames / (duration.count() / 1000.0);
                INFO("Decoded " << decoded_frames << " frames in " << duration.count() << "ms");
                INFO("Decode performance: " << fps << " FPS");
                
                REQUIRE(fps > 10.0);
            }
        }
        
        SECTION("Seek performance") {
            HwVideoDecoder decoder;
            REQUIRE(decoder.open(fixture.getTestFilePath()));
            
            double duration = decoder.getDuration();
            const int seek_count = 10;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            for (int i = 0; i < seek_count; i++) {
                double seek_time = (duration / seek_count) * i;
                REQUIRE(decoder.seekToTime(seek_time));
                
                HwVideoDecoder::DecodedFrame frame;
                REQUIRE(decoder.readNextFrame(frame));
                REQUIRE(frame.is_valid);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto seek_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            INFO("Performed " << seek_count << " seeks in " << seek_duration.count() << "ms");
            INFO("Average seek time: " << (seek_duration.count() / seek_count) << "ms");
            
            REQUIRE(seek_duration.count() < 5000);
        }
    }
}