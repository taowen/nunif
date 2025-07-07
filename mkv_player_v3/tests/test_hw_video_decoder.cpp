#include <catch2/catch_test_macros.hpp>
#include "../src/hw_video_decoder.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

TEST_CASE("HwVideoDecoder Basic Test", "[HwVideoDecoder]") {
    fs::path test_file = fs::current_path() / "test_data" / "sample_hw.mkv";
    
    if (!fs::exists(test_file)) {
        WARN("Test file not found: " << test_file.string() << ". Skipping tests.");
        return;
    }
    
    HwVideoDecoder decoder;
    REQUIRE(decoder.open(test_file.string()));
    REQUIRE(decoder.isOpen());
}

