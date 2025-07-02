#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "../src/video_file_info.h"
#include <filesystem>
#include <fstream>

using namespace spike;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("VideoFileInfoReader functionality", "[video_file_info_reader]") {
    VideoFileInfoReader reader;
    auto result = reader.read_from_file(R"(c:\Users\taowen\Downloads\iw3\test.mkv)");
    REQUIRE(result.has_value());
}
