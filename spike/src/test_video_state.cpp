#include <catch2/catch_test_macros.hpp>
#include "ffmpeg_handler.h"
#include "video_state.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

TEST_CASE("Video state happy path", "[ffmpeg][video]") {
    bool created = createFFmpegHandler("06 4k.mp4");
    REQUIRE(created == true);
    
    // 获取视频编解码上下文
    AVCodecContext* videoCodecContext = getVideoCodecContext();
    REQUIRE(videoCodecContext != nullptr);
    
    // 验证编解码上下文的基本属性
    REQUIRE(videoCodecContext->width > 0);
    REQUIRE(videoCodecContext->height > 0);
    
    // 清理
    destroyFFmpegHandler();
}
