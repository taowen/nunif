#include <catch2/catch_test_macros.hpp>
#include "../src/video_decoder.h"
#include "../src/video_file_info.h"

namespace spike {

// Mock实现用于测试
class MockVideoDecoder : public VideoDecoder {
private:
    DecoderState state_ = DecoderState::Uninitialized;
    VideoFileInfo video_info_;
    std::string last_error_;
    double current_position_ = 0.0;
    int frame_count_ = 0;
    static constexpr int MAX_FRAMES = 100; // 模拟100帧

public:
    bool initialize(const VideoFileInfo& video_info) override {
        if (!video_info.is_valid()) {
            last_error_ = "Invalid video info";
            state_ = DecoderState::Error;
            return false;
        }
        video_info_ = video_info;
        state_ = DecoderState::Ready;
        return true;
    }
    
    bool start_decoding() override {
        if (state_ != DecoderState::Ready) {
            last_error_ = "Decoder not ready";
            return false;
        }
        state_ = DecoderState::Decoding;
        return true;
    }
    
    std::unique_ptr<VideoFrame> decode_next_frame() override {
        if (state_ != DecoderState::Decoding) {
            return nullptr;
        }
        
        if (frame_count_ >= MAX_FRAMES) {
            state_ = DecoderState::EndOfStream;
            return nullptr;
        }
        
        auto frame = std::make_unique<VideoFrame>();
        frame->width = video_info_.width;
        frame->height = video_info_.height;
        frame->timestamp_seconds = frame_count_ / video_info_.frame_rate;
        frame->data_size = frame->width * frame->height * 4; // RGBA
        frame->data = std::make_unique<uint8_t[]>(frame->data_size);
        
        current_position_ = frame->timestamp_seconds;
        frame_count_++;
        
        return frame;
    }
    
    bool seek_to_time(double seconds) override {
        if (state_ == DecoderState::Error) {
            return false;
        }
        if (seconds < 0 || seconds > video_info_.duration_seconds) {
            last_error_ = "Seek position out of range";
            return false;
        }
        
        frame_count_ = static_cast<int>(seconds * video_info_.frame_rate);
        current_position_ = seconds;
        
        if (state_ == DecoderState::EndOfStream) {
            state_ = DecoderState::Decoding;
        }
        
        return true;
    }
    
    void stop_decoding() override {
        if (state_ == DecoderState::Decoding || state_ == DecoderState::EndOfStream) {
            state_ = DecoderState::Ready;
        }
    }
    
    DecoderState get_state() const override { return state_; }
    const std::string& get_last_error() const override { return last_error_; }
    double get_current_position() const override { return current_position_; }
};

class MockVideoDecoderFactory : public VideoDecoderFactory {
public:
    std::unique_ptr<VideoDecoder> create_decoder() override {
        return std::make_unique<MockVideoDecoder>();
    }
    
    bool supports_format(const std::string& codec_name) const override {
        return codec_name == "h264" || codec_name == "h265";
    }
};

} // namespace spike

TEST_CASE("VideoFrame validation", "[VideoDecoder]") {
    spike::VideoFrame frame;
    
    SECTION("默认构造的帧无效") {
        REQUIRE_FALSE(frame.is_valid());
    }
    
    SECTION("完整的帧有效") {
        frame.width = 1920;
        frame.height = 1080;
        frame.data_size = 1920 * 1080 * 4;
        frame.data = std::make_unique<uint8_t[]>(frame.data_size);
        
        REQUIRE(frame.is_valid());
    }
}

TEST_CASE("VideoDecoder初始化", "[VideoDecoder]") {
    spike::MockVideoDecoder decoder;
    
    SECTION("初始状态为未初始化") {
        REQUIRE(decoder.get_state() == spike::DecoderState::Uninitialized);
    }
    
    SECTION("无效视频信息初始化失败") {
        spike::VideoFileInfo invalid_info; // 默认构造无效
        REQUIRE_FALSE(decoder.initialize(invalid_info));
        REQUIRE(decoder.get_state() == spike::DecoderState::Error);
        REQUIRE_FALSE(decoder.get_last_error().empty());
    }
    
    SECTION("有效视频信息初始化成功") {
        spike::VideoFileInfo info;
        info.width = 1920;
        info.height = 1080;
        info.duration_seconds = 60.0;
        info.frame_rate = 30.0;
        info.codec_name = "h264";
        
        REQUIRE(decoder.initialize(info));
        REQUIRE(decoder.get_state() == spike::DecoderState::Ready);
    }
}

TEST_CASE("VideoDecoder解码流程", "[VideoDecoder]") {
    spike::MockVideoDecoder decoder;
    spike::VideoFileInfo info;
    info.width = 1920;
    info.height = 1080;
    info.duration_seconds = 3.0;  // 3秒，90帧
    info.frame_rate = 30.0;
    info.codec_name = "h264";
    
    REQUIRE(decoder.initialize(info));
    
    SECTION("未开始解码时无法获取帧") {
        auto frame = decoder.decode_next_frame();
        REQUIRE(frame == nullptr);
    }
    
    SECTION("正常解码流程") {
        REQUIRE(decoder.start_decoding());
        REQUIRE(decoder.get_state() == spike::DecoderState::Decoding);
        
        // 解码几帧
        for (int i = 0; i < 5; ++i) {
            auto frame = decoder.decode_next_frame();
            REQUIRE(frame != nullptr);
            REQUIRE(frame->is_valid());
            REQUIRE(frame->width == 1920);
            REQUIRE(frame->height == 1080);
            REQUIRE(frame->timestamp_seconds == i / 30.0);  // 30fps
        }
        
        REQUIRE(decoder.get_current_position() > 0.0);
    }
    
    SECTION("解码结束") {
        REQUIRE(decoder.start_decoding());
        
        // 解码所有帧直到结束
        int frame_count = 0;
        while (auto frame = decoder.decode_next_frame()) {
            frame_count++;
            if (frame_count > 200) break; // 防止无限循环
        }
        
        REQUIRE(decoder.get_state() == spike::DecoderState::EndOfStream);
    }
}

TEST_CASE("VideoDecoder跳转功能", "[VideoDecoder]") {
    spike::MockVideoDecoder decoder;
    spike::VideoFileInfo info;
    info.width = 1920;
    info.height = 1080;
    info.duration_seconds = 10.0;
    info.frame_rate = 30.0;
    info.codec_name = "h264";
    
    REQUIRE(decoder.initialize(info));
    REQUIRE(decoder.start_decoding());
    
    SECTION("有效跳转") {
        REQUIRE(decoder.seek_to_time(5.0));
        REQUIRE(decoder.get_current_position() == 5.0);
        
        auto frame = decoder.decode_next_frame();
        REQUIRE(frame != nullptr);
        REQUIRE(frame->timestamp_seconds >= 5.0);
    }
    
    SECTION("跳转到开头") {
        // 先解码几帧
        decoder.decode_next_frame();
        decoder.decode_next_frame();
        
        REQUIRE(decoder.seek_to_time(0.0));
        REQUIRE(decoder.get_current_position() == 0.0);
    }
    
    SECTION("超出范围的跳转失败") {
        REQUIRE_FALSE(decoder.seek_to_time(-1.0));
        REQUIRE_FALSE(decoder.seek_to_time(15.0));
        REQUIRE_FALSE(decoder.get_last_error().empty());
    }
}

TEST_CASE("VideoDecoderFactory", "[VideoDecoder]") {
    spike::MockVideoDecoderFactory factory;
    
    SECTION("创建解码器") {
        auto decoder = factory.create_decoder();
        REQUIRE(decoder != nullptr);
        REQUIRE(decoder->get_state() == spike::DecoderState::Uninitialized);
    }
    
    SECTION("格式支持检查") {
        REQUIRE(factory.supports_format("h264"));
        REQUIRE(factory.supports_format("h265"));
        REQUIRE_FALSE(factory.supports_format("unknown"));
    }
} 