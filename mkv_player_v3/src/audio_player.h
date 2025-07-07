#pragma once

#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

#include "async_audio_decoder.h"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/samplefmt.h>
}

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#endif

class AudioPlayer {
public:
    enum class State {
        Stopped,
        Playing,
        Paused,
        Error
    };

    AudioPlayer();
    ~AudioPlayer();

    bool initialize();
    bool loadFile(const std::string& filepath);
    
    bool play();
    bool pause();
    bool stop();
    
    bool seekToTime(double seconds);
    
    State getState() const { return state_; }
    double getCurrentTime() const { return current_time_; }
    double getDuration() const { return duration_; }
    
    void setVolume(float volume);
    float getVolume() const { return volume_; }
    
    void onTimer(double current_time);

private:
    std::unique_ptr<AsyncAudioDecoder> decoder_;
    
    State state_;
    std::atomic<double> current_time_;
    std::atomic<double> duration_;
    std::atomic<float> volume_;
    
    
    std::thread audio_thread_;
    std::atomic<bool> should_stop_;
    std::atomic<bool> should_pause_;
    std::mutex state_mutex_;
    std::condition_variable state_cv_;
    
#ifdef _WIN32
    // WASAPI components
    IMMDeviceEnumerator* device_enumerator_;
    IMMDevice* audio_device_;
    IAudioClient* audio_client_;
    IAudioRenderClient* render_client_;
    WAVEFORMATEX* wave_format_;
    UINT32 buffer_frame_count_;
    HANDLE audio_event_;
#endif
    
    void audioThreadFunc();
    bool initializeWASAPI();
    void cleanupWASAPI();
    
    bool processAudioFrame();
    bool convertAndFillBuffer(const AsyncAudioDecoder::DecodedFrame& frame);
    
    void checkAndFillAudioBuffer();
    
    static void sampleFormatToWaveFormat(AVSampleFormat sample_fmt, int channels, int sample_rate, WAVEFORMATEX* wave_format);
};