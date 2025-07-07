#include "audio_player.h"
#include <iostream>
#include <cstring>
#include <algorithm>

extern "C" {
    #include <libswresample/swresample.h>
}

AudioPlayer::AudioPlayer() 
    : state_(State::Stopped)
    , current_time_(0.0)
    , duration_(0.0)
    , volume_(1.0f)
    , should_stop_(false)
    , should_pause_(false)
#ifdef _WIN32
    , device_enumerator_(nullptr)
    , audio_device_(nullptr)
    , audio_client_(nullptr)
    , render_client_(nullptr)
    , wave_format_(nullptr)
    , buffer_frame_count_(0)
    , audio_event_(nullptr)
#endif
{
    decoder_ = std::make_unique<AsyncAudioDecoder>();
}

AudioPlayer::~AudioPlayer() {
    stop();
    cleanupWASAPI();
}

bool AudioPlayer::initialize() {
#ifdef _WIN32
    if (!initializeWASAPI()) {
        state_ = State::Error;
        return false;
    }
#endif
    return true;
}

bool AudioPlayer::loadFile(const std::string& filepath) {
    if (!decoder_->open(filepath)) {
        state_ = State::Error;
        return false;
    }
    
    // 估算文件时长（简化实现）
    MKVStreamReader* reader = decoder_->getStreamReader();
    if (reader) {
        duration_ = 60.0; // 临时硬编码，实际应该从文件元数据获取
    }
    
    state_ = State::Stopped;
    return true;
}

bool AudioPlayer::play() {
    if (state_ == State::Error) {
        return false;
    }
    
    // 检查是否已加载文件
    if (!decoder_ || !decoder_->getStreamReader() || !decoder_->getStreamReader()->isOpen()) {
        return false;
    }
    
    if (state_ == State::Paused) {
        should_pause_ = false;
        state_cv_.notify_all();
        state_ = State::Playing;
        return true;
    }
    
    if (state_ == State::Playing) {
        return true;
    }
    
    should_stop_ = false;
    should_pause_ = false;
    
    audio_thread_ = std::thread(&AudioPlayer::audioThreadFunc, this);
    state_ = State::Playing;
    
    return true;
}

bool AudioPlayer::pause() {
    if (state_ != State::Playing) {
        return false;
    }
    
    should_pause_ = true;
    state_ = State::Paused;
    return true;
}

bool AudioPlayer::stop() {
    if (state_ == State::Stopped) {
        return true;
    }
    
    should_stop_ = true;
    should_pause_ = false;
    state_cv_.notify_all();
    
    if (audio_thread_.joinable()) {
        audio_thread_.join();
    }
    
    current_time_ = 0.0;
    state_ = State::Stopped;
    return true;
}

bool AudioPlayer::seekToTime(double seconds) {
    if (!decoder_ || !decoder_->getStreamReader() || !decoder_->getStreamReader()->isOpen()) {
        return false;
    }
    
    // 检查seek时间是否在有效范围内
    if (seconds < 0.0 || seconds > duration_) {
        return false;
    }
    
    bool was_playing = (state_ == State::Playing);
    
    if (was_playing) {
        pause();
    }
    
    bool result = decoder_->seekToTime(seconds);
    if (result) {
        current_time_ = seconds;
    }
    
    if (was_playing) {
        play();
    }
    
    return result;
}

void AudioPlayer::setVolume(float volume) {
    volume_ = std::clamp(volume, 0.0f, 1.0f);
}

void AudioPlayer::onTimer(double current_time) {
    current_time_ = current_time;
    
    // 检查并填充音频缓冲区
    checkAndFillAudioBuffer();
}

#ifdef _WIN32
bool AudioPlayer::initializeWASAPI() {
    HRESULT hr;
    
    // 初始化COM
    hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize COM" << std::endl;
        return false;
    }
    
    // 创建设备枚举器
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL, 
                          __uuidof(IMMDeviceEnumerator), (void**)&device_enumerator_);
    if (FAILED(hr)) {
        std::cerr << "Failed to create device enumerator" << std::endl;
        return false;
    }
    
    // 获取默认音频设备
    hr = device_enumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &audio_device_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get default audio endpoint" << std::endl;
        return false;
    }
    
    // 激活音频客户端
    hr = audio_device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audio_client_);
    if (FAILED(hr)) {
        std::cerr << "Failed to activate audio client" << std::endl;
        return false;
    }
    
    // 获取设备支持的格式
    hr = audio_client_->GetMixFormat(&wave_format_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get mix format" << std::endl;
        return false;
    }
    
    // 初始化音频客户端
    hr = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 
                                   AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                                   10000000, // 1秒缓冲
                                   0,
                                   wave_format_,
                                   nullptr);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client" << std::endl;
        return false;
    }
    
    // 获取缓冲区大小
    hr = audio_client_->GetBufferSize(&buffer_frame_count_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get buffer size" << std::endl;
        return false;
    }
    
    // 创建事件
    audio_event_ = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!audio_event_) {
        std::cerr << "Failed to create audio event" << std::endl;
        return false;
    }
    
    // 设置事件回调
    hr = audio_client_->SetEventHandle(audio_event_);
    if (FAILED(hr)) {
        std::cerr << "Failed to set event handle" << std::endl;
        return false;
    }
    
    // 获取渲染客户端
    hr = audio_client_->GetService(__uuidof(IAudioRenderClient), (void**)&render_client_);
    if (FAILED(hr)) {
        std::cerr << "Failed to get render client" << std::endl;
        return false;
    }
    
    return true;
}

void AudioPlayer::cleanupWASAPI() {
    if (audio_client_) {
        audio_client_->Stop();
    }
    
    if (audio_event_) {
        CloseHandle(audio_event_);
        audio_event_ = nullptr;
    }
    
    if (render_client_) {
        render_client_->Release();
        render_client_ = nullptr;
    }
    
    if (audio_client_) {
        audio_client_->Release();
        audio_client_ = nullptr;
    }
    
    if (wave_format_) {
        CoTaskMemFree(wave_format_);
        wave_format_ = nullptr;
    }
    
    if (audio_device_) {
        audio_device_->Release();
        audio_device_ = nullptr;
    }
    
    if (device_enumerator_) {
        device_enumerator_->Release();
        device_enumerator_ = nullptr;
    }
    
    CoUninitialize();
}
#endif

void AudioPlayer::audioThreadFunc() {
#ifdef _WIN32
    if (!audio_client_) {
        return;
    }
    
    HRESULT hr = audio_client_->Start();
    if (FAILED(hr)) {
        std::cerr << "Failed to start audio client" << std::endl;
        return;
    }
    
    while (!should_stop_) {
        // 处理暂停
        if (should_pause_) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            state_cv_.wait(lock, [this] { return !should_pause_ || should_stop_; });
            if (should_stop_) break;
        }
        
        // 等待音频事件
        DWORD wait_result = WaitForSingleObject(audio_event_, 1000);
        if (wait_result != WAIT_OBJECT_0) {
            continue;
        }
        
        if (!processAudioFrame()) {
            break;
        }
        
        // 音频线程不再主动更新时间，等待外部onTimer调用
    }
    
    audio_client_->Stop();
#endif
}

bool AudioPlayer::processAudioFrame() {
#ifdef _WIN32
    UINT32 padding;
    HRESULT hr = audio_client_->GetCurrentPadding(&padding);
    if (FAILED(hr)) {
        return false;
    }
    
    UINT32 available_frames = buffer_frame_count_ - padding;
    if (available_frames == 0) {
        return true;
    }
    
    AsyncAudioDecoder::DecodedFrame frame;
    if (!decoder_->readNextFrame(frame)) {
        return false;
    }
    
    if (frame.is_eof) {
        return false;
    }
    
    if (!frame.is_valid) {
        return true;
    }
    
    return convertAndFillBuffer(frame);
#else
    return false;
#endif
}

bool AudioPlayer::convertAndFillBuffer(const AsyncAudioDecoder::DecodedFrame& frame) {
#ifdef _WIN32
    if (!frame.frame) {
        return false;
    }
    
    BYTE* buffer_data;
    HRESULT hr = render_client_->GetBuffer(buffer_frame_count_, &buffer_data);
    if (FAILED(hr)) {
        return false;
    }
    
    // 简化的音频格式转换（假设输入是float格式）
    AVFrame* audio_frame = frame.frame;
    int channels = audio_frame->ch_layout.nb_channels;
    int samples_per_frame = audio_frame->nb_samples;
    
    // 计算要拷贝的帧数
    UINT32 frames_to_copy = (std::min)(static_cast<UINT32>(samples_per_frame), buffer_frame_count_);
    
    // 简单的音频数据拷贝（需要根据实际格式调整）
    if (audio_frame->format == AV_SAMPLE_FMT_FLTP) {
        float* output = (float*)buffer_data;
        for (UINT32 i = 0; i < frames_to_copy; i++) {
            for (int ch = 0; ch < channels; ch++) {
                float* channel_data = (float*)audio_frame->data[ch];
                *output++ = channel_data[i] * volume_;
            }
        }
    } else {
        // 其他格式的转换...
        memset(buffer_data, 0, frames_to_copy * channels * sizeof(float));
    }
    
    hr = render_client_->ReleaseBuffer(frames_to_copy, 0);
    return SUCCEEDED(hr);
#else
    return false;
#endif
}

void AudioPlayer::checkAndFillAudioBuffer() {
#ifdef _WIN32
    if (state_ != State::Playing || !audio_client_) {
        return;
    }
    
    UINT32 padding;
    HRESULT hr = audio_client_->GetCurrentPadding(&padding);
    if (FAILED(hr)) {
        return;
    }
    
    UINT32 available_frames = buffer_frame_count_ - padding;
    if (available_frames == 0) {
        return;
    }
    
    AsyncAudioDecoder::DecodedFrame frame;
    if (!decoder_->readNextFrame(frame)) {
        return;
    }
    
    if (frame.is_eof) {
        return;
    }
    
    if (frame.is_valid) {
        convertAndFillBuffer(frame);
    }
#endif
}

void AudioPlayer::sampleFormatToWaveFormat(AVSampleFormat sample_fmt, int channels, int sample_rate, WAVEFORMATEX* wave_format) {
    wave_format->wFormatTag = WAVE_FORMAT_PCM;
    wave_format->nChannels = channels;
    wave_format->nSamplesPerSec = sample_rate;
    
    switch (sample_fmt) {
        case AV_SAMPLE_FMT_U8:
            wave_format->wBitsPerSample = 8;
            break;
        case AV_SAMPLE_FMT_S16:
            wave_format->wBitsPerSample = 16;
            break;
        case AV_SAMPLE_FMT_S32:
        case AV_SAMPLE_FMT_FLT:
            wave_format->wBitsPerSample = 32;
            break;
        default:
            wave_format->wBitsPerSample = 16;
            break;
    }
    
    wave_format->nBlockAlign = channels * wave_format->wBitsPerSample / 8;
    wave_format->nAvgBytesPerSec = wave_format->nSamplesPerSec * wave_format->nBlockAlign;
    wave_format->cbSize = 0;
}