#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
}

#include <d3d11.h>
#include <dxgi.h>

// 数据流架构设计
namespace MKVPlayer {

// 1. 数据包结构
struct MediaPacket {
    AVPacket* packet;
    int stream_index;
    double pts;  // 播放时间戳
    double dts;  // 解码时间戳
};

struct VideoFrame {
    AVFrame* frame;
    double pts;
    bool is_hw_frame;  // 是否硬件帧
};

struct AudioFrame {
    AVFrame* frame;
    double pts;
    int nb_samples;
};

// 2. 线程安全队列模板
template<typename T>
class ThreadSafeQueue {
private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable condition_;
    
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        condition_.notify_one();
    }
    
    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms < 0) {
            condition_.wait(lock, [this] { return !queue_.empty(); });
        } else {
            if (!condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                   [this] { return !queue_.empty(); })) {
                return false;
            }
        }
        item = queue_.front();
        queue_.pop();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

// 3. 数据流组件接口
class IDataStreamComponent {
public:
    virtual ~IDataStreamComponent() = default;
    virtual bool initialize() = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void cleanup() = 0;
};

// 4. 文件读取器 - 解封装
class Demuxer : public IDataStreamComponent {
private:
    AVFormatContext* format_ctx_ = nullptr;
    std::string file_path_;
    int video_stream_index_ = -1;
    int audio_stream_index_ = -1;
    
    ThreadSafeQueue<MediaPacket>* video_packet_queue_;
    ThreadSafeQueue<MediaPacket>* audio_packet_queue_;
    
    std::thread demux_thread_;
    bool running_ = false;
    
public:
    Demuxer(const std::string& file_path,
            ThreadSafeQueue<MediaPacket>* video_queue,
            ThreadSafeQueue<MediaPacket>* audio_queue)
        : file_path_(file_path), video_packet_queue_(video_queue), audio_packet_queue_(audio_queue) {}
    
    bool initialize() override {
        // 打开文件
        if (avformat_open_input(&format_ctx_, file_path_.c_str(), nullptr, nullptr) < 0) {
            return false;
        }
        
        // 查找流信息
        if (avformat_find_stream_info(format_ctx_, nullptr) < 0) {
            return false;
        }
        
        // 找到视频和音频流
        for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
            if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && video_stream_index_ < 0) {
                video_stream_index_ = i;
            } else if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audio_stream_index_ < 0) {
                audio_stream_index_ = i;
            }
        }
        
        return video_stream_index_ >= 0;
    }
    
    void start() override {
        running_ = true;
        demux_thread_ = std::thread(&Demuxer::demux_loop, this);
    }
    
    void stop() override {
        running_ = false;
        if (demux_thread_.joinable()) {
            demux_thread_.join();
        }
    }
    
    void cleanup() override {
        if (format_ctx_) {
            avformat_close_input(&format_ctx_);
        }
    }
    
    int get_video_stream_index() const { return video_stream_index_; }
    int get_audio_stream_index() const { return audio_stream_index_; }
    AVStream* get_video_stream() const { return format_ctx_->streams[video_stream_index_]; }
    AVStream* get_audio_stream() const { return format_ctx_->streams[audio_stream_index_]; }
    
private:
    void demux_loop() {
        AVPacket* packet = av_packet_alloc();
        
        while (running_) {
            if (av_read_frame(format_ctx_, packet) < 0) {
                break; // EOF或错误
            }
            
            MediaPacket media_packet;
            media_packet.packet = av_packet_clone(packet);
            media_packet.stream_index = packet->stream_index;
            media_packet.pts = packet->pts * av_q2d(format_ctx_->streams[packet->stream_index]->time_base);
            media_packet.dts = packet->dts * av_q2d(format_ctx_->streams[packet->stream_index]->time_base);
            
            if (packet->stream_index == video_stream_index_) {
                video_packet_queue_->push(media_packet);
            } else if (packet->stream_index == audio_stream_index_) {
                audio_packet_queue_->push(media_packet);
            } else {
                av_packet_free(&media_packet.packet);
            }
            
            av_packet_unref(packet);
        }
        
        av_packet_free(&packet);
    }
};

// 5. 视频解码器 - DirectX11硬件加速
class VideoDecoder : public IDataStreamComponent {
private:
    AVCodecContext* codec_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    ID3D11Device* d3d11_device_ = nullptr;
    
    ThreadSafeQueue<MediaPacket>* input_queue_;
    ThreadSafeQueue<VideoFrame>* output_queue_;
    
    std::thread decode_thread_;
    bool running_ = false;
    
public:
    VideoDecoder(ThreadSafeQueue<MediaPacket>* input_queue,
                ThreadSafeQueue<VideoFrame>* output_queue,
                ID3D11Device* d3d11_device)
        : input_queue_(input_queue), output_queue_(output_queue), d3d11_device_(d3d11_device) {}
    
    bool initialize(AVStream* video_stream) {
        // 创建硬件设备上下文
        if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0) < 0) {
            return false;
        }
        
        // 设置D3D11设备
        AVHWDeviceContext* hw_device_ctx = (AVHWDeviceContext*)hw_device_ctx_->data;
        AVD3D11VADeviceContext* d3d11_ctx = (AVD3D11VADeviceContext*)hw_device_ctx->hwctx;
        d3d11_ctx->device = d3d11_device_;
        d3d11_device_->AddRef();
        
        // 初始化解码器
        const AVCodec* codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
        if (!codec) {
            return false;
        }
        
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codec_ctx_, video_stream->codecpar) < 0) {
            return false;
        }
        
        codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
        
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            return false;
        }
        
        return true;
    }
    
    void start() override {
        running_ = true;
        decode_thread_ = std::thread(&VideoDecoder::decode_loop, this);
    }
    
    void stop() override {
        running_ = false;
        if (decode_thread_.joinable()) {
            decode_thread_.join();
        }
    }
    
    void cleanup() override {
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
        }
        if (hw_device_ctx_) {
            av_buffer_unref(&hw_device_ctx_);
        }
    }
    
private:
    void decode_loop() {
        AVFrame* frame = av_frame_alloc();
        
        while (running_) {
            MediaPacket packet;
            if (!input_queue_->pop(packet, 100)) {
                continue;
            }
            
            if (avcodec_send_packet(codec_ctx_, packet.packet) >= 0) {
                while (avcodec_receive_frame(codec_ctx_, frame) >= 0) {
                    VideoFrame video_frame;
                    video_frame.frame = av_frame_clone(frame);
                    video_frame.pts = packet.pts;
                    video_frame.is_hw_frame = (frame->format == AV_PIX_FMT_D3D11);
                    
                    output_queue_->push(video_frame);
                    av_frame_unref(frame);
                }
            }
            
            av_packet_free(&packet.packet);
        }
        
        av_frame_free(&frame);
    }
};

// 6. 音频解码器
class AudioDecoder : public IDataStreamComponent {
private:
    AVCodecContext* codec_ctx_ = nullptr;
    ThreadSafeQueue<MediaPacket>* input_queue_;
    ThreadSafeQueue<AudioFrame>* output_queue_;
    
    std::thread decode_thread_;
    bool running_ = false;
    
public:
    AudioDecoder(ThreadSafeQueue<MediaPacket>* input_queue,
                ThreadSafeQueue<AudioFrame>* output_queue)
        : input_queue_(input_queue), output_queue_(output_queue) {}
    
    bool initialize(AVStream* audio_stream) {
        const AVCodec* codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);
        if (!codec) {
            return false;
        }
        
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (avcodec_parameters_to_context(codec_ctx_, audio_stream->codecpar) < 0) {
            return false;
        }
        
        if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
            return false;
        }
        
        return true;
    }
    
    void start() override {
        running_ = true;
        decode_thread_ = std::thread(&AudioDecoder::decode_loop, this);
    }
    
    void stop() override {
        running_ = false;
        if (decode_thread_.joinable()) {
            decode_thread_.join();
        }
    }
    
    void cleanup() override {
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
        }
    }
    
private:
    void decode_loop() {
        AVFrame* frame = av_frame_alloc();
        
        while (running_) {
            MediaPacket packet;
            if (!input_queue_->pop(packet, 100)) {
                continue;
            }
            
            if (avcodec_send_packet(codec_ctx_, packet.packet) >= 0) {
                while (avcodec_receive_frame(codec_ctx_, frame) >= 0) {
                    AudioFrame audio_frame;
                    audio_frame.frame = av_frame_clone(frame);
                    audio_frame.pts = packet.pts;
                    audio_frame.nb_samples = frame->nb_samples;
                    
                    output_queue_->push(audio_frame);
                    av_frame_unref(frame);
                }
            }
            
            av_packet_free(&packet.packet);
        }
        
        av_frame_free(&frame);
    }
};

// 7. 同步控制器
class AVSyncController {
private:
    std::chrono::high_resolution_clock::time_point start_time_;
    double audio_clock_ = 0.0;
    double video_clock_ = 0.0;
    bool started_ = false;
    
public:
    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        started_ = true;
    }
    
    void update_audio_clock(double pts) {
        audio_clock_ = pts;
    }
    
    void update_video_clock(double pts) {
        video_clock_ = pts;
    }
    
    double get_master_clock() const {
        if (!started_) return 0.0;
        
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return elapsed.count() / 1000000.0;
    }
    
    // 计算视频帧应该延迟多长时间显示
    int calculate_video_delay(double frame_pts) {
        double master_clock = get_master_clock();
        double delay = frame_pts - master_clock;
        return static_cast<int>(delay * 1000); // 转换为毫秒
    }
    
    bool should_drop_video_frame(double frame_pts) {
        double master_clock = get_master_clock();
        return (master_clock - frame_pts) > 0.1; // 超过100ms就丢帧
    }
};

// 8. 主数据流控制器
class MKVPlayerDataStream {
private:
    // 数据队列
    ThreadSafeQueue<MediaPacket> video_packet_queue_;
    ThreadSafeQueue<MediaPacket> audio_packet_queue_;
    ThreadSafeQueue<VideoFrame> video_frame_queue_;
    ThreadSafeQueue<AudioFrame> audio_frame_queue_;
    
    // 组件
    std::unique_ptr<Demuxer> demuxer_;
    std::unique_ptr<VideoDecoder> video_decoder_;
    std::unique_ptr<AudioDecoder> audio_decoder_;
    std::unique_ptr<AVSyncController> sync_controller_;
    
    ID3D11Device* d3d11_device_ = nullptr;
    
    // 新增sink组件
    std::unique_ptr<VideoSink> video_sink_;
    std::unique_ptr<AudioSink> audio_sink_;
    
    HWND hwnd_ = nullptr;
    
public:
    MKVPlayerDataStream(ID3D11Device* d3d11_device, HWND hwnd) 
        : d3d11_device_(d3d11_device), hwnd_(hwnd) {
        sync_controller_ = std::make_unique<AVSyncController>();
    }
    
    bool initialize(const std::string& file_path) {
        // 创建解封装器
        demuxer_ = std::make_unique<Demuxer>(file_path, &video_packet_queue_, &audio_packet_queue_);
        if (!demuxer_->initialize()) {
            return false;
        }
        
        // 创建视频解码器
        video_decoder_ = std::make_unique<VideoDecoder>(&video_packet_queue_, &video_frame_queue_, d3d11_device_);
        if (!video_decoder_->initialize(demuxer_->get_video_stream())) {
            return false;
        }
        
        // 创建音频解码器
        audio_decoder_ = std::make_unique<AudioDecoder>(&audio_packet_queue_, &audio_frame_queue_);
        if (!audio_decoder_->initialize(demuxer_->get_audio_stream())) {
            return false;
        }
        
        // 创建视频渲染器
        video_sink_ = std::make_unique<VideoSink>(&video_frame_queue_, sync_controller_.get(), d3d11_device_, hwnd_);
        if (!video_sink_->initialize()) {
            return false;
        }
        
        // 创建音频播放器
        audio_sink_ = std::make_unique<AudioSink>(&audio_frame_queue_, sync_controller_.get());
        if (!audio_sink_->initialize()) {
            return false;
        }
        
        return true;
    }
    
    void start() {
        sync_controller_->start();
        demuxer_->start();
        video_decoder_->start();
        audio_decoder_->start();
        video_sink_->start();
        audio_sink_->start();
    }
    
    void stop() {
        video_sink_->stop();
        audio_sink_->stop();
        demuxer_->stop();
        video_decoder_->stop();
        audio_decoder_->stop();
    }
    
    void cleanup() {
        video_sink_->cleanup();
        audio_sink_->cleanup();
        demuxer_->cleanup();
        video_decoder_->cleanup();
        audio_decoder_->cleanup();
    }
};

// 9. 视频渲染器 - DirectX11
class VideoSink : public IDataStreamComponent {
private:
    ThreadSafeQueue<VideoFrame>* input_queue_;
    AVSyncController* sync_controller_;
    
    ID3D11Device* d3d11_device_ = nullptr;
    ID3D11DeviceContext* d3d11_context_ = nullptr;
    IDXGISwapChain* swap_chain_ = nullptr;
    ID3D11RenderTargetView* render_target_view_ = nullptr;
    
    std::thread render_thread_;
    bool running_ = false;
    HWND hwnd_;
    
public:
    VideoSink(ThreadSafeQueue<VideoFrame>* input_queue,
              AVSyncController* sync_controller,
              ID3D11Device* d3d11_device,
              HWND hwnd)
        : input_queue_(input_queue), sync_controller_(sync_controller), 
          d3d11_device_(d3d11_device), hwnd_(hwnd) {}
    
    bool initialize() override {
        // 获取设备上下文
        d3d11_device_->GetImmediateContext(&d3d11_context_);
        
        // 创建交换链
        DXGI_SWAP_CHAIN_DESC swap_chain_desc = {};
        swap_chain_desc.BufferCount = 2;
        swap_chain_desc.BufferDesc.Width = 0;
        swap_chain_desc.BufferDesc.Height = 0;
        swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        swap_chain_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.OutputWindow = hwnd_;
        swap_chain_desc.SampleDesc.Count = 1;
        swap_chain_desc.Windowed = TRUE;
        swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        
        IDXGIDevice* dxgi_device = nullptr;
        d3d11_device_->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgi_device);
        
        IDXGIAdapter* dxgi_adapter = nullptr;
        dxgi_device->GetAdapter(&dxgi_adapter);
        
        IDXGIFactory* dxgi_factory = nullptr;
        dxgi_adapter->GetParent(__uuidof(IDXGIFactory), (void**)&dxgi_factory);
        
        HRESULT hr = dxgi_factory->CreateSwapChain(d3d11_device_, &swap_chain_desc, &swap_chain_);
        
        // 清理临时对象
        dxgi_factory->Release();
        dxgi_adapter->Release();
        dxgi_device->Release();
        
        if (FAILED(hr)) {
            return false;
        }
        
        // 创建渲染目标视图
        ID3D11Texture2D* back_buffer = nullptr;
        swap_chain_->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&back_buffer);
        d3d11_device_->CreateRenderTargetView(back_buffer, nullptr, &render_target_view_);
        back_buffer->Release();
        
        return true;
    }
    
    void start() override {
        running_ = true;
        render_thread_ = std::thread(&VideoSink::render_loop, this);
    }
    
    void stop() override {
        running_ = false;
        if (render_thread_.joinable()) {
            render_thread_.join();
        }
    }
    
    void cleanup() override {
        if (render_target_view_) {
            render_target_view_->Release();
        }
        if (swap_chain_) {
            swap_chain_->Release();
        }
        if (d3d11_context_) {
            d3d11_context_->Release();
        }
    }
    
private:
    void render_loop() {
        while (running_) {
            VideoFrame frame;
            if (!input_queue_->pop(frame, 100)) {
                continue;
            }
            
            // 检查同步
            if (sync_controller_->should_drop_video_frame(frame.pts)) {
                av_frame_free(&frame.frame);
                continue;
            }
            
            int delay = sync_controller_->calculate_video_delay(frame.pts);
            if (delay > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            }
            
            // 渲染帧到后缓冲区
            render_frame(frame);
            
            // 更新同步时钟
            sync_controller_->update_video_clock(frame.pts);
            
            // 释放帧
            av_frame_free(&frame.frame);
            
            // 交换缓冲区
            swap_chain_->Present(1, 0);
        }
    }
    
    void render_frame(const VideoFrame& frame) {
        // 清除渲染目标
        float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        d3d11_context_->ClearRenderTargetView(render_target_view_, clear_color);
        
        // 设置渲染目标
        d3d11_context_->OMSetRenderTargets(1, &render_target_view_, nullptr);
        
        // 这里需要实现具体的视频帧渲染逻辑
        // 如果是硬件解码帧，可以直接使用D3D11纹理
        // 如果是软件解码帧，需要先上传到GPU纹理
        if (frame.is_hw_frame) {
            render_hw_frame(frame.frame);
        } else {
            render_sw_frame(frame.frame);
        }
    }
    
    void render_hw_frame(AVFrame* frame) {
        // 从硬件帧获取D3D11纹理并渲染
        // 这里需要实现具体的硬件帧渲染逻辑
    }
    
    void render_sw_frame(AVFrame* frame) {
        // 将软件帧上传到GPU纹理并渲染
        // 这里需要实现具体的软件帧渲染逻辑
    }
};

// 10. 音频播放器 - WASAPI
class AudioSink : public IDataStreamComponent {
private:
    ThreadSafeQueue<AudioFrame>* input_queue_;
    AVSyncController* sync_controller_;
    
    // WASAPI相关成员
    IMMDeviceEnumerator* device_enumerator_ = nullptr;
    IMMDevice* audio_device_ = nullptr;
    IAudioClient* audio_client_ = nullptr;
    IAudioRenderClient* render_client_ = nullptr;
    
    WAVEFORMATEX* wave_format_ = nullptr;
    UINT32 buffer_frame_count_ = 0;
    
    std::thread audio_thread_;
    bool running_ = false;
    
public:
    AudioSink(ThreadSafeQueue<AudioFrame>* input_queue,
              AVSyncController* sync_controller)
        : input_queue_(input_queue), sync_controller_(sync_controller) {}
    
    bool initialize() override {
        // 初始化COM
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        
        // 创建设备枚举器
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                                    __uuidof(IMMDeviceEnumerator), (void**)&device_enumerator_);
        if (FAILED(hr)) return false;
        
        // 获取默认音频设备
        hr = device_enumerator_->GetDefaultAudioEndpoint(eRender, eConsole, &audio_device_);
        if (FAILED(hr)) return false;
        
        // 激活音频客户端
        hr = audio_device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audio_client_);
        if (FAILED(hr)) return false;
        
        // 获取混音格式
        hr = audio_client_->GetMixFormat(&wave_format_);
        if (FAILED(hr)) return false;
        
        // 初始化音频客户端
        hr = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, wave_format_, nullptr);
        if (FAILED(hr)) return false;
        
        // 获取缓冲区大小
        hr = audio_client_->GetBufferSize(&buffer_frame_count_);
        if (FAILED(hr)) return false;
        
        // 获取渲染客户端
        hr = audio_client_->GetService(__uuidof(IAudioRenderClient), (void**)&render_client_);
        if (FAILED(hr)) return false;
        
        return true;
    }
    
    void start() override {
        running_ = true;
        audio_thread_ = std::thread(&AudioSink::audio_loop, this);
        audio_client_->Start();
    }
    
    void stop() override {
        running_ = false;
        if (audio_client_) {
            audio_client_->Stop();
        }
        if (audio_thread_.joinable()) {
            audio_thread_.join();
        }
    }
    
    void cleanup() override {
        if (render_client_) {
            render_client_->Release();
        }
        if (audio_client_) {
            audio_client_->Release();
        }
        if (audio_device_) {
            audio_device_->Release();
        }
        if (device_enumerator_) {
            device_enumerator_->Release();
        }
        if (wave_format_) {
            CoTaskMemFree(wave_format_);
        }
        CoUninitialize();
    }
    
private:
    void audio_loop() {
        while (running_) {
            AudioFrame frame;
            if (!input_queue_->pop(frame, 100)) {
                continue;
            }
            
            // 写入音频数据到播放缓冲区
            write_audio_data(frame);
            
            // 更新音频时钟
            sync_controller_->update_audio_clock(frame.pts);
            
            // 释放帧
            av_frame_free(&frame.frame);
        }
    }
    
    void write_audio_data(const AudioFrame& frame) {
        UINT32 padding_frames = 0;
        audio_client_->GetCurrentPadding(&padding_frames);
        
        UINT32 available_frames = buffer_frame_count_ - padding_frames;
        if (available_frames == 0) {
            return;
        }
        
        BYTE* buffer_data = nullptr;
        HRESULT hr = render_client_->GetBuffer(available_frames, &buffer_data);
        if (FAILED(hr)) return;
        
        // 这里需要实现音频格式转换和数据拷贝
        // 从AVFrame格式转换为WASAPI期望的格式
        convert_and_copy_audio_data(frame.frame, buffer_data, available_frames);
        
        render_client_->ReleaseBuffer(available_frames, 0);
    }
    
    void convert_and_copy_audio_data(AVFrame* frame, BYTE* buffer, UINT32 frame_count) {
        // 实现音频格式转换和数据拷贝
        // 这里需要根据具体的音频格式进行转换
        // 可能需要使用libswresample进行重采样
    }
};

} // namespace MKVPlayer
