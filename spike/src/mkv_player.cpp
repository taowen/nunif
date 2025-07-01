#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <audiopolicy.h>
#include <avrt.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <chrono>
#include <iostream>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "avrt.lib")

// 简单的顶点着色器
const char* vertexShaderSource = R"(
struct VSInput {
    float2 position : POSITION;
    float2 texCoord : TEXCOORD;
};

struct VSOutput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0f, 1.0f);
    output.texCoord = input.texCoord;
    return output;
}
)";

// 简单的像素着色器
const char* pixelShaderSource = R"(
Texture2D videoTexture : register(t0);
SamplerState textureSampler : register(s0);

struct PSInput {
    float4 position : SV_POSITION;
    float2 texCoord : TEXCOORD;
};

float4 main(PSInput input) : SV_TARGET {
    return videoTexture.Sample(textureSampler, input.texCoord);
}
)";

struct Vertex {
    float position[2];
    float texCoord[2];
};

class MKVPlayer {
private:
    // FFmpeg 相关
    AVFormatContext* formatContext;
    AVCodecContext* audioCodecContext;
    SwsContext* swsContext;
    SwrContext* swrContext;
    int videoStreamIndex;
    int audioStreamIndex;
    
    // DirectX11 相关
    ID3D11Device* device;
    ID3D11DeviceContext* deviceContext;
    IDXGISwapChain* swapChain;
    ID3D11RenderTargetView* renderTargetView;
    ID3D11VertexShader* vertexShader;
    ID3D11PixelShader* pixelShader;
    ID3D11Buffer* vertexBuffer;
    ID3D11InputLayout* inputLayout;
    ID3D11Texture2D* videoTexture;
    ID3D11ShaderResourceView* videoSRV;
    ID3D11SamplerState* samplerState;
    
    // 音频相关
    IMMDeviceEnumerator* deviceEnumerator;
    IMMDevice* audioDevice;
    IAudioClient* audioClient;
    IAudioRenderClient* renderClient;
    UINT32 bufferFrameCount;
    WAVEFORMATEX* audioFormat;
    
    // 窗口相关
    HWND hwnd;
    int windowWidth;
    int windowHeight;
    
    // 播放控制
    std::atomic<bool> shouldStop;
    std::thread decodingThread;
    std::thread audioThread;
    
    // 同步相关
    std::chrono::high_resolution_clock::time_point startTime;
    double videoTimeBase;
    double audioTimeBase;
    
    // 帧队列
    std::queue<AVFrame*> videoFrameQueue;
    std::queue<AVFrame*> audioFrameQueue;
    std::mutex videoQueueMutex;
    std::mutex audioQueueMutex;
    const size_t maxQueueSize = 10;

public:
    // Public members for external access
    std::atomic<bool> playing;
    AVCodecContext* videoCodecContext;
    
    MKVPlayer() : 
        formatContext(nullptr), videoCodecContext(nullptr), audioCodecContext(nullptr),
        swsContext(nullptr), swrContext(nullptr), videoStreamIndex(-1), audioStreamIndex(-1),
        device(nullptr), deviceContext(nullptr), swapChain(nullptr), renderTargetView(nullptr),
        vertexShader(nullptr), pixelShader(nullptr), vertexBuffer(nullptr), inputLayout(nullptr),
        videoTexture(nullptr), videoSRV(nullptr), samplerState(nullptr),
        deviceEnumerator(nullptr), audioDevice(nullptr), audioClient(nullptr), renderClient(nullptr),
        bufferFrameCount(0), audioFormat(nullptr), hwnd(nullptr), windowWidth(800), windowHeight(600),
        playing(false), shouldStop(false), videoTimeBase(0), audioTimeBase(0) {
    }
    
    ~MKVPlayer() {
        cleanup();
    }
    
    bool initialize(const char* filename, HWND window) {
        hwnd = window;
        
        // 初始化 FFmpeg
        if (!initializeFFmpeg(filename)) {
            std::cerr << "Failed to initialize FFmpeg" << std::endl;
            return false;
        }
        
        // 初始化 DirectX11
        if (!initializeDirectX11()) {
            std::cerr << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        // 初始化音频
        if (!initializeAudio()) {
            std::cerr << "Failed to initialize audio" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool initializeFFmpeg(const char* filename) {
        // 打开文件
        if (avformat_open_input(&formatContext, filename, nullptr, nullptr) < 0) {
            return false;
        }
        
        // 获取流信息
        if (avformat_find_stream_info(formatContext, nullptr) < 0) {
            return false;
        }
        
        // 查找视频和音频流
        for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
            if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO && videoStreamIndex == -1) {
                videoStreamIndex = i;
            } else if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO && audioStreamIndex == -1) {
                audioStreamIndex = i;
            }
        }
        
        // 初始化视频解码器
        if (videoStreamIndex >= 0) {
            AVStream* videoStream = formatContext->streams[videoStreamIndex];
            const AVCodec* videoCodec = avcodec_find_decoder(videoStream->codecpar->codec_id);
            if (!videoCodec) return false;
            
            videoCodecContext = avcodec_alloc_context3(videoCodec);
            if (!videoCodecContext) return false;
            
            if (avcodec_parameters_to_context(videoCodecContext, videoStream->codecpar) < 0) return false;
            if (avcodec_open2(videoCodecContext, videoCodec, nullptr) < 0) return false;
            
            videoTimeBase = av_q2d(videoStream->time_base);
            
            // 初始化 swscale
            swsContext = sws_getContext(
                videoCodecContext->width, videoCodecContext->height, videoCodecContext->pix_fmt,
                videoCodecContext->width, videoCodecContext->height, AV_PIX_FMT_RGBA,
                SWS_BILINEAR, nullptr, nullptr, nullptr
            );
        }
        
        // 初始化音频解码器
        if (audioStreamIndex >= 0) {
            AVStream* audioStream = formatContext->streams[audioStreamIndex];
            const AVCodec* audioCodec = avcodec_find_decoder(audioStream->codecpar->codec_id);
            if (!audioCodec) return false;
            
            audioCodecContext = avcodec_alloc_context3(audioCodec);
            if (!audioCodecContext) return false;
            
            if (avcodec_parameters_to_context(audioCodecContext, audioStream->codecpar) < 0) return false;
            if (avcodec_open2(audioCodecContext, audioCodec, nullptr) < 0) return false;
            
            audioTimeBase = av_q2d(audioStream->time_base);
            
            // 初始化 swresample - 使用兼容的方法
            swrContext = swr_alloc();
            if (!swrContext) return false;
            
            // 创建输出通道布局
            AVChannelLayout out_ch_layout = AV_CHANNEL_LAYOUT_STEREO;
            
            // 设置输出格式
            av_opt_set_chlayout(swrContext, "out_chlayout", &out_ch_layout, 0);
            av_opt_set_int(swrContext, "out_sample_rate", 48000, 0);
            av_opt_set_sample_fmt(swrContext, "out_sample_fmt", AV_SAMPLE_FMT_S16, 0);
            
            // 设置输入格式
            av_opt_set_chlayout(swrContext, "in_chlayout", &audioCodecContext->ch_layout, 0);
            av_opt_set_int(swrContext, "in_sample_rate", audioCodecContext->sample_rate, 0);
            av_opt_set_sample_fmt(swrContext, "in_sample_fmt", audioCodecContext->sample_fmt, 0);
            
            if (swr_init(swrContext) < 0) return false;
        }
        
        return true;
    }
    
    bool initializeDirectX11() {
        // Get screen dimensions for fullscreen
        windowWidth = GetSystemMetrics(SM_CXSCREEN);
        windowHeight = GetSystemMetrics(SM_CYSCREEN);
        
        // 创建设备和交换链
        DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
        swapChainDesc.BufferCount = 1;
        swapChainDesc.BufferDesc.Width = windowWidth;
        swapChainDesc.BufferDesc.Height = windowHeight;
        swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.OutputWindow = hwnd;
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.Windowed = TRUE;
        
        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            nullptr, 0, D3D11_SDK_VERSION,
            &swapChainDesc, &swapChain, &device, &featureLevel, &deviceContext
        );
        
        if (FAILED(hr)) return false;
        
        // 创建渲染目标视图
        ID3D11Texture2D* backBuffer;
        swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
        device->CreateRenderTargetView(backBuffer, nullptr, &renderTargetView);
        backBuffer->Release();
        
        // 创建着色器
        if (!createShaders()) return false;
        
        // 创建顶点缓冲区
        if (!createVertexBuffer()) return false;
        
        // 创建采样器状态
        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        device->CreateSamplerState(&samplerDesc, &samplerState);
        
        return true;
    }
    
    bool createShaders() {
        ID3DBlob* vsBlob = nullptr;
        ID3DBlob* psBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;
        
        // 编译顶点着色器
        HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr,
            nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) errorBlob->Release();
            return false;
        }
        
        // 编译像素着色器
        hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr,
            nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) errorBlob->Release();
            vsBlob->Release();
            return false;
        }
        
        // 创建着色器
        device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vertexShader);
        device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pixelShader);
        
        // 创建输入布局
        D3D11_INPUT_ELEMENT_DESC inputElements[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8, D3D11_INPUT_PER_VERTEX_DATA, 0}
        };
        
        device->CreateInputLayout(inputElements, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &inputLayout);
        
        vsBlob->Release();
        psBlob->Release();
        return true;
    }
    
    bool createVertexBuffer() {
        Vertex vertices[] = {
            {{-1.0f, -1.0f}, {0.0f, 1.0f}},  // 左下
            {{-1.0f,  1.0f}, {0.0f, 0.0f}},  // 左上
            {{ 1.0f, -1.0f}, {1.0f, 1.0f}},  // 右下
            {{ 1.0f,  1.0f}, {1.0f, 0.0f}}   // 右上
        };
        
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.ByteWidth = sizeof(vertices);
        bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        
        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = vertices;
        
        return SUCCEEDED(device->CreateBuffer(&bufferDesc, &initData, &vertexBuffer));
    }
    
    bool initializeAudio() {
        CoInitialize(nullptr);
        
        // 创建设备枚举器
        HRESULT hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
            __uuidof(IMMDeviceEnumerator), (void**)&deviceEnumerator);
        if (FAILED(hr)) return false;
        
        // 获取默认音频设备
        hr = deviceEnumerator->GetDefaultAudioEndpoint(eRender, eConsole, &audioDevice);
        if (FAILED(hr)) return false;
        
        // 激活音频客户端
        hr = audioDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&audioClient);
        if (FAILED(hr)) return false;
        
        // 设置音频格式
        WAVEFORMATEX format = {};
        format.wFormatTag = WAVE_FORMAT_PCM;
        format.nChannels = 2;
        format.nSamplesPerSec = 48000;
        format.wBitsPerSample = 16;
        format.nBlockAlign = (format.nChannels * format.wBitsPerSample) / 8;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;
        
        // 初始化音频客户端
        hr = audioClient->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10000000, 0, &format, nullptr);
        if (FAILED(hr)) return false;
        
        // 获取缓冲区大小
        audioClient->GetBufferSize(&bufferFrameCount);
        
        // 获取渲染客户端
        hr = audioClient->GetService(__uuidof(IAudioRenderClient), (void**)&renderClient);
        if (FAILED(hr)) return false;
        
        return true;
    }
    
    void play() {
        if (playing) return;
        
        playing = true;
        shouldStop = false;
        startTime = std::chrono::high_resolution_clock::now();
        
        // 启动解码线程
        decodingThread = std::thread(&MKVPlayer::decodingLoop, this);
        
        // 启动音频线程
        if (audioStreamIndex >= 0) {
            audioThread = std::thread(&MKVPlayer::audioLoop, this);
            audioClient->Start();
        }
        
        // 启动渲染循环
        renderLoop();
    }
    
    void stop() {
        shouldStop = true;
        playing = false;
        
        if (decodingThread.joinable()) {
            decodingThread.join();
        }
        
        if (audioThread.joinable()) {
            audioThread.join();
        }
        
        if (audioClient) {
            audioClient->Stop();
        }
    }
    
    void decodingLoop() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        
        while (!shouldStop && av_read_frame(formatContext, packet) >= 0) {
            if (packet->stream_index == videoStreamIndex) {
                if (avcodec_send_packet(videoCodecContext, packet) == 0) {
                    while (avcodec_receive_frame(videoCodecContext, frame) == 0) {
                        std::lock_guard<std::mutex> lock(videoQueueMutex);
                        if (videoFrameQueue.size() < maxQueueSize) {
                            AVFrame* clonedFrame = av_frame_clone(frame);
                            videoFrameQueue.push(clonedFrame);
                        }
                    }
                }
            } else if (packet->stream_index == audioStreamIndex) {
                if (avcodec_send_packet(audioCodecContext, packet) == 0) {
                    while (avcodec_receive_frame(audioCodecContext, frame) == 0) {
                        std::lock_guard<std::mutex> lock(audioQueueMutex);
                        if (audioFrameQueue.size() < maxQueueSize) {
                            AVFrame* clonedFrame = av_frame_clone(frame);
                            audioFrameQueue.push(clonedFrame);
                        }
                    }
                }
            }
            
            av_packet_unref(packet);
            
            // 简单的队列大小控制
            if (videoFrameQueue.size() >= maxQueueSize && audioFrameQueue.size() >= maxQueueSize) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        av_packet_free(&packet);
        av_frame_free(&frame);
    }
    
    void audioLoop() {
        BYTE* audioBuffer;
        UINT32 numFramesPadding;
        
        while (!shouldStop) {
            audioClient->GetCurrentPadding(&numFramesPadding);
            UINT32 numFramesAvailable = bufferFrameCount - numFramesPadding;
            
            if (numFramesAvailable > 0) {
                std::lock_guard<std::mutex> lock(audioQueueMutex);
                if (!audioFrameQueue.empty()) {
                    AVFrame* frame = audioFrameQueue.front();
                    audioFrameQueue.pop();
                    
                    if (SUCCEEDED(renderClient->GetBuffer(numFramesAvailable, &audioBuffer))) {
                        // 重采样音频数据
                        int outputSamples = swr_convert(swrContext,
                            &audioBuffer, numFramesAvailable,
                            (const uint8_t**)frame->data, frame->nb_samples);
                        
                        renderClient->ReleaseBuffer(outputSamples, 0);
                    }
                    
                    av_frame_free(&frame);
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void renderLoop() {
        while (playing && !shouldStop) {
            renderFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        }
    }
    
    void renderFrame() {
        // 获取当前时间
        auto currentTime = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - startTime);
        double currentSeconds = elapsed.count() / 1000000.0;
        
        // 从队列中获取视频帧
        AVFrame* frame = nullptr;
        {
            std::lock_guard<std::mutex> lock(videoQueueMutex);
            while (!videoFrameQueue.empty()) {
                AVFrame* candidate = videoFrameQueue.front();
                double frameTime = candidate->pts * videoTimeBase;
                
                if (frameTime <= currentSeconds + 0.04) { // 40ms 容差
                    videoFrameQueue.pop();
                    if (frame) av_frame_free(&frame);
                    frame = candidate;
                } else {
                    break;
                }
            }
        }
        
        if (frame) {
            updateVideoTexture(frame);
            av_frame_free(&frame);
        }
        
        // 渲染
        float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
        deviceContext->ClearRenderTargetView(renderTargetView, clearColor);
        
        deviceContext->OMSetRenderTargets(1, &renderTargetView, nullptr);
        
        D3D11_VIEWPORT viewport = {};
        viewport.Width = static_cast<float>(windowWidth);
        viewport.Height = static_cast<float>(windowHeight);
        viewport.MaxDepth = 1.0f;
        deviceContext->RSSetViewports(1, &viewport);
        
        deviceContext->IASetInputLayout(inputLayout);
        deviceContext->VSSetShader(vertexShader, nullptr, 0);
        deviceContext->PSSetShader(pixelShader, nullptr, 0);
        
        if (videoSRV) {
            deviceContext->PSSetShaderResources(0, 1, &videoSRV);
            deviceContext->PSSetSamplers(0, 1, &samplerState);
        }
        
        UINT stride = sizeof(Vertex);
        UINT offset = 0;
        deviceContext->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
        deviceContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        
        deviceContext->Draw(4, 0);
        
        swapChain->Present(1, 0);
    }
    
    void updateVideoTexture(AVFrame* frame) {
        if (!videoTexture || !swsContext) return;
        
        // 创建临时 RGBA 缓冲区
        int width = videoCodecContext->width;
        int height = videoCodecContext->height;
        std::vector<uint8_t> rgbaBuffer(width * height * 4);
        
        uint8_t* rgbaData[1] = { rgbaBuffer.data() };
        int rgbaLinesize[1] = { width * 4 };
        
        // 转换为 RGBA
        sws_scale(swsContext, frame->data, frame->linesize, 0, height, rgbaData, rgbaLinesize);
        
        // 更新纹理
        D3D11_MAPPED_SUBRESOURCE mappedResource;
        if (SUCCEEDED(deviceContext->Map(videoTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &mappedResource))) {
            uint8_t* dest = static_cast<uint8_t*>(mappedResource.pData);
            uint8_t* src = rgbaBuffer.data();
            
            for (int y = 0; y < height; y++) {
                memcpy(dest + y * mappedResource.RowPitch, src + y * width * 4, width * 4);
            }
            
            deviceContext->Unmap(videoTexture, 0);
        }
    }
    
    void createVideoTexture(int width, int height) {
        if (videoTexture) {
            videoTexture->Release();
            videoTexture = nullptr;
        }
        if (videoSRV) {
            videoSRV->Release();
            videoSRV = nullptr;
        }
        
        D3D11_TEXTURE2D_DESC textureDesc = {};
        textureDesc.Width = width;
        textureDesc.Height = height;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.Usage = D3D11_USAGE_DYNAMIC;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        
        device->CreateTexture2D(&textureDesc, nullptr, &videoTexture);
        device->CreateShaderResourceView(videoTexture, nullptr, &videoSRV);
    }
    
    void cleanup() {
        stop();
        
        // 清理队列
        {
            std::lock_guard<std::mutex> lock(videoQueueMutex);
            while (!videoFrameQueue.empty()) {
                AVFrame* frame = videoFrameQueue.front();
                videoFrameQueue.pop();
                av_frame_free(&frame);
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(audioQueueMutex);
            while (!audioFrameQueue.empty()) {
                AVFrame* frame = audioFrameQueue.front();
                audioFrameQueue.pop();
                av_frame_free(&frame);
            }
        }
        
        // 清理 FFmpeg
        if (swsContext) sws_freeContext(swsContext);
        if (swrContext) swr_free(&swrContext);
        if (videoCodecContext) avcodec_free_context(&videoCodecContext);
        if (audioCodecContext) avcodec_free_context(&audioCodecContext);
        if (formatContext) avformat_close_input(&formatContext);
        
        // 清理 DirectX11
        if (samplerState) samplerState->Release();
        if (videoSRV) videoSRV->Release();
        if (videoTexture) videoTexture->Release();
        if (inputLayout) inputLayout->Release();
        if (vertexBuffer) vertexBuffer->Release();
        if (pixelShader) pixelShader->Release();
        if (vertexShader) vertexShader->Release();
        if (renderTargetView) renderTargetView->Release();
        if (swapChain) swapChain->Release();
        if (deviceContext) deviceContext->Release();
        if (device) device->Release();
        
        // 清理音频
        if (renderClient) renderClient->Release();
        if (audioClient) audioClient->Release();
        if (audioDevice) audioDevice->Release();
        if (deviceEnumerator) deviceEnumerator->Release();
        
        CoUninitialize();
    }
};

// 窗口过程
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    static MKVPlayer* player = nullptr;
    
    switch (uMsg) {
    case WM_CREATE:
        player = new MKVPlayer();
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(player));
        break;
        
    case WM_DESTROY:
        if (player) {
            delete player;
            player = nullptr;
        }
        PostQuitMessage(0);
        break;
        
    case WM_KEYDOWN:
        if (wParam == VK_SPACE && player) {
            // 空格键暂停/播放（简单实现）
            if (player->playing) {
                player->stop();
            }
        }
        break;
        
    default:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <mkv_file>" << std::endl;
        return -1;
    }
    
    // Get screen dimensions
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    
    // 注册窗口类 - 使用 ANSI 字符串
    WNDCLASSA wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = "MKVPlayer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    
    RegisterClassA(&wc);
    
    // 创建全屏窗口 - 使用 ANSI 版本
    HWND hwnd = CreateWindowExA(
        WS_EX_TOPMOST, "MKVPlayer", "MKV Player",
        WS_POPUP | WS_VISIBLE,
        0, 0, screenWidth, screenHeight,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );
    
    if (!hwnd) {
        std::cerr << "Failed to create window" << std::endl;
        return -1;
    }
    
    ShowWindow(hwnd, SW_MAXIMIZE);
    UpdateWindow(hwnd);
    
    // 获取播放器实例
    MKVPlayer* player = reinterpret_cast<MKVPlayer*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
    
    // 初始化播放器
    if (player && player->initialize(argv[1], hwnd)) {
        // 创建视频纹理
        if (player->videoCodecContext) {
            player->createVideoTexture(player->videoCodecContext->width, player->videoCodecContext->height);
        }
        
        // 开始播放
        std::thread playThread([player]() {
            player->play();
        });
        
        // 消息循环
        MSG msg = {};
        while (GetMessage(&msg, nullptr, 0, 0)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        
        playThread.join();
    } else {
        std::cerr << "Failed to initialize player" << std::endl;
        MessageBoxA(hwnd, "Failed to initialize player", "Error", MB_OK | MB_ICONERROR);
    }
    
    return 0;
}
