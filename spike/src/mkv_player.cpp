#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <iostream>
#include <string>
#include <memory>
#include <chrono>
#include <thread>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_d3d11va.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

class MKVPlayer {
private:
    // DirectX11 resources
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGISwapChain* swapChain = nullptr;
    ID3D11RenderTargetView* renderTargetView = nullptr;
    ID3D11Texture2D* backBuffer = nullptr;
    ID3D11VertexShader* vertexShader = nullptr;
    ID3D11PixelShader* pixelShader = nullptr;
    ID3D11Buffer* vertexBuffer = nullptr;
    ID3D11InputLayout* inputLayout = nullptr;
    ID3D11SamplerState* samplerState = nullptr;
    ID3D11ShaderResourceView* textureView = nullptr;
    ID3D11Texture2D* videoTexture = nullptr;
    
    // FFmpeg resources
    AVFormatContext* formatContext = nullptr;
    AVCodecContext* codecContext = nullptr;
    const AVCodec* codec = nullptr;
    AVStream* videoStream = nullptr;
    AVBufferRef* hwDeviceRef = nullptr;
    int videoStreamIndex = -1;
    
    // Window and timing
    HWND hwnd = nullptr;
    int windowWidth = 1280;
    int windowHeight = 720;
    int videoWidth = 0;
    int videoHeight = 0;
    
    // Vertex structure for rendering
    struct Vertex {
        float pos[3];
        float tex[2];
    };

public:
    bool Initialize(const std::string& filename, HWND window) {
        hwnd = window;
        
        if (!InitializeDirectX11()) {
            std::cerr << "Failed to initialize DirectX11" << std::endl;
            return false;
        }
        
        if (!InitializeFFmpeg(filename)) {
            std::cerr << "Failed to initialize FFmpeg" << std::endl;
            return false;
        }
        
        if (!CreateShaders()) {
            std::cerr << "Failed to create shaders" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool InitializeDirectX11() {
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
            nullptr,
            D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            nullptr,
            0,
            D3D11_SDK_VERSION,
            &swapChainDesc,
            &swapChain,
            &device,
            &featureLevel,
            &context
        );
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create D3D11 device and swap chain" << std::endl;
            return false;
        }
        
        // Create render target view
        hr = swapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (void**)&backBuffer);
        if (FAILED(hr)) return false;
        
        hr = device->CreateRenderTargetView(backBuffer, nullptr, &renderTargetView);
        if (FAILED(hr)) return false;
        
        context->OMSetRenderTargets(1, &renderTargetView, nullptr);
        
        // Set viewport
        D3D11_VIEWPORT viewport = {};
        viewport.Width = static_cast<float>(windowWidth);
        viewport.Height = static_cast<float>(windowHeight);
        viewport.MinDepth = 0.0f;
        viewport.MaxDepth = 1.0f;
        context->RSSetViewports(1, &viewport);
        
        return true;
    }
    
    bool InitializeFFmpeg(const std::string& filename) {
        // Initialize hardware device context
        int ret = av_hwdevice_ctx_create(&hwDeviceRef, AV_HWDEVICE_TYPE_D3D11VA, nullptr, nullptr, 0);
        if (ret < 0) {
            std::cerr << "Failed to create D3D11VA device context" << std::endl;
            return false;
        }
        
        // Open input file
        ret = avformat_open_input(&formatContext, filename.c_str(), nullptr, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open input file" << std::endl;
            return false;
        }
        
        // Find stream info
        ret = avformat_find_stream_info(formatContext, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to find stream info" << std::endl;
            return false;
        }
        
        // Find video stream
        videoStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
        if (videoStreamIndex < 0) {
            std::cerr << "Failed to find video stream" << std::endl;
            return false;
        }
        
        videoStream = formatContext->streams[videoStreamIndex];
        
        // Create codec context
        codecContext = avcodec_alloc_context3(codec);
        if (!codecContext) {
            std::cerr << "Failed to allocate codec context" << std::endl;
            return false;
        }
        
        ret = avcodec_parameters_to_context(codecContext, videoStream->codecpar);
        if (ret < 0) {
            std::cerr << "Failed to copy codec parameters" << std::endl;
            return false;
        }
        
        // Set hardware device context
        codecContext->hw_device_ctx = av_buffer_ref(hwDeviceRef);
        
        // Open codec
        ret = avcodec_open2(codecContext, codec, nullptr);
        if (ret < 0) {
            std::cerr << "Failed to open codec" << std::endl;
            return false;
        }
        
        videoWidth = codecContext->width;
        videoHeight = codecContext->height;
        
        std::cout << "Video: " << videoWidth << "x" << videoHeight << std::endl;
        
        return true;
    }
    
    bool CreateShaders() {
        // Simple vertex shader
        const char* vertexShaderSource = R"(
            struct VSInput {
                float3 pos : POSITION;
                float2 tex : TEXCOORD0;
            };
            
            struct VSOutput {
                float4 pos : SV_POSITION;
                float2 tex : TEXCOORD0;
            };
            
            VSOutput main(VSInput input) {
                VSOutput output;
                output.pos = float4(input.pos, 1.0f);
                output.tex = input.tex;
                return output;
            }
        )";
        
        // Simple pixel shader
        const char* pixelShaderSource = R"(
            Texture2D videoTexture : register(t0);
            SamplerState videoSampler : register(s0);
            
            struct PSInput {
                float4 pos : SV_POSITION;
                float2 tex : TEXCOORD0;
            };
            
            float4 main(PSInput input) : SV_TARGET {
                return videoTexture.Sample(videoSampler, input.tex);
            }
        )";
        
        ID3DBlob* vsBlob = nullptr;
        ID3DBlob* psBlob = nullptr;
        ID3DBlob* errorBlob = nullptr;
        
        // Compile vertex shader
        HRESULT hr = D3DCompile(vertexShaderSource, strlen(vertexShaderSource), nullptr, nullptr, nullptr,
                               "main", "vs_5_0", 0, 0, &vsBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "Vertex shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
                errorBlob->Release();
            }
            return false;
        }
        
        // Compile pixel shader
        hr = D3DCompile(pixelShaderSource, strlen(pixelShaderSource), nullptr, nullptr, nullptr,
                       "main", "ps_5_0", 0, 0, &psBlob, &errorBlob);
        if (FAILED(hr)) {
            if (errorBlob) {
                std::cerr << "Pixel shader compilation error: " << (char*)errorBlob->GetBufferPointer() << std::endl;
                errorBlob->Release();
            }
            vsBlob->Release();
            return false;
        }
        
        // Create shaders
        hr = device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vertexShader);
        if (FAILED(hr)) {
            vsBlob->Release();
            psBlob->Release();
            return false;
        }
        
        hr = device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &pixelShader);
        if (FAILED(hr)) {
            vsBlob->Release();
            psBlob->Release();
            return false;
        }
        
        // Create input layout
        D3D11_INPUT_ELEMENT_DESC inputElementDesc[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0}
        };
        
        hr = device->CreateInputLayout(inputElementDesc, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &inputLayout);
        
        vsBlob->Release();
        psBlob->Release();
        
        if (FAILED(hr)) {
            return false;
        }
        
        // Create vertex buffer
        Vertex vertices[] = {
            {{-1.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
            {{-1.0f,  1.0f, 0.0f}, {0.0f, 0.0f}},
            {{ 1.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
            {{ 1.0f,  1.0f, 0.0f}, {1.0f, 0.0f}}
        };
        
        D3D11_BUFFER_DESC bufferDesc = {};
        bufferDesc.Usage = D3D11_USAGE_DEFAULT;
        bufferDesc.ByteWidth = sizeof(vertices);
        bufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        
        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = vertices;
        
        hr = device->CreateBuffer(&bufferDesc, &initData, &vertexBuffer);
        if (FAILED(hr)) {
            return false;
        }
        
        // Create sampler state
        D3D11_SAMPLER_DESC samplerDesc = {};
        samplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        samplerDesc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        samplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        samplerDesc.MinLOD = 0;
        samplerDesc.MaxLOD = D3D11_FLOAT32_MAX;
        
        hr = device->CreateSamplerState(&samplerDesc, &samplerState);
        if (FAILED(hr)) {
            return false;
        }
        
        return true;
    }
    
    void PlayLoop() {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();
        
        auto lastTime = std::chrono::high_resolution_clock::now();
        
        while (true) {
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    av_packet_free(&packet);
                    av_frame_free(&frame);
                    return;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            
            int ret = av_read_frame(formatContext, packet);
            if (ret < 0) {
                // End of file or error, restart
                av_seek_frame(formatContext, videoStreamIndex, 0, AVSEEK_FLAG_BACKWARD);
                continue;
            }
            
            if (packet->stream_index != videoStreamIndex) {
                av_packet_unref(packet);
                continue;
            }
            
            ret = avcodec_send_packet(codecContext, packet);
            av_packet_unref(packet);
            
            if (ret < 0) {
                std::cerr << "Error sending packet to decoder" << std::endl;
                continue;
            }
            
            while (ret >= 0) {
                ret = avcodec_receive_frame(codecContext, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                    break;
                }
                if (ret < 0) {
                    std::cerr << "Error receiving frame from decoder" << std::endl;
                    break;
                }
                
                // Render frame
                if (!RenderFrame(frame)) {
                    std::cerr << "Failed to render frame" << std::endl;
                }
                
                av_frame_unref(frame);
                
                // Simple timing control
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime);
                
                // Assuming 30 FPS for simplicity
                if (timeDiff.count() < 33) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(33 - timeDiff.count()));
                }
                lastTime = std::chrono::high_resolution_clock::now();
            }
        }
        
        av_packet_free(&packet);
        av_frame_free(&frame);
    }
    
    bool RenderFrame(AVFrame* frame) {
        if (!frame || frame->format != AV_PIX_FMT_D3D11) {
            std::cerr << "Frame is not in D3D11 format" << std::endl;
            return false;
        }
        
        // Get D3D11 texture from frame
        ID3D11Texture2D* frameTexture = (ID3D11Texture2D*)frame->data[0];
        
        // Create shader resource view if needed
        if (!textureView) {
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
            srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            srvDesc.Texture2D.MipLevels = 1;
            
            HRESULT hr = device->CreateShaderResourceView(frameTexture, &srvDesc, &textureView);
            if (FAILED(hr)) {
                std::cerr << "Failed to create shader resource view" << std::endl;
                return false;
            }
        }
        
        // Clear render target
        float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        context->ClearRenderTargetView(renderTargetView, clearColor);
        
        // Set shaders and resources
        context->VSSetShader(vertexShader, nullptr, 0);
        context->PSSetShader(pixelShader, nullptr, 0);
        context->PSSetShaderResources(0, 1, &textureView);
        context->PSSetSamplers(0, 1, &samplerState);
        
        // Set input layout and vertex buffer
        context->IASetInputLayout(inputLayout);
        UINT stride = sizeof(Vertex);
        UINT offset = 0;
        context->IASetVertexBuffers(0, 1, &vertexBuffer, &stride, &offset);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
        
        // Draw
        context->Draw(4, 0);
        
        // Present
        swapChain->Present(1, 0);
        
        return true;
    }
    
    void Cleanup() {
        // Release DirectX resources
        if (textureView) textureView->Release();
        if (videoTexture) videoTexture->Release();
        if (samplerState) samplerState->Release();
        if (inputLayout) inputLayout->Release();
        if (vertexBuffer) vertexBuffer->Release();
        if (pixelShader) pixelShader->Release();
        if (vertexShader) vertexShader->Release();
        if (renderTargetView) renderTargetView->Release();
        if (backBuffer) backBuffer->Release();
        if (swapChain) swapChain->Release();
        if (context) context->Release();
        if (device) device->Release();
        
        // Release FFmpeg resources
        if (codecContext) avcodec_free_context(&codecContext);
        if (formatContext) avformat_close_input(&formatContext);
        if (hwDeviceRef) av_buffer_unref(&hwDeviceRef);
    }
};

// Window procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mkv_file>" << std::endl;
        return 1;
    }
    
    // Register window class - use WNDCLASSW for Unicode
    WNDCLASSW wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.lpszClassName = L"MKVPlayer";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClassW(&wc);
    
    // Create window - use CreateWindowExW for Unicode
    HWND hwnd = CreateWindowExW(
        0,
        L"MKVPlayer",
        L"DirectX11 MKV Player",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 1280, 720,
        nullptr, nullptr, GetModuleHandle(nullptr), nullptr
    );
    
    if (!hwnd) {
        std::cerr << "Failed to create window" << std::endl;
        return 1;
    }
    
    ShowWindow(hwnd, SW_SHOWDEFAULT);
    
    // Initialize and run player
    MKVPlayer player;
    if (!player.Initialize(argv[1], hwnd)) {
        std::cerr << "Failed to initialize player" << std::endl;
        player.Cleanup();
        return 1;
    }
    
    std::cout << "Starting playback..." << std::endl;
    player.PlayLoop();
    
    player.Cleanup();
    return 0;
}
