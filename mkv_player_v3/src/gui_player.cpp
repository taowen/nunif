#include <Windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <string>
#include <iostream>
#include <memory>
#include "rgb_video_decoder.h"
#include "video_player.h"  // For RenderRgbFrame and shader helpers

using Microsoft::WRL::ComPtr;

// Globals
ComPtr<ID3D11Device> g_device;
ComPtr<ID3D11DeviceContext> g_context;
ComPtr<IDXGISwapChain1> g_swapchain;
std::unique_ptr<RgbVideoDecoder> g_decoder;
ComPtr<ID3D11VertexShader> g_vertex_shader;
ComPtr<ID3D11PixelShader> g_pixel_shader;
ComPtr<ID3D11InputLayout> g_input_layout;
ComPtr<ID3D11Buffer> g_vertex_buffer;
ComPtr<ID3D11SamplerState> g_sampler_state;
int g_video_width = 0;
int g_video_height = 0;

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_SIZE:
        // Handle resize: Recreate backbuffer if needed
        return 0;
    }
    return DefWindowProc(hwnd, message, wParam, lParam);
}

bool InitializeDeviceAndSwapchain(HWND hwnd) {
    ComPtr<IDXGIFactory2> factory;
    HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return false;

    UINT flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, &g_device, nullptr, &g_context);
    if (FAILED(hr)) return false;

    DXGI_SWAP_CHAIN_DESC1 swap_desc = {};
    swap_desc.Width = 640;  // Initial size
    swap_desc.Height = 480;
    swap_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    swap_desc.SampleDesc.Count = 1;
    swap_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swap_desc.BufferCount = 2;
    swap_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    hr = factory->CreateSwapChainForHwnd(g_device.Get(), hwnd, &swap_desc, nullptr, nullptr, &g_swapchain);
    if (FAILED(hr)) return false;

    return true;
}

bool InitializeShadersAndQuad() {
    // Copy from VideoPlayer::initializeShaders and createQuad
    const std::string vs_code = R"(
        struct VS_INPUT { float4 pos : POSITION; float2 tex : TEXCOORD0; };
        struct PS_INPUT { float4 pos : SV_POSITION; float2 tex : TEXCOORD0; };
        PS_INPUT main(VS_INPUT input) { PS_INPUT output; output.pos = input.pos; output.tex = input.tex; return output; }
    )";
    
    const std::string ps_code = R"(
        Texture2D tex : register(t0); SamplerState sam : register(s0);
        float4 main(float4 pos : SV_POSITION, float2 texcoord : TEXCOORD0) : SV_TARGET { return tex.Sample(sam, texcoord); }
    )";
    
    ComPtr<ID3DBlob> vs_blob, ps_blob, error_blob;
    HRESULT hr = D3DCompile(vs_code.c_str(), vs_code.length(), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vs_blob, &error_blob);
    if (FAILED(hr)) return false;
    hr = g_device->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &g_vertex_shader);
    if (FAILED(hr)) return false;

    hr = D3DCompile(ps_code.c_str(), ps_code.length(), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &ps_blob, &error_blob);
    if (FAILED(hr)) return false;
    hr = g_device->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr, &g_pixel_shader);
    if (FAILED(hr)) return false;

    D3D11_INPUT_ELEMENT_DESC layout[] = { {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0}, {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0} };
    hr = g_device->CreateInputLayout(layout, ARRAYSIZE(layout), vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), &g_input_layout);
    if (FAILED(hr)) return false;

    D3D11_SAMPLER_DESC sampler_desc = {}; 
    sampler_desc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; 
    sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP; 
    sampler_desc.ComparisonFunc = D3D11_COMPARISON_NEVER; 
    sampler_desc.MinLOD = 0; 
    sampler_desc.MaxLOD = D3D11_FLOAT32_MAX;
    hr = g_device->CreateSamplerState(&sampler_desc, &g_sampler_state);
    if (FAILED(hr)) return false;

    struct Vertex { float x, y, z; float u, v; };
    Vertex vertices[] = { {-1.0f, -1.0f, 0.0f, 0.0f, 1.0f}, {-1.0f, 1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, -1.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f, 1.0f, 0.0f} };
    D3D11_BUFFER_DESC bd = {sizeof(vertices), D3D11_USAGE_DEFAULT, D3D11_BIND_VERTEX_BUFFER};
    D3D11_SUBRESOURCE_DATA init = {vertices};
    hr = g_device->CreateBuffer(&bd, &init, &g_vertex_buffer);
    if (FAILED(hr)) return false;

    return true;
}

int main() {
    // Register window class
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX), CS_HREDRAW | CS_VREDRAW, WndProc, 0, 0, GetModuleHandle(nullptr), nullptr, LoadCursor(nullptr, IDC_ARROW), (HBRUSH)(COLOR_WINDOW + 1), nullptr, "GuiPlayerClass", nullptr };
    RegisterClassEx(&wcex);

    HWND hwnd = CreateWindow("GuiPlayerClass", "MKV Player", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 640, 480, nullptr, nullptr, GetModuleHandle(nullptr), nullptr);
    if (!hwnd) return 1;
    ShowWindow(hwnd, SW_SHOW);

    if (!InitializeDeviceAndSwapchain(hwnd)) {
        std::cerr << "Failed to initialize device and swapchain" << std::endl;
        return 1;
    }

    if (!InitializeShadersAndQuad()) {
        std::cerr << "Failed to initialize shaders and quad" << std::endl;
        return 1;
    }

    g_decoder = std::make_unique<RgbVideoDecoder>();
    std::string filepath = "test_data/sample_hw.mkv";
    if (!g_decoder->open(filepath, g_device.Get())) {
        std::cerr << "Failed to open decoder" << std::endl;
        return 1;
    }

    RgbVideoDecoder::DecodedFrame first_frame;
    if (g_decoder->readNextFrame(first_frame)) {
        g_video_width = first_frame.rgb_frame.width;
        g_video_height = first_frame.rgb_frame.height;
        g_decoder->seekToTime(0.0);
    }

    MSG msg;
    while (true) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT) break;
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            // Render loop
            RgbVideoDecoder::DecodedFrame frame;
            if (g_decoder->readNextFrame(frame)) {
                ComPtr<ID3D11Texture2D> backbuffer;
                g_swapchain->GetBuffer(0, IID_PPV_ARGS(&backbuffer));
                ComPtr<ID3D11RenderTargetView> rtv;
                g_device->CreateRenderTargetView(backbuffer.Get(), nullptr, &rtv);
                RenderRgbFrame(g_context.Get(), rtv.Get(), frame.rgb_frame, g_video_width, g_video_height, g_vertex_shader.Get(), g_pixel_shader.Get(), g_input_layout.Get(), g_vertex_buffer.Get(), g_sampler_state.Get());
                g_swapchain->Present(1, 0);
            }
        }
    }

    return 0;
}
