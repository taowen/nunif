#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <d3dcompiler.h>
#include <chrono>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib") 
#pragma comment(lib, "d3dcompiler.lib")

// Global DirectX11 objects
ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pImmediateContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_pRenderTargetView = nullptr;

// Window handle
HWND g_hWnd = nullptr;

// Colors for cycling
float g_Colors[3][4] = {
    {1.0f, 0.0f, 0.0f, 1.0f}, // Red
    {0.0f, 1.0f, 0.0f, 1.0f}, // Green  
    {0.0f, 0.0f, 1.0f, 1.0f}  // Blue
};
int g_currentColorIndex = 0;
auto g_lastColorChange = std::chrono::steady_clock::now();

// Initialize DirectX11
bool InitD3D(HWND hWnd) {
    RECT rc;
    GetClientRect(hWnd, &rc);
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    // Create swap chain description
    DXGI_SWAP_CHAIN_DESC sd = {};
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

    // Create device and swap chain
    HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        &featureLevel,
        1,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        nullptr,
        &g_pImmediateContext
    );

    if (FAILED(hr)) {
        return false;
    }

    // Create render target view
    ID3D11Texture2D* pBackBuffer = nullptr;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
    if (FAILED(hr)) {
        return false;
    }

    hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_pRenderTargetView);
    pBackBuffer->Release();
    if (FAILED(hr)) {
        return false;
    }

    // Set render target
    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, nullptr);

    // Set viewport
    D3D11_VIEWPORT vp = {};
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pImmediateContext->RSSetViewports(1, &vp);

    return true;
}

// Cleanup DirectX11
void CleanupD3D() {
    if (g_pRenderTargetView) g_pRenderTargetView->Release();
    if (g_pSwapChain) g_pSwapChain->Release();
    if (g_pImmediateContext) g_pImmediateContext->Release();
    if (g_pd3dDevice) g_pd3dDevice->Release();
}

// Render one frame
void Render() {
    // Check if 1 second has passed to change color
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - g_lastColorChange);
    
    if (elapsed.count() >= 1000) { // 1 second
        g_currentColorIndex = (g_currentColorIndex + 1) % 3;
        g_lastColorChange = now;
    }

    // Clear with current color
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, g_Colors[g_currentColorIndex]);
    
    // Present the frame
    g_pSwapChain->Present(0, 0);
}

// Window procedure
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                PostQuitMessage(0);
            }
            return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}

// Main function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    // Register window class
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_CLASSDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.lpszClassName = "DX11Demo";
    RegisterClassEx(&wc);

    // Create window
    g_hWnd = CreateWindow(
        "DX11Demo",
        "DirectX11 Color Demo - Press ESC to exit",
        WS_OVERLAPPEDWINDOW,
        100, 100, 800, 600,
        nullptr, nullptr, hInstance, nullptr
    );

    // Initialize DirectX11
    if (!InitD3D(g_hWnd)) {
        MessageBox(nullptr, "Failed to initialize DirectX11", "Error", MB_OK);
        return -1;
    }

    // Show window
    ShowWindow(g_hWnd, nCmdShow);
    UpdateWindow(g_hWnd);

    // Main message loop
    MSG msg = {};
    while (WM_QUIT != msg.message) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        } else {
            Render();
        }
    }

    // Cleanup
    CleanupD3D();
    UnregisterClass("DX11Demo", hInstance);

    return (int)msg.wParam;
}