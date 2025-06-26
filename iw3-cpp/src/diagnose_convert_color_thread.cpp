#include "diagnose_convert_color_thread.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <wincodec.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// Simplified function to save texture as PNG
bool save_texture_as_png(ID3D11Texture2D* texture, ID3D11Device* device, ID3D11DeviceContext* context, 
                        const std::string& filepath) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    if (FAILED(device->CreateTexture2D(&staging_desc, nullptr, &staging_texture))) {
        return false;
    }
    
    // Copy and map texture data
    context->CopyResource(staging_texture, texture);
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped))) {
        staging_texture->Release();
        return false;
    }
    
    // Convert float RGBA to uint8 RGBA
    std::vector<uint8_t> rgba8_data(desc.Width * desc.Height * 4);
    for (UINT y = 0; y < desc.Height; y++) {
        float* src_row = (float*)((uint8_t*)mapped.pData + y * mapped.RowPitch);
        uint8_t* dst_row = rgba8_data.data() + y * desc.Width * 4;
        
        for (UINT x = 0; x < desc.Width; x++) {
            dst_row[x * 4 + 0] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 0], 0.0f), 1.0f) * 255.0f);
            dst_row[x * 4 + 1] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 1], 0.0f), 1.0f) * 255.0f);
            dst_row[x * 4 + 2] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 2], 0.0f), 1.0f) * 255.0f);
            dst_row[x * 4 + 3] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 3], 0.0f), 1.0f) * 255.0f);
        }
    }
    
    context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    // Save using WIC
    ComPtr<IWICImagingFactory> wic_factory;
    if (FAILED(CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic_factory)))) {
        return false;
    }
    
    ComPtr<IWICStream> stream;
    if (FAILED(wic_factory->CreateStream(&stream))) {
        return false;
    }
    
    std::wstring wide_path(filepath.begin(), filepath.end());
    if (FAILED(stream->InitializeFromFilename(wide_path.c_str(), GENERIC_WRITE))) {
        return false;
    }
    
    ComPtr<IWICBitmapEncoder> encoder;
    if (FAILED(wic_factory->CreateEncoder(GUID_ContainerFormatPng, NULL, &encoder))) {
        return false;
    }
    
    if (FAILED(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache))) {
        return false;
    }
    
    ComPtr<IWICBitmapFrameEncode> frame_encoder;
    ComPtr<IPropertyBag2> properties;
    if (FAILED(encoder->CreateNewFrame(&frame_encoder, &properties))) {
        return false;
    }
    
    if (FAILED(frame_encoder->Initialize(properties.Get()))) {
        return false;
    }
    
    if (FAILED(frame_encoder->SetSize(desc.Width, desc.Height))) {
        return false;
    }
    
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppRGBA;
    if (FAILED(frame_encoder->SetPixelFormat(&format))) {
        return false;
    }
    
    UINT stride = desc.Width * 4;
    if (FAILED(frame_encoder->WritePixels(desc.Height, stride, rgba8_data.size(), rgba8_data.data()))) {
        return false;
    }
    
    if (FAILED(frame_encoder->Commit()) || FAILED(encoder->Commit())) {
        return false;
    }
    
    return true;
}

void start_diagnose_convert_color_thread(
    ColorConvertedFrameQueue& input_queue,
    ColorConvertedFrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context
) {
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    
    // Create debug output directory
    std::filesystem::create_directories("debug_textures");
    
    int frame_count = 0;
    
    while (true) {
        ColorConvertedFrame frame = input_queue.pop();
        
        if (frame.is_end_signal) {
            output_queue.push(ColorConvertedFrame::end_signal());
            break;
        }
        
        frame_count++;
        
        // Save frame as PNG file
        if (frame.texture && device && context) {
            std::string filename = "debug_textures/frame_" + std::to_string(frame_count) + ".png";
            save_texture_as_png(frame.texture, device, context, filename);
        }
        
        // Forward frame to output queue
        output_queue.push(std::move(frame));
    }
    
    CoUninitialize();
} 