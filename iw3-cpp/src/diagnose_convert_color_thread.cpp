#include "diagnose_convert_color_thread.h"
#include <iostream>
#include <chrono>
#include <iomanip>  // 添加这个头文件用于 std::setprecision
#include <filesystem>
#include <algorithm>  // 添加这个头文件用于 std::min/std::max
#include <wincodec.h>
#include <wrl/client.h>

using Microsoft::WRL::ComPtr;

// Helper function to save texture content as image file
bool save_texture_to_file(ID3D11Texture2D* texture, ID3D11Device* device, ID3D11DeviceContext* context, 
                         const std::string& output_dir, const std::string& filename) {
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    // Create a staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture for saving: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Copy from GPU texture to staging
    context->CopyResource(staging_texture, texture);
    
    // Map and read pixel data
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: " << std::hex << hr << std::endl;
        staging_texture->Release();
        return false;
    }
    
    // Convert float RGBA to RGBA8 format
    std::vector<uint8_t> rgba8_data(desc.Width * desc.Height * 4);
    float* src_pixels = (float*)mapped.pData;
    uint8_t* dst_pixels = rgba8_data.data();
    
    for (UINT y = 0; y < desc.Height; y++) {
        float* src_row = (float*)((uint8_t*)mapped.pData + y * mapped.RowPitch);
        uint8_t* dst_row = dst_pixels + y * desc.Width * 4;
        
        for (UINT x = 0; x < desc.Width; x++) {
            // Clamp and convert from [0.0, 1.0] float to [0, 255] uint8
            // Use parentheses to avoid Windows min/max macro interference
            dst_row[x * 4 + 0] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 0], 0.0f), 1.0f) * 255.0f); // R
            dst_row[x * 4 + 1] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 1], 0.0f), 1.0f) * 255.0f); // G
            dst_row[x * 4 + 2] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 2], 0.0f), 1.0f) * 255.0f); // B
            dst_row[x * 4 + 3] = (uint8_t)((std::min)((std::max)(src_row[x * 4 + 3], 0.0f), 1.0f) * 255.0f); // A
        }
    }
    
    context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    // Save using Windows Imaging Component (WIC)
    ComPtr<IWICImagingFactory> wic_factory;
    hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&wic_factory));
    if (FAILED(hr)) {
        std::cerr << "Failed to create WIC factory: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Create file stream
    std::wstring wide_path = std::wstring(output_dir.begin(), output_dir.end()) + L"\\" + 
                            std::wstring(filename.begin(), filename.end()) + L".png";
    ComPtr<IWICStream> stream;
    hr = wic_factory->CreateStream(&stream);
    if (FAILED(hr)) {
        std::cerr << "Failed to create WIC stream: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = stream->InitializeFromFilename(wide_path.c_str(), GENERIC_WRITE);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize stream with filename: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Create PNG encoder
    ComPtr<IWICBitmapEncoder> encoder;
    hr = wic_factory->CreateEncoder(GUID_ContainerFormatPng, NULL, &encoder);
    if (FAILED(hr)) {
        std::cerr << "Failed to create PNG encoder: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache);
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize encoder: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Create frame encoder
    ComPtr<IWICBitmapFrameEncode> frame_encoder;
    ComPtr<IPropertyBag2> properties;
    hr = encoder->CreateNewFrame(&frame_encoder, &properties);
    if (FAILED(hr)) {
        std::cerr << "Failed to create frame encoder: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = frame_encoder->Initialize(properties.Get());
    if (FAILED(hr)) {
        std::cerr << "Failed to initialize frame encoder: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = frame_encoder->SetSize(desc.Width, desc.Height);
    if (FAILED(hr)) {
        std::cerr << "Failed to set frame size: " << std::hex << hr << std::endl;
        return false;
    }
    
    WICPixelFormatGUID format = GUID_WICPixelFormat32bppRGBA;
    hr = frame_encoder->SetPixelFormat(&format);
    if (FAILED(hr)) {
        std::cerr << "Failed to set pixel format: " << std::hex << hr << std::endl;
        return false;
    }
    
    // Write pixel data
    UINT stride = desc.Width * 4;
    hr = frame_encoder->WritePixels(desc.Height, stride, rgba8_data.size(), rgba8_data.data());
    if (FAILED(hr)) {
        std::cerr << "Failed to write pixels: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = frame_encoder->Commit();
    if (FAILED(hr)) {
        std::cerr << "Failed to commit frame: " << std::hex << hr << std::endl;
        return false;
    }
    
    hr = encoder->Commit();
    if (FAILED(hr)) {
        std::cerr << "Failed to commit encoder: " << std::hex << hr << std::endl;
        return false;
    }
    
    std::cout << "Successfully saved texture to: " << output_dir << "\\" << filename << ".png" << std::endl;
    return true;
}

void start_diagnose_convert_color_thread(
    ColorConvertedFrameQueue& input_queue,
    ColorConvertedFrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context
) {
    std::cout << "=== Diagnose Convert Color Thread Started ===\n";
    
    int frame_count = 0;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize COM for WIC
    CoInitializeEx(NULL, COINIT_MULTITHREADED);
    
    while (true) {
        // Get frame from input queue
        ColorConvertedFrame frame = input_queue.pop();
        
        // Check for end signal
        if (frame.is_end_signal) {
            std::cout << "Diagnose thread received end signal\n";
            // Forward end signal to output queue
            output_queue.push(ColorConvertedFrame::end_signal());
            break;
        }
        
        frame_count++;
        
        // Validate frame data
        bool is_valid = true;
        std::string validation_errors;
        
        // Check if texture is valid
        if (!frame.texture) {
            is_valid = false;
            validation_errors += "Invalid texture pointer; ";
        }
        
        // Check dimensions
        if (frame.width == 0 || frame.height == 0) {
            is_valid = false;
            validation_errors += "Invalid dimensions (" + std::to_string(frame.width) + 
                               "x" + std::to_string(frame.height) + "); ";
        }
        
        // Check texture properties if valid
        if (frame.texture) {
            D3D11_TEXTURE2D_DESC desc;
            frame.texture->GetDesc(&desc);
            
            // Validate texture format - only check for R32G32B32A32_FLOAT
            if (desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT) {
                is_valid = false;
                validation_errors += "Expected DXGI_FORMAT_R32G32B32A32_FLOAT but got format " + 
                                   std::to_string(desc.Format) + "; ";
            }
            
            // Validate texture dimensions match frame dimensions
            if (desc.Width != frame.width || desc.Height != frame.height) {
                is_valid = false;
                validation_errors += "Texture dimensions mismatch (texture: " + 
                                   std::to_string(desc.Width) + "x" + std::to_string(desc.Height) +
                                   " vs frame: " + std::to_string(frame.width) + "x" + std::to_string(frame.height) + "); ";
            }
            
            // Check if texture has proper usage flags for model input
            if (!(desc.BindFlags & D3D11_BIND_SHADER_RESOURCE)) {
                validation_errors += "Texture missing SHADER_RESOURCE bind flag; ";
            }
            
            // Save texture to file if we have device context and frame is valid
            if (device && context && is_valid) {
                std::string output_dir = "debug_textures";
                std::string filename = "frame_" + std::to_string(frame_count);
                
                if (!save_texture_to_file(frame.texture, device, context, output_dir, filename)) {
                    validation_errors += "Failed to save texture to file; ";
                }
            }
        }
        
        // Log validation results
        if (is_valid) {
            if (frame_count % 30 == 0) { // Log every 30 frames to avoid spam
                std::cout << "Frame " << frame_count << ": VALID R32G32B32A32_FLOAT (" 
                         << frame.width << "x" << frame.height << ")\n";
            }
        } else {
            std::cerr << "Frame " << frame_count << ": INVALID - " << validation_errors << "\n";
        }
        
        // Performance statistics every 100 frames
        if (frame_count % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
            double fps = (double)frame_count / (duration.count() / 1000.0);
            std::cout << "Diagnose thread processed " << frame_count 
                     << " frames, average FPS: " << std::fixed << std::setprecision(2) << fps << "\n";
        }
        
        // Forward frame to output queue (move semantics to avoid copy)
        output_queue.push(std::move(frame));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double avg_fps = (double)frame_count / (total_duration.count() / 1000.0);
    
    std::cout << "=== Diagnose Convert Color Thread Completed ===\n";
    std::cout << "Total frames processed: " << frame_count << "\n";
    std::cout << "Total time: " << total_duration.count() << " ms\n";
    std::cout << "Average FPS: " << std::fixed << std::setprecision(2) << avg_fps << "\n";
    
    // Cleanup COM
    CoUninitialize();
} 