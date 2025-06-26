#include "diagnose_convert_color_thread.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <d3d11.h>
#include <DirectXTex.h>

using namespace DirectX;

// Simplified function to save texture as DDS
bool save_texture_as_dds(ID3D11Texture2D* texture, ID3D11Device* device, ID3D11DeviceContext* context, 
                        const std::string& filepath) {
    // First get the original texture description
    D3D11_TEXTURE2D_DESC texture_desc;
    texture->GetDesc(&texture_desc);
    
    ScratchImage scratch_image;
    HRESULT hr = CaptureTexture(device, context, texture, scratch_image);
    if (FAILED(hr)) {
        std::cerr << "Failed to capture texture, HRESULT: " << hr << std::endl;
        return false;
    }
    
    // Get and verify metadata
    TexMetadata metadata = scratch_image.GetMetadata();
    
    // Ensure metadata is correctly set based on original texture
    metadata.width = texture_desc.Width;
    metadata.height = texture_desc.Height;
    metadata.depth = 1;
    metadata.arraySize = texture_desc.ArraySize;
    metadata.mipLevels = texture_desc.MipLevels;
    metadata.format = texture_desc.Format;
    metadata.dimension = TEX_DIMENSION_TEXTURE2D;
    metadata.miscFlags = 0;
    metadata.miscFlags2 = 0;

    // 检查纹理格式
    if (metadata.format != DXGI_FORMAT_R32G32B32A32_FLOAT) {
        std::cerr << "Warning: Unexpected texture format: " << metadata.format 
                  << " (expected: " << DXGI_FORMAT_R32G32B32A32_FLOAT << ")" << std::endl;
    }
    
    // Debug output metadata
    std::cout << "Saving DDS - Width: " << metadata.width 
              << ", Height: " << metadata.height 
              << ", Format: " << metadata.format 
              << ", MipLevels: " << metadata.mipLevels << std::endl;

    // Save as DDS
    std::wstring wide_path(filepath.begin(), filepath.end());
    hr = SaveToDDSFile(scratch_image.GetImages(), scratch_image.GetImageCount(), metadata,
                       DDS_FLAGS_NONE, wide_path.c_str());

    if (FAILED(hr)) {
        std::cerr << "Failed to save DDS file, HRESULT: " << hr << std::endl;
        return false;
    }
    
    std::cout << "Successfully saved: " << filepath << std::endl;
    return true;
}

void start_diagnose_convert_color_thread(
    ColorConvertedFrameQueue& input_queue,
    ColorConvertedFrameQueue& output_queue,
    ID3D11Device* device,
    ID3D11DeviceContext* context
) {
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
        
        // Save frame as DDS file
        if (frame.texture && device && context) {
            std::string filename = "debug_textures/frame_" + std::to_string(frame_count) + ".dds";
            if (!save_texture_as_dds(frame.texture, device, context, filename)) {
                std::cerr << "Error: Could not save " << filename << std::endl;
            }
        }
        
        // Forward frame to output queue
        output_queue.push(std::move(frame));
    }
} 