#include "rgb_verification.h"
#include <iostream>
#include <cmath>
#include <algorithm>

bool RGBVerification::saveTextureAsBMP(ID3D11Device* device, ID3D11DeviceContext* context, 
                                       ID3D11Texture2D* texture, const std::string& filename) {
    std::vector<PixelData> pixel_data;
    int width, height;
    
    if (!readTextureData(device, context, texture, pixel_data, width, height)) {
        return false;
    }
    
    FILE* file = nullptr;
    errno_t err = fopen_s(&file, filename.c_str(), "wb");
    if (err != 0 || !file) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write BMP header
    if (!writeBMPHeader(file, width, height)) {
        fclose(file);
        return false;
    }
    
    // Write pixel data (BMP is bottom-up)
    for (int y = height - 1; y >= 0; y--) {
        for (int x = 0; x < width; x++) {
            const PixelData& pixel = pixel_data[y * width + x];
            fwrite(&pixel.b, 1, 1, file);  // Blue
            fwrite(&pixel.g, 1, 1, file);  // Green  
            fwrite(&pixel.r, 1, 1, file);  // Red
        }
        
        // BMP rows must be padded to 4-byte boundary
        int padding = (4 - (width * 3) % 4) % 4;
        for (int p = 0; p < padding; p++) {
            uint8_t zero = 0;
            fwrite(&zero, 1, 1, file);
        }
    }
    
    fclose(file);
    std::cout << "Saved texture as: " << filename << " (" << width << "x" << height << ")" << std::endl;
    return true;
}

bool RGBVerification::readTextureData(ID3D11Device* device, ID3D11DeviceContext* context,
                                     ID3D11Texture2D* texture, std::vector<PixelData>& pixel_data,
                                     int& width, int& height) {
    if (!texture) {
        return false;
    }
    
    // Get texture description
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    width = desc.Width;
    height = desc.Height;
    
    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.BindFlags = 0;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.MiscFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return false;
    }
    
    // Copy texture to staging
    context->CopyResource(staging_texture, texture);
    
    // Map staging texture
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: 0x" << std::hex << hr << std::endl;
        staging_texture->Release();
        return false;
    }
    
    // Read pixel data
    pixel_data.resize(width * height);
    uint8_t* src_data = (uint8_t*)mapped.pData;
    
    for (int y = 0; y < height; y++) {
        uint8_t* row_data = src_data + y * mapped.RowPitch;
        for (int x = 0; x < width; x++) {
            PixelData& pixel = pixel_data[y * width + x];
            
            // Assuming BGRA format (D3D11 default)
            pixel.b = row_data[x * 4 + 0];
            pixel.g = row_data[x * 4 + 1];
            pixel.r = row_data[x * 4 + 2];
            pixel.a = row_data[x * 4 + 3];
        }
    }
    
    context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    return true;
}

RGBVerification::ImageStats RGBVerification::calculateImageStats(const std::vector<PixelData>& pixel_data, 
                                                                 int width, int height) {
    ImageStats stats = {};
    stats.width = width;
    stats.height = height;
    
    if (pixel_data.empty()) {
        return stats;
    }
    
    // Initialize min/max
    stats.min_r = stats.min_g = stats.min_b = 255;
    stats.max_r = stats.max_g = stats.max_b = 0;
    
    // Calculate means and min/max
    double sum_r = 0, sum_g = 0, sum_b = 0;
    for (const auto& pixel : pixel_data) {
        sum_r += pixel.r;
        sum_g += pixel.g;
        sum_b += pixel.b;
        
        stats.min_r = (std::min)(stats.min_r, pixel.r);
        stats.min_g = (std::min)(stats.min_g, pixel.g);
        stats.min_b = (std::min)(stats.min_b, pixel.b);
        
        stats.max_r = (std::max)(stats.max_r, pixel.r);
        stats.max_g = (std::max)(stats.max_g, pixel.g);
        stats.max_b = (std::max)(stats.max_b, pixel.b);
    }
    
    int pixel_count = width * height;
    stats.mean_r = sum_r / pixel_count;
    stats.mean_g = sum_g / pixel_count;
    stats.mean_b = sum_b / pixel_count;
    
    // Calculate standard deviations
    double var_r = 0, var_g = 0, var_b = 0;
    for (const auto& pixel : pixel_data) {
        var_r += (pixel.r - stats.mean_r) * (pixel.r - stats.mean_r);
        var_g += (pixel.g - stats.mean_g) * (pixel.g - stats.mean_g);
        var_b += (pixel.b - stats.mean_b) * (pixel.b - stats.mean_b);
    }
    
    stats.std_r = std::sqrt(var_r / pixel_count);
    stats.std_g = std::sqrt(var_g / pixel_count);
    stats.std_b = std::sqrt(var_b / pixel_count);
    
    return stats;
}

bool RGBVerification::compareImages(const std::vector<PixelData>& image1,
                                   const std::vector<PixelData>& image2,
                                   int width, int height,
                                   double& psnr, double& mse) {
    if (image1.size() != image2.size() || image1.size() != width * height) {
        return false;
    }
    
    double sum_squared_error = 0;
    int pixel_count = width * height;
    
    for (int i = 0; i < pixel_count; i++) {
        double diff_r = image1[i].r - image2[i].r;
        double diff_g = image1[i].g - image2[i].g;
        double diff_b = image1[i].b - image2[i].b;
        
        sum_squared_error += diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    }
    
    mse = sum_squared_error / (pixel_count * 3); // 3 channels
    psnr = calculatePSNR(mse);
    
    return true;
}

void RGBVerification::printImageStats(const ImageStats& stats, const std::string& name) {
    std::cout << "=== " << name << " Statistics ===" << std::endl;
    std::cout << "Dimensions: " << stats.width << "x" << stats.height << std::endl;
    std::cout << "RGB Means: R=" << stats.mean_r << ", G=" << stats.mean_g << ", B=" << stats.mean_b << std::endl;
    std::cout << "RGB Std: R=" << stats.std_r << ", G=" << stats.std_g << ", B=" << stats.std_b << std::endl;
    std::cout << "RGB Range: R=[" << (int)stats.min_r << "-" << (int)stats.max_r << "], "
              << "G=[" << (int)stats.min_g << "-" << (int)stats.max_g << "], "
              << "B=[" << (int)stats.min_b << "-" << (int)stats.max_b << "]" << std::endl;
}

bool RGBVerification::isValidRGBRange(const std::vector<PixelData>& pixel_data) {
    for (const auto& pixel : pixel_data) {
        // Check if any pixel values are outside valid 0-255 range
        // (This shouldn't happen, but good to verify)
        if (pixel.r > 255 || pixel.g > 255 || pixel.b > 255) {
            return false;
        }
    }
    return true;
}

bool RGBVerification::detectSolidColor(const std::vector<PixelData>& pixel_data, 
                                      int width, int height,
                                      PixelData& detected_color, double tolerance) {
    if (pixel_data.empty()) {
        return false;
    }
    
    // Use first pixel as reference
    PixelData reference = pixel_data[0];
    detected_color = reference;
    
    // Check if all pixels are within tolerance of reference
    for (const auto& pixel : pixel_data) {
        double diff_r = std::abs(pixel.r - reference.r);
        double diff_g = std::abs(pixel.g - reference.g);
        double diff_b = std::abs(pixel.b - reference.b);
        
        if (diff_r > tolerance || diff_g > tolerance || diff_b > tolerance) {
            return false;
        }
    }
    
    return true;
}

bool RGBVerification::writeBMPHeader(FILE* file, int width, int height) {
    // BMP File Header (14 bytes)
    uint8_t file_header[14] = {
        'B', 'M',  // Signature
        0, 0, 0, 0,  // File size (will be filled)
        0, 0,  // Reserved
        0, 0,  // Reserved  
        54, 0, 0, 0  // Offset to pixel data
    };
    
    // Calculate file size
    int row_size = ((width * 3 + 3) / 4) * 4;  // 4-byte aligned
    int file_size = 54 + row_size * height;
    
    file_header[2] = file_size & 0xFF;
    file_header[3] = (file_size >> 8) & 0xFF;
    file_header[4] = (file_size >> 16) & 0xFF;
    file_header[5] = (file_size >> 24) & 0xFF;
    
    // BMP Info Header (40 bytes)
    uint8_t info_header[40] = {
        40, 0, 0, 0,  // Header size
        0, 0, 0, 0,   // Width (will be filled)
        0, 0, 0, 0,   // Height (will be filled)
        1, 0,         // Planes
        24, 0,        // Bits per pixel
        0, 0, 0, 0,   // Compression
        0, 0, 0, 0,   // Image size
        0, 0, 0, 0,   // X resolution
        0, 0, 0, 0,   // Y resolution
        0, 0, 0, 0,   // Colors used
        0, 0, 0, 0    // Important colors
    };
    
    // Fill width and height
    info_header[4] = width & 0xFF;
    info_header[5] = (width >> 8) & 0xFF;
    info_header[6] = (width >> 16) & 0xFF;
    info_header[7] = (width >> 24) & 0xFF;
    
    info_header[8] = height & 0xFF;
    info_header[9] = (height >> 8) & 0xFF;
    info_header[10] = (height >> 16) & 0xFF;
    info_header[11] = (height >> 24) & 0xFF;
    
    // Write headers
    if (fwrite(file_header, 1, 14, file) != 14 ||
        fwrite(info_header, 1, 40, file) != 40) {
        return false;
    }
    
    return true;
}

double RGBVerification::calculatePSNR(double mse, int max_value) {
    if (mse == 0) {
        return 100.0;  // Perfect match
    }
    return 20.0 * std::log10(max_value / std::sqrt(mse));
}