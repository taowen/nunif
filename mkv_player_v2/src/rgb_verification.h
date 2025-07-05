#pragma once

#include <d3d11.h>
#include <string>
#include <vector>

class RGBVerification {
public:
    struct PixelData {
        uint8_t b, g, r, a;  // BGRA format
    };
    
    struct ImageStats {
        double mean_r, mean_g, mean_b;
        double std_r, std_g, std_b;
        uint8_t min_r, min_g, min_b;
        uint8_t max_r, max_g, max_b;
        int width, height;
    };

    static bool saveTextureAsBMP(ID3D11Device* device, ID3D11DeviceContext* context, 
                                 ID3D11Texture2D* texture, const std::string& filename);
    
    static bool readTextureData(ID3D11Device* device, ID3D11DeviceContext* context,
                               ID3D11Texture2D* texture, std::vector<PixelData>& pixel_data,
                               int& width, int& height);
    
    static ImageStats calculateImageStats(const std::vector<PixelData>& pixel_data, 
                                         int width, int height);
    
    static bool compareImages(const std::vector<PixelData>& image1,
                             const std::vector<PixelData>& image2,
                             int width, int height,
                             double& psnr, double& mse);
    
    static void printImageStats(const ImageStats& stats, const std::string& name);
    
    static bool isValidRGBRange(const std::vector<PixelData>& pixel_data);
    
    static bool detectSolidColor(const std::vector<PixelData>& pixel_data, 
                                int width, int height,
                                PixelData& detected_color, double tolerance = 5.0);

private:
    static bool writeBMPHeader(FILE* file, int width, int height);
    static double calculatePSNR(double mse, int max_value = 255);
};