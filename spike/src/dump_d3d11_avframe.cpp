#include "main.h"
#include <fstream>
#include <vector>

// Add this to prevent Windows min/max macro conflicts
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

/*
根据这份代码分析，D3D11 NV12格式数据在显存中的实际布局如下：

## NV12格式显存布局总结

### 整体结构
```
[Y平面数据 - height * row_pitch 字节]
[UV平面数据 - (height/2) * row_pitch 字节]
```

### 关键参数
- `row_pitch`: 每行实际占用的字节数（包含内存对齐padding）
- `width`: 图像实际宽度
- `height`: 图像实际高度

### Y平面布局（亮度）
```cpp:spike/src/dump_d3d11_avframe.cpp
// Y平面：width * height 像素，每像素1字节
for (int y = 0; y < height; y++) {
    memcpy(result.y_plane.data() + y * width, 
           mapped_data + y * row_pitch,  // 注意：使用row_pitch而非width
           width);
}
```

### UV平面布局（色度，交错存储）
```cpp:spike/src/dump_d3d11_avframe.cpp
// UV平面起始位置：Y平面之后
uint8_t* uv_start = mapped_data + input_desc.Height * row_pitch;

// UV数据交错存储：UVUVUV...
for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width / 2; x++) {
        int src_idx = y * row_pitch + x * 2;  // 每2个字节一组：UV
        int dst_idx = y * (width / 2) + x;
        result.u_plane[dst_idx] = uv_start[src_idx];     // U分量
        result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V分量
    }
}
```

## 常见的CUDA读取错误

### 1. 忽略内存对齐（row_pitch）
**错误做法：**
```cpp
// 错误：直接使用width计算偏移
Y_offset = y * width;
```

**正确做法：**
```cpp
// 正确：使用row_pitch计算偏移
Y_offset = y * row_pitch;
```

### 2. UV平面位置计算错误
**错误做法：**
```cpp
// 错误：使用width * height
uv_offset = width * height;
```

**正确做法：**
```cpp
// 正确：使用height * row_pitch（考虑内存对齐）
uv_offset = height * row_pitch;
```

### 3. UV交错存储理解错误
**错误做法：**
```cpp
// 错误：把UV当作平面存储
U_value = uv_data[uv_index];
V_value = uv_data[uv_index + uv_plane_size];
```

**正确做法：**
```cpp
// 正确：UV交错存储，每2字节一组
U_value = uv_data[uv_index * 2];     // 偶数位置是U
V_value = uv_data[uv_index * 2 + 1]; // 奇数位置是V
```

## 完整的内存地址计算公式

```cpp
// Y平面某像素(x,y)的地址
Y_address = base_address + y * row_pitch + x;

// UV平面某像素组(x,y)的地址（注意x,y都是UV坐标，范围是原图的一半）
UV_base = base_address + height * row_pitch;
U_address = UV_base + y * row_pitch + x * 2;     // U分量
V_address = UV_base + y * row_pitch + x * 2 + 1; // V分量
```

**关键点：`row_pitch`通常比`width`大，因为GPU内存需要对齐（如256字节对齐），这是CUDA读取出错的最常见原因。**
*/
YUVData dump_d3d11_avframe(const FFMepgContext* ctx, AVFrame* frame, bool save_to_file) {
    YUVData result;
    
    if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return result;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return result;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return result;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return result;
    }
    
    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);

    std::cout << "[dump_d3d11_avframe] Input Texture Desc:" << std::endl;
    std::cout << "  Width: " << input_desc.Width << ", Height: " << input_desc.Height << std::endl;
    std::cout << "  MipLevels: " << input_desc.MipLevels << ", ArraySize: " << input_desc.ArraySize << std::endl;
    std::cout << "  Format: " << input_desc.Format << " (Expected DXGI_FORMAT_NV12=" << DXGI_FORMAT_NV12 << ")" << std::endl;
    std::cout << "  Texture Index: " << texture_index << std::endl;
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << input_desc.Format << std::endl;
        return result;
    }

    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = {};
    staging_desc.Width = input_desc.Width;
    staging_desc.Height = input_desc.Height;
    staging_desc.MipLevels = 1;
    staging_desc.ArraySize = 1;
    staging_desc.Format = input_desc.Format;
    staging_desc.SampleDesc.Count = 1;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.BindFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    HRESULT hr = ctx->d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    if (FAILED(hr)) {
        std::cerr << "Failed to create staging texture: 0x" << std::hex << hr << std::endl;
        return result;
    }

    // Copy from D3D11 texture to staging texture
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
    ctx->d3d_context->CopySubresourceRegion(staging_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    
    // Map the staging texture to access CPU memory
    D3D11_MAPPED_SUBRESOURCE mapped_resource;
    hr = ctx->d3d_context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped_resource);
    if (FAILED(hr)) {
        std::cerr << "Failed to map staging texture: 0x" << std::hex << hr << std::endl;
        staging_texture->Release();
        return result;
    }

    // Extract YUV data from NV12 format
    uint8_t* mapped_data = static_cast<uint8_t*>(mapped_resource.pData);
    int width = frame->width;
    int height = frame->height;
    int row_pitch = mapped_resource.RowPitch;
    
    // Calculate plane sizes
    int y_plane_size = width * height;
    int uv_plane_size = width * height / 4; // U and V planes are each 1/4 the size
    
    // Allocate buffers for separated planes
    result.y_plane.resize(y_plane_size);
    result.u_plane.resize(uv_plane_size);
    result.v_plane.resize(uv_plane_size);
    result.width = width;
    result.height = height;
    
    // Copy Y plane
    for (int y = 0; y < height; y++) {
        memcpy(result.y_plane.data() + y * width, 
               mapped_data + y * row_pitch, 
               width);
    }
    
    // Copy and separate UV plane (NV12 format has interleaved UV)
    uint8_t* uv_start = mapped_data + input_desc.Height * row_pitch;
    for (int y = 0; y < height / 2; y++) {
        for (int x = 0; x < width / 2; x++) {
            int src_idx = y * row_pitch + x * 2;
            int dst_idx = y * (width / 2) + x;
            result.u_plane[dst_idx] = uv_start[src_idx];     // U component
            result.v_plane[dst_idx] = uv_start[src_idx + 1]; // V component
        }
    }
    
    result.valid = true;
    
    // Optionally save to files
    if (save_to_file) {
        std::string base_filename = "frame_" + std::to_string(width) + "x" + std::to_string(height) + "_8";
        
        // Save interleaved YUV420 format
        std::ofstream yuv_file(base_filename + "_yuv420p.yuv", std::ios::binary);
        if (yuv_file.is_open()) {
            yuv_file.write(reinterpret_cast<const char*>(result.y_plane.data()), result.y_plane.size());
            yuv_file.write(reinterpret_cast<const char*>(result.u_plane.data()), result.u_plane.size());
            yuv_file.write(reinterpret_cast<const char*>(result.v_plane.data()), result.v_plane.size());
            yuv_file.close();
            std::cout << "Saved YUV420P to " << base_filename << "_yuv420p.yuv" << std::endl;
        }
    }
    
    // Unmap and release resources
    ctx->d3d_context->Unmap(staging_texture, 0);
    staging_texture->Release();
    
    std::cout << "Successfully extracted YUV data from D3D11 AVFrame (" 
              << width << "x" << height << ")" << std::endl;
    
    return result;
}