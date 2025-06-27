#include "main.h"

void dump_d3d11_avframe(const FFMepgContext* ctx, AVFrame* frame) {
      if (!ctx || !frame || !ctx->d3d_device || !ctx->d3d_context) {
        std::cerr << "Error: Invalid context or frame provided." << std::endl;
        return;
    }
    if (frame->format != AV_PIX_FMT_D3D11) {
        std::cerr << "Error: Expected D3D11 format, got " << frame->format << std::endl;
        return;
    }
    if (frame->width <= 0 || frame->height <= 0) {
        std::cerr << "Error: Invalid frame dimensions." << std::endl;
        return;
    }
    if (!frame->data[0]) {
        std::cerr << "Error: D3D11 texture pointer is null." << std::endl;
        return;
    }
    
    ID3D11Texture2D* input_texture = reinterpret_cast<ID3D11Texture2D*>(frame->data[0]);
    int texture_index = (int)(intptr_t)frame->data[1];
    D3D11_TEXTURE2D_DESC input_desc;
    input_texture->GetDesc(&input_desc);
    
    if (input_desc.Format != DXGI_FORMAT_NV12) {
        std::cerr << "Error: Expected DXGI_FORMAT_NV12 format, got " << frame->format << std::endl;
        return;
    }

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = input_desc.Width;
    desc.Height = input_desc.Height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = input_desc.Format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    
    ID3D11Texture2D* intermediate_texture = nullptr;
    HRESULT hr = ctx->d3d_device->CreateTexture2D(&desc, nullptr, &intermediate_texture);
    if (FAILED(hr)) {
        throw std::runtime_error("Failed to create intermediate texture");
    }
    UINT src_subresource = D3D11CalcSubresource(0, texture_index, input_desc.MipLevels);
    ctx->d3d_context->CopySubresourceRegion(intermediate_texture, 0, 0, 0, 0, input_texture, src_subresource, nullptr);
    std::cout << "Dump d3d11 avframe done" << std::endl;
}