/******************************************************************************
 *  File : dx11_nv12_cuda_interop.cu
 *
 *  用途 : 演示 DirectX-11 NV12 纹理 ⇄ CUDA 外部内存互操作
 *         - 创建 NV12 纹理 → 取 NT 句柄 → cudaImportExternalMemory
 *         - 将两个平面映射为 CUDA array → kernel 转成 RGBA8 → 导回 CPU
 ******************************************************************************/

#include <windows.h>
#include <d3d11_4.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cassert>

/*-------------------------------------------------------------------------*/
/* 1. 一些简单的工具                                                       */
/*-------------------------------------------------------------------------*/
static void check(HRESULT hr, const char *msg)
{
    if (FAILED(hr)) { fprintf(stderr, "%s (0x%08x)\n", msg, hr); std::exit(EXIT_FAILURE); }
}
static void check(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) { fprintf(stderr, "%s (%s)\n", msg, cudaGetErrorString(e)); std::exit(EXIT_FAILURE); }
}

/* 写一张裸的 BGRA8 TGA，方便在任何图片查看器里验证结果 ---------------------*/
static void write_tga(const char *fn, const void *rgba, int w, int h)
{
    FILE *fp = std::fopen(fn, "wb");
    if (!fp) return;
    uint8_t header[18] = {};
    header[2]  = 2;              // un-compressed RGB
    header[12] =  w        &0xFF;
    header[13] = (w >> 8)  &0xFF;
    header[14] =  h        &0xFF;
    header[15] = (h >> 8)  &0xFF;
    header[16] = 32;             // 32-bit
    header[17] = 0x20;           // origin at top-left
    std::fwrite(header, 1, 18, fp);
    std::fwrite(rgba, 1, w*h*4, fp);
    std::fclose(fp);
}

/*-------------------------------------------------------------------------*/
/* 2. 设备 / 纹理初始化 (DirectX 11)                                        */
/*-------------------------------------------------------------------------*/
struct DxObjects {
    ID3D11Device          *dev  = nullptr;
    ID3D11DeviceContext   *ctx  = nullptr;
    ID3D11Texture2D       *tex  = nullptr;
    HANDLE                 ntHandle = nullptr;
    UINT                   width, height;
};

static void initDx11(DxObjects &dx, UINT w, UINT h)
{
    dx.width  = w;
    dx.height = h;

    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#if defined(_DEBUG)
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL level;
    check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                            nullptr, 0, D3D11_SDK_VERSION, &dx.dev, &level, &dx.ctx),
          "D3D11CreateDevice failed");

    /* 创建 NV12 纹理 (必须带 shared NT handle flag) */
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width            = w;
    desc.Height           = h;
    desc.MipLevels        = 1;
    desc.ArraySize        = 1;
    desc.Format           = DXGI_FORMAT_NV12;
    desc.SampleDesc.Count = 1;
    desc.Usage            = D3D11_USAGE_DEFAULT;
    desc.BindFlags        = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    desc.MiscFlags        = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;   // 关键!!

    check(dx.dev->CreateTexture2D(&desc, nullptr, &dx.tex), "CreateTexture2D NV12 failed");

    /* 获取 NT 句柄 */
    IDXGIResource1 *res1 = nullptr;
    check(dx.tex->QueryInterface(__uuidof(IDXGIResource1), (void**)&res1), "QI IDXGIResource1 failed");
    check(res1->CreateSharedHandle(nullptr,
                                   DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                   nullptr,
                                   &dx.ntHandle),
          "CreateSharedHandle failed");
    res1->Release();

    /* 这里随便往 NV12 填充一点测试图案（Y=亮度梯度，UV=恒定 128/128） */
    const size_t pitchY  = w;
    const size_t pitchUV = w;
    const size_t bytes   = w * h * 3 / 2;
    uint8_t      *cpuBuf = new uint8_t[bytes];
    for (UINT y=0; y<h; ++y)  for (UINT x=0; x<w; ++x)
        cpuBuf[y*pitchY + x] = static_cast<uint8_t>(x * 255 / w);
    std::memset(cpuBuf + pitchY*h, 128, pitchUV*h/2);    // UV 平面
    dx.ctx->UpdateSubresource(dx.tex, 0, nullptr, cpuBuf, pitchY, 0);
    delete [] cpuBuf;
}

/*-------------------------------------------------------------------------*/
/* 3. CUDA 外部内存导入 & 映射                                              */
/*-------------------------------------------------------------------------*/
struct CudaObjects {
    cudaExternalMemory_t         extMem = nullptr;
    cudaMipmappedArray_t         mip[2] = {};
    cudaArray_t                  arr[2] = {};
    cudaSurfaceObject_t          surf[2]= {};
    cudaStream_t                 stream = nullptr;
    uint8_t                     *d_rgba = nullptr;
};

static void initCuda(CudaObjects &cu, const DxObjects &dx)
{
    /* a) 导入 NT 句柄为 external memory ----------------------------------*/
    const size_t sizeBytes = size_t(dx.width) * dx.height * 3 / 2; // NV12 总字节
    cudaExternalMemoryHandleDesc memDesc = {};
    memDesc.type               = cudaExternalMemoryHandleTypeD3D11Resource;
    memDesc.handle.win32.handle= dx.ntHandle;
    memDesc.size               = sizeBytes;
    memDesc.flags              = cudaExternalMemoryDedicated;

    check(cudaImportExternalMemory(&cu.extMem, &memDesc), "cudaImportExternalMemory");

    /* b) 映射两个平面 -> mipmapped array -> level-0 array ----------------*/
    const size_t planeOffset[2] = { 0, size_t(dx.width)*dx.height }; // Y, then UV

    /* ---- Y 平面 (8-bit) ---- */
    cudaExternalMemoryMipmappedArrayDesc mipDescY = {};
    mipDescY.offset     = planeOffset[0];
    mipDescY.formatDesc = cudaCreateChannelDesc(8,0,0,0,cudaChannelFormatKindUnsigned);
    mipDescY.extent     = make_cudaExtent(dx.width, dx.height, 0);
    mipDescY.numLevels  = 1;
    check(cudaExternalMemoryGetMappedMipmappedArray(&cu.mip[0], cu.extMem, &mipDescY),
          "GetMappedMipmappedArray Y");
    check(cudaGetMipmappedArrayLevel(&cu.arr[0], cu.mip[0], 0), "array level Y");

    /* ---- UV 平面 (interleaved 8U 8V) => 格式 unsigned char2 ---- */
    cudaExternalMemoryMipmappedArrayDesc mipDescUV = {};
    mipDescUV.offset     = planeOffset[1];
    mipDescUV.formatDesc = cudaCreateChannelDesc<uchar2>();
    mipDescUV.extent     = make_cudaExtent(dx.width/2, dx.height/2, 0);
    mipDescUV.numLevels  = 1;
    check(cudaExternalMemoryGetMappedMipmappedArray(&cu.mip[1], cu.extMem, &mipDescUV),
          "GetMappedMipmappedArray UV");
    check(cudaGetMipmappedArrayLevel(&cu.arr[1], cu.mip[1], 0), "array level UV");

    /* c) 把 array 封装成 surface object，后面 kernel 里方便读 ------------*/
    for (int i=0;i<2;++i) {
        cudaResourceDesc rdesc{};
        rdesc.resType         = cudaResourceTypeArray;
        rdesc.res.array.array = cu.arr[i];
        check(cudaCreateSurfaceObject(&cu.surf[i], &rdesc), "create surface");
    }

    check(cudaStreamCreate(&cu.stream), "stream create");
    check(cudaMalloc(&cu.d_rgba, dx.width*dx.height*4), "device rgba");
}

/*-------------------------------------------------------------------------*/
/* 4. CUDA 内核: NV12 → BGRA (非常简陋，仅示范)                              */
/*-------------------------------------------------------------------------*/
__device__ static inline uchar4 yuv2rgba(uint8_t y, uint8_t u, uint8_t v)
{
    int c = int(y) - 16;
    int d = int(u) - 128;
    int e = int(v) - 128;
    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;
    r = min(max(r,0),255); g = min(max(g,0),255); b = min(max(b,0),255);
    return make_uchar4(b,g,r,255);          // BGRA
}

__global__ void nv12_to_bgra(cudaSurfaceObject_t surfY,
                              cudaSurfaceObject_t surfUV,
                              uchar4 *dst,
                              int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x>=W || y>=H) return;

    /* 读 Y */
    uint8_t Y;
    surf2Dread(&Y, surfY, x, y);

    /* 读 UV - NV12: 每 2x2 share one UV */
    uint8_t2 UV;
    surf2Dread(&UV, surfUV, (x/2), (y/2));
    uchar4 c = yuv2rgba(Y, UV.x, UV.y);
    dst[y*W + x] = c;
}

/*-------------------------------------------------------------------------*/
/* 5. 主函数                                                               */
/*-------------------------------------------------------------------------*/
int main()
{
    const UINT W = 1280, H = 720;

    DxObjects   dx;
    CudaObjects cu;

    /* ① DX11 创建 NV12 + NT handle */
    initDx11(dx, W, H);

    /* ② CUDA 导入 & 映射 */
    initCuda(cu, dx);

    /* ③ 启动 CUDA kernel */
    dim3 blk(16,16);
    dim3 grd( (W+blk.x-1)/blk.x, (H+blk.y-1)/blk.y );
    nv12_to_bgra<<<grd,blk,0,cu.stream>>>(cu.surf[0], cu.surf[1],
                                          (uchar4*)cu.d_rgba, W, H);
    check(cudaStreamSynchronize(cu.stream), "kernel sync");

    /* ④ 拷回 CPU & 保存图片 */
    std::vector<uint8_t> host(W*H*4);
    check(cudaMemcpy(host.data(), cu.d_rgba, host.size(), cudaMemcpyDeviceToHost),
          "memcpy D→H");
    write_tga("nv12_cuda_out.tga", host.data(), W, H);
    printf("Wrote nv12_cuda_out.tga\n");

    /* ⑤ 清理 -------------------------------------------------------------*/
    cudaDestroySurfaceObject(cu.surf[0]);
    cudaDestroySurfaceObject(cu.surf[1]);
    cudaFreeMipmappedArray(cu.mip[0]);
    cudaFreeMipmappedArray(cu.mip[1]);
    cudaDestroyExternalMemory(cu.extMem);
    cudaFree(cu.d_rgba);
    cudaStreamDestroy(cu.stream);

    CloseHandle(dx.ntHandle);
    dx.tex ->Release();
    dx.ctx ->Release();
    dx.dev ->Release();
    return 0;
}