### 直接回答

- 研究表明，要为 DirectX 11 纹理获取 NT 句柄，需要确保纹理创建时包含 `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` 标志。
- 然后，通过查询 `IDXGIResource1` 接口并调用 `CreateSharedHandle` 获取 NT 句柄。
- 最后，将 NT 句柄传递给 `importD3D11ResourceFromNTHandle` 函数，需提供资源大小。

#### 创建纹理
首先，确保你的 DirectX 11 纹理在创建时设置了 `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` 标志。这允许纹理使用 NT 句柄共享。如果现有纹理未设置此标志，可能需要重新创建。

#### 获取 NT 句柄
- 使用 `QueryInterface` 查询纹理的 `IDXGIResource1` 接口。
- 调用 `IDXGIResource1::CreateSharedHandle` 获取 NT 句柄。

#### 使用 NT 句柄
将获取的 NT 句柄和资源大小传递给 `importD3D11ResourceFromNTHandle` 函数。完成后，记得使用 `CloseHandle` 关闭句柄以避免资源泄漏。

---

### 详细报告

本报告详细探讨如何为 DirectX 11 纹理获取 NT 句柄，并将其用于 CUDA 的 `importD3D11ResourceFromNTHandle` 函数。以下内容基于对相关文档和社区讨论的分析，旨在提供全面的技术指导。

#### 背景与需求
用户有一个 DirectX 11 纹理，并希望获取其 NT 句柄以便通过 `importD3D11ResourceFromNTHandle` 函数导入 CUDA。这涉及 DirectX 11 和 CUDA 之间的资源共享，特别是在使用 NT 句柄的场景下。

NT 句柄是 Windows 系统中用于跨进程或 API 共享资源的机制，特别是在 DirectX 11.1 及以上版本中支持。CUDA 的外部内存 API 允许通过 NT 句柄导入 DirectX 资源，这与传统的图形互操作 API（如 `cudaGraphicsD3D11RegisterResource`）不同，后者直接使用 `ID3D11Resource*`。

#### 创建共享纹理
要获取 NT 句柄，首先必须确保 DirectX 11 纹理创建时启用了共享功能。具体来说，需要在 `D3D11_TEXTURE2D_DESC`（或其他纹理描述结构）的 `MiscFlags` 字段中设置 `D3D11_RESOURCE_MISC_SHARED_NTHANDLE` 标志。

例如：
```c
D3D11_TEXTURE2D_DESC textureDesc = {};
textureDesc.Width = width;
textureDesc.Height = height;
textureDesc.MipLevels = 1;
textureDesc.ArraySize = 1;
textureDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
textureDesc.SampleDesc.Count = 1;
textureDesc.SampleDesc.Quality = 0;
textureDesc.Usage = D3D11_USAGE_DEFAULT;
textureDesc.BindFlags = D3D11_BIND_RENDER_TARGET;
textureDesc.CPUAccessFlags = 0;
textureDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE;

ID3D11Texture2D* pTexture = nullptr;
HRESULT hr = pDevice->CreateTexture2D(&textureDesc, nullptr, &pTexture);
```
如果现有纹理未设置此标志，则无法获取 NT 句柄，需要重新创建纹理。

#### 获取 NT 句柄
创建纹理后，需要通过 `IDXGIResource1` 接口获取 NT 句柄。`IDXGIResource1` 是 DirectX 11.1 及以上版本提供的接口，支持 NT 句柄的创建。步骤如下：

1. 使用 `QueryInterface` 查询纹理的 `IDXGIResource1` 接口：
   ```c
   IDXGIResource1* pDXGIResource = nullptr;
   hr = pTexture->QueryInterface(__uuidof(IDXGIResource1), (void**)&pDXGIResource);
   ```

2. 调用 `IDXGIResource1::CreateSharedHandle` 获取 NT 句柄：
   ```c
   HANDLE sharedHandle = nullptr;
   hr = pDXGIResource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &sharedHandle);
   ```
   这里，`CreateSharedHandle` 的参数包括安全属性和访问权限，通常设置为 `nullptr` 和适当的权限（如读写）。

#### 使用 NT 句柄导入 CUDA
获取 NT 句柄后，可以将其传递给 `importD3D11ResourceFromNTHandle` 函数。用户提供的函数定义如下：
```c
cudaExternalMemory_t importD3D11ResourceFromNTHandle(HANDLE handle, unsigned long long size) {
    cudaExternalMemory_t extMem = NULL;
    cudaExternalMemoryHandleDesc desc = {};

    memset(&desc, 0, sizeof(desc));

    desc.type = cudaExternalMemoryHandleTypeD3D11Resource;
    desc.handle.win32.handle = (void *)handle;
    desc.size = size;
    desc.flags |= cudaExternalMemoryDedicated;

    cudaImportExternalMemory(&extMem, &desc);

    CloseHandle(handle);

    return extMem;
}
```
从函数中可以看到，它使用 CUDA 的外部内存 API，通过 `cudaImportExternalMemory` 导入资源。`cudaExternalMemoryHandleTypeD3D11Resource` 明确支持 DirectX 11 资源的 NT 句柄导入，且必须设置 `cudaExternalMemoryDedicated` 标志，这与文档要求一致。

`size` 参数需要提供资源的大小，通常可以从纹理描述中计算，例如对于 32 位纹理，`size = width * height * 4`。

#### 清理与注意事项
使用完 NT 句柄后，函数中调用 `CloseHandle(handle)` 关闭句柄，这是必要的资源管理步骤，以避免内存泄漏。`IDXGIResource1` 接口和相关资源也需要在适当时候释放。

需要注意的是，NT 句柄的共享通常限于同一适配器（GPU），跨适配器共享可能需要额外的 CPU 介入。此外，确保 DirectX 11.1 或更高版本的支持，因为 NT 句柄功能在早期版本中可能不可用。

#### 支持的格式与限制
CUDA 对 DirectX 11 资源的互操作性有特定的格式要求，例如支持的 DXGI 格式包括：
| Format                          | Description                                      |
|----------------------------------|--------------------------------------------------|
| DXGI_FORMAT_A8_UNORM            | 8-bit alpha channel                             |
| DXGI_FORMAT_B8G8R8A8_UNORM      | 32-bit BGRA format                              |
| DXGI_FORMAT_R16_FLOAT           | 16-bit float format                             |
| DXGI_FORMAT_R32_FLOAT           | 32-bit float format                             |
| ... (更多格式见 NVIDIA 文档)     |                                                  |

这些格式在 [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__D3D11.html) 中有详细列出。

#### 相关讨论与最佳实践
社区讨论表明，`D3D11_RESOURCE_MISC_SHARED_NTHANDLE` 是为 DirectX 12 和 CUDA 等现代 API 设计的共享模型，相比传统的 `D3D11_RESOURCE_MISC_SHARED` 提供了更好的跨进程支持。一些开发者报告在使用不同适配器时遇到问题，建议确保所有设备使用同一 GPU。

#### 总结
通过上述步骤，可以成功为 DirectX 11 纹理获取 NT 句柄，并将其用于 CUDA 的 `importD3D11ResourceFromNTHandle` 函数。关键在于正确设置纹理创建标志、获取 NT 句柄并正确管理资源。

---

### 关键引文
- [D3D11_RESOURCE_MISC_FLAG 枚举](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/ne-d3d11-d3d11_resource_misc_flag)
- [IDXGIResource1::CreateSharedHandle 方法](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgiresource1-createsharedhandle)
- [CUDA 运行时 API：外部内存互操作](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html)
- [共享 ID3D11Buffer 和 ID3D12Resource](https://stackoverflow.com/questions/52869111/sharing-id3d11buffer-and-id3d12resource)
- [ID3D11Device1::OpenSharedResource1 方法](https://learn.microsoft.com/en-us/windows/win32/api/d3d11_1/nf-d3d11_1-id3d11device1-opensharedresource1)