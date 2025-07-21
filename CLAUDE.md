# Nunif Project Instructions

## Virtual Environment (venv) Usage

### Correct way to activate and use venv:

**Windows (Git Bash/WSL):**
```bash
source venv/Scripts/activate
python -m iw3.onnx.export_iw3
deactivate  # when done
```

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
python -m iw3.onnx.export_iw3
deactivate
```

**Windows (cmd):**
```cmd
venv\Scripts\activate.bat
python -m iw3.onnx.export_iw3
deactivate
```

**Linux/Mac:**
```bash
source venv/bin/activate
python -m iw3.onnx.export_iw3
deactivate
```

### What NOT to do:
- ❌ `venv/Scripts/python.exe -m module` (跳过激活)
- ❌ 直接用系统Python而不激活venv

### Notes:
- 激活venv后，prompt会显示 `(venv)` 前缀
- 激活后所有python命令都会使用venv中的Python和包
- 运行完成后用 `deactivate` 退出虚拟环境

## 子项目
- 3d-player: 参考 3d-player/CLAUDE.md 和 3d-player/README.md

## OpenVINO测试依赖修复
3d-player的OpenVINO测试需要export_iw3导出的ONNX模型，修复步骤：

```bash
# 1. 激活虚拟环境并生成ONNX模型
source venv/Scripts/activate
python -m iw3.onnx.export_iw3

# 2. 创建3d-player需要的目录结构
mkdir -p "3d-player/deps/nunif"

# 3. 复制ONNX模型到正确位置
cp stereo_module_half_sbs.onnx "3d-player/deps/nunif/"

# 4. 构建并运行测试
cd "3d-player"
powershell -ExecutionPolicy Bypass -File build.ps1

# 5. 运行OpenVINO测试验证修复
build/Debug/integration-test.exe "Stereo Depth NPU ONNX inference" -s
```

## Compilation and Running

- 记录刚才的编译运行命令