import subprocess
import os
from pathlib import Path

def build_engines_for_targets():
    """为不同目标平台构建引擎"""
    targets = [
        # (compute_capability, cuda_version, arch_name)
        ("7.5", "11.8", "sm_75"),  # RTX 2080, GTX 1660 等
        ("8.0", "11.8", "sm_80"),  # A100
        ("8.6", "12.0", "sm_86"),  # RTX 3080, 3090 等
        ("8.9", "12.0", "sm_89"),  # RTX 4080, 4090 等
    ]
    
    base_onnx = "stereo_module_half_sbs.onnx"
    
    for cc, cuda_ver, arch in targets:
        engine_name = f"stereo_module_half_sbs_{arch}.trt"
        
        cmd = [
            "trtexec",
            f"--onnx={base_onnx}",
            f"--saveEngine={engine_name}",
            "--fp16",
            "--optShapes=input:1x3x392x392",
            "--maxShapes=input:2x3x2160x3840",
            f"--buildOnly",
        ]
        
        print(f"Building engine for {arch} (CUDA {cuda_ver})...")
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    build_engines_for_targets() 