import torch
import torch.onnx
import os
from os import path
from nunif.utils.ui import HiddenPrints, TorchHubDir
from iw3.dilation import dilate_edge  # 新增导入

# README:
# venv/Scripts/Activate
# python -m iw3.export_onnx


HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")

class DistillAnyDepthWithDilation(torch.nn.Module):
    def __init__(self, base_model, edge_dilation=2):
        super().__init__()
        self.base_model = base_model
        self.edge_dilation = edge_dilation

    def forward(self, x):
        out = self.base_model(x)
        # out: (B, H, W) or (B, 1, H, W)
        if out.ndim == 3:
            out = out.unsqueeze(1)
        out = dilate_edge(out, self.edge_dilation)
        return out

def export_distill_any_depth_to_onnx(model_size='s', input_size=392, edge_dilation=2):  # 新增 edge_dilation 参数
    """
    Export Distill Any Depth model to ONNX format, with optional edge dilation.
    Args:
        model_size: 's' for small, 'b' for base, 'l' for large
        input_size: input image size (must be multiple of 14)
        edge_dilation: number of dilation iterations (int)
    """
    # Map model size to encoder type
    size_to_encoder = {
        's': 'v2_vits',
        'b': 'v2_vitb',
        'l': 'v2_vitl'
    }
    encoder = size_to_encoder[model_size]
    
    print(f"Exporting Distill Any Depth {model_size.upper()} model...")
    
    # Ensure input size is multiple of 14
    if input_size % 14 != 0:
        input_size = input_size + (14 - input_size % 14)
        print(f"Adjusted input size to {input_size} to be multiple of 14")
    
    # Load the model
    if not os.getenv("IW3_DEBUG"):
        with HiddenPrints():
            model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                   "DistillAnyDepth", encoder=encoder,
                                   verbose=False, trust_repo=True)
    else:
        model = torch.hub.load("../Depth-Anything_iw3",
                               "DistillAnyDepth", encoder=encoder, source="local",
                               verbose=False, trust_repo=True)
    
    model.eval()
    
    # 包装模型
    model = DistillAnyDepthWithDilation(model, edge_dilation=edge_dilation)

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Output filename
    output_file = path.join(HUB_MODEL_DIR, f"distill_any_depth_{model_size}.onnx")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"ONNX model exported to {output_file}")

if __name__ == "__main__":
    with TorchHubDir(HUB_MODEL_DIR):
        # Export small model with default input size and dilation
        export_distill_any_depth_to_onnx('s', edge_dilation=2)
