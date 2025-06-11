import torch
import torch.onnx
import os
from os import path
from nunif.utils.ui import HiddenPrints, TorchHubDir




HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")

def export_distill_any_depth_to_onnx():
    # Load the model (similar to how it's done in your depth_anything_model.py)
    encoder = "v2_vits"  # or "v2_vitb", "v2_vitl" for different sizes
    
    if not os.getenv("IW3_DEBUG"):
        model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                               "DistillAnyDepth", encoder=encoder,
                               verbose=False, trust_repo=True)
    else:
        model = torch.hub.load("../Depth-Anything_iw3",
                               "DistillAnyDepth", encoder=encoder, source="local",
                               verbose=False, trust_repo=True)
    
    model.eval()
    
    # Create dummy input (adjust size as needed)
    # The model expects input size to be multiple of 14
    dummy_input = torch.randn(1, 3, 392, 392)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f"{HUB_MODEL_DIR}\distill_any_depth.onnx",
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
    print("ONNX model exported to distill_any_depth.onnx")

if __name__ == "__main__":
    with TorchHubDir(HUB_MODEL_DIR):
        export_distill_any_depth_to_onnx()
