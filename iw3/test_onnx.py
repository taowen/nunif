import torch
import onnxruntime
import numpy as np
from iw3.export_onnx import export_distill_any_depth_to_onnx, HUB_MODEL_DIR
from nunif.utils.ui import TorchHubDir
from os import path
from iw3.dilation import dilate_edge

def test_onnx_model(model_size='s', input_size=392, edge_dilation=2):
    """
    Test ONNX model output against PyTorch model output
    """
    print(f"Testing Distill Any Depth {model_size.upper()} ONNX model...")
    
    # First export the model if not exists
    onnx_path = path.join(HUB_MODEL_DIR, f"distill_any_depth_{model_size}.onnx")
    if not path.exists(onnx_path):
        with TorchHubDir(HUB_MODEL_DIR):
            export_distill_any_depth_to_onnx(model_size, input_size, edge_dilation=edge_dilation)
    
    # Load PyTorch model
    size_to_encoder = {
        's': 'v2_vits',
        'b': 'v2_vitb',
        'l': 'v2_vitl'
    }
    encoder = size_to_encoder[model_size]
    
    model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                          "DistillAnyDepth", encoder=encoder,
                          verbose=False, trust_repo=True)
    model.eval()
    
    # Create test input
    test_input = torch.randn(1, 3, input_size, input_size)
    
    # Get PyTorch output
    with torch.no_grad():
        torch_output = model(test_input)
        if torch_output.ndim == 3:
            torch_output = torch_output.unsqueeze(1)
        torch_output = dilate_edge(torch_output, edge_dilation)
        torch_output = torch_output.numpy()
    
    # Get ONNX output
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare outputs
    max_diff = np.max(np.abs(torch_output - ort_output))
    mean_diff = np.mean(np.abs(torch_output - ort_output))
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if the differences are within acceptable range
    if max_diff > 1e-4:
        print("WARNING: Large difference detected between PyTorch and ONNX outputs!")
    else:
        print("ONNX model output matches PyTorch model output within acceptable range.")

if __name__ == "__main__":
    # Test small model
    with TorchHubDir(HUB_MODEL_DIR):
        test_onnx_model('s', edge_dilation=2) 