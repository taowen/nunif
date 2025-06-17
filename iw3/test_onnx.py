import torch
import onnxruntime
import numpy as np
from iw3.export_onnx import (
    export_end2end_to_onnx, HUB_MODEL_DIR,
    End2EndStereoModel, DistillAnyDepthWithDilation
)
from nunif.utils.ui import TorchHubDir, HiddenPrints
from os import path
from iw3.utils import ROW_FLOW_V3_SYM_URL
from nunif.models import load_model


def test_end2end_onnx_model(model_size='s', input_size=392, edge_dilation=2, divergence=2.0, convergence=0.5):
    """
    Test End2End ONNX model output against PyTorch model output
    """
    print(f"Testing End2End Stereo {model_size.upper()} ONNX model...")

    # 1. Export the model if it does not exist
    onnx_path = path.join(HUB_MODEL_DIR, f"end2end_stereo_{model_size}.onnx")
    if not path.exists(onnx_path):
        with TorchHubDir(HUB_MODEL_DIR):
            export_end2end_to_onnx(model_size=model_size, input_size=input_size,
                                   edge_dilation=edge_dilation, divergence=divergence, convergence=convergence)

    # 2. Load PyTorch model for verification
    # This logic must be the same as in export_end2end_to_onnx
    encoder = {'s': 'v2_vits', 'b': 'v2_vitb', 'l': 'v2_vitl'}[model_size]
    with HiddenPrints(), TorchHubDir(HUB_MODEL_DIR):
        depth_model_pt = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                        "DistillAnyDepth", encoder=encoder,
                                        verbose=False, trust_repo=True)
    depth_model_pt.eval()
    depth_model_pt = DistillAnyDepthWithDilation(depth_model_pt, edge_dilation=edge_dilation)

    side_model_path = ROW_FLOW_V3_SYM_URL
    with TorchHubDir(HUB_MODEL_DIR):
        side_model_pt = load_model(side_model_path, weights_only=True, device_ids=[-1])[0].eval()
    side_model_pt.delta_output = True
    side_model_pt.symmetric = True

    model_pt = End2EndStereoModel(depth_model_pt, side_model_pt, divergence, convergence)
    model_pt.eval()

    # 3. Create test input
    test_input = torch.randn(1, 3, input_size, input_size)

    # 4. Get PyTorch output
    with torch.no_grad():
        torch_left, torch_right = model_pt(test_input)
        torch_left = torch_left.numpy()
        torch_right = torch_right.numpy()

    # 5. Get ONNX output
    ort_session = onnxruntime.InferenceSession(onnx_path)
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    ort_left, ort_right = ort_session.run(None, ort_inputs)

    # 6. Compare outputs
    # Left eye
    max_diff_left = np.max(np.abs(torch_left - ort_left))
    mean_diff_left = np.mean(np.abs(torch_left - ort_left))

    print(f"Left Eye - Maximum absolute difference: {max_diff_left}")
    print(f"Left Eye - Mean absolute difference: {mean_diff_left}")

    # Right eye
    max_diff_right = np.max(np.abs(torch_right - ort_right))
    mean_diff_right = np.mean(np.abs(torch_right - ort_right))

    print(f"Right Eye - Maximum absolute difference: {max_diff_right}")
    print(f"Right Eye - Mean absolute difference: {mean_diff_right}")

    # Check if the differences are within acceptable range
    if max_diff_left > 5e-4 or max_diff_right > 5e-4:
        print("WARNING: Large difference detected between PyTorch and ONNX outputs!")
    else:
        print("ONNX model output matches PyTorch model output within acceptable range.")


if __name__ == "__main__":
    # Test small model
    with TorchHubDir(HUB_MODEL_DIR):
        test_end2end_onnx_model('s', edge_dilation=2) 