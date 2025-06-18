import torch
import torch.onnx
import os
from os import path
from nunif.utils.ui import HiddenPrints, TorchHubDir
from iw3.dilation import dilate_edge  # 新增导入
from iw3.depth_model_factory import create_depth_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_SYM_URL
from nunif.models import load_model
from iw3.backward_warp import apply_divergence_nn_LR

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

class End2EndStereoModel(torch.nn.Module):
    def __init__(self, depth_model, side_model, divergence=2.0, convergence=0.5):
        super().__init__()
        self.depth_model = depth_model
        self.side_model = side_model
        self.divergence = divergence
        self.convergence = convergence

    def forward(self, x):
        # x: (B, 3, H, W), float32, 0-1
        # 1. 深度推理
        depth = self.depth_model(x)
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        # 2. minmax 归一化
        min_val = depth.amin(dim=[2, 3], keepdim=True)
        max_val = depth.amax(dim=[2, 3], keepdim=True)
        depth = (depth - min_val) / (max_val - min_val + 1e-8)
        # 3. row_flow_v3 左右眼
        left, right = apply_divergence_nn_LR(
            self.side_model,
            x,
            depth,
            divergence=self.divergence,
            convergence=self.convergence,
            steps=None,
            mapper='none',
            synthetic_view='both',
            preserve_screen_border=False,
            enable_amp=False  # ONNX 不支持 AMP
        )
        return left, right

def export_end2end_to_onnx(model_size='s', input_size=(392, 392), edge_dilation=2, divergence=2.0, convergence=0.5):
    # 1. 加载深度模型
    encoder = {'s': 'v2_vits', 'b': 'v2_vitb', 'l': 'v2_vitl'}[model_size]
    with HiddenPrints():
        depth_model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                                    "DistillAnyDepth", encoder=encoder,
                                    verbose=False, trust_repo=True)
    depth_model.eval()
    depth_model = DistillAnyDepthWithDilation(depth_model, edge_dilation=edge_dilation)
    # 2. 加载 row_flow_v3
    side_model_path = ROW_FLOW_V3_SYM_URL
    side_model = load_model(side_model_path, weights_only=True, device_ids=[-1])[0].eval()
    side_model.delta_output = True
    side_model.symmetric = True
    # 3. 包装
    model = End2EndStereoModel(depth_model, side_model, divergence, convergence)
    # 4. dummy input
    if isinstance(input_size, int):
        dummy_input = torch.randn(1, 3, input_size, input_size)
    else:
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    # 5. 导出
    output_file = path.join(HUB_MODEL_DIR, f"end2end_stereo_{model_size}.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['left', 'right'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'left': {0: 'batch_size', 2: 'height', 3: 'width'},
            'right': {0: 'batch_size', 2: 'height', 3: 'width'}
        },
        verbose=True
    )
    print(f"ONNX model exported to {output_file}")

if __name__ == "__main__":
    with TorchHubDir(HUB_MODEL_DIR):
        # 用常见视频分辨率导出，确保模型支持非正方形
        export_end2end_to_onnx('s', input_size=(14 * 30, 14 * 20), edge_dilation=2)
