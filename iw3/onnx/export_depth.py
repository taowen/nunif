# input a rgb image from disk: waifu2x/docs/images/miku_128.png
# calc depth map using distll any depth small
# use row_flow_v3 with delta_output=True
# generate left right eye using the delta
# output left right eye rgb image to disk

import torch
from torchvision.transforms import functional as TF
from PIL import Image
import os
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F

from iw3.mapper import get_mapper, resolve_mapper_name
from nunif.utils.ui import TorchHubDir
from nunif.logger import logger
from iw3.backward_warp import apply_divergence_nn_LR

img_path1 = "iw3/figure/convergence.png"
img_path2 = "iw3/figure/divergence.png"
logger.debug(f"load two images")
img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")
x1 = TF.to_tensor(img1)
x2 = TF.to_tensor(img2)
logger.debug(f"x1 shape: {x1.shape}, x2 shape: {x2.shape}")
x = torch.stack([x1, x2], dim=0)  # BCHW, float32, 0-1
logger.debug(f"stacked x shape: {x.shape}")

from iw3.depth_model_factory import create_depth_model
depth_model = create_depth_model("Distill_Any_S")
depth_model.load(gpu=[0], resolution=392)
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_SYM_URL

side_model_path = ROW_FLOW_V3_SYM_URL
with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(side_model_path, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = True
side_model.symmetric = True

mapper_name = resolve_mapper_name(mapper=None, foreground_scale=0.9, metric_depth=False)
logger.debug(f"mapper_name: {mapper_name}")
mapper = get_mapper(mapper_name)

class StereoDepthModule(nn.Module):
    def __init__(self, depth_model, side_model):
        super().__init__()
        self.depth_model_wrapper = depth_model
        self.depth_model = depth_model.model
        self.side_model_wrapper = side_model
        self.side_model = side_model.model if hasattr(side_model, "model") else side_model

    def forward(self, x):
        # x: BCHW, float32, 0-1, BGRA format from D3D11 texture (4 channels)
        # D3D11 textures are typically in BGRA format, need to convert to RGB

        # Reorder BGRA to RGB and drop alpha channel
        x_rgb = x[:, [2, 1, 0], :, :]  # BGRA to RGB

        depth = self.depth_model_wrapper.infer(
            x_rgb, tta=False, low_vram=False, enable_amp=False, edge_dilation=1, depth_aa=False
        )
        depth = self.depth_model_wrapper.minmax_normalize_chw(depth)  # BCHW
        depth = mapper(depth)

        left, right = apply_divergence_nn_LR(
            self.side_model_wrapper,
            x_rgb,  # Use RGB data for side model
            depth,
            divergence=2.0,
            convergence=0.5,
            steps=None,
            mapper='none',
            synthetic_view='both',
            preserve_screen_border=False,
            enable_amp=False
        )
        # resize left/right to half width
        B, C, H, W = left.shape
        left_half = F.interpolate(left, size=(H, W // 2), mode="bilinear", align_corners=False)
        right_half = F.interpolate(right, size=(H, W // 2), mode="bilinear", align_corners=False)
        # concat along width
        half_sbs = torch.cat([left_half, right_half], dim=3)  # (B, C, H, W)
        
        # Convert RGB output back to BGRA format for D3D11 texture compatibility
        half_sbs_bgr = half_sbs[:, [2, 1, 0], :, :]  # RGB to BGR

        # Add alpha channel (set to 1.0)
        B, C, H, W = half_sbs.shape
        alpha_channel = torch.ones((B, 1, H, W), dtype=half_sbs.dtype, device=half_sbs.device)
        half_sbs_bgra_final = torch.cat([half_sbs_bgr, alpha_channel], dim=1)  # (B, 4, H, W) - BGRA
        
        return half_sbs_bgra_final

# 构建模型
stereo_module = StereoDepthModule(depth_model, side_model).eval()

# 修改测试数据为 BGRA 格式以匹配 D3D11 纹理格式
logger.debug(f"Converting RGB to BGRA format to match D3D11 texture format")
# 将 RGB 转换为 BGRA (重新排列通道顺序)
x_rgb = torch.stack([x1, x2], dim=0)
x_bgr = x_rgb[:, [2, 1, 0], :, :]  # RGB to BGR
B, C, H, W = x_rgb.shape
alpha_channel = torch.ones((B, 1, H, W), dtype=x_rgb.dtype, device=x_rgb.device)
x = torch.cat([x_bgr, alpha_channel], dim=1)  # (B, 4, H, W) - BGRA

logger.debug(f"BGRA stacked x shape: {x.shape}")  # Should be (2, 4, H, W)

with torch.inference_mode():
    x = x.to(depth_model.device)
    logger.debug(f"x moved to device: {x.device}, shape: {x.shape}")
    half_sbs = stereo_module(x)

    # export to onnx
    output_path = "stereo_module_half_sbs.onnx"
    logger.info(f"Exporting ONNX model to {output_path}")
    torch.onnx.export(
        stereo_module,
        (x,),
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["half_sbs"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "half_sbs": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    logger.info(f"ONNX model saved to {output_path}")
    logger.info(f"Model expects BGRA input (4 channels) from D3D11 texture and outputs BGRA (4 channels) for D3D11 compatibility")

os.makedirs("tmp", exist_ok=True)
for idx in range(half_sbs.shape[0]):
    logger.debug(f"Saving half_sbs_{idx}.png shape: {half_sbs[idx].shape}")
    TF.to_pil_image(half_sbs[idx]).save(f"tmp/half_sbs_{idx}.png")
print('done')

# 验证 ONNX 推理结果和 PyTorch 输出一致性
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("stereo_module_half_sbs.onnx", providers=['CPUExecutionProvider'])
x_numpy = x.cpu().numpy()
onnx_outputs = ort_session.run(None, {"input": x_numpy})
onnx_half_sbs = onnx_outputs[0]
torch_half_sbs = half_sbs.cpu().numpy()

def compare_outputs(torch_out, onnx_out, name):
    diff = np.abs(torch_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

compare_outputs(torch_half_sbs, onnx_half_sbs, "Half Side-by-Side")