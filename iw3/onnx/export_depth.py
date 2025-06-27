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

img_path1 = "C:/games/nunif/debug_textures/frame_5_color_conv.png"
img_path2 = "C:/games/nunif/debug_textures/frame_5_color_conv.png"
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
        # x: BCHW, float32, 0-1, RGBA format from D3D11 texture (4 channels)
        # D3D11 DXGI_FORMAT_R32G32B32A32_FLOAT textures are in RGBA order, not BGRA

        # Extract RGB channels (drop alpha channel for processing)
        x_rgb = x[:, :3, :, :]  # Take first 3 channels (RGB)
        B, C, H, W = x_rgb.shape
        alpha_channel = torch.ones((B, 1, H, W), dtype=x_rgb.dtype, device=x_rgb.device)
        x_rgb_restored = torch.cat([x_rgb, alpha_channel], dim=1)  # (B, 4, H, W) - RGBA
        # 转换为NHWC格式输出
        return x_rgb_restored.permute(0, 2, 3, 1)  # (B, H, W, 4)

        # depth = self.depth_model_wrapper.infer(
        #     x_rgb, tta=False, low_vram=False, enable_amp=False, edge_dilation=1, depth_aa=False
        # )
        # depth = self.depth_model_wrapper.minmax_normalize_chw(depth)  # BCHW
        # depth = mapper(depth)

        # left, right = apply_divergence_nn_LR(
        #     self.side_model_wrapper,
        #     x_rgb,  # Use RGB data for side model
        #     depth,
        #     divergence=2.0,
        #     convergence=0.5,
        #     steps=None,
        #     mapper='none',
        #     synthetic_view='both',
        #     preserve_screen_border=False,
        #     enable_amp=False
        # )
        
        # # Only output left eye for color diagnostics
        # # Keep RGB output as RGB format (no channel reordering needed)  
        # # Add alpha channel (set to 1.0)
        # B, C, H, W = left.shape
        # alpha_channel = torch.ones((B, 1, H, W), dtype=left.dtype, device=left.device)
        # left_rgba = torch.cat([left, alpha_channel], dim=1)  # (B, 4, H, W) - RGBA
        
        # # 转换为NHWC格式输出
        # left_rgba_nhwc = left_rgba.permute(0, 2, 3, 1)  # (B, H, W, 4)
        
        # return left_rgba_nhwc

# 构建模型
stereo_module = StereoDepthModule(depth_model, side_model).eval()

# 修改测试数据为 RGBA 格式以匹配 D3D11 纹理格式
logger.debug(f"Using RGBA format to match D3D11 DXGI_FORMAT_R32G32B32A32_FLOAT texture format")
# 保持 RGB 顺序，添加 alpha 通道
x_rgb = torch.stack([x1, x2], dim=0)
B, C, H, W = x_rgb.shape
alpha_channel = torch.ones((B, 1, H, W), dtype=x_rgb.dtype, device=x_rgb.device)
x = torch.cat([x_rgb, alpha_channel], dim=1)  # (B, 4, H, W) - RGBA

logger.debug(f"RGBA stacked x shape: {x.shape}")  # Should be (2, 4, H, W)

with torch.inference_mode():
    x = x.to(depth_model.device)
    logger.debug(f"x moved to device: {x.device}, shape: {x.shape}")
    left_eye = stereo_module(x)

    # export to onnx
    output_path = "stereo_module_left_eye.onnx"
    logger.info(f"Exporting ONNX model to {output_path}")
    torch.onnx.export(
        stereo_module,
        (x,),
        output_path,
        opset_version=18,
        input_names=["input"],
        output_names=["left_eye"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "left_eye": {0: "batch_size", 2: "height", 3: "width"},
        },
    )
    logger.info(f"ONNX model saved to {output_path}")
    logger.info(f"Model expects RGBA input (4 channels) from D3D11 texture and outputs left eye RGBA (4 channels)")

os.makedirs("tmp", exist_ok=True)
for idx in range(left_eye.shape[0]):
    logger.debug(f"Saving left_eye_{idx}.png shape: {left_eye[idx].shape}")
    # 由于输出已经是NHWC格式(H, W, 4)，需要转换为CHW格式用于保存
    left_eye_chw = left_eye[idx].permute(2, 0, 1)  # (H, W, 4) -> (4, H, W)
    # 只保存RGB通道
    left_eye_rgb = left_eye_chw[:3]  # 取前3个通道 (3, H, W)
    TF.to_pil_image(left_eye_rgb).save(f"tmp/left_eye_{idx}.png")
print('done')

# 验证 ONNX 推理结果和 PyTorch 输出一致性
import onnxruntime as ort
import numpy as np

ort_session = ort.InferenceSession("stereo_module_left_eye.onnx", providers=['CPUExecutionProvider'])
x_numpy = x.cpu().numpy()
onnx_outputs = ort_session.run(None, {"input": x_numpy})
onnx_left_eye = onnx_outputs[0]
torch_left_eye = left_eye.cpu().numpy()

def compare_outputs(torch_out, onnx_out, name):
    diff = np.abs(torch_out - onnx_out)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

compare_outputs(torch_left_eye, onnx_left_eye, "Left Eye")

# 保存 ONNX 推理输出的图片
for idx in range(onnx_left_eye.shape[0]):
    logger.debug(f"Saving onnx_left_eye_{idx}.png shape: {onnx_left_eye[idx].shape}")
    # ONNX输出也是NHWC格式(H, W, 4)，需要转换为CHW格式用于保存
    onnx_tensor = torch.from_numpy(onnx_left_eye[idx])  # (H, W, 4)
    onnx_tensor_chw = onnx_tensor.permute(2, 0, 1)  # (H, W, 4) -> (4, H, W)
    # 只保存RGB通道
    onnx_tensor_rgb = onnx_tensor_chw[:3]  # 取前3个通道 (3, H, W)
    TF.to_pil_image(onnx_tensor_rgb).save(f"tmp/onnx_left_eye_{idx}.png")

print(f"Saved PyTorch outputs: tmp/left_eye_0.png, tmp/left_eye_1.png")
print(f"Saved ONNX outputs: tmp/onnx_left_eye_0.png, tmp/onnx_left_eye_1.png")