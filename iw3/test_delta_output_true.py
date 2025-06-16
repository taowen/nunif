# input a rgb image from disk: waifu2x/docs/images/miku_128.png
# calc depth map using distll any depth small
# use row_flow_v3 with delta_output=True
# generate left right eye using the delta
# output left right eye rgb image to disk

import torch
from torchvision.transforms import functional as TF
from PIL import Image
import os

from nunif.utils.ui import TorchHubDir

# 1. 读取图片
img_path = "waifu2x/docs/images/miku_128.png"
img = Image.open(img_path).convert("RGB")
x = TF.to_tensor(img).unsqueeze(0)  # BCHW, float32, 0-1

# 2. 加载深度模型
from iw3.depth_model_factory import create_depth_model
depth_model = create_depth_model("Distill_Any_S")
depth_model.load(gpu=[0], resolution=None)

# 3. 推理深度
with torch.inference_mode():
    x = x.to(depth_model.device)
    depth = depth_model.infer(x[0], tta=False, low_vram=False, enable_amp=False)
    depth = depth_model.minmax_normalize_chw(depth)  # 1CHW

# 4. 构造 row_flow_v3 输入
rgb = x[0]  # 3CHW
depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=rgb.shape[1:], mode="bilinear", align_corners=True).squeeze(0)
divergence = torch.zeros_like(depth)
convergence = torch.zeros_like(depth)
_, h, w = depth.shape
yy, xx = torch.meshgrid(
    torch.linspace(-1, 1, h),
    torch.linspace(-1, 1, w),
    indexing="ij"
)
grid = torch.stack([xx, yy], dim=0).to(depth)
input_8ch = torch.cat([rgb, depth, divergence, convergence, grid], dim=0).unsqueeze(0)  # 1,8,H,W

# 5. 加载 row_flow_v3
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_URL

with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = True  # 关键
side_model.symmetric = True     # 生成左右眼

# 6. 推理 delta
with torch.inference_mode():
    # 只取 depth, divergence, convergence 这3个通道
    input_3ch = input_8ch[:, 3:6, :, :]  # 1,3,H,W
    delta = side_model(input_3ch)
    if isinstance(delta, tuple):
        delta = delta[0]
    delta = delta.squeeze(0)  # 不要 .cpu()

# 7. 用 delta 合成左右眼
def warp(rgb, grid, delta, delta_scale):
    # 参考 RowFlowV3._warp
    rgb = rgb.unsqueeze(0).to(torch.float32)  # 1,3,H,W
    grid = grid.unsqueeze(0).to(torch.float32)  # 1,2,H,W
    delta = delta.unsqueeze(0).to(torch.float32)  # 1,2,H,W
    # 保证 delta_scale 在同一 device
    delta_scale = torch.tensor(delta_scale, dtype=torch.float32, device=grid.device)
    # grid + delta * scale
    grid_left = grid + delta * delta_scale
    grid_right = grid - delta * delta_scale
    # grid_sample 需要 (B,H,W,2)
    grid_left = grid_left.permute(0,2,3,1)
    grid_right = grid_right.permute(0,2,3,1)
    left = torch.nn.functional.grid_sample(rgb, grid_left, mode="bilinear", padding_mode="border", align_corners=True)
    right = torch.nn.functional.grid_sample(rgb, grid_right, mode="bilinear", padding_mode="border", align_corners=True)
    return left.squeeze(0).cpu(), right.squeeze(0).cpu()

delta_scale = float(side_model.delta_scale)
left, right = warp(rgb, grid, delta, delta_scale)

# 8. 保存
os.makedirs("tmp", exist_ok=True)
TF.to_pil_image(left).save("tmp/left_true.png")
TF.to_pil_image(right).save("tmp/right_true.png")
print('done')