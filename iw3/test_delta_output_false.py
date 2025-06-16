# input a rgb image from disk: waifu2x/docs/images/miku_128.png
# calc depth map using distll any depth small
# use row_flow_v3 with delta_output=False
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
# 输入格式: [rgb(3), depth(1), divergence(1), convergence(1), grid(2)] 共8通道
# 这里只用最基础的合成，divergence/convergence/grid用默认值
rgb = x[0]  # 3CHW
# resize depth to match rgb
depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=rgb.shape[1:], mode="bilinear", align_corners=True).squeeze(0)

# divergence, convergence, grid 需要构造
divergence = torch.zeros_like(depth)
convergence = torch.zeros_like(depth)
# grid: 2CHW, meshgrid normalized to [-1, 1]
_, h, w = depth.shape
yy, xx = torch.meshgrid(
    torch.linspace(-1, 1, h),
    torch.linspace(-1, 1, w),
    indexing="ij"
)
grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # 1,2,H,W
grid = grid.to(depth)
grid = grid[0]

# 拼接成8通道
input_8ch = torch.cat([rgb, depth, divergence, convergence, grid], dim=0).unsqueeze(0)  # 1,8,H,W

# 5. 加载 row_flow_v3
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_URL

with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(ROW_FLOW_V3_URL, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = False  # 关键
side_model.symmetric = True      # 新增这一行

# 6. 推理左右眼
with torch.inference_mode():
    output = side_model(input_8ch)
    # output: 1,3,H,W (合成后的左右眼并排图像)
    if isinstance(output, tuple):
        output = output[0]
    output = output.squeeze(0).cpu()
    print("output.shape:", output.shape)

# 7. 拆分左右眼
if output.shape[2] == output.shape[1] * 2:
    # (3, H, 2W)
    left = output[:, :, :output.shape[2]//2]
    right = output[:, :, output.shape[2]//2:]
elif output.shape[0] == 6 and output.shape[1] == output.shape[2]:
    # (6, H, W) -> 2x(3, H, W)
    left = output[:3, :, :]
    right = output[3:, :, :]
else:
    raise ValueError(f"Unexpected output shape: {output.shape}")

# 8. 保存
os.makedirs("tmp", exist_ok=True)
TF.to_pil_image(left).save("tmp/left.png")
TF.to_pil_image(right).save("tmp/right.png")
print('done')