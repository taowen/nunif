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
from nunif.logger import logger
from iw3.backward_warp import apply_divergence_nn_LR

# 1. 读取图片
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

# 2. 加载深度模型
from iw3.depth_model_factory import create_depth_model
depth_model = create_depth_model("Distill_Any_S")
depth_model.load(gpu=[0], resolution=None)

# 3. 推理深度
with torch.inference_mode():
    x = x.to(depth_model.device)
    logger.debug(f"x moved to device: {x.device}, shape: {x.shape}")
    depth = depth_model.infer(x, tta=False, low_vram=False, enable_amp=True, edge_dilation=0, depth_aa=False)
    logger.debug(f"depth raw output shape: {depth.shape}, dtype: {depth.dtype}")
    depth = depth_model.minmax_normalize_chw(depth)  # BCHW
    logger.debug(f"depth normalized shape: {depth.shape}, min: {depth.min().item()}, max: {depth.max().item()}")

# 4. 调用 apply_divergence_nn_LR
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_SYM_URL

side_model_path = ROW_FLOW_V3_SYM_URL
with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(side_model_path, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = True
side_model.symmetric = True

with torch.inference_mode():
    left_list = []
    right_list = []
    for i in range(x.shape[0]):
        logger.debug(f"apply_divergence_nn_LR input x[{i}] shape: {x[i:i+1].shape}, depth[{i}] shape: {depth[i:i+1].shape}")
        left, right = apply_divergence_nn_LR(
            side_model,
            x[i:i+1],  # 保持 batch 维度
            depth[i:i+1],  # 这里去掉 .unsqueeze(0)
            divergence=2.0,
            convergence=0.5,
            steps=None,
            mapper='none',
            synthetic_view='both',
            preserve_screen_border=False,
            enable_amp=True
        )
        logger.debug(f"left output shape: {left.shape}, right output shape: {right.shape}")
        left_list.append(left.squeeze(0).cpu())
        right_list.append(right.squeeze(0).cpu())

# 8. 保存
os.makedirs("tmp", exist_ok=True)
for idx, (left, right) in enumerate(zip(left_list, right_list)):
    logger.debug(f"Saving left_eye_{idx}.png shape: {left.shape}, right_eye_{idx}.png shape: {right.shape}")
    TF.to_pil_image(left).save(f"tmp/left_eye_{idx}.png")
    TF.to_pil_image(right).save(f"tmp/right_eye_{idx}.png")
print('done')