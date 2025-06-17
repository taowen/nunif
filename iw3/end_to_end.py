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
from iw3.backward_warp import apply_divergence_nn_LR

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
    depth = depth_model.infer(x[0], tta=False, low_vram=False, enable_amp=True)
    depth = depth_model.minmax_normalize_chw(depth)  # 1CHW

# 4. 调用 apply_divergence_nn_LR
from nunif.models import load_model
from iw3.utils import HUB_MODEL_DIR, ROW_FLOW_V3_SYM_URL

side_model_path = ROW_FLOW_V3_SYM_URL
with TorchHubDir(HUB_MODEL_DIR):
    side_model = load_model(side_model_path, weights_only=True, device_ids=[0])[0].eval()
side_model.delta_output = True
side_model.symmetric = True

with torch.inference_mode():
    left, right = apply_divergence_nn_LR(
        side_model,
        x,
        depth.unsqueeze(0),
        divergence=2.0,
        convergence=0.5,
        steps=None,
        mapper='none',
        synthetic_view='both',
        preserve_screen_border=False,
        enable_amp=True
    )
    left = left.squeeze(0).cpu()
    right = right.squeeze(0).cpu()

# 8. 保存
os.makedirs("tmp", exist_ok=True)
TF.to_pil_image(left).save("tmp/left_eye.png")
TF.to_pil_image(right).save("tmp/right_eye.png")
print('done')