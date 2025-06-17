# python -m run end_to_end
# python -m run iw3
# compare the left right eye image

import subprocess
from PIL import Image, ImageChops
import numpy as np

# 1. 运行 end_to_end.py, 获得 tmp/left_eye.png, tmp/right_eye.png
subprocess.run("python -m iw3.end_to_end", shell=True, check=True)

# 2. 用 CLI 跑一遍，获得 tmp/side_by_side.png
subprocess.run('python -m iw3 --input waifu2x/docs/images/miku_128.png --output tmp/side_by_side.png --method row_flow_v3_sym --depth-model Distill_Any_S --format png --edge-dilation 0', shell=True,check=True)

# 3. 比较图片

# 读取 end_to_end.py 生成的左右眼图片
left_eye = Image.open("tmp/left_eye.png").convert("RGB")
right_eye = Image.open("tmp/right_eye.png").convert("RGB")

# 读取 side_by_side.png，并分割为左右两半
side_by_side = Image.open("tmp/side_by_side.png").convert("RGB")
w, h = side_by_side.size
left_cli = side_by_side.crop((0, 0, w // 2, h))
right_cli = side_by_side.crop((w // 2, 0, w, h))

# 转为 numpy 数组
left_eye_np = np.array(left_eye)
right_eye_np = np.array(right_eye)
left_cli_np = np.array(left_cli)
right_cli_np = np.array(right_cli)

# 计算差异
left_diff = np.abs(left_eye_np.astype(np.int32) - left_cli_np.astype(np.int32))
right_diff = np.abs(right_eye_np.astype(np.int32) - right_cli_np.astype(np.int32))

print("左眼最大像素差:", left_diff.max())
print("右眼最大像素差:", right_diff.max())

# 判断是否完全一致
if left_diff.max() == 0 and right_diff.max() == 0:
    print("图片完全一致")
else:
    print("图片有差异")