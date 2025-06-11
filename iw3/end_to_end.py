import os
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import argparse
import numpy as np


def batch_preprocess_depth(x, lower_bound=392, max_aspect_ratio=4):
    """预处理图像用于深度估计 - 从 depth_anything_model.py 简化"""
    B, C, H, W = x.shape
    
    # resize
    ensure_multiple_of = 14
    if W < H:
        scale_factor = lower_bound / W
    else:
        scale_factor = lower_bound / H
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)
    
    # Limit aspect ratio to avoid OOM
    if new_h < new_w:
        new_w = min(new_w, int(max_aspect_ratio * new_h))
    else:
        new_h = min(new_h, int(max_aspect_ratio * new_w))
    
    new_h -= new_h % ensure_multiple_of
    new_w -= new_w % ensure_multiple_of
    if new_h < lower_bound:
        new_h = lower_bound
    if new_w < lower_bound:
        new_w = lower_bound
    
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    x = torch.clamp(x, 0, 1)
    
    # normalize
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    stdv = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).reshape(1, 3, 1, 1)
    x = (x - mean) / stdv
    return x


@torch.inference_mode()
def infer_depth(model, im, device="cpu"):
    """使用深度估计模型推理深度"""
    # 转换PIL图片到tensor
    if not torch.is_tensor(im):
        x = TF.to_tensor(im).unsqueeze(0).to(device)
    else:
        x = im.to(device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
    
    # 预处理
    x = batch_preprocess_depth(x, lower_bound=392)
    
    # 推理
    with torch.autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', enabled=device.type == 'cuda'):
        out = model(x).unsqueeze(dim=1)
    
    if out.dtype != torch.float32:
        out = out.to(torch.float32)
    out = torch.nan_to_num(out)
    
    # 反转用于兼容
    out = -out
    
    # 归一化到0-1
    out = out.squeeze(0)
    return out


def get_none_mapper():
    """简化的mapper - 不做任何变换"""
    return lambda x: x


def make_grid(batch, width, height, device):
    """创建网格坐标"""
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, height, device=device),
                                    torch.linspace(-1, 1, width, device=device), indexing="ij")
    mesh_y = mesh_y.reshape(1, 1, height, width).expand(batch, 1, height, width)
    mesh_x = mesh_x.reshape(1, 1, height, width).expand(batch, 1, height, width)
    grid = torch.cat((mesh_x, mesh_y), dim=1)
    return grid


def backward_warp(c, grid, delta, delta_scale):
    """后向变形 - 从 backward_warp.py 简化"""
    grid = grid + delta * delta_scale
    if c.shape[2] != grid.shape[2] or c.shape[3] != grid.shape[3]:
        grid = F.interpolate(grid, size=c.shape[-2:],
                             mode="bilinear", align_corners=True, antialias=False)
    grid = grid.permute(0, 2, 3, 1)
    
    z = F.grid_sample(c, grid, mode="bicubic", padding_mode="border", align_corners=True)
    z = torch.clamp(z, 0, 1)
    return z


def make_input_tensor_for_symmetric(depth, divergence, convergence, image_width):
    """为对称模型创建输入张量 - 从 backward_warp.py 简化"""
    H, W = depth.shape[-2:]
    
    # 创建散度和收敛特征
    divergence_value = divergence * 0.01
    convergence_value = convergence
    
    divergence_feat = torch.full((H, W), divergence_value, device=depth.device, dtype=depth.dtype)
    convergence_feat = torch.full((H, W), convergence_value, device=depth.device, dtype=depth.dtype)
    
    # 创建网格
    mesh_y, mesh_x = torch.meshgrid(torch.linspace(-1, 1, H, device=depth.device),
                                    torch.linspace(-1, 1, W, device=depth.device), indexing="ij")
    grid = torch.stack((mesh_x, mesh_y), 0)  # CHW
    
    # 组合输入
    input_tensor = torch.cat([
        depth,
        divergence_feat.unsqueeze(0),
        convergence_feat.unsqueeze(0),
        grid,
    ], dim=0)
    
    return input_tensor


def apply_divergence_symmetric(model, c, depth, divergence, convergence, enable_amp=True):
    """使用对称模型应用散度 - 从 backward_warp.py 简化"""
    B, _, H, W = depth.shape
    
    # 准备输入
    x = torch.stack([make_input_tensor_for_symmetric(depth[i], divergence, convergence, W)
                     for i in range(depth.shape[0])])
    
    # 推理
    with torch.autocast(device_type='cuda' if depth.device.type == 'cuda' else 'cpu', enabled=enable_amp):
        delta = model(x)
    
    # 创建网格和变形
    grid = make_grid(B, W, H, c.device)
    delta_scale = 1.0 / (W // 2 - 1)
    
    left_eye = backward_warp(c, grid, delta, delta_scale)
    right_eye = backward_warp(c, grid, -delta, delta_scale)
    
    return left_eye, right_eye


def to_pil_image(x):
    """转换tensor到PIL图像"""
    x = torch.clamp(x, 0, 1)
    x = (x * 255).round_().to(torch.uint8).cpu()
    return TF.to_pil_image(x)


def load_depth_model(device):
    """加载 Distill Any Depth Small 模型"""
    try:
        # 使用torch.hub加载模型
        model = torch.hub.load("nagadomi/Depth-Anything_iw3:main",
                               "DistillAnyDepth", encoder="v2_vits",
                               verbose=False, trust_repo=True)
        model = model.to(device).eval()
        print("Successfully loaded Distill Any Depth Small model")
        return model
    except Exception as e:
        print(f"Error loading depth model: {e}")
        print("Please make sure you have internet connection and the model can be downloaded")
        raise


def load_stereo_model(device):
    """加载 row_flow_v3_sym 模型"""
    try:
        # 尝试通过 torch.hub 加载（需要从代码库中）
        print("Attempting to load stereo model...")
        
        # 由于需要模型架构定义，这里简化为返回 None
        # 在实际使用中，用户需要提供完整的模型架构或使用预编译的模型
        print("Stereo model loading requires complete model architecture")
        print("Falling back to grid sampling method...")
        return None
        
    except Exception as e:
        print(f"Error loading stereo model: {e}")
        print("Using fallback grid sampling method instead")
        return None


def process_single_image(input_path, output_dir="./output", divergence=2.0, convergence=0.5):
    """处理单张图像生成左右眼图像"""
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    print(f"Loading image: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    rgb_image = Image.open(input_path).convert("RGB")
    rgb_tensor = TF.to_tensor(rgb_image).to(device)
    print(f"Image size: {rgb_image.size}")
    
    # 加载模型
    print("Loading models...")
    depth_model = load_depth_model(device)
    
    try:
        stereo_model = load_stereo_model(device)
    except Exception as e:
        print(f"Could not load stereo model: {e}")
        print("Falling back to grid sample method...")
        stereo_model = None
    
    # 推理深度
    print("Estimating depth...")
    depth = infer_depth(depth_model, rgb_image, device)
    print(f"Depth shape: {depth.shape}")
    
    # 调整深度图尺寸匹配RGB
    if depth.shape[1:] != rgb_tensor.shape[1:]:
        print(f"Resizing depth from {depth.shape[1:]} to {rgb_tensor.shape[1:]}")
        depth = F.interpolate(
            depth.unsqueeze(0), 
            size=rgb_tensor.shape[1:], 
            mode="bilinear", 
            align_corners=True, 
            antialias=True
        ).squeeze(0)
    
    # 最小-最大归一化深度
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    
    # 生成立体图像
    print("Generating stereo images...")
    rgb_batch = rgb_tensor.unsqueeze(0)  # 添加batch维度
    depth_batch = depth.unsqueeze(0)     # 添加batch维度
    
    if stereo_model is not None:
        # 使用神经网络模型
        left_eye, right_eye = apply_divergence_symmetric(
            stereo_model, rgb_batch, depth_batch, divergence, convergence
        )
    else:
        # 使用简单的网格采样方法作为后备
        print("Using simple grid sampling as fallback...")
        shift_size = divergence * 0.01
        index_shift = depth_batch * shift_size - (shift_size * convergence)
        delta = torch.cat([index_shift, torch.zeros_like(index_shift)], dim=1)
        grid = make_grid(1, rgb_tensor.shape[2], rgb_tensor.shape[1], device)
        
        left_eye = backward_warp(rgb_batch, grid, -delta, 1)
        right_eye = backward_warp(rgb_batch, grid, delta, 1)
    
    # 移除batch维度
    left_eye = left_eye.squeeze(0)
    right_eye = right_eye.squeeze(0)
    
    print(f"Left eye shape: {left_eye.shape}")
    print(f"Right eye shape: {right_eye.shape}")
    
    # 转换为PIL图像并保存
    left_pil = to_pil_image(left_eye)
    right_pil = to_pil_image(right_eye)
    
    # 保存图像
    left_path = os.path.join(output_dir, "left_eye.png")
    right_path = os.path.join(output_dir, "right_eye.png")
    depth_path = os.path.join(output_dir, "depth.png")
    
    left_pil.save(left_path)
    right_pil.save(right_path)
    
    # 保存深度图
    depth_pil = to_pil_image(depth)
    depth_pil.save(depth_path)
    
    # 创建并排图像
    sbs_width = left_pil.width + right_pil.width
    sbs_height = max(left_pil.height, right_pil.height)
    sbs_image = Image.new('RGB', (sbs_width, sbs_height))
    sbs_image.paste(left_pil, (0, 0))
    sbs_image.paste(right_pil, (left_pil.width, 0))
    
    sbs_path = os.path.join(output_dir, "side_by_side.png")
    sbs_image.save(sbs_path)
    
    print(f"Results saved to:")
    print(f"  Left eye: {left_path}")
    print(f"  Right eye: {right_path}")
    print(f"  Depth map: {depth_path}")
    print(f"  Side-by-side: {sbs_path}")
    
    return left_path, right_path, depth_path, sbs_path


# Simple test example
def test_with_default_image():
    """使用默认测试图像进行测试"""
    test_image_path = "waifu2x/docs/images/miku_128.png"
    if os.path.exists(test_image_path):
        print(f"Testing with default image: {test_image_path}")
        process_single_image(test_image_path, "./test_output")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide an input image using --input argument")

HUB_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "hub")
class TorchHubDir:
    def __init__(self, hub_dir):
        self.hub_dir = hub_dir
        self.original_hub_dir = None

    def __enter__(self):
        self.original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(self.hub_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.hub.set_dir(self.original_hub_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end stereo image generation")
    parser.add_argument("--input", type=str, help="Input image path")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--divergence", type=float, default=2.0, help="Divergence strength (0-5)")
    parser.add_argument("--convergence", type=float, default=0.5, help="Convergence plane (0-1)")
    parser.add_argument("--test", action="store_true", help="Run test with default image")
    
    args = parser.parse_args()
    
    try:
        with TorchHubDir(HUB_MODEL_DIR): 
            if args.test:
                test_with_default_image()
            elif args.input:
                process_single_image(
                    input_path=args.input,
                    output_dir=args.output,
                    divergence=args.divergence,
                    convergence=args.convergence
                )
                print("\nProcessing completed successfully!")
            else:
                print("Please provide --input or use --test flag")
                parser.print_help()
    except Exception as e:
        print(f"Error during processing: {e}")
        raise