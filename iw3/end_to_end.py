import os
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import argparse
from packaging import version as packaging_version
import torch.nn as nn
import copy

import logging

PYTORCH2 = packaging_version.parse(torch.__version__).major >= 2
logger = logging.getLogger("nunif")


def create_device_name(device_id):
    if isinstance(device_id, (list, tuple)):
        assert len(device_id) > 0
        device_id = device_id[0]
    if device_id < 0:
        device_name = "cpu"
    else:
        if torch.cuda.is_available():
            device_name = 'cuda:%d' % device_id
        else:
            raise ValueError("No cuda available. Use `--gpu -1` for CPU.")

    return device_name

def create_device(device_id):
    return torch.device(create_device_name(device_id))

# 添加缺失的模块实现
def pixel_shuffle(x, downscale_factor):
    """Pixel shuffle implementation"""
    if isinstance(downscale_factor, int):
        downscale_factor = (downscale_factor, downscale_factor)
    
    batch_size, channels, height, width = x.size()
    out_channels = channels // (downscale_factor[0] * downscale_factor[1])
    
    x = x.view(batch_size, out_channels, downscale_factor[0], downscale_factor[1], height, width)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(batch_size, out_channels, height * downscale_factor[0], width * downscale_factor[1])
    
    return x


def pixel_unshuffle(x, downscale_factor):
    """Pixel unshuffle implementation"""
    if isinstance(downscale_factor, int):
        downscale_factor = (downscale_factor, downscale_factor)
    
    batch_size, channels, height, width = x.size()
    
    x = x.view(batch_size, channels, height // downscale_factor[0], downscale_factor[0], 
               width // downscale_factor[1], downscale_factor[1])
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(batch_size, channels * downscale_factor[0] * downscale_factor[1], 
               height // downscale_factor[0], width // downscale_factor[1])
    
    return x


def replication_pad2d_naive(x, pad, detach=False):
    """Naive replication padding implementation"""
    left, right, top, bottom = pad
    if detach:
        x = x.detach()
    return F.pad(x, pad, mode='replicate')


class ReplicationPad2d(nn.Module):
    """Replication padding module"""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding
    
    def forward(self, x):
        return F.pad(x, self.padding, mode='replicate')


def bchw_to_bnc(x, window_size):
    """Convert BCHW to BNC format for window attention"""
    B, C, H, W = x.shape
    window_h, window_w = window_size
    
    # Pad if necessary
    pad_h = (window_h - H % window_h) % window_h
    pad_w = (window_w - W % window_w) % window_w
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        H, W = H + pad_h, W + pad_w
    
    # Reshape to windows
    x = x.view(B, C, H // window_h, window_h, W // window_w, window_w)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, nH, nW, wH, wW, C
    x = x.view(B * (H // window_h) * (W // window_w), window_h * window_w, C)
    
    return x


def bnc_to_bchw(x, out_shape, window_size):
    """Convert BNC back to BCHW format"""
    B, C, H, W = out_shape
    window_h, window_w = window_size
    
    # Calculate padded dimensions
    pad_h = (window_h - H % window_h) % window_h
    pad_w = (window_w - W % window_w) % window_w
    H_pad, W_pad = H + pad_h, W + pad_w
    
    # Reshape back
    x = x.view(B, H_pad // window_h, W_pad // window_w, window_h, window_w, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, nH, wH, nW, wW
    x = x.view(B, C, H_pad, W_pad)
    
    # Remove padding
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :H, :W]
    
    return x


def sliced_sdp(q, k, v, num_heads, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Sliced scaled dot product attention"""
    B, N, C = q.shape
    head_dim = C // num_heads
    
    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # B, H, N, D
    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)  # B, H, N, D
    v = v.view(B, N, num_heads, head_dim).transpose(1, 2)  # B, H, N, D
    
    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    if attn_mask is not None:
        if attn_mask.ndim == 2:  # (N, N)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        elif attn_mask.ndim == 3:  # (H, N, N)
            attn_mask = attn_mask.unsqueeze(0)  # (1, H, N, N)
        attn = attn + attn_mask
    
    attn = F.softmax(attn, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=False)
    
    out = torch.matmul(attn, v)  # B, H, N, D
    out = out.transpose(1, 2).contiguous().view(B, N, C)  # B, N, C
    
    return out


class MHA(nn.Module):
    """Multi-Head Attention module to match nunif structure"""
    def __init__(self, embed_dim, num_heads, qkv_dim=None):
        super().__init__()
        if qkv_dim is None:
            assert embed_dim % num_heads == 0
            qkv_dim = embed_dim // num_heads
        self.qkv_dim = qkv_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, qkv_dim * num_heads * 3)
        self.head_proj = nn.Linear(qkv_dim * num_heads, embed_dim)

    def forward(self, x, attn_mask=None, dropout_p=0.0, is_causal=False):
        # x.shape: batch, sequence, feature
        q, k, v = self.qkv_proj(x).split(self.qkv_dim * self.num_heads, dim=-1)
        x = sliced_sdp(q, k, v, self.num_heads, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
        x = self.head_proj(x)
        return x


class WindowMHA2d(nn.Module):
    """Window-based Multi-Head Attention for 2D tensors - matches nunif structure"""
    def __init__(self, in_channels, num_heads, window_size, qkv_dim=None, shift=False):
        super().__init__()
        self.window_size = (window_size if isinstance(window_size, (tuple, list))
                            else (window_size, window_size))
        self.shift = shift
        self.num_heads = num_heads
        self.mha = MHA(in_channels, num_heads, qkv_dim)
        
    def forward(self, x, attn_mask=None, layer_norm=None):
        out_shape = x.shape
        x = bchw_to_bnc(x, self.window_size)
        if layer_norm is not None:
            x = layer_norm(x)
        x = self.mha(x, attn_mask=attn_mask)
        x = bnc_to_bchw(x, out_shape, self.window_size)
        return x


class WindowScoreBias(nn.Module):
    """Window score bias for attention - matches nunif structure"""
    def __init__(self, window_size, hidden_dim=None, reduction=1, num_heads=None):
        super().__init__()
        if isinstance(window_size, int):
            window_size1 = [window_size, window_size]
        else:
            window_size1 = window_size

        assert window_size1[0] % reduction == 0 and window_size1[1] % reduction == 0

        window_size2 = [window_size1[0] // reduction, window_size1[1] // reduction]

        self.window_size1 = window_size1
        self.window_size2 = window_size2
        self.num_heads = num_heads

        index, unique_delta = self._gen_window_score_bias_input(self.window_size1, self.window_size2, reduction)
        self.register_buffer("index", index)
        self.register_buffer("delta", unique_delta)
        if hidden_dim is None:
            hidden_dim = int((self.window_size1[0] * self.window_size1[1]) ** 0.5) * 2
        if self.num_heads is None:
            output_dim = 1
        else:
            output_dim = num_heads

        self.to_bias = nn.Sequential(
            nn.Linear(2, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=True))

    @torch.no_grad()
    def _gen_window_score_bias_input(self, window_size1, window_size2, reduction):
        N1 = window_size1[0] * window_size1[1]
        N2 = window_size2[0] * window_size2[1]

        positions1 = torch.stack(
            torch.meshgrid(torch.arange(0, window_size1[0]),
                           torch.arange(0, window_size1[1]), indexing="ij"), dim=2).reshape(N1, 2)

        positions2 = torch.stack(
            torch.meshgrid(torch.arange(0, window_size2[0]),
                           torch.arange(0, window_size2[1]), indexing="ij"), dim=2).reshape(N2, 2)
        positions2.mul_(reduction)

        delta = torch.zeros((N1, N2, 2), dtype=torch.long)
        for i in range(N1):
            for j in range(N2):
                delta[i][j] = positions1[i] - positions2[j]

        delta = delta.view(N1 * N2, 2)
        delta = [tuple(p) for p in delta.tolist()]
        unique_delta = sorted(list(set(delta)))
        index = [unique_delta.index(d) for d in delta]
        index = torch.tensor(index, dtype=torch.int64)
        unique_delta = torch.tensor(unique_delta, dtype=torch.float32)
        unique_delta = unique_delta / unique_delta.abs().max()
        return index, unique_delta

    def forward(self):
        N1 = self.window_size1[0] * self.window_size1[1]
        N2 = self.window_size2[0] * self.window_size2[1]
        bias = self.to_bias(self.delta)
        bias = bias[self.index]
        if self.num_heads is None:
            # (N,N) float attention score bias
            bias = bias.reshape(N1, N2)
        else:
            # (H,N,N) float attention score bias
            bias = bias.permute(1, 0).contiguous().reshape(self.num_heads, N1, N2)
        return bias


class WABlock(nn.Module):
    """Window Attention Block - matches nunif structure"""
    def __init__(self, in_channels, window_size, layer_norm=False):
        super(WABlock, self).__init__()
        self.mha = WindowMHA2d(in_channels, num_heads=2, window_size=window_size)
        self.conv_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.GELU(),
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.1, inplace=True))
        self.bias = WindowScoreBias(window_size)

    def forward(self, x):
        x = x + self.mha(x, attn_mask=self.bias())
        x = x + self.conv_mlp(x)
        return x


OFFSET = 32

class RowFlowV3(nn.Module):
    name = "sbs.row_flow_v3"

    def __init__(self):
        super().__init__()
        self.downscaling_factor = (1, 8)
        self.mod = 4 * 3
        pack = self.downscaling_factor[0] * self.downscaling_factor[1]
        C = 64
        assert C >= pack
        self.blocks = nn.Sequential(
            nn.Conv2d(3 * pack, C, kernel_size=1, stride=1, padding=0),
            WABlock(C, (4, 4)),
            WABlock(C, (3, 3)),
        )
        self.last_layer = nn.Sequential(
            ReplicationPad2d((1, 1, 1, 1)),
            nn.Conv2d(C // pack, 1, kernel_size=3, stride=1, padding=0)
        )
        self.register_buffer("delta_scale", torch.tensor(1.0 / 127.0))
        self.delta_output = False
        self.symmetric = False

    def to_script_module(self):
        net = copy.deepcopy(self)
        net.eval()
        return torch.jit.script(net)

    def _forward(self, x):
        input_height, input_width = x.shape[2:]
        pad1 = (self.mod * self.downscaling_factor[1]) - input_width % (self.mod * self.downscaling_factor[1])
        pad2 = (self.mod * self.downscaling_factor[0]) - input_height % (self.mod * self.downscaling_factor[0])
        x = replication_pad2d_naive(x, (0, pad1, 0, pad2), detach=True)
        x = pixel_unshuffle(x, self.downscaling_factor)
        x = self.blocks(x)
        x = pixel_shuffle(x, self.downscaling_factor)
        x = F.pad(x, (0, -pad1, 0, -pad2), mode="constant")
        x = self.last_layer(x)
        return x

    def _warp(self, rgb, grid, delta, delta_scale):
        output_dtye = rgb.dtype
        rgb = rgb.to(torch.float32)
        grid = grid.to(torch.float32)
        delta = delta.to(torch.float32)
        delta_scale = delta_scale.to(torch.float32)

        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        grid = grid + delta * delta_scale
        grid = grid.permute(0, 2, 3, 1)
        z = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
        return z.to(output_dtye)

    def _forward_default(self, x):
        rgb = x[:, 0:3, :, ]
        grid = x[:, 6:8, :, ]
        x = x[:, 3:6, :, ]  # depth + diverdence feature + convergence

        delta = self._forward(x)
        if self.symmetric:
            left = self._warp(rgb, grid, delta, self.delta_scale)
            right = self._warp(rgb, grid, -delta, self.delta_scale)
            left = F.pad(left, (-OFFSET,) * 4)
            right = F.pad(right, (-OFFSET,) * 4)
            z = torch.cat([left, right], dim=1)
        else:
            z = self._warp(rgb, grid, delta, self.delta_scale)
            z = F.pad(z, (-OFFSET,) * 4)

        if self.training:
            return z, ((grid[:, 0:1, :, :] / self.delta_scale).detach() + delta)
        else:
            return torch.clamp(z, 0., 1.)

    def _forward_delta_only(self, x):
        assert not self.training
        delta = self._forward(x)
        delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
        return delta

    def forward(self, x):
        if not self.delta_output:
            return self._forward_default(x)
        else:
            return self._forward_delta_only(x)

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


def make_input_tensor(x, depth, divergence, convergence, image_width, mapper="none"):
    """创建神经网络输入tensor - 从 backward_warp.py 简化"""
    B, _, H, W = depth.shape
    
    # 归一化深度
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    
    # 应用mapper (这里简化为不做变换)
    if mapper == "none":
        processed_depth = depth
    else:
        # 可以根据需要添加其他mapper逻辑
        processed_depth = depth
    
    # 计算视差
    shift_size = divergence * 0.01
    disparity = processed_depth * shift_size - (shift_size * convergence)
    
    # 创建输入tensor：深度图 + 视差图
    input_tensor = torch.cat([processed_depth, disparity], dim=1)
    
    return input_tensor


def apply_divergence_symmetric(model, c, depth, divergence, convergence, enable_amp=True):
    """使用对称模型应用散度 - 基于 backward_warp.py 的 apply_divergence_nn_symmetric"""
    B, _, H, W = depth.shape
    
    # 计算视差特征
    shift_size = divergence * 0.01
    disparity = depth * shift_size - (shift_size * convergence)
    
    # 创建收敛特征
    convergence_tensor = torch.full_like(depth, convergence)
    
    # 创建3通道输入：depth + disparity + convergence
    x = torch.cat([
        depth,                # depth (1 channel)  
        disparity,            # disparity (1 channel)
        convergence_tensor,   # convergence (1 channel)
    ], dim=1)  # Total: 3 channels
    
    # 推理 - 只调用_forward，这会处理pixel_unshuffle和网格
    with torch.autocast(device_type='cuda' if depth.device.type == 'cuda' else 'cpu', enabled=enable_amp):
        delta = model._forward(x)
    
    # 创建网格用于变形
    grid = make_grid(B, W, H, c.device)
    
    # 添加Y维度到delta使其成为2D位移
    delta = torch.cat([delta, torch.zeros_like(delta)], dim=1)
    
    # 使用模型的delta_scale
    delta_scale = model.delta_scale
    
    left_eye = backward_warp(c, grid, delta, delta_scale)
    right_eye = backward_warp(c, grid, -delta, delta_scale)
    
    return left_eye, right_eye


def to_pil_image(x):
    """转换tensor到PIL图像"""
    x = torch.clamp(x, 0, 1)
    x = (x * 255).round_().to(torch.uint8).cpu()
    return TF.to_pil_image(x)


# 拷贝的TorchHubDir类
class TorchHubDir:
    def __init__(self, hub_dir):
        self.hub_dir = hub_dir
        self.original_hub_dir = None

    def __enter__(self):
        self.original_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(self.hub_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.hub.set_dir(self.original_hub_dir)


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
        print("Loading row_flow_v3_sym stereo model...")
        
        # 模型URL
        ROW_FLOW_V3_SYM_URL = "https://github.com/nagadomi/nunif/releases/download/0.0.0/iw3_row_flow_v3_sym_20240424.pth"
        
        # 转换device对象为device_id
        if device.type == "cuda":
            device_id = device.index if device.index is not None else 0
        else:
            device_id = -1  # CPU
        
        # 直接加载模型而不使用load_model函数
        map_location = "cpu"
        weights_only = True
        if not PYTORCH2:
            weights_only = False
        if "mps" in str(map_location):
            map_location = "cpu"
        
        # 加载模型数据
        data = torch.hub.load_state_dict_from_url(ROW_FLOW_V3_SYM_URL, weights_only=True, map_location=map_location)
        
        assert ("nunif_model" in data)
        
        # 创建模型并直接移动到设备
        side_model = RowFlowV3()
        device_obj = create_device(device_id)
        side_model = side_model.to(device_obj)
        
        # 加载状态字典
        side_model.load_state_dict(data["state_dict"], strict=True)
        if "updated_at" in data:
            side_model.updated_at = data["updated_at"]
        data.pop("state_dict")
        
        side_model = side_model.eval()
            
        # 设置模型属性
        side_model.symmetric = True
        side_model.delta_output = True
            
        print("Successfully loaded row_flow_v3_sym stereo model")
        return side_model
        
    except Exception as e:
        print(f"Error loading stereo model: {e}")
        print("Falling back to grid sampling method...")
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
    
    # 使用神经网络模型
    left_eye, right_eye = apply_divergence_symmetric(
        stereo_model, rgb_batch, depth_batch, divergence, convergence
    )
    
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


# 常量定义
HUB_MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "hub")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end stereo image generation")
    parser.add_argument("--input", type=str, default="waifu2x/docs/images/miku_128.png", help="Input image path")
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