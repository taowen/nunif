#!/usr/bin/env python3
"""
DDS文件颜色转换分析工具
用于对比同一帧在颜色转换前后的差异，分析颜色输出异常的原因
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Optional, Tuple, Dict, Any
import struct

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available, will use basic DDS reading")

class DDSReader:
    """DDS文件读取器"""
    
    # DDS格式常量
    DDS_MAGIC = b'DDS '
    DDSD_CAPS = 0x1
    DDSD_HEIGHT = 0x2
    DDSD_WIDTH = 0x4
    DDSD_PITCH = 0x8
    DDSD_PIXELFORMAT = 0x1000
    DDSD_MIPMAPCOUNT = 0x20000
    DDSD_LINEARSIZE = 0x80000
    DDSD_DEPTH = 0x800000
    
    # DDPF Flags
    DDPF_RGB = 0x40
    DDPF_FOURCC = 0x4
    
    # DXGI格式
    DXGI_FORMAT_R32G32B32A32_FLOAT = 2
    # DX9 formats (FourCC)
    FOURCC_A32B32G32R32F = 116
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.header = None
        self.data = None
        
    def read_dds(self) -> Optional[np.ndarray]:
        """读取DDS文件"""
        try:
            with open(self.filepath, 'rb') as f:
                # 读取DDS magic
                magic = f.read(4)
                if magic != self.DDS_MAGIC:
                    print(f"Error: {self.filepath} is not a valid DDS file")
                    return None
                
                # 读取DDS头部 (124字节)
                header_data = f.read(124)
                header = self._parse_header(header_data)
                
                if header is None:
                    return None
                
                self.header = header
                
                # 检查是否有DX10扩展头部
                has_dx10_header = False
                dxgi_format = 0
                if header['ddspf_fourcc'] == b'DX10':
                    has_dx10_header = True
                    dx10_header = f.read(20)  # DX10头部20字节
                    dxgi_format = struct.unpack('<I', dx10_header[:4])[0]
                
                # 计算数据大小
                width = header['width']
                height = header['height']
                
                if has_dx10_header:
                    if dxgi_format == self.DXGI_FORMAT_R32G32B32A32_FLOAT:
                        print("Detected R32G32B32A32_FLOAT format")
                        expected_size = width * height * 16
                        data = f.read(expected_size)
                        float_data = np.frombuffer(data, dtype=np.float32)
                        return float_data.reshape((height, width, 4))
                    else:
                        print(f"Warning: Unexpected DXGI format: {dxgi_format}")
                else:
                    # Legacy DX9 path
                    flags = header['ddspf_flags']
                    
                    if flags & self.DDPF_FOURCC:
                        if header['ddspf_fourcc'] == b'DX10':
                            has_dx10_header = True
                            dx10_header = f.read(20)  # DX10头部20字节
                            dxgi_format = struct.unpack('<I', dx10_header[:4])[0]
                        else:
                            fourcc_val = struct.unpack('<I', header['ddspf_fourcc'])[0]
                            if fourcc_val == self.FOURCC_A32B32G32R32F:
                                print("Detected A32B32G32R32F (legacy DX9) format")
                                expected_size = width * height * 16
                                data = f.read(expected_size)
                                float_data = np.frombuffer(data, dtype=np.float32)
                                return float_data.reshape((height, width, 4))
                            else:
                                print(f"Unsupported FourCC: {header['ddspf_fourcc']} ({fourcc_val})")
                    elif flags & self.DDPF_RGB:
                        if header['ddspf_bitcount'] == 128:
                            print("Detected uncompressed RGBA 128-bit (likely float) format")
                            expected_size = width * height * 16
                            data = f.read(expected_size)
                            float_data = np.frombuffer(data, dtype=np.float32)
                            return float_data.reshape((height, width, 4))
                        elif header['ddspf_bitcount'] == 32:
                            print("Detected uncompressed 32-bit RGBA format")
                            expected_size = width * height * 4
                            data = f.read(expected_size)
                            byte_data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
                            
                            r_mask = header['ddspf_rbitmask']
                            a_mask = header['ddspf_abitmask']

                            rgba_image = np.zeros((height, width, 4), dtype=np.float32)

                            if r_mask == 0x00ff0000:  # BGRA
                                print("Converting from BGRA to RGBA")
                                rgba_image[..., :3] = byte_data[..., [2, 1, 0]] / 255.0
                            elif r_mask == 0x000000ff:  # RGBA
                                print("Assuming RGBA byte order")
                                rgba_image[..., :3] = byte_data[..., :3] / 255.0
                            else:
                                print(f"Unsupported bitmask for 32bpp: R={r_mask:x}. Treating as RGBA.")
                                rgba_image[..., :3] = byte_data[..., :3] / 255.0
                            
                            if a_mask != 0:
                                rgba_image[..., 3] = byte_data[..., 3] / 255.0
                            else:
                                rgba_image[..., 3] = 1.0
                            
                            return rgba_image
                        elif header['ddspf_bitcount'] == 24:
                            print("Detected uncompressed 24-bit RGB format")
                            expected_size = width * height * 3
                            data = f.read(expected_size)
                            byte_data = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))

                            rgba_image = np.zeros((height, width, 4), dtype=np.float32)
                            r_mask = header['ddspf_rbitmask']

                            if r_mask == 0x00ff0000: # BGR
                                 print("Converting from BGR to RGBA")
                                 rgba_image[..., :3] = byte_data[..., [2, 1, 0]] / 255.0
                            elif r_mask == 0x000000ff: # RGB
                                 print("Assuming RGB byte order")
                                 rgba_image[..., :3] = byte_data[..., :3] / 255.0
                            else:
                                print(f"Unsupported bitmask for 24bpp: R={r_mask:x}. Treating as RGB.")
                                rgba_image[..., :3] = byte_data[..., :3] / 255.0
                            
                            rgba_image[..., 3] = 1.0

                            return rgba_image
                        else:
                            print(f"Unsupported bitcount for uncompressed RGB: {header['ddspf_bitcount']}")
                    else:
                        print(f"Unsupported pixel format flags: {flags:#x}")

                print(f"Unsupported format in {self.filepath}")
                return None
                    
        except Exception as e:
            print(f"Error reading {self.filepath}: {e}")
            return None
    
    def _parse_header(self, header_data: bytes) -> Optional[Dict[str, Any]]:
        """解析DDS头部"""
        try:
            # 解析基本头部信息
            header = {}
            
            # 偏移量定义
            header['size'] = struct.unpack('<I', header_data[0:4])[0]
            header['flags'] = struct.unpack('<I', header_data[4:8])[0]
            header['height'] = struct.unpack('<I', header_data[8:12])[0]
            header['width'] = struct.unpack('<I', header_data[12:16])[0]
            header['pitch'] = struct.unpack('<I', header_data[16:20])[0]
            header['depth'] = struct.unpack('<I', header_data[20:24])[0]
            header['mipmapcount'] = struct.unpack('<I', header_data[24:28])[0]
            
            # 像素格式 (从偏移76开始的32字节)
            pf_start = 72
            header['ddspf_size'] = struct.unpack('<I', header_data[pf_start:pf_start+4])[0]
            header['ddspf_flags'] = struct.unpack('<I', header_data[pf_start+4:pf_start+8])[0]
            header['ddspf_fourcc'] = header_data[pf_start+8:pf_start+12]
            header['ddspf_bitcount'] = struct.unpack('<I', header_data[pf_start+12:pf_start+16])[0]
            header['ddspf_rbitmask'] = struct.unpack('<I', header_data[pf_start+16:pf_start+20])[0]
            header['ddspf_gbitmask'] = struct.unpack('<I', header_data[pf_start+20:pf_start+24])[0]
            header['ddspf_bbitmask'] = struct.unpack('<I', header_data[pf_start+24:pf_start+28])[0]
            header['ddspf_abitmask'] = struct.unpack('<I', header_data[pf_start+28:pf_start+32])[0]
            
            return header
            
        except Exception as e:
            print(f"Error parsing DDS header: {e}")
            return None

class ColorAnalyzer:
    """颜色分析器"""
    
    def __init__(self):
        self.results = {}
        
    def analyze_frame_pair(self, frame_num: int, before_data: np.ndarray, after_data: np.ndarray) -> Dict[str, Any]:
        """分析同一帧转换前后的差异"""
        analysis = {
            'frame_number': frame_num,
            'before_shape': before_data.shape,
            'after_shape': after_data.shape,
            'shape_match': before_data.shape == after_data.shape
        }
        
        if not analysis['shape_match']:
            print(f"Warning: Frame {frame_num} shape mismatch - Before: {before_data.shape}, After: {after_data.shape}")
            return analysis
        
        # NOTE: Compare only the left half, as 'after' is a side-by-side image
        # and its right half is a synthetic view.
        width = before_data.shape[1]
        half_width = width // 2
        before_data_left = before_data[:, :half_width, :]
        after_data_left = after_data[:, :half_width, :]

        # 计算基本统计信息
        analysis['before_stats'] = self._calculate_stats(before_data_left)
        analysis['after_stats'] = self._calculate_stats(after_data_left)
        
        # 计算差异
        diff = after_data_left - before_data_left
        analysis['diff_stats'] = self._calculate_stats(diff)
        
        # 计算各通道的差异
        analysis['channel_diff'] = {}
        channel_names = ['R', 'G', 'B', 'A']
        for i, channel in enumerate(channel_names):
            if i < before_data.shape[2]:
                before_channel = before_data_left[:, :, i]
                after_channel = after_data_left[:, :, i]
                diff_channel = after_channel - before_channel
                
                analysis['channel_diff'][channel] = {
                    'mean_diff': np.mean(diff_channel),
                    'max_diff': np.max(np.abs(diff_channel)),
                    'std_diff': np.std(diff_channel),
                    'before_range': (np.min(before_channel), np.max(before_channel)),
                    'after_range': (np.min(after_channel), np.max(after_channel))
                }
        
        # 检测异常区域
        analysis['anomalies'] = self._detect_anomalies(before_data_left, after_data_left, diff)
        
        return analysis
    
    def _calculate_stats(self, data: np.ndarray) -> Dict[str, float]:
        """计算数据统计信息"""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data)
        }
    
    def _detect_anomalies(self, before: np.ndarray, after: np.ndarray, diff: np.ndarray) -> Dict[str, Any]:
        """检测异常区域"""
        anomalies = {}
        
        # 检测大幅度变化的区域
        abs_diff = np.abs(diff)
        threshold = np.std(abs_diff) * 3  # 3-sigma rule
        
        anomaly_mask = abs_diff > threshold
        anomaly_count = np.sum(anomaly_mask)
        total_pixels = anomaly_mask.size
        
        anomalies['large_changes'] = {
            'threshold': threshold,
            'count': int(anomaly_count),
            'percentage': (anomaly_count / total_pixels) * 100,
            'locations': np.where(anomaly_mask)
        }
        
        # 检测值域异常
        anomalies['value_range'] = {
            'before_out_of_range': {
                'negative': np.sum(before < 0),
                'above_one': np.sum(before > 1.0)
            },
            'after_out_of_range': {
                'negative': np.sum(after < 0),
                'above_one': np.sum(after > 1.0)
            }
        }
        
        return anomalies
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """生成分析报告"""
        report = []
        report.append(f"=== Frame {analysis['frame_number']} Analysis Report ===")
        report.append(f"Shape match: {analysis['shape_match']}")
        
        if analysis['shape_match']:
            report.append("\n--- Channel Differences ---")
            for channel, diff_info in analysis['channel_diff'].items():
                report.append(f"{channel} Channel:")
                report.append(f"  Mean difference: {diff_info['mean_diff']:.6f}")
                report.append(f"  Max difference: {diff_info['max_diff']:.6f}")
                report.append(f"  Std difference: {diff_info['std_diff']:.6f}")
                report.append(f"  Before range: {diff_info['before_range']}")
                report.append(f"  After range: {diff_info['after_range']}")
            
            report.append("\n--- Anomaly Detection ---")
            anomalies = analysis['anomalies']
            large_changes = anomalies['large_changes']
            report.append(f"Large changes (>{large_changes['threshold']:.6f}):")
            report.append(f"  Count: {large_changes['count']} ({large_changes['percentage']:.2f}%)")
            
            value_range = anomalies['value_range']
            report.append(f"Value range issues:")
            report.append(f"  Before: {value_range['before_out_of_range']['negative']} negative, {value_range['before_out_of_range']['above_one']} >1.0")
            report.append(f"  After: {value_range['after_out_of_range']['negative']} negative, {value_range['after_out_of_range']['above_one']} >1.0")
        
        return "\n".join(report)
    
    def create_visualization(self, frame_num: int, before_data: np.ndarray, after_data: np.ndarray, 
                           analysis: Dict[str, Any], output_dir: str):
        """创建可视化图表"""
        if not analysis['shape_match']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Frame {frame_num} Color Conversion Analysis (Left Half)', fontsize=16)
        
        # Get left half for comparison
        width = before_data.shape[1]
        half_width = width // 2
        before_data_left = before_data[:, :half_width, :]
        after_data_left = after_data[:, :half_width, :]

        # 显示原始图像和转换后图像 (只显示RGB通道)
        before_rgb = np.clip(before_data_left[:, :, :3], 0, 1)
        after_rgb = np.clip(after_data_left[:, :, :3], 0, 1)
        
        axes[0, 0].imshow(before_rgb)
        axes[0, 0].set_title('Before Conversion (Left Half)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(after_rgb)
        axes[0, 1].set_title('After Conversion (Left of SBS)')
        axes[0, 1].axis('off')
        
        # 显示差异
        diff = after_data_left - before_data_left
        diff_rgb = diff[:, :, :3]
        # 将差异映射到可视化范围
        diff_vis = (diff_rgb - np.min(diff_rgb)) / (np.max(diff_rgb) - np.min(diff_rgb) + 1e-8)
        
        axes[0, 2].imshow(diff_vis)
        axes[0, 2].set_title('Difference Visualization (Left Half)')
        axes[0, 2].axis('off')
        
        # 显示各通道的直方图
        channel_names = ['R', 'G', 'B']
        colors = ['red', 'green', 'blue']
        
        for i, (channel, color) in enumerate(zip(channel_names, colors)):
            if i < before_data.shape[2]:
                axes[1, i].hist(before_data_left[:, :, i].flatten(), bins=50, alpha=0.5, 
                               label='Before', color=color, density=True)
                axes[1, i].hist(after_data_left[:, :, i].flatten(), bins=50, alpha=0.5, 
                               label='After', color=color, density=True, histtype='step')
                axes[1, i].set_title(f'{channel} Channel Histogram (Left Half)')
                axes[1, i].legend()
                axes[1, i].set_xlabel('Value')
                axes[1, i].set_ylabel('Density')
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_dir, f'frame_{frame_num}_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")

def find_frame_pairs(debug_dir: str) -> Dict[int, Dict[str, str]]:
    """查找成对的DDS文件"""
    pairs = {}
    
    debug_path = Path(debug_dir)
    if not debug_path.exists():
        print(f"Debug directory not found: {debug_dir}")
        return pairs
    
    # 查找所有DDS文件
    dds_files = list(debug_path.glob("*.dds"))
    
    for dds_file in dds_files:
        filename = dds_file.name
        # 解析文件名: frame_{number}_{suffix}.dds
        if filename.startswith("frame_") and filename.endswith(".dds"):
            parts = filename[6:-4].split("_")  # 去掉"frame_"前缀和".dds"后缀
            if len(parts) >= 2:
                try:
                    frame_num = int(parts[0])
                    suffix = "_".join(parts[1:])
                    
                    if frame_num not in pairs:
                        pairs[frame_num] = {}
                    
                    pairs[frame_num][suffix] = str(dds_file)
                except ValueError:
                    continue
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description='DDS颜色转换分析工具')
    parser.add_argument('--debug-dir', default='debug_textures', 
                        help='包含DDS文件的调试目录 (默认: debug_textures)')
    parser.add_argument('--output-dir', default='analysis_output',
                        help='分析结果输出目录 (默认: analysis_output)')
    parser.add_argument('--frame', type=int, help='分析特定帧号 (可选)')
    parser.add_argument('--max-frames', type=int, default=10, 
                        help='最大分析帧数 (默认: 10)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找DDS文件对
    frame_pairs = find_frame_pairs(args.debug_dir)
    
    if not frame_pairs:
        print(f"No DDS file pairs found in {args.debug_dir}")
        return
    
    print(f"Found {len(frame_pairs)} frames with DDS files")
    
    # 初始化分析器
    analyzer = ColorAnalyzer()
    
    # 分析帧
    frames_to_analyze = []
    if args.frame is not None:
        if args.frame in frame_pairs:
            frames_to_analyze = [args.frame]
        else:
            print(f"Frame {args.frame} not found")
            return
    else:
        frames_to_analyze = sorted(frame_pairs.keys())[:args.max_frames]
    
    reports = []
    
    for frame_num in frames_to_analyze:
        frame_files = frame_pairs[frame_num]
        print(f"\nAnalyzing frame {frame_num}...")
        
        # 检查是否有所需的文件
        if 'color_conv' not in frame_files or 'sbs_infer' not in frame_files:
            print(f"Missing required files for frame {frame_num}")
            print(f"Available: {list(frame_files.keys())}")
            continue
        
        # 读取DDS文件
        before_reader = DDSReader(frame_files['color_conv'])
        after_reader = DDSReader(frame_files['sbs_infer'])
        
        before_data = before_reader.read_dds()
        after_data = after_reader.read_dds()
        
        if before_data is None or after_data is None:
            print(f"Failed to read DDS files for frame {frame_num}")
            continue
        
        # 分析差异
        analysis = analyzer.analyze_frame_pair(frame_num, before_data, after_data)
        
        # 生成报告
        report = analyzer.generate_report(analysis)
        reports.append(report)
        print(report)
        
        # 创建可视化
        try:
            analyzer.create_visualization(frame_num, before_data, after_data, 
                                        analysis, args.output_dir)
        except Exception as e:
            print(f"Failed to create visualization for frame {frame_num}: {e}")
    
    # 保存完整报告
    if reports:
        report_path = os.path.join(args.output_dir, 'color_analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DDS Color Conversion Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n\n".join(reports))
        
        print(f"\nComplete analysis report saved to: {report_path}")

if __name__ == "__main__":
    main() 