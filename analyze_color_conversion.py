#!/usr/bin/env python3
"""
Analysis script for color conversion debugging
Reads PNG files exported from the C++ diagnose thread and analyzes color conversion quality
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from pathlib import Path

def analyze_single_frame(image_path):
    """Analyze a single frame for color conversion issues"""
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        
        height, width, channels = img_array.shape
        
        # Basic statistics
        stats = {
            'path': image_path,
            'dimensions': (width, height),
            'mean_rgb': np.mean(img_array, axis=(0,1)),
            'std_rgb': np.std(img_array, axis=(0,1)),
            'min_rgb': np.min(img_array, axis=(0,1)),
            'max_rgb': np.max(img_array, axis=(0,1)),
        }
        
        # Check for common issues
        issues = []
        
        # Check for clipped values (pure black/white regions)
        black_pixels = np.sum(np.all(img_array < 0.01, axis=2))
        white_pixels = np.sum(np.all(img_array > 0.99, axis=2))
        total_pixels = width * height
        
        black_ratio = black_pixels / total_pixels
        white_ratio = white_pixels / total_pixels
        
        if black_ratio > 0.1:  # More than 10% black pixels
            issues.append(f"High black pixel ratio: {black_ratio:.3f}")
        if white_ratio > 0.1:  # More than 10% white pixels
            issues.append(f"High white pixel ratio: {white_ratio:.3f}")
            
        # Check for unusual color distribution
        gray_pixels = np.sum(np.abs(img_array[:,:,0] - img_array[:,:,1]) < 0.05) / total_pixels
        if gray_pixels > 0.8:
            issues.append(f"Image appears too grayscale: {gray_pixels:.3f}")
            
        # Check for color range issues
        color_range = stats['max_rgb'] - stats['min_rgb']
        if np.any(color_range < 0.1):
            issues.append(f"Low color range detected: R={color_range[0]:.3f}, G={color_range[1]:.3f}, B={color_range[2]:.3f}")
            
        # Check for color bias (one channel dominant)
        channel_dominance = stats['mean_rgb'] / np.mean(stats['mean_rgb'])
        if np.max(channel_dominance) > 1.5 or np.min(channel_dominance) < 0.5:
            issues.append(f"Color bias detected: R={channel_dominance[0]:.3f}, G={channel_dominance[1]:.3f}, B={channel_dominance[2]:.3f}")
            
        stats['issues'] = issues
        stats['black_ratio'] = black_ratio
        stats['white_ratio'] = white_ratio
        stats['gray_ratio'] = gray_pixels
        
        return stats, img_array
        
    except Exception as e:
        print(f"Error analyzing {image_path}: {e}")
        return None, None

def create_analysis_plots(frames_data, output_dir):
    """Create analysis plots"""
    if not frames_data:
        print("No valid frames to plot")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    frame_numbers = []
    mean_r, mean_g, mean_b = [], [], []
    std_r, std_g, std_b = [], [], []
    black_ratios, white_ratios, gray_ratios = [], [], []
    
    for i, (stats, _) in enumerate(frames_data):
        if stats:
            frame_numbers.append(i)
            mean_r.append(stats['mean_rgb'][0])
            mean_g.append(stats['mean_rgb'][1]) 
            mean_b.append(stats['mean_rgb'][2])
            std_r.append(stats['std_rgb'][0])
            std_g.append(stats['std_rgb'][1])
            std_b.append(stats['std_rgb'][2])
            black_ratios.append(stats['black_ratio'])
            white_ratios.append(stats['white_ratio'])
            gray_ratios.append(stats['gray_ratio'])
    
    # Plot 1: Mean RGB values over frames
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(frame_numbers, mean_r, 'r-', label='Red', alpha=0.7)
    plt.plot(frame_numbers, mean_g, 'g-', label='Green', alpha=0.7)
    plt.plot(frame_numbers, mean_b, 'b-', label='Blue', alpha=0.7)
    plt.ylabel('Mean Color Value')
    plt.xlabel('Frame Number')
    plt.title('Mean RGB Values per Frame')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation
    plt.subplot(2, 2, 2)
    plt.plot(frame_numbers, std_r, 'r-', label='Red', alpha=0.7)
    plt.plot(frame_numbers, std_g, 'g-', label='Green', alpha=0.7)
    plt.plot(frame_numbers, std_b, 'b-', label='Blue', alpha=0.7)
    plt.ylabel('Standard Deviation')
    plt.xlabel('Frame Number')
    plt.title('RGB Standard Deviation per Frame')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Clipping ratios
    plt.subplot(2, 2, 3)
    plt.plot(frame_numbers, black_ratios, 'k-', label='Black pixels', alpha=0.7)
    plt.plot(frame_numbers, white_ratios, 'gray', label='White pixels', alpha=0.7)
    plt.ylabel('Pixel Ratio')
    plt.xlabel('Frame Number')
    plt.title('Clipping Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Gray ratio
    plt.subplot(2, 2, 4)
    plt.plot(frame_numbers, gray_ratios, 'purple', alpha=0.7)
    plt.ylabel('Grayscale Pixel Ratio')
    plt.xlabel('Frame Number')
    plt.title('Grayscale Content Detection')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/color_analysis.png", dpi=150, bbox_inches='tight')
    print(f"Analysis plots saved to {output_dir}/color_analysis.png")
    
    # Create histogram of first valid frame
    for stats, img_array in frames_data:
        if stats and img_array is not None:
            plt.figure(figsize=(10, 6))
            
            # RGB histograms
            plt.subplot(1, 2, 1)
            plt.hist(img_array[:,:,0].flatten(), bins=50, alpha=0.7, color='red', label='Red')
            plt.hist(img_array[:,:,1].flatten(), bins=50, alpha=0.7, color='green', label='Green')
            plt.hist(img_array[:,:,2].flatten(), bins=50, alpha=0.7, color='blue', label='Blue')
            plt.xlabel('Pixel Value')
            plt.ylabel('Frequency')
            plt.title(f'RGB Histogram - {Path(stats["path"]).name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Luminance histogram
            plt.subplot(1, 2, 2)
            # Calculate luminance using standard weights
            luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
            plt.hist(luminance.flatten(), bins=50, alpha=0.7, color='gray')
            plt.xlabel('Luminance Value')
            plt.ylabel('Frequency')
            plt.title('Luminance Histogram')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/histogram_sample.png", dpi=150, bbox_inches='tight')
            print(f"Sample histogram saved to {output_dir}/histogram_sample.png")
            break

def main():
    parser = argparse.ArgumentParser(description='Analyze color conversion from exported PNG files')
    parser.add_argument('--input-dir', default='debug_textures', 
                       help='Directory containing exported PNG files (default: debug_textures)')
    parser.add_argument('--output-dir', default='analysis_output',
                       help='Directory to save analysis results (default: analysis_output)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to analyze')
    parser.add_argument('--show-plots', action='store_true',
                       help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Find PNG files
    png_pattern = os.path.join(args.input_dir, "frame_*.png")
    png_files = sorted(glob.glob(png_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not png_files:
        print(f"No PNG files found in {args.input_dir}")
        print(f"Looking for pattern: {png_pattern}")
        return
        
    print(f"Found {len(png_files)} PNG files")
    
    if args.max_frames:
        png_files = png_files[:args.max_frames]
        print(f"Analyzing first {len(png_files)} frames")
    
    # Analyze all frames
    frames_data = []
    issue_count = 0
    
    print("\nAnalyzing frames...")
    for i, png_file in enumerate(png_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing frame {i + 1}/{len(png_files)}")
            
        stats, img_array = analyze_single_frame(png_file)
        frames_data.append((stats, img_array))
        
        if stats and stats['issues']:
            issue_count += 1
            print(f"  Issues in {Path(png_file).name}: {', '.join(stats['issues'])}")
    
    # Print summary
    print(f"\n=== Analysis Summary ===")
    print(f"Total frames analyzed: {len(png_files)}")
    print(f"Frames with issues: {issue_count}")
    
    valid_frames = [stats for stats, _ in frames_data if stats]
    if valid_frames:
        # Overall statistics
        all_means = np.array([stats['mean_rgb'] for stats in valid_frames])
        all_stds = np.array([stats['std_rgb'] for stats in valid_frames])
        
        print(f"\nOverall Statistics:")
        print(f"  Mean RGB: R={np.mean(all_means[:,0]):.3f}, G={np.mean(all_means[:,1]):.3f}, B={np.mean(all_means[:,2]):.3f}")
        print(f"  Std RGB:  R={np.mean(all_stds[:,0]):.3f}, G={np.mean(all_stds[:,1]):.3f}, B={np.mean(all_stds[:,2]):.3f}")
        
        # Check for common issues across all frames
        avg_black_ratio = np.mean([stats['black_ratio'] for stats in valid_frames])
        avg_white_ratio = np.mean([stats['white_ratio'] for stats in valid_frames])
        avg_gray_ratio = np.mean([stats['gray_ratio'] for stats in valid_frames])
        
        print(f"  Average black pixel ratio: {avg_black_ratio:.3f}")
        print(f"  Average white pixel ratio: {avg_white_ratio:.3f}")
        print(f"  Average grayscale ratio: {avg_gray_ratio:.3f}")
        
        # Color conversion quality assessment
        print(f"\n=== Color Conversion Assessment ===")
        if avg_black_ratio > 0.05:
            print("⚠️  HIGH BLACK PIXEL RATIO - possible underexposure or clipping")
        if avg_white_ratio > 0.05:
            print("⚠️  HIGH WHITE PIXEL RATIO - possible overexposure or clipping")
        if avg_gray_ratio > 0.7:
            print("⚠️  HIGH GRAYSCALE RATIO - colors may be desaturated")
        
        mean_channel_ratio = np.mean(all_means, axis=0)
        mean_channel_ratio = mean_channel_ratio / np.mean(mean_channel_ratio)
        if np.max(mean_channel_ratio) > 1.3 or np.min(mean_channel_ratio) < 0.7:
            print(f"⚠️  COLOR BIAS DETECTED - channel ratios: R={mean_channel_ratio[0]:.3f}, G={mean_channel_ratio[1]:.3f}, B={mean_channel_ratio[2]:.3f}")
        
        if issue_count == 0:
            print("✅ No major issues detected in color conversion")
    
    # Create plots
    create_analysis_plots(frames_data, args.output_dir)
    
    if args.show_plots:
        plt.show()

if __name__ == "__main__":
    main() 