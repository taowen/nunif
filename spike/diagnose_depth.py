import torch
import torch.onnx
import onnxruntime as ort
import cv2
import numpy as np
import os
from pathlib import Path

def preprocess_image_python(input_image, lower_bound=392):
    """
    Python version of the C++ preprocessing function for comparison
    """
    # Convert to float32 and normalize to [0,1]
    image = input_image.astype(np.float32) / 255.0
    
    H, W = image.shape[:2]
    
    # Calculate scale factor
    scale_factor = lower_bound / min(W, H)
    new_h = int(H * scale_factor)
    new_w = int(W * scale_factor)
    
    # Limit aspect ratio (max 4:1)
    if new_h < new_w:
        new_w = min(new_w, 4 * new_h)
    else:
        new_h = min(new_h, 4 * new_w)
    
    # Ensure multiple of 14
    new_h = new_h - (new_h % 14)
    new_w = new_w - (new_w % 14)
    
    if new_h < lower_bound:
        new_h = lower_bound
    if new_w < lower_bound:
        new_w = lower_bound
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize: ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    normalized = (resized - mean) / std
    
    return normalized

def image_to_chw_format(image):
    """
    Convert HWC to CHW format (channels first)
    """
    return np.transpose(image, (2, 0, 1))

def postprocess_output_python(output_data, width, height):
    """
    Python version of the C++ postprocessing
    """
    depth_map = output_data.reshape(height, width)
    
    # Handle NaN values
    depth_map = np.nan_to_num(depth_map, nan=0.0)
    
    # Invert for compatibility (DistillAnyDepth outputs inverted depth)
    depth_map = -depth_map
    
    return depth_map

def test_with_onnx(image_path, onnx_path):
    """
    Test inference using ONNX Runtime
    """
    print("=== Testing with ONNX Runtime ===")
    
    # Load image
    if os.path.exists(image_path):
        input_image = cv2.imread(image_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        print(f"Loaded image: {input_image.shape}")
    else:
        print(f"Image {image_path} not found, creating dummy image")
        input_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    # Preprocess
    preprocessed = preprocess_image_python(input_image)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
    
    # Convert to CHW and add batch dimension
    input_tensor = image_to_chw_format(preprocessed)
    input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
    
    # Ensure float32 data type (CRITICAL FIX)
    input_tensor = input_tensor.astype(np.float32)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Input tensor dtype: {input_tensor.dtype}")
    
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        return None, None
    
    # Load ONNX model
    try:
        session = ort.InferenceSession(onnx_path)
        print("ONNX model loaded successfully")
        
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Input name: {input_name}")
        print(f"Output name: {output_name}")
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})
        output_data = outputs[0]
        
        print(f"Raw output shape: {output_data.shape}")
        print(f"Raw output range: [{output_data.min():.6f}, {output_data.max():.6f}]")
        print(f"Raw output mean: {output_data.mean():.6f}")
        print(f"Raw output std: {output_data.std():.6f}")
        
        # Check for all zeros
        if np.all(output_data == 0):
            print("WARNING: All output values are zero!")
        elif np.all(np.abs(output_data) < 1e-6):
            print("WARNING: All output values are very close to zero!")
        
        # Postprocess
        if len(output_data.shape) == 4:  # NCHW
            depth_data = output_data[0, 0, :, :]  # Remove batch and channel dims
        elif len(output_data.shape) == 3:  # NHW
            depth_data = output_data[0, :, :]  # Remove batch dim
        else:
            depth_data = output_data
        
        depth_map = postprocess_output_python(depth_data, depth_data.shape[1], depth_data.shape[0])
        
        print(f"Processed depth range: [{depth_map.min():.6f}, {depth_map.max():.6f}]")
        
        return depth_map, input_tensor
        
    except Exception as e:
        print(f"Error running ONNX inference: {e}")
        return None, None

def analyze_depth_output(depth_map, output_path="depth_analysis.png"):
    """
    Analyze and visualize depth map
    """
    if depth_map is None:
        print("No depth map to analyze")
        return
    
    print("\n=== Depth Map Analysis ===")
    print(f"Shape: {depth_map.shape}")
    print(f"Data type: {depth_map.dtype}")
    print(f"Min value: {depth_map.min():.6f}")
    print(f"Max value: {depth_map.max():.6f}")
    print(f"Mean value: {depth_map.mean():.6f}")
    print(f"Std deviation: {depth_map.std():.6f}")
    
    # Check for special values
    nan_count = np.isnan(depth_map).sum()
    inf_count = np.isinf(depth_map).sum()
    zero_count = (depth_map == 0).sum()
    
    print(f"NaN values: {nan_count}")
    print(f"Inf values: {inf_count}")
    print(f"Zero values: {zero_count}")
    print(f"Total pixels: {depth_map.size}")
    
    # Create visualizations
    if depth_map.max() > depth_map.min():
        # Normalize for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Save visualization
        cv2.imwrite(output_path, depth_colored)
        print(f"Depth visualization saved to {output_path}")
        
        # Also save raw depth as 16-bit
        depth_16bit = (depth_normalized * 65535).astype(np.uint16)
        cv2.imwrite(output_path.replace('.png', '_16bit.png'), depth_16bit)
        print(f"16-bit depth saved to {output_path.replace('.png', '_16bit.png')}")
    else:
        print("Depth map has no variation - cannot create meaningful visualization")

def test_cpp_output():
    """
    Analyze the C++ output if it exists
    """
    cpp_output_path = "depth_output.png"
    if os.path.exists(cpp_output_path):
        print("\n=== Analyzing C++ Output ===")
        cpp_depth = cv2.imread(cpp_output_path, cv2.IMREAD_UNCHANGED)
        if cpp_depth is not None:
            print(f"C++ output shape: {cpp_depth.shape}")
            print(f"C++ output dtype: {cpp_depth.dtype}")
            print(f"C++ output range: [{cpp_depth.min()}, {cpp_depth.max()}]")
            
            if np.all(cpp_depth == 0):
                print("C++ output is all black (all zeros)")
            elif cpp_depth.max() == cpp_depth.min():
                print("C++ output has no variation")
            else:
                print("C++ output appears to have some variation")
        else:
            print("Could not load C++ output image")
    else:
        print("C++ output file not found")

def main():
    print("Depth Estimation Diagnostic Tool")
    print("=" * 50)
    
    # Paths
    image_path = "miku_128.png"
    onnx_path = "../iw3/pretrained_models/hub/distill_any_depth.onnx"
    
    # Alternative paths to check
    alt_paths = [
        "distill_any_depth.onnx",
        "pretrained_models/hub/distill_any_depth.onnx",
        "../pretrained_models/hub/distill_any_depth.onnx"
    ]
    
    # Find ONNX model
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}")
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                onnx_path = alt_path
                print(f"Found ONNX model at {alt_path}")
                break
        else:
            print("No ONNX model found. Please export the model first.")
            print("Run: python ../iw3/export_onnx.py")
            return
    
    # Test with ONNX
    depth_map, input_tensor = test_with_onnx(image_path, onnx_path)
    
    # Analyze results
    analyze_depth_output(depth_map)
    
    # Test C++ output
    test_cpp_output()
    
    # Save input for debugging
    if input_tensor is not None:
        print(f"\nInput tensor stats:")
        print(f"Shape: {input_tensor.shape}")
        print(f"Range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
        print(f"Mean: {input_tensor.mean():.6f}")
        print(f"Std: {input_tensor.std():.6f}")
        
        # Save input visualization (convert back to image)
        if input_tensor.shape[0] == 1 and input_tensor.shape[1] == 3:
            # Convert CHW back to HWC
            input_vis = np.transpose(input_tensor[0], (1, 2, 0))
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            input_vis = input_vis * std + mean
            input_vis = np.clip(input_vis * 255, 0, 255).astype(np.uint8)
            input_vis = cv2.cvtColor(input_vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite("input_preprocessed.png", input_vis)
            print("Preprocessed input saved to input_preprocessed.png")

if __name__ == "__main__":
    main()