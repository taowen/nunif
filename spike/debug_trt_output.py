import numpy as np
import cv2
import os
import struct

def read_raw_tensor_file(filepath, shape, dtype=np.float32):
    """
    Read raw tensor data from a binary file
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found")
        return None
    
    try:
        data = np.fromfile(filepath, dtype=dtype)
        if len(shape) > 1:
            data = data.reshape(shape)
        return data
    except Exception as e:
        print(f"Error reading tensor file: {e}")
        return None

def analyze_trt_inference():
    """
    Analyze TensorRT inference output by examining the raw values
    """
    print("=== TensorRT Output Analysis ===")
    
    # Check if depth_output.png exists and analyze it
    depth_output_path = "depth_output.png"
    if os.path.exists(depth_output_path):
        # Read as different formats to see what we get
        img_8bit = cv2.imread(depth_output_path, cv2.IMREAD_GRAYSCALE)
        img_16bit = cv2.imread(depth_output_path, cv2.IMREAD_UNCHANGED)
        
        print(f"8-bit read shape: {img_8bit.shape if img_8bit is not None else 'None'}")
        print(f"16-bit read shape: {img_16bit.shape if img_16bit is not None else 'None'}")
        
        if img_8bit is not None:
            print(f"8-bit range: [{img_8bit.min()}, {img_8bit.max()}]")
            print(f"8-bit unique values: {len(np.unique(img_8bit))}")
            
        if img_16bit is not None:
            print(f"16-bit dtype: {img_16bit.dtype}")
            print(f"16-bit range: [{img_16bit.min()}, {img_16bit.max()}]")
            print(f"16-bit unique values: {len(np.unique(img_16bit))}")
            
            # Check if it's actually all black
            if np.all(img_16bit == 0):
                print("IMAGE IS ALL BLACK (all zeros)")
            elif img_16bit.max() == img_16bit.min():
                print(f"IMAGE HAS NO VARIATION (all values are {img_16bit.max()})")
            else:
                print("Image has some variation")
    else:
        print("depth_output.png not found")

def create_debug_version_cpp():
    """
    Create a debug version of the C++ code that outputs raw tensor values
    """
    debug_code = '''
// Add this to your C++ code right after inference, before postprocessing:

// Debug: Save raw output to file for analysis
std::ofstream debug_file("debug_raw_output.bin", std::ios::binary);
if (debug_file.is_open()) {
    debug_file.write(reinterpret_cast<const char*>(output_data.data()), 
                     output_data.size() * sizeof(float));
    debug_file.close();
    std::cout << "Raw output saved to debug_raw_output.bin" << std::endl;
    
    // Print some statistics
    float min_val = *std::min_element(output_data.begin(), output_data.end());
    float max_val = *std::max_element(output_data.begin(), output_data.end());
    float sum = std::accumulate(output_data.begin(), output_data.end(), 0.0f);
    float mean = sum / output_data.size();
    
    std::cout << "Raw output stats:" << std::endl;
    std::cout << "  Size: " << output_data.size() << std::endl;
    std::cout << "  Min: " << min_val << std::endl;
    std::cout << "  Max: " << max_val << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    
    // Check for all zeros
    bool all_zero = std::all_of(output_data.begin(), output_data.end(), 
                               [](float f) { return f == 0.0f; });
    if (all_zero) {
        std::cout << "  WARNING: All values are zero!" << std::endl;
    }
    
    // Check for very small values
    bool all_tiny = std::all_of(output_data.begin(), output_data.end(), 
                               [](float f) { return std::abs(f) < 1e-6f; });
    if (all_tiny) {
        std::cout << "  WARNING: All values are very close to zero!" << std::endl;
    }
}
'''
    
    print("=== Debug C++ Code ===")
    print("Add this code to your C++ file to debug raw tensor output:")
    print(debug_code)

def analyze_raw_debug_output():
    """
    Analyze the raw debug output if it exists
    """
    debug_file = "debug_raw_output.bin"
    if os.path.exists(debug_file):
        print("\n=== Raw Debug Output Analysis ===")
        
        # Read raw float data
        raw_data = np.fromfile(debug_file, dtype=np.float32)
        print(f"Raw data size: {raw_data.size}")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Raw data range: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
        print(f"Raw data mean: {raw_data.mean():.6f}")
        print(f"Raw data std: {raw_data.std():.6f}")
        
        # Check for problematic values
        zero_count = (raw_data == 0).sum()
        nan_count = np.isnan(raw_data).sum()
        inf_count = np.isinf(raw_data).sum()
        
        print(f"Zero values: {zero_count} / {raw_data.size}")
        print(f"NaN values: {nan_count}")
        print(f"Inf values: {inf_count}")
        
        if zero_count == raw_data.size:
            print("ALL VALUES ARE ZERO - This is the problem!")
        elif np.all(np.abs(raw_data) < 1e-6):
            print("ALL VALUES ARE VERY CLOSE TO ZERO - Likely inference issue")
        
        # Try to determine likely shape based on common depth map sizes
        likely_shapes = [
            (392, 392), (512, 512), (640, 480), (480, 640),
            (224, 224), (256, 256), (416, 416)
        ]
        
        for h, w in likely_shapes:
            if h * w == raw_data.size:
                print(f"Data could be shaped as {h}x{w}")
                reshaped = raw_data.reshape(h, w)
                
                # Save visualization if data has variation
                if reshaped.max() > reshaped.min():
                    normalized = (reshaped - reshaped.min()) / (reshaped.max() - reshaped.min())
                    vis = (normalized * 255).astype(np.uint8)
                    cv2.imwrite(f"debug_output_{h}x{w}.png", vis)
                    print(f"Visualization saved as debug_output_{h}x{w}.png")
                break
    else:
        print("\nNo debug_raw_output.bin found. Run the modified C++ code first.")

def check_model_files():
    """
    Check if required model files exist
    """
    print("\n=== Model Files Check ===")
    
    files_to_check = [
        "distill_any_depth.trt",
        "../iw3/pretrained_models/hub/distill_any_depth.trt",
        "../iw3/pretrained_models/hub/distill_any_depth.onnx",
        "../iw3/pretrained_models/hub/distill_any_depth.onnx"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} (not found)")

def main():
    print("TensorRT Debug Analysis")
    print("=" * 50)
    
    # Check model files
    check_model_files()
    
    # Analyze TensorRT output
    analyze_trt_inference()
    
    # Check for raw debug output
    analyze_raw_debug_output()
    
    # Provide debug code
    create_debug_version_cpp()
    
    print("\n=== Recommendations ===")
    print("1. First, run the ONNX diagnostic: python diagnose_depth.py")
    print("2. Add the debug code to your C++ file and recompile")
    print("3. Run your C++ program to generate debug_raw_output.bin")
    print("4. Run this script again to analyze the raw output")
    print("5. Compare ONNX vs TensorRT outputs to identify the issue")

if __name__ == "__main__":
    main() 