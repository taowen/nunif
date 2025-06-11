if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF
    from PIL import Image
    import os
    import argparse
    from .utils import (
        load_sbs_model, apply_divergence, to_pil_image, 
        ROW_FLOW_V3_SYM_URL, HUB_MODEL_DIR,
        preprocess_image, postprocess_image
    )
    from .base_depth_model import BaseDepthModel
    from nunif.utils.ui import TorchHubDir
    from nunif.models import load_model
    from nunif.utils.pil_io import load_image_simple
    
    def create_test_args():
        """Create a mock args object with default parameters for testing"""
        class MockArgs:
            def __init__(self):
                # Divergence parameters
                self.divergence = 2.0
                self.convergence = 0.5
                self.method = "row_flow_v3_sym"
                self.synthetic_view = "both"
                self.warp_steps = None
                self.mapper = "none"
                self.preserve_screen_border = False
                
                # Image processing
                self.rotate_left = False
                self.rotate_right = False
                self.max_output_height = None
                self.disable_amp = False
                self.edge_dilation = 0
                
                # Output format
                self.half_sbs = False
                self.vr180 = False
                self.tb = False
                self.half_tb = False
                self.cross_eyed = False
                self.anaglyph = None
                self.rgbd = False
                self.half_rgbd = False
                self.debug_depth = False
                
                # Padding and sizing
                self.pad = None
                self.pad_mode = "tblr"
                self.ipd_offset = 0
                self.max_output_width = None
                self.keep_aspect_ratio = False
                
                # Stereo processing (missing attribute that caused the error)
                self.stereo_width = None
                
                # Additional attributes that might be needed
                self.tta = False
                self.low_vram = False
                self.depth_aa = False
                self.foreground_scale = 0
                self.resolution = None
                
                # State
                self.state = {
                    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    "inpaint_model": None
                }
                
        return MockArgs()
    
    def test_row_flow_v3_sym(rgb_path, depth_path, output_dir="./output"):
        """
        Test the row_flow_v3_sym model with RGB image and depth map
        
        Args:
            rgb_path: Path to RGB image
            depth_path: Path to depth map (PNG format)
            output_dir: Output directory for left and right eye images
        """
        print(f"Loading RGB image from: {rgb_path}")
        print(f"Loading depth map from: {depth_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create mock args
        args = create_test_args()
        device = args.state["device"]
        print(f"Using device: {device}")
        
        # Load RGB image
        rgb_image, _ = load_image_simple(rgb_path, color="rgb")
        rgb_tensor = TF.to_tensor(rgb_image).to(device)
        print(f"RGB image shape: {rgb_tensor.shape}")
        
        # Load depth map
        depth_tensor = BaseDepthModel.load_depth(depth_path)[0].to(device)
        print(f"Depth map shape: {depth_tensor.shape}")
        
        # Ensure depth and RGB have same spatial dimensions
        if rgb_tensor.shape[1:] != depth_tensor.shape[1:]:
            print(f"Resizing depth from {depth_tensor.shape[1:]} to {rgb_tensor.shape[1:]}")
            depth_tensor = F.interpolate(
                depth_tensor.unsqueeze(0), 
                size=rgb_tensor.shape[1:], 
                mode="bilinear", 
                align_corners=True, 
                antialias=True
            ).squeeze(0)
        
        # Load row_flow_v3_sym model
        print("Loading row_flow_v3_sym model...")
        with TorchHubDir(HUB_MODEL_DIR):
            side_model = load_model(ROW_FLOW_V3_SYM_URL, weights_only=True, device_ids=[0])[0].eval()
            side_model.symmetric = True
            side_model.delta_output = True
        
        print("Model loaded successfully")
        
        # Preprocessing
        rgb_processed = preprocess_image(rgb_tensor, args)
        print(f"Processed RGB shape: {rgb_processed.shape}")
        
        # Apply divergence to generate left and right eye views
        print("Generating left and right eye views...")
        with torch.inference_mode():
            left_eye, right_eye = apply_divergence(
                depth_tensor.unsqueeze(0),  # Add batch dimension
                rgb_processed.unsqueeze(0),  # Add batch dimension
                args, 
                side_model
            )
            
            # Remove batch dimension
            left_eye = left_eye.squeeze(0)
            right_eye = right_eye.squeeze(0)
        
        print(f"Left eye shape: {left_eye.shape}")
        print(f"Right eye shape: {right_eye.shape}")
        
        # Convert to PIL images
        left_pil = to_pil_image(left_eye)
        right_pil = to_pil_image(right_eye)
        
        # Save images
        left_path = os.path.join(output_dir, "left_eye.png")
        right_path = os.path.join(output_dir, "right_eye.png")
        
        left_pil.save(left_path)
        right_pil.save(right_path)
        
        print(f"Left eye saved to: {left_path}")
        print(f"Right eye saved to: {right_path}")
        
        # Also create side-by-side image for easy viewing
        sbs_width = left_pil.width + right_pil.width
        sbs_height = max(left_pil.height, right_pil.height)
        sbs_image = Image.new('RGB', (sbs_width, sbs_height))
        sbs_image.paste(left_pil, (0, 0))
        sbs_image.paste(right_pil, (left_pil.width, 0))
        
        sbs_path = os.path.join(output_dir, "side_by_side.png")
        sbs_image.save(sbs_path)
        print(f"Side-by-side image saved to: {sbs_path}")
        
        return left_path, right_path, sbs_path
    
    # Example usage
    parser = argparse.ArgumentParser(description="Test row_flow_v3_sym model")
    parser.add_argument("--rgb", type=str, default="waifu2x/docs/images/miku_128.png",required=False, help="Path to RGB image")
    parser.add_argument("--depth", type=str, default="./tmp/depth_anything_out.png", required=False, help="Path to depth map")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--divergence", type=float, default=2.0, help="Divergence strength (0-5)")
    parser.add_argument("--convergence", type=float, default=0.5, help="Convergence plane (0-1)")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.rgb):
        raise FileNotFoundError(f"RGB image not found: {args.rgb}")
    if not os.path.exists(args.depth):
        raise FileNotFoundError(f"Depth map not found: {args.depth}")
    
    # Run test
    try:
        left_path, right_path, sbs_path = test_row_flow_v3_sym(
            args.rgb, 
            args.depth, 
            args.output
        )
        print("\nTest completed successfully!")
        print(f"You can view the results in: {args.output}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        raise