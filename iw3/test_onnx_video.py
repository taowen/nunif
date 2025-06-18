import torch
import onnxruntime as ort
import numpy as np
from nunif.utils import video as VU
from os import path
import sys
from PIL import Image
import torch.nn.functional as F

ort.set_default_logger_severity(0)

def run_onnx_stereo_video(
    input_video, output_video, onnx_path
):
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def process_frame(frame):
        if frame is None:
            return None  # Handle flush call at end of video processing
        # Convert VideoFrame to numpy array, then to PIL Image, resize, then back to numpy
        x = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float() / 255.0  # CHW, float32, 0-1
        x = x.unsqueeze(0).numpy()  # BCHW, numpy
        ort_inputs = {ort_session.get_inputs()[0].name: x}
        left, right = ort_session.run(None, ort_inputs)
        left = torch.from_numpy(left[0])
        right = torch.from_numpy(right[0])

        # === 新增：resize left/right 到输入帧分辨率 ===
        input_h, input_w = frame.height, frame.width
        # left/right: CHW, 0-1
        left = F.interpolate(left.unsqueeze(0), size=(input_h, input_w), mode="bicubic", align_corners=True).squeeze(0)
        right = F.interpolate(right.unsqueeze(0), size=(input_h, input_w), mode="bicubic", align_corners=True).squeeze(0)
        # === end ===

        sbs = torch.cat([left, right], dim=2)  # CHW, W*2
        sbs = torch.clamp(sbs, 0., 1.)  # 保证范围
        return VU.to_frame(sbs)

    def config_callback(stream):
        fps = VU.get_fps(stream)
        return VU.VideoOutputConfig(
            fps=fps,
            options={"preset": "ultrafast", "crf": "20"}
        )

    VU.process_video(
        input_video,
        output_video,
        config_callback=config_callback,
        frame_callback=process_frame,
        title=path.basename(input_video)
    )

HUB_MODEL_DIR = path.join(path.dirname(__file__), "pretrained_models", "hub")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_onnx_video.py input.mp4 output.mp4")
        exit(1)
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    onnx_path = path.join(HUB_MODEL_DIR, f"end2end_stereo_s.onnx")
    run_onnx_stereo_video(input_video, output_video, onnx_path) 