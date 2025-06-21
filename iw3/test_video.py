import tensorrt as trt
import torch
from torchvision.transforms import functional as TF
from PIL import Image
from nunif.logger import logger
import numpy as np
import cuda.cudart as cudart
from nunif.utils.video import process_video, VideoOutputConfig, get_fps

def check_cuda_error(cuda_ret):
    err, *rest = cuda_ret
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {cudart.cudaGetErrorString(err)[1].decode('utf-8')}")
    if len(rest) == 1:
        return rest[0]
    return rest


class TensorRTProcessor:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.bindings = None
        self.input_bindings_dev = []
        self.output_bindings_dev = {}
        self.output_tensors_host = {}

    def _initialize_buffers(self, x_batch):
        self.context.set_input_shape('input', x_batch.shape)

        self.bindings = [0] * self.engine.num_io_tensors
        for binding_idx in range(self.engine.num_io_tensors):
            binding_name = self.engine.get_tensor_name(binding_idx)
            if self.engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
                if binding_name == 'input':
                    h_input = x_batch.contiguous()
                    d_input = check_cuda_error(cudart.cudaMalloc(h_input.nbytes))
                    self.bindings[binding_idx] = int(d_input)
                    self.input_bindings_dev.append(d_input)
            else:
                output_shape = self.context.get_tensor_shape(binding_name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(binding_name))
                output_dtype = torch.from_numpy(np.array(0, dtype=dtype)).dtype
                h_output = torch.empty(tuple(output_shape), dtype=output_dtype, device="cpu")
                d_output = check_cuda_error(cudart.cudaMalloc(h_output.nbytes))
                self.bindings[binding_idx] = int(d_output)
                self.output_bindings_dev[binding_name] = d_output
                self.output_tensors_host[binding_name] = h_output

    def process(self, frame):
        if frame is None:
            return None

        x = TF.to_tensor(frame.to_image())
        x_batch = x.unsqueeze(0)

        if self.bindings is None:
            self._initialize_buffers(x_batch)

        h_input = x_batch.contiguous()
        check_cuda_error(cudart.cudaMemcpy(self.input_bindings_dev[0], h_input.data_ptr(), h_input.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))

        self.context.execute_v2(self.bindings)

        for name, d_output in self.output_bindings_dev.items():
            h_output = self.output_tensors_host[name]
            check_cuda_error(cudart.cudaMemcpy(h_output.data_ptr(), d_output, h_output.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

        left_batch = self.output_tensors_host['left']
        right_batch = self.output_tensors_host['right']

        combined = torch.cat([left_batch[0], right_batch[0]], dim=2)
        combined_pil = TF.to_pil_image(combined)

        return frame.from_image(combined_pil)

    def release(self):
        if self.input_bindings_dev:
            all_output_bindings = list(self.output_bindings_dev.values())
            for d_mem in self.input_bindings_dev + all_output_bindings:
                check_cuda_error(cudart.cudaFree(d_mem))
        self.bindings = None
        self.input_bindings_dev = []
        self.output_bindings_dev = {}
        self.output_tensors_host = {}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--engine", "-e", default="stereo_module.trt", help="TensorRT engine path")
    args = parser.parse_args()

    processor = TensorRTProcessor(args.engine)

    def make_config(stream):
        fps = get_fps(stream)
        if fps > 30:
            fps = 30
        return VideoOutputConfig(
            fps=fps,
            options={"preset": "ultrafast", "crf": "20"}
        )

    def frame_callback(frame):
        return processor.process(frame)

    try:
        process_video(args.input, args.output,
                      frame_callback=frame_callback,
                      config_callback=make_config)
    finally:
        processor.release()


if __name__ == "__main__":
    main()
