python -m onnxsim stereo_module_half_sbs.onnx sim.onnx
trtexec --onnx=sim.onnx --saveEngine=stereo_module_half_sbs.trt --fp16 --optShapes=input:1x3x392x392 --maxShapes=input:2x3x2160x3840