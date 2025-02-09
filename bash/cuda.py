import torch
import sys

print(f"Python version: {sys.version}")

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.cuda.get_device_name(0)
    print(f"GPU ({device}) is available and CUDA is supported.")
else:
    print("GPU and CUDA are not available.")
