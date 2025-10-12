import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"当前设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")