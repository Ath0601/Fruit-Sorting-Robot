import torch
import sys

print("=== RTX 4060 PyTorch Verification ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

if torch.cuda.is_available():
    print(f"🎉 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test GPU tensor operations
    x = torch.randn(3, 3).cuda()
    y = torch.randn(3, 3).cuda()
    z = x + y
    print(f"✅ GPU computation test passed!")
    
    # Check memory usage
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
else:
    print("❌ CUDA not available in PyTorch")
    print("Make sure you installed the CUDA version of PyTorch")

print(f"Available devices: {torch.cuda.device_count()}")