"""
Check CUDA availability and PyTorch setup.

Run this script to verify your GPU setup before training.
"""

import torch
import sys


def check_cuda():
    """Check CUDA availability and print device information."""
    print("=" * 60)
    print("CUDA Availability Check")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
            print(f"  Available Memory: {torch.cuda.mem_get_info(i)[0] / 1e9:.2f} GB")

        # Test CUDA functionality
        print("\n" + "-" * 60)
        print("Testing CUDA functionality...")
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            print("✓ CUDA matrix multiplication test passed!")
        except Exception as e:
            print(f"✗ CUDA test failed: {e}")
            return False

        print("\n" + "=" * 60)
        print("[OK] CUDA is ready for training!")
        print("=" * 60)
        return True

    else:
        print("\n" + "=" * 60)
        print("[WARNING] CUDA is not available.")
        print("=" * 60)
        print("\nFallback options:")
        print("1. Check if GPU drivers are installed")
        print("2. Reinstall PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("3. Or use CPU (slower but will work)")
        print("\nTo use CPU in training, set device: 'cpu' in config.yaml")
        return False


if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
