import torch
import sys


def check_pytorch_environment():
    print("=" * 60)
    print("PyTorch Environment Check Script")
    print("=" * 60)

    # 1. Python Version Check
    print(f"\n[1] Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        print(
            "    WARNING: Python version is older than 3.12 (Recommended for your setup)"
        )
    else:
        print("    STATUS: OK")

    # 2. PyTorch Version Check
    print(f"\n[2] PyTorch Version: {torch.__version__}")

    # 3. CUDA Availability Check
    cuda_available = torch.cuda.is_available()
    print(f"\n[3] CUDA Available: {cuda_available}")

    if not cuda_available:
        print("    CRITICAL ERROR: CUDA is not available to PyTorch.")
        print(
            "    Possible causes: Docker not run with --gpus all, incorrect PyTorch build, or driver issues."
        )
        return

    # 4. CUDA Version Check (Reported by PyTorch)
    print(f"    CUDA Version (PyTorch): {torch.version.cuda}")

    # 5. GPU Device Check
    device_count = torch.cuda.device_count()
    print(f"\n[4] Detected GPUs: {device_count}")

    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        print(f"    GPU {i}: {device_name}")
        print(f"    Compute Capability: {capability[0]}.{capability[1]}")

        # Check for RTX 5080 (Blackwell architecture)
        if "5080" in device_name:
            print("    STATUS: RTX 5080 Detected.")

    # 6. Tensor Computation & CUDA Test
    print("\n[5] Running Tensor Computation Test...")
    try:
        # Perform a simple matrix multiplication on the GPU
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)

        # Ensure synchronization to catch any async CUDA errors
        torch.cuda.synchronize()

        print("    Success: Matrix multiplication on GPU completed.")
        print(f"    Result shape: {z.shape}")
        print("    Device: " + str(z.device))

    except Exception as e:
        print(f"    ERROR: Computation failed. {str(e)}")
        return

    # 7. cuDNN Check
    print(f"\n[6] cuDNN Enabled: {torch.backends.cudnn.enabled}")
    if torch.backends.cudnn.is_available():
        print(f"    cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("    WARNING: cuDNN not available")

    print("\n" + "=" * 60)
    print("SUMMARY: Environment is operational.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        check_pytorch_environment()
    except ImportError:
        print("CRITICAL: PyTorch not installed. Please run 'pip install torch'")
