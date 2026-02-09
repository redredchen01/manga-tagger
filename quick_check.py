import torch
import sys


def check_environment():
    """Check RTX 5090 environment configuration"""
    print("=== RTX 5090 Environment Check ===")

    # Check Python version
    print(f"Python Version: {sys.version}")

    # Check PyTorch version
    try:
        pytorch_version = torch.__version__
    except:
        pytorch_version = "Unknown"
    print(f"PyTorch Version: {pytorch_version}")

    # Check CUDA availability
    if torch.cuda.is_available():
        print("CUDA Available: YES")

        # Check GPU devices
        device_count = torch.cuda.device_count()
        print(f"GPU Count: {device_count}")

        if device_count > 0:
            # Check GPU details
            gpu_props = torch.cuda.get_device_properties(0)
            print(f"GPU Name: {gpu_props.name}")
            print(f"Total VRAM: {gpu_props.total_memory / 1024**3:.1f}GB")

            # Check current memory usage
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Current VRAM Usage: {current_memory:.2f}GB")

            return True
        else:
            print("ERROR: No GPU devices detected")
            return False
    else:
        print("ERROR: CUDA not available")
        return False


if __name__ == "__main__":
    print("RTX 5090 Chinese Embedding Models Environment Verification")
    print("=============================================")

    # Environment check
    env_ok = check_environment()

    if not env_ok:
        print("ERROR: Environment check failed, please install CUDA drivers first")
        sys.exit(1)
    else:
        print("\nSUCCESS: RTX 5090 environment is ready!")
        print("You can now proceed to load Chinese embedding models.")
