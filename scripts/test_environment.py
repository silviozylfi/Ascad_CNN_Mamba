import platform
import sys

import torch


def main() -> None:
    print(f"Operating system: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the NVIDIA GPU.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    a = torch.randn(2000, 2000, device=device)
    b = torch.randn(2000, 2000, device=device)
    c = a @ b

    print(f"Result device: {c.device}")
    print(f"Result shape: {tuple(c.shape)}")
    print("CUDA test completed successfully.")


if __name__ == "__main__":
    main()