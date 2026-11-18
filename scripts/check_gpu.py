"""Windows NVIDIA CUDA installation smoke test; does not train a model."""

import platform


def main():
    print("OS:", platform.platform())
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is not installed. Use https://pytorch.org/get-started/locally/") from exc

    print("PyTorch:", torch.__version__)
    print("Built with CUDA:", torch.version.cuda)
    available = torch.cuda.is_available()
    print("CUDA available:", available)
    if not available:
        raise SystemExit("CUDA unavailable: check NVIDIA driver and CUDA-enabled PyTorch build.")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    print("GPU:", props.name)
    print("VRAM (GiB):", round(props.total_memory / 1024**3, 2))
    tensor = torch.randn(1024, 1024, device=device)
    result = tensor @ tensor.T
    torch.cuda.synchronize(device)
    print("CUDA tensor test: OK", tuple(result.shape))
    print("Allocated VRAM (MiB):", round(torch.cuda.memory_allocated(device) / 1024**2, 2))


if __name__ == "__main__":
    main()
