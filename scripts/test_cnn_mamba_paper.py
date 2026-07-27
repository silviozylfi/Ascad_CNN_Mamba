import torch

from models.cnn_mamba_paper import PaperCNNMambaModel


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. "
            "This Mamba model requires an NVIDIA GPU with CUDA."
        )

    device = torch.device("cuda")

    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    batch_size = 4
    trace_length = 700

    model = PaperCNNMambaModel(num_classes=256).to(device)
    model.eval()

    traces = torch.randn(
        batch_size,
        trace_length,
        device=device,
    )

    with torch.no_grad():
        logits = model(traces)

    print("Input shape:", traces.shape)
    print("Input device:", traces.device)

    print("Output shape:", logits.shape)
    print("Output device:", logits.device)

    expected_shape = (batch_size, 256)
    assert logits.shape == expected_shape

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_parameter_count:,}")
    print("Model test completed successfully.")


if __name__ == "__main__":
    main()