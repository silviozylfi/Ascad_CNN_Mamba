from __future__ import annotations

import torch

from models.cnn_mamba_mlp_ascad import (
    CnnMambaMlpAscad,
    count_trainable_parameters
)


def main() -> None:
    """Run a CUDA smoke test for the CNN-Mamba-MLP model."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The installed Mamba implementation "
            "requires an NVIDIA GPU with CUDA support."
        )

    device = torch.device("cuda")

    model = CnnMambaMlpAscad().to(device)
    model.eval()

    inputs = torch.randn(
        4,
        1,
        700,
        device=device,
        dtype=torch.float32
    )

    with torch.inference_mode():
        cnn_features = model.feature_extractor(inputs)
        mamba_input = cnn_features.transpose(1, 2).contiguous()

        mamba_output = mamba_input

        for block in model.mamba_blocks:
            mamba_output = block(mamba_output)

        pooled_output = mamba_output.mean(dim=1)
        mlp_output = model.mlp(pooled_output)
        logits = model.classifier(mlp_output)

    expected_output_shape = (
        inputs.shape[0],
        256
    )

    if tuple(logits.shape) != expected_output_shape:
        raise RuntimeError(
            "Unexpected output shape: "
            f"expected {expected_output_shape}, "
            f"found {tuple(logits.shape)}."
        )

    print("CNN-Mamba-MLP ASCAD smoke test")
    print("------------------------------")
    print(f"Device:             {device}")
    print(f"Input:              {tuple(inputs.shape)}")
    print(f"CNN output:         {tuple(cnn_features.shape)}")
    print(f"Mamba input:        {tuple(mamba_input.shape)}")
    print(f"Mamba output:       {tuple(mamba_output.shape)}")
    print(f"Global pooling:     {tuple(pooled_output.shape)}")
    print(f"MLP output:         {tuple(mlp_output.shape)}")
    print(f"Classifier output:  {tuple(logits.shape)}")
    print(
        "Trainable parameters: "
        f"{count_trainable_parameters(model):,}"
    )
    print()
    print("Result: OK")


if __name__ == "__main__":
    main()
