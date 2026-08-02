from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from models.ascad_cnn_original import ASCADOriginalCNN
from models.mamba_cnn_ascad import MambaCnnAscad


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/ascad_cnn_official_converted.pt"
)

OUTPUT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/mamba_cnn_ascad/pretrained_initialization.pt"
)


def extract_state_dict(
    checkpoint: Any
) -> Mapping[str, Tensor]:
    """Extract a model state dictionary from a checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "The checkpoint must contain a mapping."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "The checkpoint does not contain a valid state dictionary."
        )

    for parameter_name, parameter_value in state_dict.items():
        if not isinstance(parameter_name, str):
            raise TypeError(
                "Every state-dictionary key must be a string."
            )

        if not isinstance(parameter_value, Tensor):
            raise TypeError(
                "Every state-dictionary value must be a tensor. "
                f"Invalid entry: {parameter_name}."
            )

    return cast(Mapping[str, Tensor], state_dict)


def copy_conv1d_parameters(
    source_layer: nn.Conv1d,
    target_layer: nn.Conv1d
) -> None:
    """Copy parameters between compatible Conv1d layers."""
    if source_layer.weight.shape != target_layer.weight.shape:
        raise ValueError(
            "Conv1d weight shapes do not match: "
            f"source={tuple(source_layer.weight.shape)}, "
            f"target={tuple(target_layer.weight.shape)}."
        )

    if source_layer.bias is None or target_layer.bias is None:
        raise ValueError(
            "Both Conv1d layers must contain a bias."
        )

    if source_layer.bias.shape != target_layer.bias.shape:
        raise ValueError(
            "Conv1d bias shapes do not match: "
            f"source={tuple(source_layer.bias.shape)}, "
            f"target={tuple(target_layer.bias.shape)}."
        )

    with torch.no_grad():
        target_layer.weight.copy_(source_layer.weight)
        target_layer.bias.copy_(source_layer.bias)


def copy_linear_parameters(
    source_layer: nn.Linear,
    target_layer: nn.Linear
) -> None:
    """Copy parameters between compatible Linear layers."""
    if source_layer.weight.shape != target_layer.weight.shape:
        raise ValueError(
            "Linear weight shapes do not match: "
            f"source={tuple(source_layer.weight.shape)}, "
            f"target={tuple(target_layer.weight.shape)}."
        )

    if source_layer.bias is None or target_layer.bias is None:
        raise ValueError(
            "Both Linear layers must contain a bias."
        )

    if source_layer.bias.shape != target_layer.bias.shape:
        raise ValueError(
            "Linear bias shapes do not match: "
            f"source={tuple(source_layer.bias.shape)}, "
            f"target={tuple(target_layer.bias.shape)}."
        )

    with torch.no_grad():
        target_layer.weight.copy_(source_layer.weight)
        target_layer.bias.copy_(source_layer.bias)


def copy_pretrained_convolutions(
    official_model: ASCADOriginalCNN,
    target_model: MambaCnnAscad
) -> None:
    """Copy official convolutional layers 2 through 5."""
    layer_mappings = (
        ("Conv2", 3),
        ("Conv3", 6),
        ("Conv4", 9),
        ("Conv5", 12)
    )

    for layer_name, layer_index in layer_mappings:
        source_layer = cast(
            nn.Conv1d,
            official_model.features[layer_index]
        )

        target_layer = cast(
            nn.Conv1d,
            target_model.feature_extractor[layer_index]
        )

        copy_conv1d_parameters(
            source_layer=source_layer,
            target_layer=target_layer
        )

        print(f"{layer_name} copied successfully.")


def copy_pretrained_classifier(
    official_model: ASCADOriginalCNN,
    target_model: MambaCnnAscad
) -> None:
    """Copy the complete official ASCAD classifier."""
    layer_mappings = (
        ("FC1", 1),
        ("FC2", 3),
        ("Predictions", 5)
    )

    for layer_name, layer_index in layer_mappings:
        source_layer = cast(
            nn.Linear,
            official_model.classifier[layer_index]
        )

        target_layer = cast(
            nn.Linear,
            target_model.classifier[layer_index]
        )

        copy_linear_parameters(
            source_layer=source_layer,
            target_layer=target_layer
        )

        print(f"{layer_name} copied successfully.")


def print_parameter_summary(
    model: MambaCnnAscad
) -> None:
    """Print the number of parameters in each model section."""
    input_projection_parameters = sum(
        parameter.numel()
        for parameter in model.input_projection.parameters()
    )

    mamba_parameters = sum(
        parameter.numel()
        for parameter in model.mamba_blocks.parameters()
    )

    feature_extractor_parameters = sum(
        parameter.numel()
        for parameter in model.feature_extractor.parameters()
    )

    classifier_parameters = sum(
        parameter.numel()
        for parameter in model.classifier.parameters()
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print("Parameter summary")
    print("-----------------")
    print(
        f"Input projection:  "
        f"{input_projection_parameters:,}"
    )
    print(f"Mamba blocks:      {mamba_parameters:,}")
    print(
        f"Feature extractor: "
        f"{feature_extractor_parameters:,}"
    )
    print(f"Classifier:        {classifier_parameters:,}")
    print(f"Total:             {total_parameters:,}")


def main() -> None:
    """Initialize Mamba-CNN using compatible official ASCAD weights."""
    if not OFFICIAL_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            "Official ASCAD checkpoint not found: "
            f"{OFFICIAL_CHECKPOINT_PATH}"
        )

    print(f"Official checkpoint: {OFFICIAL_CHECKPOINT_PATH}")
    print(f"Output checkpoint:   {OUTPUT_CHECKPOINT_PATH}")
    print()

    loaded_checkpoint = torch.load(
        OFFICIAL_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True
    )

    official_state_dict = extract_state_dict(
        loaded_checkpoint
    )

    official_model = ASCADOriginalCNN()

    official_model.load_state_dict(
        official_state_dict,
        strict=True
    )

    target_model = MambaCnnAscad(
        num_classes=256,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_blocks=2
    )

    print("Randomly initialized components:")
    print("- Input projection")
    print("- Mamba blocks")
    print("- Conv1")
    print()

    print("Loading compatible pretrained components:")

    copy_pretrained_convolutions(
        official_model=official_model,
        target_model=target_model
    )

    copy_pretrained_classifier(
        official_model=official_model,
        target_model=target_model
    )

    print_parameter_summary(target_model)

    OUTPUT_CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_checkpoint = {
        "model_state_dict": target_model.state_dict(),
        "source_checkpoint": str(OFFICIAL_CHECKPOINT_PATH),
        "initialization": {
            "input_projection": "random",
            "mamba_blocks": "random",
            "conv1": "random",
            "conv2_to_conv5": "official_ascad",
            "classifier": "official_ascad"
        },
        "model_configuration": {
            "num_classes": 256,
            "d_model": 64,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_mamba_blocks": 2
        }
    }

    torch.save(
        output_checkpoint,
        OUTPUT_CHECKPOINT_PATH
    )

    print()
    print(
        "Pretrained initialization checkpoint saved to: "
        f"{OUTPUT_CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()