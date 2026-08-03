from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from models.ascad_cnn_original import ASCADOriginalCNN
from models.cnn_mamba_mlp_ascad import CnnMambaMlpAscad


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OFFICIAL_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/ascad_cnn_official_converted.pt"
)

OUTPUT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/cnn_mamba_mlp_ascad/pretrained_initialization.pt"
)


def extract_state_dict(
    checkpoint: object
) -> Mapping[str, Tensor]:
    """Extract and validate a model state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "The checkpoint must contain a mapping."
        )

    checkpoint_mapping = cast(
        Mapping[str, object],
        checkpoint
    )

    state_dict_object = checkpoint_mapping.get(
        "model_state_dict",
        checkpoint
    )

    if not isinstance(state_dict_object, Mapping):
        raise TypeError(
            "The checkpoint does not contain a valid model state."
        )

    validated_state_dict: dict[str, Tensor] = {}

    for parameter_name, parameter_value in state_dict_object.items():
        if not isinstance(parameter_name, str):
            raise TypeError(
                "Every state-dictionary key must be a string."
            )

        if not isinstance(parameter_value, Tensor):
            raise TypeError(
                "Every state-dictionary value must be a tensor. "
                f"Invalid entry: {parameter_name}."
            )

        validated_state_dict[parameter_name] = parameter_value

    return validated_state_dict


def get_conv1d_layer(
    sequential: nn.Sequential,
    layer_index: int,
    layer_name: str
) -> nn.Conv1d:
    """Return and validate one Conv1d layer."""
    layer = sequential[layer_index]

    if not isinstance(layer, nn.Conv1d):
        raise TypeError(
            f"{layer_name} must be an nn.Conv1d layer."
        )

    return layer


def get_linear_layer(
    sequential: nn.Sequential,
    layer_index: int,
    layer_name: str
) -> nn.Linear:
    """Return and validate one Linear layer."""
    layer = sequential[layer_index]

    if not isinstance(layer, nn.Linear):
        raise TypeError(
            f"{layer_name} must be an nn.Linear layer."
        )

    return layer


def copy_conv1d_parameters(
    source_layer: nn.Conv1d,
    target_layer: nn.Conv1d,
    layer_name: str
) -> None:
    """Copy compatible Conv1d parameters."""
    if source_layer.weight.shape != target_layer.weight.shape:
        raise ValueError(
            f"{layer_name} weight shapes do not match: "
            f"source={tuple(source_layer.weight.shape)}, "
            f"target={tuple(target_layer.weight.shape)}."
        )

    if source_layer.bias is None or target_layer.bias is None:
        raise ValueError(
            f"{layer_name} requires biases in both models."
        )

    if source_layer.bias.shape != target_layer.bias.shape:
        raise ValueError(
            f"{layer_name} bias shapes do not match: "
            f"source={tuple(source_layer.bias.shape)}, "
            f"target={tuple(target_layer.bias.shape)}."
        )

    with torch.no_grad():
        target_layer.weight.copy_(source_layer.weight)
        target_layer.bias.copy_(source_layer.bias)


def copy_linear_parameters(
    source_layer: nn.Linear,
    target_layer: nn.Linear,
    layer_name: str
) -> None:
    """Copy compatible Linear parameters."""
    if source_layer.weight.shape != target_layer.weight.shape:
        raise ValueError(
            f"{layer_name} weight shapes do not match: "
            f"source={tuple(source_layer.weight.shape)}, "
            f"target={tuple(target_layer.weight.shape)}."
        )

    if source_layer.bias is None or target_layer.bias is None:
        raise ValueError(
            f"{layer_name} requires biases in both models."
        )

    if source_layer.bias.shape != target_layer.bias.shape:
        raise ValueError(
            f"{layer_name} bias shapes do not match: "
            f"source={tuple(source_layer.bias.shape)}, "
            f"target={tuple(target_layer.bias.shape)}."
        )

    with torch.no_grad():
        target_layer.weight.copy_(source_layer.weight)
        target_layer.bias.copy_(source_layer.bias)


def initialize_batch_norm_as_identity(
    model: CnnMambaMlpAscad
) -> None:
    """Initialize every BatchNorm1d layer as an identity transform."""
    batch_norm_count = 0

    for module in model.feature_extractor.modules():
        if not isinstance(module, nn.BatchNorm1d):
            continue

        with torch.no_grad():
            if module.weight is not None:
                module.weight.fill_(1.0)

            if module.bias is not None:
                module.bias.zero_()

            if module.running_mean is not None:
                module.running_mean.zero_()

            if module.running_var is not None:
                module.running_var.fill_(1.0)

            if module.num_batches_tracked is not None:
                module.num_batches_tracked.zero_()

        batch_norm_count += 1

    if batch_norm_count != 5:
        raise RuntimeError(
            "Expected exactly five BatchNorm1d layers, "
            f"but found {batch_norm_count}."
        )

    print(
        f"Initialized {batch_norm_count} BatchNorm1d layers as identity."
    )


def copy_official_convolutions(
    official_model: ASCADOriginalCNN,
    target_model: CnnMambaMlpAscad
) -> None:
    """Copy all five official ASCAD convolutional layers."""
    layer_mappings = (
        ("Conv1", 0, 0),
        ("Conv2", 3, 4),
        ("Conv3", 6, 8),
        ("Conv4", 9, 12),
        ("Conv5", 12, 16)
    )

    for layer_name, source_index, target_index in layer_mappings:
        source_layer = get_conv1d_layer(
            sequential=official_model.features,
            layer_index=source_index,
            layer_name=f"Official {layer_name}"
        )

        target_layer = get_conv1d_layer(
            sequential=target_model.feature_extractor,
            layer_index=target_index,
            layer_name=f"Target {layer_name}"
        )

        copy_conv1d_parameters(
            source_layer=source_layer,
            target_layer=target_layer,
            layer_name=layer_name
        )

        print(
            f"{layer_name} copied successfully: "
            f"{tuple(source_layer.weight.shape)}."
        )


def copy_compatible_classifier_layers(
    official_model: ASCADOriginalCNN,
    target_model: CnnMambaMlpAscad
) -> None:
    """Copy the exactly compatible official classifier layers."""
    layer_mappings = (
        ("FC2", 3, 3),
        ("Predictions", 5, 5)
    )

    for layer_name, source_index, target_index in layer_mappings:
        source_layer = get_linear_layer(
            sequential=official_model.classifier,
            layer_index=source_index,
            layer_name=f"Official {layer_name}"
        )

        target_layer = get_linear_layer(
            sequential=target_model.classifier,
            layer_index=target_index,
            layer_name=f"Target {layer_name}"
        )

        copy_linear_parameters(
            source_layer=source_layer,
            target_layer=target_layer,
            layer_name=layer_name
        )

        print(
            f"{layer_name} copied successfully: "
            f"{tuple(source_layer.weight.shape)}."
        )


def count_parameters(
    module: nn.Module
) -> int:
    """Return the total number of parameters in a module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def print_parameter_summary(
    model: CnnMambaMlpAscad
) -> None:
    """Print parameter counts for every major model section."""
    feature_extractor_parameters = count_parameters(
        model.feature_extractor
    )

    mamba_parameters = count_parameters(
        model.mamba_blocks
    )

    mlp_parameters = count_parameters(
        model.mlp
    )

    classifier_parameters = count_parameters(
        model.classifier
    )

    total_parameters = count_parameters(
        model
    )

    print()
    print("Parameter summary")
    print("-----------------")
    print(
        "Feature extractor: "
        f"{feature_extractor_parameters:,}"
    )
    print(f"Mamba blocks:      {mamba_parameters:,}")
    print(f"MLP:               {mlp_parameters:,}")
    print(f"Classifier:        {classifier_parameters:,}")
    print(f"Total:             {total_parameters:,}")


def main() -> None:
    """Create a partially pretrained CNN-Mamba-MLP checkpoint."""
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

    target_model = CnnMambaMlpAscad(
        num_classes=256,
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_blocks=3
    )

    initialize_batch_norm_as_identity(
        target_model
    )

    print()
    print("Loading compatible official ASCAD weights")
    print("-----------------------------------------")

    copy_official_convolutions(
        official_model=official_model,
        target_model=target_model
    )

    copy_compatible_classifier_layers(
        official_model=official_model,
        target_model=target_model
    )

    print()
    print("Randomly initialized components")
    print("-------------------------------")
    print("- Three Residual Mamba blocks")
    print("- Two-layer intermediate MLP")
    print("- Classifier LayerNorm")
    print("- Classifier 512 -> 4096 layer")
    print()
    print("Not copied")
    print("----------")
    print(
        "- Official FC1 10752 -> 4096 because it is incompatible "
        "with the new 512-dimensional representation"
    )

    print_parameter_summary(
        target_model
    )

    OUTPUT_CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_checkpoint = {
        "model_state_dict": target_model.state_dict(),
        "source_checkpoint": str(
            OFFICIAL_CHECKPOINT_PATH
        ),
        "initialization": {
            "conv1_to_conv5": "official_ascad",
            "batch_norm": "identity",
            "mamba_blocks": "random",
            "intermediate_mlp": "random",
            "classifier_layer_norm": "random",
            "classifier_fc1": "random",
            "classifier_fc2": "official_ascad",
            "classifier_predictions": "official_ascad"
        },
        "model_configuration": {
            "num_classes": 256,
            "d_model": 512,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "num_mamba_blocks": 3
        }
    }

    torch.save(
        output_checkpoint,
        OUTPUT_CHECKPOINT_PATH
    )

    print()
    print(
        "Initialization checkpoint saved successfully to: "
        f"{OUTPUT_CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()
