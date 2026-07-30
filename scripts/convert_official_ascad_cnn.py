from pathlib import Path
from typing import cast

import h5py
import numpy as np
import torch
import torch.nn as nn

from models.ascad_cnn_original import ASCADOriginalCNN


MODEL_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_trained_models/"
    "cnn_best_ascad_desync0_epochs75_classes256_batchsize200.h5"
)

OUTPUT_PATH = Path(
    "checkpoints/ascad_cnn_official_converted.pt"
)


def load_h5_array(
    weights_group: h5py.Group,
    dataset_path: str,
) -> np.ndarray:
    dataset = cast(
        h5py.Dataset,
        weights_group[dataset_path]
    )

    return np.asarray(dataset, dtype=np.float32)


def copy_conv1d_weights(
    pytorch_layer: nn.Conv1d,
    keras_layer_name: str,
    weights_group: h5py.Group,
) -> None:
    kernel = load_h5_array(
        weights_group,
        f"{keras_layer_name}/{keras_layer_name}/kernel:0",
    )

    bias = load_h5_array(
        weights_group,
        f"{keras_layer_name}/{keras_layer_name}/bias:0",
    )

    # Keras Conv1D:
    # (kernel_size, in_channels, out_channels)
    #
    # PyTorch Conv1d:
    # (out_channels, in_channels, kernel_size)
    kernel = np.transpose(kernel, (2, 1, 0)).copy()

    kernel_tensor = torch.from_numpy(kernel)
    bias_tensor = torch.from_numpy(bias)

    if pytorch_layer.weight.shape != kernel_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {keras_layer_name} kernel: "
            f"PyTorch={tuple(pytorch_layer.weight.shape)}, "
            f"Keras={tuple(kernel_tensor.shape)}"
        )

    if pytorch_layer.bias is None:
        raise ValueError(
            f"PyTorch layer {keras_layer_name} has no bias."
        )

    if pytorch_layer.bias.shape != bias_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {keras_layer_name} bias: "
            f"PyTorch={tuple(pytorch_layer.bias.shape)}, "
            f"Keras={tuple(bias_tensor.shape)}"
        )

    with torch.no_grad():
        pytorch_layer.weight.copy_(kernel_tensor)
        pytorch_layer.bias.copy_(bias_tensor)

    print(
        f"Loaded {keras_layer_name}: "
        f"kernel={tuple(kernel_tensor.shape)}, "
        f"bias={tuple(bias_tensor.shape)}"
    )


def copy_linear_weights(
    pytorch_layer: nn.Linear,
    keras_layer_name: str,
    weights_group: h5py.Group
) -> None:
    kernel = load_h5_array(
        weights_group,
        f"{keras_layer_name}/{keras_layer_name}/kernel:0",
    )

    bias = load_h5_array(
        weights_group,
        f"{keras_layer_name}/{keras_layer_name}/bias:0",
    )

    # Keras Dense:
    # (in_features, out_features)
    #
    # PyTorch Linear:
    # (out_features, in_features)
    if keras_layer_name == "fc1":
        # Keras Flatten:
        # (batch, temporal_length, channels)
        #
        # PyTorch Flatten:
        # (batch, channels, temporal_length)
        #
        # L'output dell'ultimo blocco convoluzionale è:
        # Keras:   (21, 512)
        # PyTorch: (512, 21)
        kernel = kernel.reshape(
            21,
            512,
            4096
        )

        kernel = np.transpose(
            kernel,
            (1, 0, 2)
        )

        kernel = kernel.reshape(
            10752,
            4096
        )

    kernel = kernel.T.copy()

    kernel_tensor = torch.from_numpy(kernel)
    bias_tensor = torch.from_numpy(bias)

    if pytorch_layer.weight.shape != kernel_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {keras_layer_name} kernel: "
            f"PyTorch={tuple(pytorch_layer.weight.shape)}, "
            f"Keras={tuple(kernel_tensor.shape)}"
        )

    if pytorch_layer.bias is None:
        raise ValueError(
            f"PyTorch layer {keras_layer_name} has no bias."
        )

    if pytorch_layer.bias.shape != bias_tensor.shape:
        raise ValueError(
            f"Shape mismatch for {keras_layer_name} bias: "
            f"PyTorch={tuple(pytorch_layer.bias.shape)}, "
            f"Keras={tuple(bias_tensor.shape)}"
        )

    with torch.no_grad():
        pytorch_layer.weight.copy_(kernel_tensor)
        pytorch_layer.bias.copy_(bias_tensor)

    print(
        f"Loaded {keras_layer_name}: "
        f"kernel={tuple(kernel_tensor.shape)}, "
        f"bias={tuple(bias_tensor.shape)}"
    )


def load_keras_weights_into_pytorch(
    model: ASCADOriginalCNN,
    h5_path: Path,
) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Official ASCAD model not found: {h5_path}"
        )

    with h5py.File(h5_path, "r") as h5_file:
        weights_group = cast(
            h5py.Group,
            h5_file["model_weights"],
        )

        conv_mapping: list[tuple[str, nn.Conv1d]] = [
            ("block1_conv1", cast(nn.Conv1d, model.features[0])),
            ("block2_conv1", cast(nn.Conv1d, model.features[3])),
            ("block3_conv1", cast(nn.Conv1d, model.features[6])),
            ("block4_conv1", cast(nn.Conv1d, model.features[9])),
            ("block5_conv1", cast(nn.Conv1d, model.features[12]))
        ]

        dense_mapping: list[tuple[str, nn.Linear]] = [
            ("fc1", cast(nn.Linear, model.classifier[1])),
            ("fc2", cast(nn.Linear, model.classifier[3])),
            ("predictions", cast(nn.Linear, model.classifier[5]))
        ]

        for keras_name, pytorch_layer in conv_mapping:
            copy_conv1d_weights(
                pytorch_layer=pytorch_layer,
                keras_layer_name=keras_name,
                weights_group=weights_group,
            )

        for keras_name, pytorch_layer in dense_mapping:
            copy_linear_weights(
                pytorch_layer=pytorch_layer,
                keras_layer_name=keras_name,
                weights_group=weights_group,
            )


def verify_forward_pass(
    model: ASCADOriginalCNN
) -> None:
    model.eval()

    dummy_input = torch.zeros(
        size=(2, 1, 700),
        dtype=torch.float32
    )

    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=1)

    expected_shape = (2, 256)

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            f"Unexpected output shape: "
            f"expected={expected_shape}, "
            f"obtained={tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise ValueError(
            "The model produced non-finite logits."
        )

    probability_sums = probabilities.sum(dim=1)

    if not torch.allclose(
        probability_sums,
        torch.ones_like(probability_sums),
        atol=1e-5,
    ):
        raise ValueError(
            "Softmax probabilities do not sum to 1."
        )

    print(
        f"Forward-pass verification successful: "
        f"input={tuple(dummy_input.shape)}, "
        f"output={tuple(logits.shape)}"
    )


def main() -> None:
    print(f"Official Keras model: {MODEL_PATH}")
    print(f"Output checkpoint:    {OUTPUT_PATH}")

    model = ASCADOriginalCNN()

    load_keras_weights_into_pytorch(
        model=model,
        h5_path=MODEL_PATH
    )

    verify_forward_pass(model)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_model": str(MODEL_PATH),
            "architecture": "ASCADOriginalCNN",
            "input_length": 700,
            "num_classes": 256,
            "desynchronization": 0,
            "training_epochs": 75,
            "training_batch_size": 200
        },
        OUTPUT_PATH,
    )

    print(
        f"Converted model saved successfully to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()