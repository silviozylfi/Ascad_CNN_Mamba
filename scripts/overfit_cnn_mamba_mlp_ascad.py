from __future__ import annotations

import argparse
import random
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from models.cnn_mamba_mlp_ascad import CnnMambaMlpAscad
from scripts.ascad_dataloaders import create_ascad_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints/cnn_mamba_mlp_ascad/pretrained_initialization.pt"
)

NUM_CLASSES = 256


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the CNN-Mamba-MLP model can overfit one small "
            "fixed batch of normalized ASCAD traces."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Partially pretrained initialization checkpoint."
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of fixed training samples used by the overfitting test."
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Maximum number of optimization steps."
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate."
    )

    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm."
    )

    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=1.0,
        help="Accuracy required for a successful test."
    )

    parser.add_argument(
        "--log-interval",
        type=int,
        default=25,
        help="Number of optimization steps between progress messages."
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility."
    )

    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if arguments.samples <= 1:
        raise ValueError(
            "--samples must be greater than one."
        )

    if arguments.steps <= 0:
        raise ValueError(
            "--steps must be greater than zero."
        )

    if arguments.learning_rate <= 0.0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if arguments.gradient_clip_norm <= 0.0:
        raise ValueError(
            "--gradient-clip-norm must be greater than zero."
        )

    if not 0.0 < arguments.target_accuracy <= 1.0:
        raise ValueError(
            "--target-accuracy must be in the interval (0, 1]."
        )

    if arguments.log_interval <= 0:
        raise ValueError(
            "--log-interval must be greater than zero."
        )


def set_random_seed(random_seed: int) -> None:
    """Configure random generators for reproducibility."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    """Select the CUDA device required by the installed Mamba kernels."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. The installed Mamba implementation "
            "requires an NVIDIA GPU with CUDA support."
        )

    return torch.device("cuda")


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
                "Every model-state key must be a string."
            )

        if not isinstance(parameter_value, Tensor):
            raise TypeError(
                "Every model-state value must be a tensor. "
                f"Invalid entry: {parameter_name}."
            )

        validated_state_dict[parameter_name] = parameter_value

    return validated_state_dict


def load_model(
    checkpoint_path: Path,
    device: torch.device
) -> CnnMambaMlpAscad:
    """Load the partially pretrained CNN-Mamba-MLP model."""
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = CnnMambaMlpAscad(
        num_classes=NUM_CLASSES,
        d_model=512,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_blocks=3
    )

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    state_dict = extract_state_dict(
        loaded_checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    return model.to(device)


def load_fixed_batch(
    sample_count: int,
    random_seed: int,
    device: torch.device
) -> tuple[Tensor, Tensor]:
    """Load one fixed normalized batch from the ASCAD training split."""
    dataloaders = create_ascad_dataloaders(
        batch_size=sample_count,
        num_workers=0,
        random_seed=random_seed
    )

    training_iterator = iter(
        dataloaders.training
    )

    try:
        traces, labels = next(
            training_iterator
        )
    except StopIteration as error:
        raise RuntimeError(
            "The training DataLoader produced no samples."
        ) from error

    if int(traces.shape[0]) != sample_count:
        raise RuntimeError(
            "Unexpected fixed-batch size: "
            f"expected {sample_count}, found {int(traces.shape[0])}."
        )

    traces = traces.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True
    )

    labels = labels.to(
        device=device,
        dtype=torch.long,
        non_blocking=True
    )

    return traces, labels


def compute_accuracy(
    logits: Tensor,
    labels: Tensor
) -> float:
    """Return classification accuracy for one fixed batch."""
    predictions = logits.argmax(dim=1)

    return float(
        (predictions == labels).float().mean().item()
    )


def main() -> None:
    """Run the fixed-batch overfitting test."""
    arguments = parse_arguments()
    validate_arguments(arguments)

    set_random_seed(
        arguments.random_seed
    )

    device = select_device()

    model = load_model(
        checkpoint_path=arguments.checkpoint,
        device=device
    )

    traces, labels = load_fixed_batch(
        sample_count=arguments.samples,
        random_seed=arguments.random_seed,
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=arguments.learning_rate
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("CNN-Mamba-MLP ASCAD overfitting test")
    print("------------------------------------")
    print(f"Device:               {device}")
    print(f"Checkpoint:           {arguments.checkpoint.resolve()}")
    print(f"Fixed samples:        {arguments.samples}")
    print(f"Maximum steps:        {arguments.steps}")
    print(f"Learning rate:        {arguments.learning_rate}")
    print(
        "Gradient clip norm:  "
        f"{arguments.gradient_clip_norm}"
    )
    print(
        "Target accuracy:     "
        f"{arguments.target_accuracy:.2%}"
    )
    print(
        "Trainable parameters:"
        f" {trainable_parameter_count:,}"
    )
    print()

    target_reached = False
    final_loss = float("inf")
    final_accuracy = 0.0

    model.train()

    for step in range(1, arguments.steps + 1):
        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            traces
        )

        expected_shape = (
            arguments.samples,
            NUM_CLASSES
        )

        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                "Unexpected model output shape: "
                f"expected {expected_shape}, "
                f"found {tuple(logits.shape)}."
            )

        loss = criterion(
            logits,
            labels
        )

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "The overfitting loss is NaN or infinite."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=arguments.gradient_clip_norm
        )

        optimizer.step()

        final_loss = float(
            loss.item()
        )

        final_accuracy = compute_accuracy(
            logits=logits,
            labels=labels
        )

        should_print = (
            step == 1
            or step % arguments.log_interval == 0
            or final_accuracy >= arguments.target_accuracy
            or step == arguments.steps
        )

        if should_print:
            print(
                f"Step {step:04d}/{arguments.steps:04d} | "
                f"loss: {final_loss:.6f} | "
                f"accuracy: {final_accuracy:.2%}"
            )

        if final_accuracy >= arguments.target_accuracy:
            target_reached = True
            break

    print()
    print("Overfitting test result")
    print("-----------------------")
    print(f"Final loss:      {final_loss:.6f}")
    print(f"Final accuracy:  {final_accuracy:.2%}")

    if not target_reached:
        raise RuntimeError(
            "Overfitting test failed: the model did not reach "
            f"{arguments.target_accuracy:.2%} accuracy within "
            f"{arguments.steps} steps."
        )

    print(
        "Overfitting test completed successfully."
    )


if __name__ == "__main__":
    main()
