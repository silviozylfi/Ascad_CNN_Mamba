import argparse
import json
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.mamba_cnn_ascad import MambaCnnAscad
from scripts.ascad_dataloaders import create_ascad_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INITIALIZATION_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/mamba_cnn_ascad/pretrained_initialization.pt"
)

DEFAULT_CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "checkpoints/mamba_cnn_ascad/phase_1"
)

NUM_CLASSES = 256


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for phase-one Mamba-CNN training."""

    epochs: int
    batch_size: int
    learning_rate: float
    gradient_clip_norm: float
    num_workers: int
    random_seed: int
    patience: int
    initialization_checkpoint: Path
    checkpoint_directory: Path
    resume: Path | None


def parse_arguments() -> TrainingConfig:
    """Read and validate command-line training arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the Mamba-CNN front-end while keeping the pretrained "
            "ASCAD Conv2-Conv5 layers and classifier frozen."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of traces in each batch."
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
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes."
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for reproducibility."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early-stopping patience measured in epochs."
    )

    parser.add_argument(
        "--initialization-checkpoint",
        type=Path,
        default=DEFAULT_INITIALIZATION_CHECKPOINT,
        help="Checkpoint containing the partially pretrained Mamba-CNN."
    )

    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
        help="Directory in which phase-one checkpoints are saved."
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Phase-one checkpoint from which training should resume."
    )

    arguments = parser.parse_args()

    config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        gradient_clip_norm=arguments.gradient_clip_norm,
        num_workers=arguments.num_workers,
        random_seed=arguments.random_seed,
        patience=arguments.patience,
        initialization_checkpoint=arguments.initialization_checkpoint,
        checkpoint_directory=arguments.checkpoint_directory,
        resume=arguments.resume
    )

    validate_config(config)

    return config


def validate_config(config: TrainingConfig) -> None:
    """Validate the training configuration."""
    if config.epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")

    if config.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")

    if config.learning_rate <= 0.0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if config.gradient_clip_norm <= 0.0:
        raise ValueError(
            "--gradient-clip-norm must be greater than zero."
        )

    if config.num_workers < 0:
        raise ValueError(
            "--num-workers cannot be negative."
        )

    if config.patience <= 0:
        raise ValueError(
            "--patience must be greater than zero."
        )


def set_random_seed(random_seed: int) -> None:
    """Configure random generators for reproducible experiments."""
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


def extract_model_state_dict(
    checkpoint: object
) -> Mapping[str, Tensor]:
    """Extract and validate a model state dictionary."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            "The initialization checkpoint must contain a mapping."
        )

    checkpoint_mapping = cast(Mapping[str, object], checkpoint)
    state_dict_object = checkpoint_mapping.get("model_state_dict")

    if not isinstance(state_dict_object, Mapping):
        raise TypeError(
            "The initialization checkpoint does not contain a valid "
            "model_state_dict."
        )

    state_dict_mapping = cast(Mapping[object, object], state_dict_object)
    validated_state_dict: dict[str, Tensor] = {}

    for parameter_name, parameter_value in state_dict_mapping.items():
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


def load_pretrained_initialization(
    model: MambaCnnAscad,
    checkpoint_path: Path
) -> None:
    """Load the partially pretrained Mamba-CNN initialization."""
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Initialization checkpoint not found: "
            f"{checkpoint_path}"
        )

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    state_dict = extract_model_state_dict(
        loaded_checkpoint
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print(
        "Initialization checkpoint loaded: "
        f"{checkpoint_path}"
    )


def freeze_pretrained_components(
    model: MambaCnnAscad
) -> None:
    """
    Freeze Conv2 through Conv5 and the complete classifier.

    Trainable components:
        - input projection
        - Mamba blocks
        - Conv1
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.input_projection.parameters():
        parameter.requires_grad = True

    for parameter in model.mamba_blocks.parameters():
        parameter.requires_grad = True

    first_convolution = model.feature_extractor[0]

    if not isinstance(first_convolution, nn.Conv1d):
        raise TypeError(
            "The first feature-extractor layer must be nn.Conv1d."
        )

    for parameter in first_convolution.parameters():
        parameter.requires_grad = True


def get_trainable_parameters(
    model: nn.Module
) -> list[nn.Parameter]:
    """Return all parameters that will be updated by the optimizer."""
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError(
            "The model contains no trainable parameters."
        )

    return trainable_parameters


def print_parameter_summary(
    model: nn.Module
) -> None:
    """Print trainable parameter names and parameter counts."""
    total_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameter_count = (
        total_parameter_count - trainable_parameter_count
    )

    print()
    print("Trainable components")
    print("--------------------")

    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(
                f"{parameter_name}: {parameter.numel():,}"
            )

    print()
    print(f"Total parameters:     {total_parameter_count:,}")
    print(f"Trainable parameters: {trainable_parameter_count:,}")
    print(f"Frozen parameters:    {frozen_parameter_count:,}")


def compute_batch_accuracy(
    logits: Tensor,
    labels: Tensor
) -> tuple[int, int]:
    """Return the number of correct predictions and samples."""
    predictions = logits.argmax(dim=1)

    correct_predictions = int(
        (predictions == labels).sum().item()
    )

    sample_count = int(labels.shape[0])

    return correct_predictions, sample_count


def validate_model_output(
    logits: Tensor,
    batch_size: int
) -> None:
    """Validate the model output shape."""
    expected_shape = (
        batch_size,
        NUM_CLASSES
    )

    if tuple(logits.shape) != expected_shape:
        raise ValueError(
            "Unexpected model output shape: "
            f"expected {expected_shape}, found {tuple(logits.shape)}."
        )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: list[nn.Parameter],
    device: torch.device,
    gradient_clip_norm: float
) -> tuple[float, float]:
    """Train the unfrozen model components for one epoch."""
    model.train()

    accumulated_loss = 0.0
    accumulated_correct = 0
    accumulated_samples = 0

    for traces, labels in dataloader:
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

        optimizer.zero_grad(set_to_none=True)

        logits = model(traces)

        batch_size = int(labels.shape[0])

        validate_model_output(
            logits=logits,
            batch_size=batch_size
        )

        loss = criterion(
            logits,
            labels
        )

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "The training loss is NaN or infinite."
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=gradient_clip_norm
        )

        optimizer.step()

        correct_predictions, sample_count = (
            compute_batch_accuracy(
                logits=logits,
                labels=labels
            )
        )

        accumulated_loss += float(loss.item()) * batch_size
        accumulated_correct += correct_predictions
        accumulated_samples += sample_count

    if accumulated_samples == 0:
        raise RuntimeError(
            "The training DataLoader produced no samples."
        )

    average_loss = accumulated_loss / accumulated_samples
    average_accuracy = accumulated_correct / accumulated_samples

    return average_loss, average_accuracy


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Evaluate the model without updating its parameters."""
    model.eval()

    accumulated_loss = 0.0
    accumulated_correct = 0
    accumulated_samples = 0

    for traces, labels in dataloader:
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

        logits = model(traces)

        batch_size = int(labels.shape[0])

        validate_model_output(
            logits=logits,
            batch_size=batch_size
        )

        loss = criterion(
            logits,
            labels
        )

        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(
                "The validation loss is NaN or infinite."
            )

        correct_predictions, sample_count = (
            compute_batch_accuracy(
                logits=logits,
                labels=labels
            )
        )

        accumulated_loss += float(loss.item()) * batch_size
        accumulated_correct += correct_predictions
        accumulated_samples += sample_count

    if accumulated_samples == 0:
        raise RuntimeError(
            "The validation DataLoader produced no samples."
        )

    average_loss = accumulated_loss / accumulated_samples
    average_accuracy = accumulated_correct / accumulated_samples

    return average_loss, average_accuracy


def config_to_serializable_dict(
    config: TrainingConfig
) -> dict[str, Any]:
    """Convert the training configuration into serializable values."""
    config_dictionary = asdict(config)

    config_dictionary["initialization_checkpoint"] = str(
        config.initialization_checkpoint.resolve()
    )

    config_dictionary["checkpoint_directory"] = str(
        config.checkpoint_directory.resolve()
    )

    config_dictionary["resume"] = (
        None
        if config.resume is None
        else str(config.resume.resolve())
    )

    return config_dictionary


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    best_validation_loss: float,
    epochs_without_improvement: int,
    config: TrainingConfig
) -> None:
    """Save the complete phase-one training state."""
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "training_phase": 1,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "training_configuration": config_to_serializable_dict(config)
    }

    torch.save(
        checkpoint,
        checkpoint_path
    )


def save_training_history(
    history: list[dict[str, float | int]],
    history_path: Path
) -> None:
    """Save phase-one training metrics in JSON format."""
    history_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with history_path.open(
        mode="w",
        encoding="utf-8"
    ) as history_file:
        json.dump(
            history,
            history_file,
            indent=4
        )


def load_training_history(
    history_path: Path
) -> list[dict[str, float | int]]:
    """Load a previously saved phase-one training history."""
    if not history_path.is_file():
        return []

    with history_path.open(
        mode="r",
        encoding="utf-8"
    ) as history_file:
        loaded_history: object = json.load(history_file)

    if not isinstance(loaded_history, list):
        raise TypeError(
            "The training history must contain a list."
        )

    validated_history: list[dict[str, float | int]] = []

    for item_index, item in enumerate(loaded_history):
        if not isinstance(item, dict):
            raise TypeError(
                "Every training-history entry must be a dictionary. "
                f"Invalid entry index: {item_index}."
            )

        validated_item: dict[str, float | int] = {}

        for key, value in item.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Every training-history key must be a string."
                )

            if isinstance(value, bool) or not isinstance(
                value,
                (int, float)
            ):
                raise TypeError(
                    "Every training-history value must be numeric. "
                    f"Invalid key: {key}."
                )

            validated_item[key] = value

        validated_history.append(validated_item)

    return validated_history


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path
) -> tuple[int, float, int]:
    """
    Restore model, optimizer and phase-one progress from a checkpoint.

    Returns:
        completed_epoch,
        best_validation_loss,
        epochs_without_improvement
    """
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {checkpoint_path}"
        )

    loaded_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if not isinstance(loaded_checkpoint, Mapping):
        raise TypeError(
            "The resume checkpoint must contain a mapping."
        )

    checkpoint_mapping = cast(
        Mapping[str, object],
        loaded_checkpoint
    )

    model_state_object = checkpoint_mapping.get(
        "model_state_dict"
    )

    optimizer_state_object = checkpoint_mapping.get(
        "optimizer_state_dict"
    )

    completed_epoch_object = checkpoint_mapping.get(
        "epoch"
    )

    validation_loss_object = checkpoint_mapping.get(
        "validation_loss"
    )

    best_validation_loss_object = checkpoint_mapping.get(
        "best_validation_loss",
        validation_loss_object
    )

    epochs_without_improvement_object = checkpoint_mapping.get(
        "epochs_without_improvement",
        0
    )

    if not isinstance(model_state_object, Mapping):
        raise TypeError(
            "The resume checkpoint does not contain a valid "
            "model_state_dict."
        )

    if not isinstance(optimizer_state_object, dict):
        raise TypeError(
            "The resume checkpoint does not contain a valid "
            "optimizer_state_dict."
        )

    if isinstance(completed_epoch_object, bool) or not isinstance(
        completed_epoch_object,
        int
    ):
        raise TypeError(
            "The resume checkpoint does not contain a valid epoch."
        )

    if isinstance(best_validation_loss_object, bool) or not isinstance(
        best_validation_loss_object,
        (int, float)
    ):
        raise TypeError(
            "The resume checkpoint does not contain a valid best "
            "validation loss."
        )

    if isinstance(epochs_without_improvement_object, bool) or not isinstance(
        epochs_without_improvement_object,
        int
    ):
        raise TypeError(
            "The resume checkpoint does not contain a valid early-stopping "
            "counter."
        )

    validated_model_state: dict[str, Tensor] = {}

    for parameter_name, parameter_value in model_state_object.items():
        if not isinstance(parameter_name, str):
            raise TypeError(
                "Every resumed model-state key must be a string."
            )

        if not isinstance(parameter_value, Tensor):
            raise TypeError(
                "Every resumed model-state value must be a tensor. "
                f"Invalid entry: {parameter_name}."
            )

        validated_model_state[parameter_name] = parameter_value

    model.load_state_dict(
        validated_model_state,
        strict=True
    )

    optimizer.load_state_dict(
        optimizer_state_object
    )

    completed_epoch = completed_epoch_object
    best_validation_loss = float(
        best_validation_loss_object
    )
    epochs_without_improvement = (
        epochs_without_improvement_object
    )

    print()
    print(f"Training resumed from:        {checkpoint_path}")
    print(f"Completed epoch:              {completed_epoch}")
    print(
        "Best validation loss:        "
        f"{best_validation_loss:.6f}"
    )
    print(
        "Epochs without improvement:  "
        f"{epochs_without_improvement}"
    )
    print(
        "Training will start at epoch "
        f"{completed_epoch + 1}."
    )
    print()

    return (
        completed_epoch,
        best_validation_loss,
        epochs_without_improvement
    )


def format_duration(duration_seconds: float) -> str:
    """Format a duration as hours, minutes and seconds."""
    hours = int(duration_seconds // 3600)
    minutes = int((duration_seconds % 3600) // 60)
    seconds = duration_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"


def main() -> None:
    """Run phase-one training for the partially pretrained Mamba-CNN."""
    config = parse_arguments()

    set_random_seed(
        config.random_seed
    )

    device = select_device()

    checkpoint_directory = (
        config.checkpoint_directory.resolve()
    )

    best_checkpoint_path = (
        checkpoint_directory / "best_model.pt"
    )

    last_checkpoint_path = (
        checkpoint_directory / "last_model.pt"
    )

    history_path = (
        checkpoint_directory / "training_history.json"
    )

    print()
    print("Mamba-CNN ASCAD phase-one training")
    print("----------------------------------")
    print(f"Device:                    {device}")
    print(f"Epochs:                    {config.epochs}")
    print(f"Batch size:                {config.batch_size}")
    print(f"Learning rate:             {config.learning_rate}")
    print(
        "Gradient clip norm:        "
        f"{config.gradient_clip_norm}"
    )
    print(f"Random seed:               {config.random_seed}")
    print(
        "Initialization checkpoint: "
        f"{config.initialization_checkpoint.resolve()}"
    )
    print(f"Best checkpoint:           {best_checkpoint_path}")
    print(
        "Resume checkpoint:         "
        f"{config.resume.resolve() if config.resume is not None else None}"
    )
    print()

    dataloaders = create_ascad_dataloaders(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        random_seed=config.random_seed
    )

    model = MambaCnnAscad(
        num_classes=NUM_CLASSES,
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        num_mamba_blocks=2
    )

    load_pretrained_initialization(
        model=model,
        checkpoint_path=config.initialization_checkpoint
    )

    freeze_pretrained_components(
        model
    )

    model = model.to(device)

    print_parameter_summary(
        model
    )

    trainable_parameters = get_trainable_parameters(
        model
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        trainable_parameters,
        lr=config.learning_rate
    )

    start_epoch = 1
    history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    if config.resume is not None:
        (
            completed_epoch,
            best_validation_loss,
            epochs_without_improvement
        ) = resume_training(
            model=model,
            optimizer=optimizer,
            checkpoint_path=config.resume
        )

        start_epoch = completed_epoch + 1

        history = load_training_history(
            history_path
        )

        if history:
            last_history_epoch = history[-1].get("epoch")

            if last_history_epoch != completed_epoch:
                raise ValueError(
                    "The training history and resume checkpoint refer "
                    "to different completed epochs."
                )

    if start_epoch > config.epochs:
        raise ValueError(
            "The requested total number of epochs has already been "
            "completed. "
            f"Completed: {start_epoch - 1}, requested: {config.epochs}."
        )

    training_start_time = time.perf_counter()

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.perf_counter()

        training_loss, training_accuracy = train_one_epoch(
            model=model,
            dataloader=dataloaders.training,
            criterion=criterion,
            optimizer=optimizer,
            trainable_parameters=trainable_parameters,
            device=device,
            gradient_clip_norm=config.gradient_clip_norm
        )

        validation_loss, validation_accuracy = evaluate_model(
            model=model,
            dataloader=dataloaders.validation,
            criterion=criterion,
            device=device
        )

        epoch_duration = (
            time.perf_counter() - epoch_start_time
        )

        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "duration_seconds": epoch_duration
        }

        history.append(
            epoch_metrics
        )

        save_training_history(
            history=history,
            history_path=history_path
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs:03d} | "
            f"train loss: {training_loss:.6f} | "
            f"train acc: {training_accuracy:.4%} | "
            f"val loss: {validation_loss:.6f} | "
            f"val acc: {validation_accuracy:.4%} | "
            f"time: {epoch_duration:.1f}s"
        )


        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                config=config
            )

            print("  New best checkpoint saved.")
        else:
            epochs_without_improvement += 1

            print(
                "  Epochs without improvement: "
                f"{epochs_without_improvement}/{config.patience}"
            )

        save_checkpoint(
            last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            best_validation_loss=best_validation_loss,
            epochs_without_improvement=epochs_without_improvement,
            config=config
        )

        if epochs_without_improvement >= config.patience:
            print()
            print(
                "Early stopping: validation loss did not improve "
                f"for {config.patience} epochs."
            )
            break

    total_duration = (
        time.perf_counter() - training_start_time
    )

    print()
    print("Phase-one training completed")
    print("----------------------------")
    print(
        "Total duration:       "
        f"{format_duration(total_duration)}"
    )
    print(
        "Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(f"Best checkpoint:      {best_checkpoint_path}")
    print(f"Last checkpoint:      {last_checkpoint_path}")
    print(f"Training history:     {history_path}")


if __name__ == "__main__":
    main()