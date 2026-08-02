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

DEFAULT_PHASE_ONE_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints/mamba_cnn_ascad/phase_1/best_model.pt"
)

DEFAULT_CHECKPOINT_DIRECTORY = (
    PROJECT_ROOT
    / "checkpoints/mamba_cnn_ascad/phase_2"
)

NUM_CLASSES = 256

FRONT_END_GROUP_NAME = "front_end"
CNN_GROUP_NAME = "conv2_conv3"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for phase-two Mamba-CNN fine-tuning."""

    epochs: int
    batch_size: int
    front_end_learning_rate: float
    cnn_learning_rate: float
    gradient_clip_norm: float
    num_workers: int
    random_seed: int
    patience: int
    phase_one_checkpoint: Path
    checkpoint_directory: Path
    resume: Path | None


def parse_arguments() -> TrainingConfig:
    """Read and validate command-line training arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run phase-two Mamba-CNN fine-tuning by training the Mamba "
            "front-end, Conv1, Conv2 and Conv3 while keeping Conv4, Conv5 "
            "and the classifier frozen."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum total number of phase-two epochs."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of traces in each batch."
    )

    parser.add_argument(
        "--front-end-learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for the input projection, Mamba blocks and Conv1."
    )

    parser.add_argument(
        "--cnn-learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate for Conv2 and Conv3."
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
        "--phase-one-checkpoint",
        type=Path,
        default=DEFAULT_PHASE_ONE_CHECKPOINT,
        help="Best phase-one checkpoint used to initialize phase two."
    )

    parser.add_argument(
        "--checkpoint-directory",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
        help="Directory in which phase-two checkpoints are saved."
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Phase-two checkpoint from which training should resume."
    )

    arguments = parser.parse_args()

    config = TrainingConfig(
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        front_end_learning_rate=arguments.front_end_learning_rate,
        cnn_learning_rate=arguments.cnn_learning_rate,
        gradient_clip_norm=arguments.gradient_clip_norm,
        num_workers=arguments.num_workers,
        random_seed=arguments.random_seed,
        patience=arguments.patience,
        phase_one_checkpoint=arguments.phase_one_checkpoint,
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

    if config.front_end_learning_rate <= 0.0:
        raise ValueError(
            "--front-end-learning-rate must be greater than zero."
        )

    if config.cnn_learning_rate <= 0.0:
        raise ValueError(
            "--cnn-learning-rate must be greater than zero."
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
            "The checkpoint must contain a mapping."
        )

    checkpoint_mapping = cast(
        Mapping[str, object],
        checkpoint
    )

    state_dict_object = checkpoint_mapping.get(
        "model_state_dict"
    )

    if not isinstance(state_dict_object, Mapping):
        raise TypeError(
            "The checkpoint does not contain a valid model_state_dict."
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


def load_phase_one_checkpoint(
    model: MambaCnnAscad,
    checkpoint_path: Path
) -> None:
    """Load the best phase-one model weights."""
    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Phase-one checkpoint not found: {checkpoint_path}"
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
        "Phase-one checkpoint loaded: "
        f"{checkpoint_path}"
    )


def get_convolution(
    model: MambaCnnAscad,
    layer_index: int,
    layer_name: str
) -> nn.Conv1d:
    """Return and validate one convolutional layer."""
    layer = model.feature_extractor[layer_index]

    if not isinstance(layer, nn.Conv1d):
        raise TypeError(
            f"{layer_name} must be an nn.Conv1d layer."
        )

    return layer


def configure_phase_two_trainable_components(
    model: MambaCnnAscad
) -> None:
    """
    Configure the phase-two trainable and frozen components.

    Trainable:
        - input projection
        - Mamba blocks
        - Conv1
        - Conv2
        - Conv3

    Frozen:
        - Conv4
        - Conv5
        - complete classifier
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.input_projection.parameters():
        parameter.requires_grad = True

    for parameter in model.mamba_blocks.parameters():
        parameter.requires_grad = True

    trainable_convolution_indices = (
        (0, "Conv1"),
        (3, "Conv2"),
        (6, "Conv3")
    )

    for layer_index, layer_name in trainable_convolution_indices:
        convolution = get_convolution(
            model=model,
            layer_index=layer_index,
            layer_name=layer_name
        )

        for parameter in convolution.parameters():
            parameter.requires_grad = True

    frozen_convolution_indices = (
        (9, "Conv4"),
        (12, "Conv5")
    )

    for layer_index, layer_name in frozen_convolution_indices:
        convolution = get_convolution(
            model=model,
            layer_index=layer_index,
            layer_name=layer_name
        )

        if any(
            parameter.requires_grad
            for parameter in convolution.parameters()
        ):
            raise RuntimeError(
                f"{layer_name} must remain frozen during phase two."
            )

    if any(
        parameter.requires_grad
        for parameter in model.classifier.parameters()
    ):
        raise RuntimeError(
            "The classifier must remain frozen during phase two."
        )


def collect_parameter_groups(
    model: MambaCnnAscad,
    config: TrainingConfig
) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    """Create the Adam parameter groups used during phase two."""
    conv1 = get_convolution(
        model=model,
        layer_index=0,
        layer_name="Conv1"
    )

    conv2 = get_convolution(
        model=model,
        layer_index=3,
        layer_name="Conv2"
    )

    conv3 = get_convolution(
        model=model,
        layer_index=6,
        layer_name="Conv3"
    )

    front_end_parameters = [
        *model.input_projection.parameters(),
        *model.mamba_blocks.parameters(),
        *conv1.parameters()
    ]

    cnn_parameters = [
        *conv2.parameters(),
        *conv3.parameters()
    ]

    all_trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    grouped_parameter_ids = {
        id(parameter)
        for parameter in (
            front_end_parameters
            + cnn_parameters
        )
    }

    trainable_parameter_ids = {
        id(parameter)
        for parameter in all_trainable_parameters
    }

    if grouped_parameter_ids != trainable_parameter_ids:
        raise RuntimeError(
            "The optimizer parameter groups do not match the complete "
            "set of trainable parameters."
        )

    parameter_groups: list[dict[str, Any]] = [
        {
            "params": front_end_parameters,
            "lr": config.front_end_learning_rate,
            "group_name": FRONT_END_GROUP_NAME
        },
        {
            "params": cnn_parameters,
            "lr": config.cnn_learning_rate,
            "group_name": CNN_GROUP_NAME
        }
    ]

    return parameter_groups, all_trainable_parameters


def print_parameter_summary(
    model: MambaCnnAscad
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


def print_optimizer_groups(
    optimizer: torch.optim.Optimizer
) -> None:
    """Print the learning rate and parameter count of each group."""
    print()
    print("Optimizer groups")
    print("----------------")

    for group_index, parameter_group in enumerate(
        optimizer.param_groups,
        start=1
    ):
        group_name_object = parameter_group.get(
            "group_name",
            f"group_{group_index}"
        )

        group_name = str(group_name_object)
        learning_rate = float(parameter_group["lr"])

        parameters_object = parameter_group["params"]

        if not isinstance(parameters_object, list):
            raise TypeError(
                "Optimizer parameter groups must contain parameter lists."
            )

        parameter_count = sum(
            parameter.numel()
            for parameter in parameters_object
            if isinstance(parameter, Tensor)
        )

        print(
            f"{group_name}: "
            f"lr={learning_rate:.2e}, "
            f"parameters={parameter_count:,}"
        )


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
    """Train the phase-two components for one epoch."""
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

    config_dictionary["phase_one_checkpoint"] = str(
        config.phase_one_checkpoint.resolve()
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
    """Save the complete phase-two training state."""
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "training_phase": 2,
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
    """Save phase-two training metrics in JSON format."""
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
    """Load a previously saved phase-two training history."""
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


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> None:
    """Move all restored optimizer tensors to the selected device."""
    for optimizer_state in optimizer.state.values():
        for state_name, state_value in optimizer_state.items():
            if isinstance(state_value, Tensor):
                optimizer_state[state_name] = state_value.to(device)


def resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device
) -> tuple[int, float, int]:
    """
    Restore model, optimizer and phase-two progress from a checkpoint.

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

    training_phase_object = checkpoint_mapping.get(
        "training_phase"
    )

    if training_phase_object != 2:
        raise ValueError(
            "The resume checkpoint is not a phase-two checkpoint."
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

    move_optimizer_state_to_device(
        optimizer=optimizer,
        device=device
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
    """Run phase-two fine-tuning for the Mamba-CNN model."""
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
    print("Mamba-CNN ASCAD phase-two training")
    print("----------------------------------")
    print(f"Device:                    {device}")
    print(f"Epochs:                    {config.epochs}")
    print(f"Batch size:                {config.batch_size}")
    print(
        "Front-end learning rate:   "
        f"{config.front_end_learning_rate}"
    )
    print(
        "Conv2-Conv3 learning rate: "
        f"{config.cnn_learning_rate}"
    )
    print(
        "Gradient clip norm:        "
        f"{config.gradient_clip_norm}"
    )
    print(f"Random seed:               {config.random_seed}")
    print(
        "Phase-one checkpoint:      "
        f"{config.phase_one_checkpoint.resolve()}"
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

    load_phase_one_checkpoint(
        model=model,
        checkpoint_path=config.phase_one_checkpoint
    )

    configure_phase_two_trainable_components(
        model
    )

    model = model.to(device)

    print_parameter_summary(
        model
    )

    parameter_groups, trainable_parameters = (
        collect_parameter_groups(
            model=model,
            config=config
        )
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        parameter_groups
    )

    print_optimizer_groups(
        optimizer
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
            checkpoint_path=config.resume,
            device=device
        )

        start_epoch = completed_epoch + 1

        history = load_training_history(
            history_path
        )

        if history:
            last_history_epoch = history[-1].get(
                "epoch"
            )

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
            "front_end_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
            "cnn_learning_rate": float(
                optimizer.param_groups[1]["lr"]
            ),
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
    print("Phase-two training completed")
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
