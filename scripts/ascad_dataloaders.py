from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from scripts.ascad_dataset import ASCADDataset
from scripts.normalized_ascad_dataset import (
    ASCADDatasetProtocol,
    NormalizedASCADDataset
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ASCAD_PATH = (
    PROJECT_ROOT
    / "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

DEFAULT_SPLIT_PATH = (
    PROJECT_ROOT
    / "data/processed/ascad_train_validation_split.npz"
)

DEFAULT_NORMALIZATION_PATH = (
    PROJECT_ROOT
    / "data/processed/ascad_training_normalization.npz"
)

EXPECTED_TRAINING_SAMPLES = 45_000
EXPECTED_VALIDATION_SAMPLES = 5_000
EXPECTED_ATTACK_SAMPLES = 10_000
EXPECTED_TRACE_LENGTH = 700


@dataclass(frozen=True)
class ASCADDataLoaders:
    """Container for the ASCAD DataLoaders."""

    training: DataLoader[tuple[Tensor, Tensor]]
    validation: DataLoader[tuple[Tensor, Tensor]]
    attack: DataLoader[tuple[Tensor, Tensor]]


def load_train_validation_indices(
    split_path: str | Path
) -> tuple[list[int], list[int]]:
    """Load and validate the deterministic dataset split."""
    split_path = Path(split_path)

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Dataset split file not found: {split_path}"
        )

    with np.load(split_path) as split_file:
        training_indices = np.asarray(
            split_file["training_indices"],
            dtype=np.int64
        )

        validation_indices = np.asarray(
            split_file["validation_indices"],
            dtype=np.int64
        )

    if training_indices.ndim != 1:
        raise ValueError(
            "Training indices must be one-dimensional."
        )

    if validation_indices.ndim != 1:
        raise ValueError(
            "Validation indices must be one-dimensional."
        )

    if training_indices.size != EXPECTED_TRAINING_SAMPLES:
        raise ValueError(
            "Unexpected number of training indices: "
            f"expected {EXPECTED_TRAINING_SAMPLES}, "
            f"found {training_indices.size}."
        )

    if validation_indices.size != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(
            "Unexpected number of validation indices: "
            f"expected {EXPECTED_VALIDATION_SAMPLES}, "
            f"found {validation_indices.size}."
        )

    if np.intersect1d(
        training_indices,
        validation_indices
    ).size != 0:
        raise ValueError(
            "Training and validation indices overlap."
        )

    return (
        training_indices.tolist(),
        validation_indices.tolist()
    )


def collate_ascad_batch(
    batch: list[tuple[Tensor, Tensor]]
) -> tuple[Tensor, Tensor]:
    """
    Combine ASCAD samples into a model-ready batch.

    Trace shape:
        (700,) -> (batch_size, 1, 700)

    Label shape:
        scalar -> (batch_size,)
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    traces, labels = zip(*batch)

    trace_batch = torch.stack(
        tuple(traces),
        dim=0
    ).to(dtype=torch.float32)

    label_batch = torch.stack(
        tuple(labels),
        dim=0
    ).to(dtype=torch.long)

    if trace_batch.ndim != 2:
        raise ValueError(
            "Unexpected trace batch dimensions: "
            f"expected 2, found {trace_batch.ndim}."
        )

    if trace_batch.shape[1] != EXPECTED_TRACE_LENGTH:
        raise ValueError(
            "Unexpected trace length: "
            f"expected {EXPECTED_TRACE_LENGTH}, "
            f"found {trace_batch.shape[1]}."
        )

    label_batch = label_batch.reshape(-1)

    if label_batch.shape[0] != trace_batch.shape[0]:
        raise ValueError(
            "The number of labels does not match "
            "the number of traces."
        )

    trace_batch = trace_batch.unsqueeze(dim=1)

    return trace_batch, label_batch


def create_ascad_dataloaders(
    batch_size: int,
    *,
    ascad_path: str | Path = DEFAULT_ASCAD_PATH,
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    normalization_path: str | Path = DEFAULT_NORMALIZATION_PATH,
    num_workers: int = 0,
    random_seed: int = 42,
    pin_memory: bool | None = None
) -> ASCADDataLoaders:
    """Create training, validation and attack DataLoaders."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    ascad_path = Path(ascad_path)
    split_path = Path(split_path)
    normalization_path = Path(normalization_path)

    if not ascad_path.is_file():
        raise FileNotFoundError(
            f"ASCAD database not found: {ascad_path}"
        )

    if not normalization_path.is_file():
        raise FileNotFoundError(
            "Normalization file not found: "
            f"{normalization_path}"
        )

    training_indices, validation_indices = (
        load_train_validation_indices(split_path)
    )

    profiling_dataset = ASCADDataset(
        ascad_path=ascad_path,
        variant="700",
        split="profiling"
    )

    attack_dataset = ASCADDataset(
        ascad_path=ascad_path,
        variant="700",
        split="attack"
    )

    if len(profiling_dataset) != 50_000:
        raise ValueError(
            "Unexpected profiling dataset size: "
            f"expected 50000, found {len(profiling_dataset)}."
        )

    if len(attack_dataset) != EXPECTED_ATTACK_SAMPLES:
        raise ValueError(
            "Unexpected attack dataset size: "
            f"expected {EXPECTED_ATTACK_SAMPLES}, "
            f"found {len(attack_dataset)}."
        )

    training_subset = Subset(
        profiling_dataset,
        training_indices
    )

    validation_subset = Subset(
        profiling_dataset,
        validation_indices
    )

    normalized_training_dataset = NormalizedASCADDataset(
        dataset=cast(
            ASCADDatasetProtocol,
            training_subset
        ),
        normalization_path=normalization_path
    )

    normalized_validation_dataset = NormalizedASCADDataset(
        dataset=cast(
            ASCADDatasetProtocol,
            validation_subset
        ),
        normalization_path=normalization_path
    )

    normalized_attack_dataset = NormalizedASCADDataset(
        dataset=cast(
            ASCADDatasetProtocol,
            attack_dataset
        ),
        normalization_path=normalization_path
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    training_generator = torch.Generator()
    training_generator.manual_seed(random_seed)

    common_arguments = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "persistent_workers": num_workers > 0,
        "collate_fn": collate_ascad_batch
    }

    training_dataloader = DataLoader(
        normalized_training_dataset,
        shuffle=True,
        generator=training_generator,
        **common_arguments
    )

    validation_dataloader = DataLoader(
        normalized_validation_dataset,
        shuffle=False,
        **common_arguments
    )

    attack_dataloader = DataLoader(
        normalized_attack_dataset,
        shuffle=False,
        **common_arguments
    )

    return ASCADDataLoaders(
        training=training_dataloader,
        validation=validation_dataloader,
        attack=attack_dataloader
    )