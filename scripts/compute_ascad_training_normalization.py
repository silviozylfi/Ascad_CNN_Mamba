from pathlib import Path
from typing import cast

import h5py
import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASCAD_PATH = (
    PROJECT_ROOT
    / "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data/processed/ascad_train_validation_split.npz"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/ascad_training_normalization.npz"
)

TRACE_LENGTH = 700
BATCH_SIZE = 1_000
MINIMUM_STANDARD_DEVIATION = 1e-8


def load_training_indices() -> NDArray[np.int64]:
    """Load the deterministic ASCAD training indices."""
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset split file not found: {SPLIT_PATH}"
        )

    with np.load(SPLIT_PATH) as split_file:
        training_indices = np.asarray(
            split_file["training_indices"],
            dtype=np.int64
        )

    if training_indices.ndim != 1:
        raise ValueError(
            "Training indices must be a one-dimensional array."
        )

    if training_indices.size != 45_000:
        raise ValueError(
            "Unexpected number of training indices: "
            f"expected 45000, found {training_indices.size}."
        )

    return training_indices


def compute_normalization_statistics(
    training_indices: NDArray[np.int64]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Compute feature-wise mean and standard deviation."""
    if not ASCAD_PATH.is_file():
        raise FileNotFoundError(
            f"ASCAD database not found: {ASCAD_PATH}"
        )

    sum_values = np.zeros(TRACE_LENGTH, dtype=np.float64)
    sum_squared_values = np.zeros(TRACE_LENGTH, dtype=np.float64)

    with h5py.File(ASCAD_PATH, "r") as h5_file:
        traces = cast(
            h5py.Dataset,
            h5_file["Profiling_traces/traces"]
        )

        if traces.shape != (50_000, TRACE_LENGTH):
            raise ValueError(
                "Unexpected profiling trace shape: "
                f"expected {(50_000, TRACE_LENGTH)}, "
                f"found {traces.shape}."
            )

        total_batches = (
            training_indices.size + BATCH_SIZE - 1
        ) // BATCH_SIZE

        for batch_number, start in enumerate(
            range(0, training_indices.size, BATCH_SIZE),
            start=1
        ):
            end = min(
                start + BATCH_SIZE,
                training_indices.size
            )

            batch_indices = training_indices[start:end]

            batch = np.asarray(
                traces[batch_indices],
                dtype=np.float64
            )

            sum_values += batch.sum(axis=0)
            sum_squared_values += np.square(batch).sum(axis=0)

            print(
                f"Batch {batch_number:02d}/{total_batches:02d} "
                f"processed."
            )

    sample_count = float(training_indices.size)

    mean = sum_values / sample_count

    variance = (
        sum_squared_values / sample_count
        - np.square(mean)
    )

    variance = np.maximum(
        variance,
        0.0
    )

    standard_deviation = np.sqrt(variance)

    standard_deviation = np.maximum(
        standard_deviation,
        MINIMUM_STANDARD_DEVIATION
    )

    return (
        mean.astype(np.float32),
        standard_deviation.astype(np.float32)
    )


def validate_statistics(
    mean: NDArray[np.float32],
    standard_deviation: NDArray[np.float32]
) -> None:
    """Validate the computed normalization statistics."""
    expected_shape = (TRACE_LENGTH,)

    if mean.shape != expected_shape:
        raise ValueError(
            "Unexpected mean shape: "
            f"expected {expected_shape}, found {mean.shape}."
        )

    if standard_deviation.shape != expected_shape:
        raise ValueError(
            "Unexpected standard deviation shape: "
            f"expected {expected_shape}, "
            f"found {standard_deviation.shape}."
        )

    if not np.isfinite(mean).all():
        raise ValueError(
            "The mean contains NaN or infinite values."
        )

    if not np.isfinite(standard_deviation).all():
        raise ValueError(
            "The standard deviation contains NaN or infinite values."
        )

    if np.any(standard_deviation <= 0.0):
        raise ValueError(
            "The standard deviation contains non-positive values."
        )


def main() -> None:
    """Compute and save ASCAD training normalization statistics."""
    training_indices = load_training_indices()

    mean, standard_deviation = compute_normalization_statistics(
        training_indices
    )

    validate_statistics(
        mean=mean,
        standard_deviation=standard_deviation
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        mean=mean,
        standard_deviation=standard_deviation,
        training_sample_count=np.asarray(
            training_indices.size,
            dtype=np.int64
        )
    )

    print()
    print(f"Normalization saved to: {OUTPUT_PATH}")
    print(f"Mean shape:               {mean.shape}")
    print(
        "Standard deviation shape: "
        f"{standard_deviation.shape}"
    )
    print(f"Mean minimum:             {mean.min():.6f}")
    print(f"Mean maximum:             {mean.max():.6f}")
    print(
        "Standard deviation minimum: "
        f"{standard_deviation.min():.6f}"
    )
    print(
        "Standard deviation maximum: "
        f"{standard_deviation.max():.6f}"
    )
    print("Result: OK")


if __name__ == "__main__":
    main()