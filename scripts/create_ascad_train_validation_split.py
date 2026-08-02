from pathlib import Path

import numpy as np
from numpy.typing import NDArray


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/ascad_train_validation_split.npz"
)

NUM_PROFILING_TRACES = 50_000
NUM_VALIDATION_TRACES = 5_000
RANDOM_SEED = 42


def create_split() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Create deterministic training and validation indices."""
    random_generator = np.random.default_rng(RANDOM_SEED)

    shuffled_indices = random_generator.permutation(
        NUM_PROFILING_TRACES
    ).astype(np.int64)

    validation_indices = np.sort(
        shuffled_indices[:NUM_VALIDATION_TRACES]
    )

    training_indices = np.sort(
        shuffled_indices[NUM_VALIDATION_TRACES:]
    )

    return training_indices, validation_indices


def validate_split(
    training_indices: NDArray[np.int64],
    validation_indices: NDArray[np.int64]
) -> None:
    """Validate the generated dataset split."""
    if training_indices.size != 45_000:
        raise ValueError(
            "Unexpected number of training indices: "
            f"expected 45000, found {training_indices.size}."
        )

    if validation_indices.size != 5_000:
        raise ValueError(
            "Unexpected number of validation indices: "
            f"expected 5000, found {validation_indices.size}."
        )

    all_indices = np.concatenate(
        (training_indices, validation_indices)
    )

    unique_indices = np.unique(all_indices)

    if unique_indices.size != NUM_PROFILING_TRACES:
        raise ValueError(
            "Training and validation indices overlap or omit traces."
        )

    if int(unique_indices.min()) != 0:
        raise ValueError(
            "The minimum profiling index is not zero."
        )

    if int(unique_indices.max()) != NUM_PROFILING_TRACES - 1:
        raise ValueError(
            "The maximum profiling index is invalid."
        )


def main() -> None:
    """Create and save the deterministic ASCAD dataset split."""
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Split file already exists: {OUTPUT_PATH}"
        )

    training_indices, validation_indices = create_split()

    validate_split(
        training_indices=training_indices,
        validation_indices=validation_indices
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,
        training_indices=training_indices,
        validation_indices=validation_indices,
        random_seed=np.asarray(RANDOM_SEED, dtype=np.int64)
    )

    print(f"Split saved to: {OUTPUT_PATH}")
    print(f"Training traces:   {training_indices.size}")
    print(f"Validation traces: {validation_indices.size}")
    print(f"Random seed:       {RANDOM_SEED}")
    print("Training and validation overlap: 0")
    print("Result: OK")


if __name__ == "__main__":
    main()