from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.cnn_mamba_paper import PaperCNNMambaModel
from scripts.ascad_dataset import ASCADDataset
from scripts.check_ascad_label_semantics import AES_SBOX
from scripts.transforms import NormalizeTrace

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ASCAD_PATH = Path("data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5")

NORMALIZATION_PATH = Path(
    "data/processed/ascad_700_normalization.npz"
)

CHECKPOINT_PATH = Path(
    "checkpoints/cnn_mamba_paper_best.pt"
)

OUTPUT_DIR = Path(
    "results/cnn_mamba_paper"
)

ATTACK_GROUP_NAME = "Attack_traces"

TARGET_BYTE = 2

BATCH_SIZE = 256
NUM_WORKERS = 0

NUM_GE_REPETITIONS = 100

RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------

def load_model(
    checkpoint_path: Path
) -> PaperCNNMambaModel:
    """
    Load the trained CNN-Mamba model from a checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = PaperCNNMambaModel(
        num_classes=256
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # The checkpoint itself may already be a state_dict.
            state_dict = checkpoint

    else:
        raise TypeError(
            "Unsupported checkpoint format."
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model

# ------------------------------------------------------------------
# ASCAD metadata
# ------------------------------------------------------------------

def load_attack_metadata(
    ascad_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load plaintext byte 2 and the corresponding real key byte
    from the ASCAD attack metadata.

    Returns:
        plaintext_bytes:
            Array of shape [num_attack_traces].

        key_bytes:
            Array of shape [num_attack_traces].
    """

    with h5py.File(ascad_path, "r") as ascad_file:

        group_object = ascad_file.get(
            ATTACK_GROUP_NAME
        )

        if not isinstance(group_object, h5py.Group):
            raise TypeError(
                f"'{ATTACK_GROUP_NAME}' is not a valid HDF5 group."
            )

        metadata_object = group_object.get("metadata")

        if not isinstance(metadata_object, h5py.Dataset):
            raise TypeError(
                "'metadata' is not a valid HDF5 dataset."
            )

        metadata_records: Any = metadata_object[:]

        plaintexts = np.asarray(
            metadata_records["plaintext"],
            dtype=np.uint8
        )

        keys = np.asarray(
            metadata_records["key"],
            dtype=np.uint8
        )

    plaintext_bytes = plaintexts[:, TARGET_BYTE]
    key_bytes = keys[:, TARGET_BYTE]

    return plaintext_bytes, key_bytes

# ------------------------------------------------------------------
# Model predictions
# ------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on all attack traces.

    Returns:
        logits:
            Shape [num_attack_traces, 256].

        labels:
            Shape [num_attack_traces].
    """

    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()

    for traces, labels in dataloader:

        traces = traces.to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(traces)

        all_logits.append(
            logits.cpu().numpy()
        )

        all_labels.append(
            labels.numpy()
        )

    logits_array = np.concatenate(
        all_logits,
        axis=0
    )

    labels_array = np.concatenate(
        all_labels,
        axis=0
    )

    return logits_array, labels_array

# ------------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------------

def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray
) -> dict[str, float]:
    """
    Compute attack-set classification metrics.
    """

    logits_tensor = torch.from_numpy(
        logits
    ).float()

    labels_tensor = torch.from_numpy(
        labels
    ).long()

    loss = nn.CrossEntropyLoss()(
        logits_tensor,
        labels_tensor
    ).item()

    top1_predictions = logits_tensor.argmax(
        dim=1
    )

    top1_accuracy = (
        top1_predictions == labels_tensor
    ).float().mean().item()

    top5_predictions = torch.topk(
        logits_tensor,
        k=5,
        dim=1
    ).indices

    top5_accuracy = (
        top5_predictions == labels_tensor.unsqueeze(1)
    ).any(dim=1).float().mean().item()

    probabilities = torch.softmax(
        logits_tensor,
        dim=1
    )

    correct_class_probabilities = probabilities[
        torch.arange(labels_tensor.size(0)),
        labels_tensor
    ]

    mean_correct_class_probability = (
        correct_class_probabilities.mean().item()
    )

    max_probabilities = probabilities.max(
        dim=1
    ).values

    mean_confidence = (
        max_probabilities.mean().item()
    )

    per_class_accuracies: list[float] = []

    for class_index in range(256):

        class_mask = labels_tensor == class_index

        if class_mask.any():

            class_accuracy = (
                top1_predictions[class_mask]
                == labels_tensor[class_mask]
            ).float().mean().item()

            per_class_accuracies.append(
                class_accuracy
            )

    macro_accuracy = float(
        np.mean(per_class_accuracies)
    )

    return {
        "cross_entropy": loss,
        "top1_accuracy": top1_accuracy,
        "top5_accuracy": top5_accuracy,
        "macro_accuracy": macro_accuracy,
        "mean_confidence": mean_confidence,
        "mean_correct_class_probability": (
            mean_correct_class_probability
        )
    }

# ------------------------------------------------------------------
# Key-rank computation
# ------------------------------------------------------------------

def prepare_key_log_likelihoods(
    logits: np.ndarray,
    plaintext_bytes: np.ndarray
) -> np.ndarray:
    """
    For every attack trace and every key hypothesis, obtain the
    log-probability assigned to:

        SBOX(plaintext_byte XOR key_guess)

    Returns:
        Array with shape [num_traces, 256].
    """

    logits_tensor = torch.from_numpy(
        logits
    ).float()

    log_probabilities = torch.log_softmax(
        logits_tensor,
        dim=1
    ).double().numpy()

    key_guesses = np.arange(
        256,
        dtype=np.uint8
    )

    hypothetical_classes = AES_SBOX[
        np.bitwise_xor(
            plaintext_bytes[:, np.newaxis],
            key_guesses[np.newaxis, :]
        )
    ]

    trace_indices = np.arange(
        len(plaintext_bytes)
    )[:, np.newaxis]

    key_log_likelihoods = log_probabilities[
        trace_indices,
        hypothetical_classes
    ].astype(np.float64)

    return key_log_likelihoods

def compute_rank_curve(
    key_log_likelihoods: np.ndarray,
    real_key: int,
    trace_order: np.ndarray
) -> np.ndarray:
    """
    Compute the rank of the real key after every additional trace.

    Rank 1 means that the correct key is the best candidate.
    """

    ordered_scores = key_log_likelihoods[
        trace_order
    ]

    cumulative_scores = np.cumsum(
        ordered_scores,
        axis=0,
        dtype=np.float64
    )

    real_key_scores = cumulative_scores[
        :,
        real_key
    ]

    ranks = 1 + np.sum(
        cumulative_scores
        > real_key_scores[:, np.newaxis],
        axis=1
    )

    return ranks.astype(np.int32)

def compute_ge_and_success_rate(
    key_log_likelihoods: np.ndarray,
    real_key: int,
    repetitions: int,
    random_seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute:

    - Guessing Entropy:
        average real-key rank over random attack-trace permutations;

    - Success Rate:
        fraction of experiments where the real key has rank 1;

    - all individual rank curves.
    """

    num_traces = key_log_likelihoods.shape[0]

    rng = np.random.default_rng(
        random_seed
    )

    all_rank_curves = np.empty(
        (repetitions, num_traces),
        dtype=np.int32
    )

    for repetition in range(repetitions):

        permutation = rng.permutation(
            num_traces
        )

        all_rank_curves[repetition] = (
            compute_rank_curve(
                key_log_likelihoods=key_log_likelihoods,
                real_key=real_key,
                trace_order=permutation
            )
        )

    guessing_entropy = all_rank_curves.mean(
        axis=0
    )

    success_rate = (
        all_rank_curves == 1
    ).mean(axis=0)

    return (
        guessing_entropy,
        success_rate,
        all_rank_curves
    )

# ------------------------------------------------------------------
# Summary values
# ------------------------------------------------------------------

def first_trace_count_at_threshold(
    values: np.ndarray,
    threshold: float,
    comparison: str
) -> int | None:
    """
    Return the first trace count where a metric reaches a threshold.
    """

    if comparison == "less_equal":
        indices = np.flatnonzero(
            values <= threshold
        )

    elif comparison == "greater_equal":
        indices = np.flatnonzero(
            values >= threshold
        )

    else:
        raise ValueError(
            f"Unsupported comparison: {comparison}"
        )

    if len(indices) == 0:
        return None

    return int(indices[0] + 1)

def first_stable_rank_one(
    guessing_entropy: np.ndarray
) -> int | None:
    """
    Find the first trace count from which GE remains equal to 1
    until the end of the experiment.
    """

    rank_one = np.isclose(
        guessing_entropy,
        1.0
    )

    stable_from_here = np.logical_and.accumulate(
        rank_one[::-1]
    )[::-1]

    indices = np.flatnonzero(
        stable_from_here
    )

    if len(indices) == 0:
        return None

    return int(indices[0] + 1)

# ------------------------------------------------------------------
# Output files
# ------------------------------------------------------------------

def save_metrics_csv(
    path: Path,
    guessing_entropy: np.ndarray,
    success_rate: np.ndarray
) -> None:

    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "num_traces",
                "guessing_entropy",
                "success_rate"
            ]
        )

        for index, (
            ge_value,
            sr_value,
        ) in enumerate(
            zip(
                guessing_entropy,
                success_rate,
                strict=True
            ),
            start=1,
        ):

            writer.writerow(
                [
                    index,
                    float(ge_value),
                    float(sr_value)
                ]
            )

def save_summary(
    path: Path,
    metrics: dict[str, float],
    real_key: int,
    guessing_entropy: np.ndarray,
    success_rate: np.ndarray
) -> None:

    first_ge_one = first_trace_count_at_threshold(
        guessing_entropy,
        threshold=1.0,
        comparison="less_equal"
    )

    stable_ge_one = first_stable_rank_one(
        guessing_entropy
    )

    first_sr_90 = first_trace_count_at_threshold(
        success_rate,
        threshold=0.90,
        comparison="greater_equal"
    )

    first_sr_95 = first_trace_count_at_threshold(
        success_rate,
        threshold=0.95,
        comparison="greater_equal"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        output_file.write(
            f"Device: {DEVICE}\n"
        )

        output_file.write(
            f"Real key byte: {real_key} "
            f"(0x{real_key:02X})\n"
        )

        output_file.write("\n")

        for metric_name, metric_value in metrics.items():

            output_file.write(
                f"{metric_name}: {metric_value:.8f}\n"
            )

        output_file.write("\n")

        output_file.write(
            f"Final GE: {guessing_entropy[-1]:.4f}\n"
        )

        output_file.write(
            f"Final Success Rate: "
            f"{success_rate[-1]:.4%}\n"
        )

        output_file.write(
            f"First GE <= 1: {first_ge_one}\n"
        )

        output_file.write(
            f"Stable GE = 1: {stable_ge_one}\n"
        )

        output_file.write(
            f"First SR >= 90%: {first_sr_90}\n"
        )

        output_file.write(
            f"First SR >= 95%: {first_sr_95}\n"
        )


def save_plots(
    output_dir: Path,
    guessing_entropy: np.ndarray,
    success_rate: np.ndarray
) -> None:

    trace_counts = np.arange(
        1,
        len(guessing_entropy) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        trace_counts,
        guessing_entropy
    )

    plt.xlabel("Number of attack traces")
    plt.ylabel("Guessing Entropy")
    plt.title("Guessing Entropy vs. Attack Traces")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "guessing_entropy.png",
        dpi=200
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.plot(
        trace_counts,
        success_rate
    )

    plt.xlabel("Number of attack traces")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs. Attack Traces")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_dir / "success_rate.png",
        dpi=200
    )

    plt.close()

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    print(f"Using device: {DEVICE}")
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    trace_transform = NormalizeTrace(
        NORMALIZATION_PATH
    )

    attack_dataset = ASCADDataset(
        ascad_path=ASCAD_PATH,
        variant="700",
        split="attack",
        transform=trace_transform
    )

    attack_loader = DataLoader(
        attack_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    try:

        model = load_model(
            CHECKPOINT_PATH
        )

        plaintext_bytes, key_bytes = (
            load_attack_metadata(
                ASCAD_PATH
            )
        )

        if len(plaintext_bytes) != len(attack_dataset):
            raise RuntimeError(
                "The number of metadata records does not match "
                "the number of attack traces."
            )

        unique_keys = np.unique(
            key_bytes
        )

        if len(unique_keys) != 1:
            raise RuntimeError(
                "The attack set contains more than one key byte."
            )

        real_key = int(
            unique_keys[0]
        )

        print(
            f"Attack traces: {len(attack_dataset)}"
        )

        print(
            f"Target byte:   {TARGET_BYTE}"
        )

        print(
            f"Real key byte: {real_key} "
            f"(0x{real_key:02X})"
        )

        print()

        logits, labels = collect_predictions(
            model=model,
            dataloader=attack_loader
        )

        classification_metrics = (
            compute_classification_metrics(
                logits=logits,
                labels=labels
            )
        )

        key_log_likelihoods = (
            prepare_key_log_likelihoods(
                logits=logits,
                plaintext_bytes=plaintext_bytes
            )
        )

        guessing_entropy, success_rate, rank_curves = (
            compute_ge_and_success_rate(
                key_log_likelihoods=(
                    key_log_likelihoods
                ),
                real_key=real_key,
                repetitions=NUM_GE_REPETITIONS,
                random_seed=RANDOM_SEED
            )
        )

        np.savez_compressed(
            OUTPUT_DIR / "attack_results.npz",
            logits=logits,
            labels=labels,
            plaintext_bytes=plaintext_bytes,
            real_key=np.asarray(real_key),
            guessing_entropy=guessing_entropy,
            success_rate=success_rate,
            rank_curves=rank_curves
        )

        save_metrics_csv(
            path=OUTPUT_DIR / "ge_success_rate.csv",
            guessing_entropy=guessing_entropy,
            success_rate=success_rate
        )

        save_summary(
            path=OUTPUT_DIR / "summary.txt",
            metrics=classification_metrics,
            real_key=real_key,
            guessing_entropy=guessing_entropy,
            success_rate=success_rate
        )

        save_plots(
            output_dir=OUTPUT_DIR,
            guessing_entropy=guessing_entropy,
            success_rate=success_rate
        )

        print("Classification metrics")
        print("-" * 50)

        for metric_name, metric_value in (
            classification_metrics.items()
        ):

            print(
                f"{metric_name:35s}: "
                f"{metric_value:.6f}"
            )

        print()
        print("Side-channel metrics")
        print("-" * 50)

        print(
            f"Final Guessing Entropy: "
            f"{guessing_entropy[-1]:.4f}"
        )

        print(
            f"Final Success Rate: "
            f"{success_rate[-1]:.2%}"
        )

        first_ge_one = first_trace_count_at_threshold(
            guessing_entropy,
            threshold=1.0,
            comparison="less_equal",
        )

        print(f"First GE <= 1: {first_ge_one}")

        stable_ge_one = first_stable_rank_one(
            guessing_entropy
        )

        first_sr_90 = first_trace_count_at_threshold(
            success_rate,
            threshold=0.90,
            comparison="greater_equal",
        )

        print(f"Stable GE = 1: {stable_ge_one}")
        print(f"First SR >= 90%: {first_sr_90}")

        print()
        print(
            f"Results saved in: {OUTPUT_DIR}"
        )

    finally:

        attack_dataset.close()


if __name__ == "__main__":
    main()