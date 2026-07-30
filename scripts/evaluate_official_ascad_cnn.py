from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from models.ascad_cnn_original import ASCADOriginalCNN


UInt8Array = NDArray[np.uint8]
Float64Array = NDArray[np.float64]
Int16Array = NDArray[np.int16]


AES_SBOX: UInt8Array = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5,
        0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
        0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC,
        0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A,
        0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
        0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B,
        0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85,
        0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17,
        0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88,
        0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
        0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9,
        0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6,
        0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
        0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94,
        0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68,
        0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    ],
    dtype=np.uint8
)


class ASCADAttackDataset(Dataset[Tensor]):
    """Load ASCAD attack traces without normalization."""

    def __init__(
        self,
        h5_path: Path,
        max_traces: int | None = None
    ) -> None:
        if not h5_path.exists():
            raise FileNotFoundError(f"ASCAD dataset not found: {h5_path}")

        self.h5_path = h5_path

        with h5py.File(self.h5_path, "r") as h5_file:
            traces_dataset = cast(
                h5py.Dataset,
                h5_file["Attack_traces/traces"]
            )
            total_traces = int(traces_dataset.shape[0])

        self.num_traces = (
            total_traces
            if max_traces is None
            else min(max_traces, total_traces)
        )
        self._h5_file: h5py.File | None = None
        self._traces_dataset: h5py.Dataset | None = None

    def _open_file(self) -> None:
        if self._h5_file is not None:
            return

        self._h5_file = h5py.File(self.h5_path, "r")
        self._traces_dataset = cast(
            h5py.Dataset,
            self._h5_file["Attack_traces/traces"]
        )

    def __len__(self) -> int:
        return self.num_traces

    def __getitem__(self, index: int) -> Tensor:
        self._open_file()

        if self._traces_dataset is None:
            raise RuntimeError("Unable to open the ASCAD attack traces.")

        trace = np.asarray(
            self._traces_dataset[index],
            dtype=np.float32
        )

        # PyTorch Conv1d input shape: (channels, trace_length).
        trace = np.expand_dims(trace, axis=0)
        return torch.from_numpy(trace)

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._traces_dataset = None

    def __del__(self) -> None:
        self.close()


def load_attack_metadata(
    h5_path: Path,
    num_traces: int,
    target_byte: int
) -> tuple[UInt8Array, int]:
    with h5py.File(h5_path, "r") as h5_file:
        metadata_dataset = cast(
            h5py.Dataset,
            h5_file["Attack_traces/metadata"]
        )
        metadata = np.asarray(metadata_dataset[:num_traces])

    field_names = metadata.dtype.names
    if field_names is None:
        raise ValueError("The metadata dataset has no structured fields.")

    if "plaintext" not in field_names:
        raise ValueError("The 'plaintext' field is missing from metadata.")

    if "key" not in field_names:
        raise ValueError("The 'key' field is missing from metadata.")

    plaintexts = cast(
        UInt8Array,
        np.asarray(metadata["plaintext"][:, target_byte], dtype=np.uint8)
    )
    keys = cast(
        UInt8Array,
        np.asarray(metadata["key"][:, target_byte], dtype=np.uint8)
    )

    true_key = int(keys[0])
    if not bool(np.all(keys == true_key)):
        raise ValueError(
            "The attack traces do not use a fixed key for "
            f"target byte {target_byte}."
        )

    return plaintexts, true_key


def load_model(
    checkpoint_path: Path,
    device: torch.device
) -> ASCADOriginalCNN:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = ASCADOriginalCNN().to(device)
    checkpoint: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = cast(
            Mapping[str, Tensor],
            checkpoint["model_state_dict"]
        )
    else:
        state_dict = cast(Mapping[str, Tensor], checkpoint)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_log_probabilities(
    model: nn.Module,
    data_loader: DataLoader[Tensor],
    num_traces: int,
    device: torch.device
) -> Float64Array:
    all_log_probabilities: list[Float64Array] = []

    batch_size = data_loader.batch_size or 1

    with torch.inference_mode():
        for batch_index, traces in enumerate(data_loader, start=1):
            traces = traces.to(
                device=device,
                dtype=torch.float32,
                non_blocking=True
            )

            logits = model(traces)

            log_probabilities = torch.log_softmax(
                logits,
                dim=1
            )

            batch_probabilities = cast(
                Float64Array,
                log_probabilities.cpu().numpy().astype(np.float64)
            )

            all_log_probabilities.append(batch_probabilities)

            if batch_index % 10 == 0 or batch_index == len(data_loader):
                processed = min(
                    batch_index * batch_size,
                    num_traces
                )

                print(
                    f"Inference: {processed}/{num_traces} traces"
                )

    if not all_log_probabilities:
        raise RuntimeError(
            "The DataLoader produced no attack traces."
        )

    return cast(
        Float64Array,
        np.concatenate(
            all_log_probabilities,
            axis=0
        )
    )


def build_key_log_likelihoods(
    log_probabilities: Float64Array,
    plaintexts: UInt8Array
) -> Float64Array:
    """Build log-likelihood values for all 256 key hypotheses."""

    num_traces = log_probabilities.shape[0]
    if plaintexts.shape[0] != num_traces:
        raise ValueError(
            "The number of plaintext values does not match the number "
            "of predictions."
        )

    key_hypotheses = np.arange(256, dtype=np.uint8)
    hypothetical_classes = AES_SBOX[
        np.bitwise_xor(
            plaintexts[:, np.newaxis],
            key_hypotheses[np.newaxis, :]
        )
    ]
    trace_indices = np.arange(num_traces)[:, np.newaxis]

    return cast(
        Float64Array,
        log_probabilities[trace_indices, hypothetical_classes]
    )


def compute_rank_evolution(
    key_log_likelihoods: Float64Array,
    true_key: int,
    num_attacks: int,
    traces_per_attack: int,
    seed: int
) -> tuple[Float64Array, Float64Array, Int16Array]:
    """Compute guessing entropy and success rate across random attacks."""

    total_available_traces = key_log_likelihoods.shape[0]
    if traces_per_attack > total_available_traces:
        raise ValueError(
            f"Requested {traces_per_attack} traces per attack, but only "
            f"{total_available_traces} are available."
        )

    rng = np.random.default_rng(seed)
    all_ranks = np.empty(
        shape=(num_attacks, traces_per_attack),
        dtype=np.int16
    )

    for attack_index in range(num_attacks):
        permutation = rng.permutation(total_available_traces)
        selected_indices = permutation[:traces_per_attack]
        selected_likelihoods = key_log_likelihoods[selected_indices]
        cumulative_scores = np.cumsum(
            selected_likelihoods,
            axis=0,
            dtype=np.float64
        )
        true_key_scores = cumulative_scores[:, true_key]

        # Rank 0 means that the correct key has the highest score.
        ranks = np.sum(
            cumulative_scores > true_key_scores[:, np.newaxis],
            axis=1
        )
        all_ranks[attack_index] = ranks.astype(np.int16)

        print(
            f"Attack {attack_index + 1:3d}/{num_attacks}: "
            f"final rank = {int(ranks[-1])}"
        )

    guessing_entropy = cast(
        Float64Array,
        np.mean(all_ranks, axis=0, dtype=np.float64)
    )
    success_rate = cast(
        Float64Array,
        np.mean(all_ranks == 0, axis=0, dtype=np.float64)
    )

    return guessing_entropy, success_rate, all_ranks


def first_trace_below_rank(
    guessing_entropy: Float64Array,
    maximum_rank: float
) -> int | None:
    indices = np.flatnonzero(guessing_entropy <= maximum_rank)
    return None if indices.size == 0 else int(indices[0] + 1)


def first_trace_with_success_rate(
    success_rate: Float64Array,
    minimum_success_rate: float
) -> int | None:
    indices = np.flatnonzero(success_rate >= minimum_success_rate)
    return None if indices.size == 0 else int(indices[0] + 1)


def save_results(
    output_directory: Path,
    guessing_entropy: Float64Array,
    success_rate: Float64Array,
    all_ranks: Int16Array,
    true_key: int,
    target_byte: int,
    num_attacks: int,
    traces_per_attack: int,
    seed: int
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    results_path = output_directory / "attack_results.npz"

    np.savez_compressed(
        results_path,
        guessing_entropy=guessing_entropy,
        success_rate=success_rate,
        all_ranks=all_ranks,
        true_key=true_key,
        target_byte=target_byte,
        num_attacks=num_attacks,
        traces_per_attack=traces_per_attack,
        seed=seed
    )
    print(f"Numerical results saved to: {results_path}")


def save_guessing_entropy_plot(
    output_directory: Path,
    guessing_entropy: Float64Array
) -> None:
    traces = np.arange(1, len(guessing_entropy) + 1)
    figure = plt.figure(figsize=(10, 6))

    plt.plot(traces, guessing_entropy, label="Guessing Entropy")
    plt.axhline(y=0, linestyle="--", label="Correct-key rank = 0")
    plt.xlabel("Number of attack traces")
    plt.ylabel("Average rank of the correct key")
    plt.title("Official ASCAD CNN - Guessing Entropy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = output_directory / "guessing_entropy.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Guessing entropy plot saved to: {output_path}")


def save_success_rate_plot(
    output_directory: Path,
    success_rate: Float64Array
) -> None:
    traces = np.arange(1, len(success_rate) + 1)
    figure = plt.figure(figsize=(10, 6))

    plt.plot(traces, success_rate, label="Success Rate")
    plt.axhline(y=1.0, linestyle="--", label="SR = 100%")
    plt.xlabel("Number of attack traces")
    plt.ylabel("Success Rate")
    plt.title("Official ASCAD CNN - Success Rate")
    plt.ylim(-0.02, 1.02)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = output_directory / "success_rate.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Success rate plot saved to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the official ASCAD CNN converted to PyTorch on the "
            "ASCAD attack traces."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
        ),
        help="Path to the ASCAD HDF5 dataset."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/ascad_cnn_official_converted.pt"),
        help="PyTorch checkpoint converted from the official Keras model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/official_ascad_cnn"),
        help="Directory used to save results and plots."
    )
    parser.add_argument(
        "--target-byte",
        type=int,
        default=2,
        help="AES key-byte index targeted by the attack."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size used during inference."
    )
    parser.add_argument(
        "--num-attacks",
        type=int,
        default=100,
        help="Number of random attacks used to compute GE and SR."
    )
    parser.add_argument(
        "--traces-per-attack",
        type=int,
        default=2000,
        help="Number of traces used in each attack."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for trace permutations."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Number of DataLoader worker processes. Keep this at 0 for HDF5 "
            "files, especially when running under WSL."
        )
    )
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not 0 <= args.target_byte <= 15:
        raise ValueError("--target-byte must be between 0 and 15.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero.")
    if args.num_attacks <= 0:
        raise ValueError("--num-attacks must be greater than zero.")
    if args.traces_per_attack <= 0:
        raise ValueError("--traces-per-attack must be greater than zero.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("OFFICIAL ASCAD CNN ATTACK EVALUATION")
    print("=" * 70)
    print(f"Dataset:             {args.dataset}")
    print(f"Checkpoint:          {args.checkpoint}")
    print(f"Device:              {device}")
    print(f"Target byte:         {args.target_byte}")
    print(f"Number of attacks:   {args.num_attacks}")
    print(f"Traces per attack:   {args.traces_per_attack}")
    print(f"Batch size:          {args.batch_size}")
    print("Preprocessing:       float32 conversion, no normalization")
    print("=" * 70)

    start_time = time.perf_counter()
    dataset = ASCADAttackDataset(h5_path=args.dataset)

    try:
        print(f"Available attack traces: {len(dataset)}")
        if args.traces_per_attack > len(dataset):
            raise ValueError(
                f"The dataset contains {len(dataset)} attack traces, but "
                f"{args.traces_per_attack} were requested."
            )

        data_loader = DataLoader[Tensor](
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False
        )
        plaintexts, true_key = load_attack_metadata(
            h5_path=args.dataset,
            num_traces=len(dataset),
            target_byte=args.target_byte
        )
        print(
            f"Correct key, byte {args.target_byte}: "
            f"0x{true_key:02X} ({true_key})"
        )

        model = load_model(
            checkpoint_path=args.checkpoint,
            device=device
        )
        print("Model loaded successfully.")

        inference_start = time.perf_counter()
        log_probabilities = predict_log_probabilities(
            model=model,
            data_loader=data_loader,
            num_traces=len(dataset),
            device=device
        )
        inference_duration = time.perf_counter() - inference_start
        print(f"Inference completed in {inference_duration:.2f} seconds.")
        print(f"Log-probability shape: {log_probabilities.shape}")

        key_log_likelihoods = build_key_log_likelihoods(
            log_probabilities=log_probabilities,
            plaintexts=plaintexts
        )

        attack_start = time.perf_counter()
        guessing_entropy, success_rate, all_ranks = compute_rank_evolution(
            key_log_likelihoods=key_log_likelihoods,
            true_key=true_key,
            num_attacks=args.num_attacks,
            traces_per_attack=args.traces_per_attack,
            seed=args.seed
        )
        attack_duration = time.perf_counter() - attack_start

        ge_rank_zero_trace = first_trace_below_rank(
            guessing_entropy,
            maximum_rank=0.0
        )
        ge_rank_one_trace = first_trace_below_rank(
            guessing_entropy,
            maximum_rank=1.0
        )
        sr_90_trace = first_trace_with_success_rate(
            success_rate,
            minimum_success_rate=0.90
        )
        sr_100_trace = first_trace_with_success_rate(
            success_rate,
            minimum_success_rate=1.00
        )

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"Final GE:             {guessing_entropy[-1]:.4f}")
        print(f"Final SR:             {success_rate[-1] * 100:.2f}%")
        print(
            "Final rank min/max:   "
            f"{int(all_ranks[:, -1].min())}/"
            f"{int(all_ranks[:, -1].max())}"
        )
        print(
            "GE <= 1:              "
            f"{ge_rank_one_trace} traces"
            if ge_rank_one_trace is not None
            else "GE <= 1:              never reached"
        )
        print(
            "GE = 0:               "
            f"{ge_rank_zero_trace} traces"
            if ge_rank_zero_trace is not None
            else "GE = 0:               never reached"
        )
        print(
            "SR >= 90%:            "
            f"{sr_90_trace} traces"
            if sr_90_trace is not None
            else "SR >= 90%:            never reached"
        )
        print(
            "SR = 100%:            "
            f"{sr_100_trace} traces"
            if sr_100_trace is not None
            else "SR = 100%:            never reached"
        )
        print(f"Attack duration:      {attack_duration:.2f} seconds")

        save_results(
            output_directory=args.output_dir,
            guessing_entropy=guessing_entropy,
            success_rate=success_rate,
            all_ranks=all_ranks,
            true_key=true_key,
            target_byte=args.target_byte,
            num_attacks=args.num_attacks,
            traces_per_attack=args.traces_per_attack,
            seed=args.seed
        )
        save_guessing_entropy_plot(
            output_directory=args.output_dir,
            guessing_entropy=guessing_entropy
        )
        save_success_rate_plot(
            output_directory=args.output_dir,
            success_rate=success_rate
        )
    finally:
        dataset.close()

    total_duration = time.perf_counter() - start_time
    print(f"Total duration:       {total_duration:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    main()
