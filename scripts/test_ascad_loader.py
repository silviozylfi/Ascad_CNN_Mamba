from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scripts.ascad_dataset import ASCADDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASCAD_700_PATH = PROJECT_ROOT / "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
ASCAD_RAW_PATH = PROJECT_ROOT / "data/raw/ascad/ASCAD_data/ASCAD_databases/ATMega8515_raw_traces.h5"


def test_dataset(
    name: str,
    dataset: ASCADDataset,
    expected_trace_length: int,
    batch_size: int = 32,
) -> None:
    """Load and inspect one batch from an ASCAD dataset."""

    print("=" * 70)
    print(f"Dataset: {name}")
    print("=" * 70)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    traces, labels = next(iter(dataloader))

    print(f"Dataset length: {len(dataset)}")
    print(f"Traces shape:   {traces.shape}")
    print(f"Labels shape:   {labels.shape}")
    print(f"Traces dtype:   {traces.dtype}")
    print(f"Labels dtype:   {labels.dtype}")
    print(f"Minimum label:  {labels.min().item()}")
    print(f"Maximum label:  {labels.max().item()}")
    print(f"Trace minimum:  {traces.min().item():.4f}")
    print(f"Trace maximum:  {traces.max().item():.4f}")
    print(f"Trace mean:     {traces.mean().item():.4f}")
    print(f"Trace std:      {traces.std().item():.4f}")

    expected_shape = (batch_size, expected_trace_length)

    if traces.shape != expected_shape:
        raise ValueError(
            f"Unexpected traces shape for {name}: "
            f"expected {expected_shape}, found {tuple(traces.shape)}."
        )

    if labels.shape != (batch_size,):
        raise ValueError(
            f"Unexpected labels shape for {name}: "
            f"expected {(batch_size,)}, found {tuple(labels.shape)}."
        )

    if traces.dtype != torch.float32:
        raise TypeError(
            f"Unexpected trace dtype for {name}: "
            f"expected torch.float32, found {traces.dtype}."
        )

    if labels.dtype != torch.int64:
        raise TypeError(
            f"Unexpected label dtype for {name}: "
            f"expected torch.int64, found {labels.dtype}."
        )

    if labels.min().item() < 0 or labels.max().item() > 255:
        raise ValueError(
            f"Labels outside the expected range [0, 255] in {name}."
        )

    if not torch.isfinite(traces).all():
        raise ValueError(
            f"NaN or infinite values found in traces for {name}."
        )

    print("Result: OK")
    print()


def main() -> None:
    datasets = [
        (
            "ASCAD-700 profiling",
            ASCADDataset(
                ascad_path=ASCAD_700_PATH,
                variant="700",
                split="profiling",
            ),
            700,
        ),
        (
            "ASCAD-700 attack",
            ASCADDataset(
                ascad_path=ASCAD_700_PATH,
                variant="700",
                split="attack",
            ),
            700,
        ),
        (
            "ASCAD-100000 profiling",
            ASCADDataset(
                ascad_path=ASCAD_700_PATH,
                raw_path=ASCAD_RAW_PATH,
                variant="raw",
                split="profiling",
            ),
            100_000,
        ),
        (
            "ASCAD-100000 attack",
            ASCADDataset(
                ascad_path=ASCAD_700_PATH,
                raw_path=ASCAD_RAW_PATH,
                variant="raw",
                split="attack",
            ),
            100_000,
        ),
    ]

    try:
        for name, dataset, expected_trace_length in datasets:
            test_dataset(
                name=name,
                dataset=dataset,
                expected_trace_length=expected_trace_length,
            )
    finally:
        for _, dataset, _ in datasets:
            dataset.close()


if __name__ == "__main__":
    main()