from pathlib import Path

import scripts.ascad_dataset


ASCAD_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

RAW_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_databases/"
    "ATMega8515_raw_traces.h5"
)


def inspect_dataset(
    variant: str,
    split: str,
) -> None:
    dataset = scripts.ascad_dataset.ASCADDataset(
        ascad_path=ASCAD_PATH,
        raw_path=RAW_PATH,
        variant=variant,  # type: ignore[arg-type]
        split=split,  # type: ignore[arg-type]
    )

    trace, label = dataset[0]

    print("=" * 80)
    print(f"Variant: {variant}")
    print(f"Split: {split}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Trace shape: {tuple(trace.shape)}")
    print(f"Trace dtype: {trace.dtype}")
    print(f"Label shape: {tuple(label.shape)}")
    print(f"Label dtype: {label.dtype}")
    print(f"Label value: {label.item()}")

    dataset.close()


def main() -> None:
    inspect_dataset(
        variant="700",
        split="profiling",
    )

    inspect_dataset(
        variant="700",
        split="attack",
    )

    inspect_dataset(
        variant="raw",
        split="profiling",
    )

    inspect_dataset(
        variant="raw",
        split="attack",
    )


if __name__ == "__main__":
    main()