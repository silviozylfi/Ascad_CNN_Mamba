from pathlib import Path

import h5py
import numpy as np


ASCAD_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)

RAW_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_databases/"
    "ATMega8515_raw_traces.h5"
)

FIELDS = (
    "plaintext",
    "ciphertext",
    "key",
    "masks",
)


def get_group(
    parent: h5py.File | h5py.Group,
    name: str,
) -> h5py.Group:
    obj = parent.get(name)

    if obj is None:
        raise KeyError(f"Group not found: {parent.name}/{name}")

    if not isinstance(obj, h5py.Group):
        raise TypeError(
            f"{parent.name}/{name} should be a group, "
            f"but it is of type {type(obj).__name__}"
        )

    return obj


def get_dataset(
    parent: h5py.File | h5py.Group,
    name: str,
) -> h5py.Dataset:
    obj = parent.get(name)

    if obj is None:
        raise KeyError(f"Dataset not found: {parent.name}/{name}")

    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"{parent.name}/{name} should be a dataset, "
            f"but it is of type {type(obj).__name__}"
        )

    return obj


def compare_records(
    raw_record: np.void,
    processed_record: np.void,
    description: str,
) -> bool:
    print("=" * 80)
    print(description)
    print("=" * 80)

    all_equal = True

    for field in FIELDS:
        raw_value = np.asarray(raw_record[field])
        processed_value = np.asarray(processed_record[field])

        equal = np.array_equal(
            raw_value,
            processed_value,
        )

        print(f"{field}: {'MATCH' if equal else 'DIFFERENT'}")

        if not equal:
            all_equal = False
            print(f"  raw:       {raw_value}")
            print(f"  processed: {processed_value}")

    print()

    return all_equal


def main() -> None:
    if not ASCAD_PATH.is_file():
        raise FileNotFoundError(
            f"File not found: {ASCAD_PATH}"
        )

    if not RAW_PATH.is_file():
        raise FileNotFoundError(
            f"File not found: {RAW_PATH}"
        )

    with (
        h5py.File(ASCAD_PATH, "r") as ascad_file,
        h5py.File(RAW_PATH, "r") as raw_file,
    ):
        raw_metadata_dataset = get_dataset(
            raw_file,
            "metadata",
        )

        profiling_group = get_group(
            ascad_file,
            "Profiling_traces",
        )

        attack_group = get_group(
            ascad_file,
            "Attack_traces",
        )

        profiling_metadata_dataset = get_dataset(
            profiling_group,
            "metadata",
        )

        attack_metadata_dataset = get_dataset(
            attack_group,
            "metadata",
        )

        print("Loading metadata into memory...")

        raw_metadata = raw_metadata_dataset[:]
        profiling_metadata = profiling_metadata_dataset[:]
        attack_metadata = attack_metadata_dataset[:]

        raw_profiling_metadata = raw_metadata[:50000]
        raw_attack_metadata = raw_metadata[50000:60000]

        print("=" * 80)
        print("Checking dimensions")
        print("=" * 80)

        print(
            f"Raw profiling records: "
            f"{len(raw_profiling_metadata)}"
        )
        print(
            f"ASCAD profiling records: "
            f"{len(profiling_metadata)}"
        )
        print(
            f"Raw attack records: "
            f"{len(raw_attack_metadata)}"
        )
        print(
            f"ASCAD attack records: "
            f"{len(attack_metadata)}"
        )

        if len(raw_profiling_metadata) != len(profiling_metadata):
            raise ValueError(
                "The number of profiling records does not match."
            )

        if len(raw_attack_metadata) != len(attack_metadata):
            raise ValueError(
                "The number of attack records does not match."
            )

        print()
        print("=" * 80)
        print("Complete metadata comparison")
        print("=" * 80)

        profiling_all_match = True
        attack_all_match = True

        for field in FIELDS:
            profiling_field_match = np.array_equal(
                raw_profiling_metadata[field],
                profiling_metadata[field],
            )

            attack_field_match = np.array_equal(
                raw_attack_metadata[field],
                attack_metadata[field],
            )

            print(
                f"{field:<12} | "
                f"profiling: "
                f"{'MATCH' if profiling_field_match else 'DIFFERENT'} | "
                f"attack: "
                f"{'MATCH' if attack_field_match else 'DIFFERENT'}"
            )

            if not profiling_field_match:
                profiling_all_match = False

                different_indices = np.flatnonzero(
                    np.any(
                        raw_profiling_metadata[field]
                        != profiling_metadata[field],
                        axis=1,
                    )
                )

                print(
                    "  First profiling differences at indices:",
                    different_indices[:10],
                )

            if not attack_field_match:
                attack_all_match = False

                different_indices = np.flatnonzero(
                    np.any(
                        raw_attack_metadata[field]
                        != attack_metadata[field],
                        axis=1,
                    )
                )

                print(
                    "  First attack differences at indices:",
                    different_indices[:10],
                )

        print()
        print("=" * 80)
        print("Final result")
        print("=" * 80)
        print(
            f"All 50,000 profiling metadata records match: "
            f"{profiling_all_match}"
        )
        print(
            f"All 10,000 attack metadata records match: "
            f"{attack_all_match}"
        )

        complete_match = (
            profiling_all_match
            and attack_all_match
        )

        print(
            f"All 60,000 records match: "
            f"{complete_match}"
        )


if __name__ == "__main__":
    main()
    