from importlib import metadata
from pathlib import Path

import h5py
import numpy as np


FILE_PATH = Path(
    "data/raw/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5"
)


def get_group(
    hdf5_file: h5py.File,
    group_name: str,
) -> h5py.Group:
    """
    Gets an HDF5 group and verifies that it is not a dataset.
    """
    obj = hdf5_file.get(group_name)

    if obj is None:
        raise KeyError(f"Group not found: {group_name}")

    if not isinstance(obj, h5py.Group):
        raise TypeError(
            f"{group_name} should be a group, "
            f"but it is of type {type(obj).__name__}"
        )

    return obj


def get_dataset(
    group: h5py.Group,
    dataset_name: str,
) -> h5py.Dataset:
    """
    Gets an HDF5 dataset and verifies that it is not a group.
    """
    obj = group.get(dataset_name)

    if obj is None:
        raise KeyError(
            f"Dataset not found: {group.name}/{dataset_name}"
        )

    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"{group.name}/{dataset_name} should be a dataset, "
            f"but it is of type {type(obj).__name__}"
        )

    return obj


def inspect_split(
    hdf5_file: h5py.File,
    split_name: str,
) -> None:
    group = get_group(hdf5_file, split_name)

    labels_dataset = get_dataset(group, "labels")
    traces_dataset = get_dataset(group, "traces")
    metadata_dataset = get_dataset(group, "metadata")

    labels = np.asarray(labels_dataset[:])

    print("=" * 80)
    print(split_name)
    print("=" * 80)

    print(f"Number of traces: {traces_dataset.shape[0]}")
    print(f"Trace length: {traces_dataset.shape[1]}")
    print(f"Trace dtype: {traces_dataset.dtype}")

    print(f"Minimum label: {labels.min()}")
    print(f"Maximum label: {labels.max()}")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(f"Unique labels: {len(unique_labels)}")
    print(f"Minimum class count: {counts.min()}")
    print(f"Maximum class count: {counts.max()}")
    print(f"Mean class count: {counts.mean():.2f}")

    first_metadata = metadata_dataset[0]

    metadata = metadata_dataset[:]

    keys = metadata["key"]
    desync_values = metadata["desync"].reshape(-1)

    unique_keys = np.unique(keys, axis=0)
    unique_desync, desync_counts = np.unique(
        desync_values,
        return_counts=True,
    )

    print(f"\nUnique AES keys: {len(unique_keys)}")

    if len(unique_keys) <= 5:
        for index, key in enumerate(unique_keys):
            print(f"Key {index}: {key}")

    print(f"Unique desync values: {unique_desync}")

    if len(unique_desync) <= 10:
        print("Desync distribution:")

        for value, count in zip(unique_desync, desync_counts):
            print(f"  {value}: {count}")

    print("\nFirst metadata record:")
    print(f"Plaintext: {first_metadata['plaintext']}")
    print(f"Ciphertext: {first_metadata['ciphertext']}")
    print(f"Key: {first_metadata['key']}")
    print(f"Masks: {first_metadata['masks']}")
    print(f"Desync: {first_metadata['desync']}")

    print()


def main() -> None:
    if not FILE_PATH.exists():
        raise FileNotFoundError(
            f"File not found: {FILE_PATH}"
        )

    if not FILE_PATH.is_file():
        raise ValueError(
            f"Path does not indicate a file: {FILE_PATH}"
        )

    with h5py.File(FILE_PATH, "r") as hdf5_file:
        inspect_split(
            hdf5_file,
            "Profiling_traces",
        )

        inspect_split(
            hdf5_file,
            "Attack_traces",
        )


if __name__ == "__main__":
    main()