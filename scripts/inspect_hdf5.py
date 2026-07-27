from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py 
import numpy as np

def format_size(size_bytes: int) -> str:
    """Format a size in bytes into a human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"

def print_attributes(obj: h5py.Group | h5py.Dataset, indent: str) -> None:
    """Print the attributes of an HDF5 object."""
    if not obj.attrs:
        print(f"{indent}No attributes")
        return

    print(f"{indent}Attributes:")
    for key, value in obj.attrs.items():
        if isinstance(value, np.ndarray):
            value_str = f"array(shape={value.shape}, dtype={value.dtype})"
        else:
            value_str = str(value)
        print(f"{indent}  {key}: {value_str}")

def estimate_dataset_size(dataset: h5py.Dataset) -> int:
    """Estimate the size of a dataset in bytes."""
    return int(dataset.size * dataset.dtype.itemsize)

def print_dataset_preview(
        dataset: h5py.Dataset,
        indent: str,
        preview_items: int = 5
) -> None:
    """Print a preview of the dataset's contents."""
    if dataset.size == 0:
        print(f"{indent}Dataset is empty")
        return

    try:
        if dataset.ndim == 0:
            preview: Any = dataset[()]
        elif dataset.ndim == 1:
            preview = dataset[:preview_items]
        else:
            preview = dataset[0]

            flattened = np.asarray(preview).reshape(-1)
            preview = flattened[:preview_items]

        print(f"{indent}Preview (first {preview_items} items): {preview}")
        
    except (OSError, TypeError, ValueError) as e:
        print(f"{indent}Error reading dataset: {e}")

def inspect_object(
        name: str,
        obj: h5py.Group | h5py.Dataset,
        show_preview: bool,
        preview_items: int
) -> None:
    depth = name.count('/')
    indent = '  ' * depth
    display_name = name.split('/')[-1] if name else 'Root'

    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {display_name}")
        print_attributes(obj, indent + '  ')
        return

    if isinstance(obj, h5py.Dataset):
        estimated_size = estimate_dataset_size(obj)

        print(f"{indent}[DATASET] {display_name}")
        print(f"{indent}  Path: /{name}")
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dimensions: {obj.ndim}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Elements: {obj.size:,}")
        print(
            f"{indent}  Estimated uncompressed size: "
            f"{format_size(estimated_size)}"
        )
        print(f"{indent}  Compression: {obj.compression}")
        print(f"{indent}  Chunks: {obj.chunks}")

        print_attributes(obj, indent + "  ")

        if show_preview:
            print_dataset_preview(
                dataset=obj,
                indent=indent + "  ",
                preview_items=preview_items,
            )  

def inspect_hdf5(
    file_path: Path,
    show_preview: bool,
    preview_items: int,
) -> None:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    print("=" * 80)
    print(f"File: {file_path.resolve()}")
    print(f"File size: {format_size(file_path.stat().st_size)}")
    print("=" * 80)

    try:
        with h5py.File(file_path, "r") as hdf5_file:
            print("\nRoot keys:")
            for key in hdf5_file.keys():
                print(f"  - {key}")

            print_attributes(hdf5_file, "  ")

            print("\nComplete structure:\n")

            hdf5_file.visititems(
                lambda name, obj: inspect_object(
                    name=name,
                    obj=obj,
                    show_preview=show_preview,
                    preview_items=preview_items,
                )
            )

    except OSError as error:
        raise RuntimeError(
            f"Impossible to open the HDF5 file: {error}"
        ) from error    

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the structure of an HDF5 file."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to the HDF5 file to inspect.",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a brief preview of the datasets.",
    )

    parser.add_argument(
        "--preview-items",
        type=int,
        default=10,
        help="Maximum number of values to display in the previews.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.preview_items <= 0:
        raise ValueError("--preview-items must be a positive integer.")

    inspect_hdf5(
        file_path=args.file,
        show_preview=args.preview,
        preview_items=args.preview_items,
    )


if __name__ == "__main__":
    main()