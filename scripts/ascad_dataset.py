from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

DatasetVariant = Literal["700", "raw"]
DatasetSplit = Literal["profiling", "attack"]

class ASCADDataset(Dataset[tuple[Tensor, Tensor]]):
    """PyTorch dataset for the ASCAD fixed-key databases."""

    PROFILING_SIZE = 50_000
    ATTACK_SIZE = 10_000
    RAW_ATTACK_OFFSET = 50_000

    def __init__(
            self,
            ascad_path: str | Path,
            raw_path: str | Path | None = None,
            variant: DatasetVariant = "700",
            split: DatasetSplit = "profiling"
    ) -> None:
        self.ascad_path = Path(ascad_path)
        self.raw_path = Path(raw_path) if raw_path is not None else None
        self.variant = variant
        self.split = split

        self._trace_file: h5py.File | None = None
        self._label_file: h5py.File | None = None
        self._traces: h5py.Dataset | None = None
        self._labels: h5py.Dataset | None = None

        self._validate_arguments()

    def _validate_arguments(self) -> None:
        """Validate paths and dataset options."""

        if self.variant not in ("700", "raw"):
            raise ValueError(
                f"Invalid variant: {self.variant}. "
                f"Expected '700' or 'raw'."
            )

        if self.split not in ("profiling", "attack"):
            raise ValueError(
                f"Invalid split: {self.split}. "
                f"Expected 'profiling' or 'attack'."
            )

        if not self.ascad_path.is_file():
            raise FileNotFoundError(
                f"ASCAD database not found: {self.ascad_path}"
            )

    def __len__(self) -> int:
        if self.split == "profiling":
            return self.PROFILING_SIZE

        return self.ATTACK_SIZE

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        self._ensure_open()

        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of bounds for dataset of size {len(self)}"
            )

        traces = self._require_dataset(
            self._traces,
            "traces"
        )

        labels = self._require_dataset(
            self._labels,
            "labels"
        )

        trace_index = self._get_trace_index(index)

        trace_array = np.asarray(
            traces[trace_index],
            dtype=np.float32
        )

        label_value = int(labels[index])

        trace_tensor = torch.from_numpy(trace_array)
        label_tensor = torch.tensor(label_value, dtype=torch.long)

        return trace_tensor, label_tensor

    def _ensure_open(self) -> None:
        """Open the HDF5 files lazily in the current process"""

        if self._traces is not None and self._labels is not None:
            return

        self.close()

        if self.variant == "700":
            self._trace_file = h5py.File(self.ascad_path, "r")
            self._label_file = self._trace_file

            group_name = self._get_group_name()
            group = self._get_group(self._trace_file, group_name)

            self._traces = self._get_dataset(group, "traces")
            self._labels = self._get_dataset(group, "labels")

        else:
            raw_path = self._require_new_path()

            self._trace_file = h5py.File(raw_path, "r")
            self._label_file = h5py.File(self.ascad_path, "r")
            self._traces = self._get_dataset(self._trace_file, "traces")

            group_name = self._get_group_name()
            label_group = self._get_group(self._label_file, group_name)
            self._labels = self._get_dataset(label_group, "labels")

    def _get_trace_index(self, local_index: int) -> int:
        """Convert a split-local index into an HDF5 trace index"""

        if self.variant == "raw" and self.split == "attack":
            return self.RAW_ATTACK_OFFSET + local_index

        return local_index

    def _get_group_name(self) -> str:
        """Get the HDF5 group name for the current split"""

        if self.split == "profiling":
            return "Profiling_traces"

        return "Attack_traces"

    def _require_new_path(self) -> Path:
        if self.raw_path is None:
            raise ValueError(
                "raw_path must be provided for the 'raw' variant"
            )

        if not self.raw_path.is_file():
            raise FileNotFoundError(
                f"Raw ASCAD database not found: {self.raw_path}"
            )

        return self.raw_path

    @staticmethod
    def _require_dataset(
        dataset: h5py.Dataset | None,
        dataset_name: str
    ) -> h5py.Dataset:
        if dataset is None:
            raise RuntimeError(
                f"Dataset '{dataset_name}' is not initialized. "
                f"Call 'ensure_open()' first."
            )

        return dataset

    @staticmethod
    def _get_group(
        parent: h5py.File | h5py.Group,
        name: str
    ) -> h5py.Group:
        obj = parent.get(name)

        if obj is None:
            raise KeyError(f"Group not found: {name}")

        if not isinstance(obj, h5py.Group):
            raise TypeError(
                f"{name} should be a group, "
                f"but it is of type {type(obj).__name__}"
            )

        return obj

    @staticmethod
    def _get_dataset(
        parent: h5py.File | h5py.Group,
        name: str
    ) -> h5py.Dataset:
        obj = parent.get(name)

        if obj is None:
            raise KeyError(f"Dataset not found: {name}")

        if not isinstance(obj, h5py.Dataset):
            raise TypeError(
                f"{name} should be a dataset, "
                f"but it is of type {type(obj).__name__}"
            )

        return obj

    def close(self) -> None:
        """Close all open HDF5 file handles"""

        trace_file = self._trace_file
        label_file = self._label_file

        self._traces = None
        self._labels = None
        self._trace_file = None
        self._label_file = None

        if trace_file is not None:
            trace_file.close()

        if (label_file is not None and label_file is not trace_file):
            label_file.close()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        """Remove open HDF5 handles before worker serialization."""

        state = self.__dict__.copy()

        state["_trace_file"] = None
        state["_label_file"] = None
        state["_traces"] = None
        state["_labels"] = None

        return state