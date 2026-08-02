from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import Dataset


class ASCADDatasetProtocol(Protocol):
    """Protocol required by NormalizedASCADDataset."""

    def __len__(self) -> int:
        ...

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor]:
        ...


class NormalizedASCADDataset(
    Dataset[tuple[Tensor, Tensor]]
):
    """Apply feature-wise normalization to an ASCAD dataset."""

    def __init__(
        self,
        dataset: ASCADDatasetProtocol,
        normalization_path: str | Path
    ) -> None:
        self.dataset = dataset
        self.normalization_path = Path(normalization_path)

        (
            self.mean,
            self.standard_deviation
        ) = self._load_normalization_statistics()

    def _load_normalization_statistics(
        self
    ) -> tuple[Tensor, Tensor]:
        """Load normalization statistics from disk."""
        if not self.normalization_path.is_file():
            raise FileNotFoundError(
                "Normalization file not found: "
                f"{self.normalization_path}"
            )

        with np.load(self.normalization_path) as normalization_file:
            mean_array = np.asarray(
                normalization_file["mean"],
                dtype=np.float32
            )

            standard_deviation_array = np.asarray(
                normalization_file["standard_deviation"],
                dtype=np.float32
            )

        self._validate_statistics(
            mean=mean_array,
            standard_deviation=standard_deviation_array
        )

        mean = torch.from_numpy(mean_array)

        standard_deviation = torch.from_numpy(
            standard_deviation_array
        )

        return mean, standard_deviation

    @staticmethod
    def _validate_statistics(
        mean: NDArray[np.float32],
        standard_deviation: NDArray[np.float32]
    ) -> None:
        """Validate normalization arrays."""
        expected_shape = (700,)

        if mean.shape != expected_shape:
            raise ValueError(
                "Unexpected mean shape: "
                f"expected {expected_shape}, "
                f"found {mean.shape}."
            )

        if standard_deviation.shape != expected_shape:
            raise ValueError(
                "Unexpected standard deviation shape: "
                f"expected {expected_shape}, "
                f"found {standard_deviation.shape}."
            )

        if not np.isfinite(mean).all():
            raise ValueError(
                "The normalization mean contains "
                "NaN or infinite values."
            )

        if not np.isfinite(standard_deviation).all():
            raise ValueError(
                "The normalization standard deviation contains "
                "NaN or infinite values."
            )

        if np.any(standard_deviation <= 0.0):
            raise ValueError(
                "The normalization standard deviation contains "
                "non-positive values."
            )

    def __len__(self) -> int:
        """Return the number of samples in the wrapped dataset."""
        return len(self.dataset)

    def __getitem__(
        self,
        index: int
    ) -> tuple[Tensor, Tensor]:
        """Return one normalized trace and its label."""
        trace, label = self.dataset[index]

        if trace.ndim != 1:
            raise ValueError(
                f"Unexpected trace dimensions at index {index}: "
                f"expected 1, found {trace.ndim}."
            )

        if trace.shape != self.mean.shape:
            raise ValueError(
                f"Unexpected trace shape at index {index}: "
                f"expected {tuple(self.mean.shape)}, "
                f"found {tuple(trace.shape)}."
            )

        normalized_trace = (
            trace.to(dtype=torch.float32)
            - self.mean
        ) / self.standard_deviation

        return normalized_trace, label