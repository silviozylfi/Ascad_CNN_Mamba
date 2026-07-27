from pathlib import Path

import numpy as np
import torch


class NormalizeTrace:
    """
    Feature-wise z-score normalization.
    """

    def __init__(self, normalization_file: str | Path):

        data = np.load(normalization_file)

        self.mean = torch.from_numpy(data["mean"]).float()
        self.std = torch.from_numpy(data["std"]).float()

    def __call__(self, trace: torch.Tensor) -> torch.Tensor:

        return (trace - self.mean) / self.std