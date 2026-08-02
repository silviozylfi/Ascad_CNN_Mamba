import torch
from mamba_ssm import Mamba
from torch import Tensor, nn


class ResidualMambaBlock(nn.Module):
    """Residual Mamba block with pre-normalization."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply a pre-normalized Mamba transformation with a residual connection.

        Expected input shape:
            (batch_size, sequence_length, d_model)

        Output shape:
            (batch_size, sequence_length, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)

        return residual + x