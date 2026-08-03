from __future__ import annotations

import torch
from mamba_ssm import Mamba
from torch import Tensor, nn


class ResidualMambaBlock(nn.Module):
    """Residual Mamba block described in the reference architecture."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2
    ) -> None:
        super().__init__()

        self.normalization = nn.LayerNorm(d_model)

        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )

        self.activation = nn.GELU()

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply LayerNorm, Mamba, GELU and a residual connection."""
        residual = inputs
        outputs = self.normalization(inputs)
        outputs = self.mamba(outputs)
        outputs = self.activation(outputs)

        return residual + outputs


class CnnMambaMlpAscad(nn.Module):
    """
    CNN-Mamba-MLP model based on the reference paper.

    Expected input shape:
        (batch_size, 1, 700)

    Output shape:
        (batch_size, 256)
    """

    def __init__(
        self,
        num_classes: int = 256,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_mamba_blocks: int = 3
    ) -> None:
        super().__init__()

        if num_classes <= 0:
            raise ValueError(
                "num_classes must be greater than zero."
            )

        if d_model != 512:
            raise ValueError(
                "This architecture requires d_model=512 because the "
                "CNN feature extractor outputs 512 channels."
            )

        if num_mamba_blocks <= 0:
            raise ValueError(
                "num_mamba_blocks must be greater than zero."
            )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=11,
                stride=2,
                padding=5
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            )
        )

        self.mamba_blocks = nn.ModuleList(
            [
                ResidualMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(num_mamba_blocks)
            ]
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=512,
                out_features=512
            ),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(
                in_features=512,
                out_features=4096
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=4096,
                out_features=4096
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=4096,
                out_features=num_classes
            )
        )

    def forward_features(
        self,
        inputs: Tensor
    ) -> Tensor:
        """Extract one 512-dimensional feature vector per trace."""
        if inputs.ndim != 3:
            raise ValueError(
                "Expected a three-dimensional input tensor with shape "
                "(batch_size, channels, trace_length)."
            )

        if inputs.shape[1] != 1:
            raise ValueError(
                "Expected exactly one input channel."
            )

        features = self.feature_extractor(inputs)

        # Conv1d layout: (batch_size, channels, sequence_length).
        # Mamba layout:  (batch_size, sequence_length, d_model).
        features = features.transpose(1, 2).contiguous()

        for block in self.mamba_blocks:
            features = block(features)

        pooled_features = features.mean(dim=1)
        fused_features = self.mlp(pooled_features)

        return fused_features

    def forward(
        self,
        inputs: Tensor
    ) -> Tensor:
        """Return logits for the 256 possible key-byte classes."""
        features = self.forward_features(inputs)
        logits = self.classifier(features)

        return logits


def count_trainable_parameters(
    module: nn.Module
) -> int:
    """Return the number of trainable parameters in a module."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
