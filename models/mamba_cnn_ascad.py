import torch
from torch import Tensor, nn

from models.mamba_block import ResidualMambaBlock


class MambaCnnAscad(nn.Module):
    """Mamba front-end followed by the ASCAD CNN classifier."""

    def __init__(
        self,
        num_classes: int = 256,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        num_mamba_blocks: int = 2
    ) -> None:
        super().__init__()

        self.input_projection = nn.Linear(
            in_features=1,
            out_features=d_model
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

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model,
                out_channels=64,
                kernel_size=11,
                padding=5
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=11,
                padding=5
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=11,
                padding=5
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=11,
                padding=5
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=11,
                padding=5
            ),
            nn.ReLU(),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=512 * 21,
                out_features=4096
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4096,
                out_features=4096
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=4096,
                out_features=num_classes
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Run a batch of ASCAD traces through the model.

        Expected input shape:
            (batch_size, 1, 700)

        Output shape:
            (batch_size, num_classes)
        """
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        for block in self.mamba_blocks:
            x = block(x)

        x = x.transpose(1, 2)

        x = self.feature_extractor(x)
        x = self.classifier(x)

        return x