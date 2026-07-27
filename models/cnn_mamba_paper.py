import torch
from torch import nn
from mamba_ssm import Mamba

class CNNFeatureExtractor(nn.Module):
    """
    Five-block 1D CNN described in Section 3.3.1 of the paper.

    Input:
    [batch_size, trace_length]
    or [batch_size, 1, trace_length]

    Output:
    [batch_size, 512, reduced_length]
    """

    def __init__(self) -> None:
       super().__init__()

       self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2)
       )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        if x.ndim != 3:
            raise ValueError(
                "CNN input must have shape [B, L] or [B, 1, L], "
                f"but received {tuple(x.shape)}"
            )

        if x.shape[1] != 1:
            raise ValueError(
                f"Expected one input channel, but received {x.shape[1]}"
            )

        return self.features(x)

class ResidualMambaBlock(nn.Module):
    """
    Residual Mamba block:

    output = GELU(Mamba(LayerNorm(x))) + x
    """

    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model = d_model, d_state = d_state, d_conv = d_conv, expand = expand)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.activation(x)
        return x + residual

class ResidualMambaEncoder(nn.Module):
    """
    Three consecutive Residual Mamba blocks followed by
    global average pooling over the sequence dimension.
    """

    def __init__(self, d_model: int = 512, d_state: int = 16, d_conv: int = 4, expand: int = 2, num_blocks: int = 3) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [ResidualMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected Mamba shape: [B, sequence_length, d_model]
        for block in self.blocks:
            x = block(x)

        # Global average pooling along sequence dimension
        return x.mean(dim=1)    

class FeatureFusionMLP(nn.Module):
    """
    MLP for feature fusion and classification.
    """

    def __init__(self, feature_dim: int = 512) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ClassificationHead(nn.Module):
    """
    Classification head for 256 possible AES byte values.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 4096, num_classes: int = 256) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class PaperCNNMambaModel(nn.Module):
    """
    Complete model architecture as described in the paper.

    Pipeline:
        CNN
        -> transpose
        -> 3 Residual Mamba blocks
        -> global average pooling
        -> MLP
        -> classification head
    """

    def __init__(self, num_classes: int = 256) -> None:
        super().__init__()

        self.cnn_extractor = CNNFeatureExtractor()
        self.mamba_encoder = ResidualMambaEncoder(d_model=512, d_state=16, d_conv=4, expand=2, num_blocks=3)
        self.mlp = FeatureFusionMLP(feature_dim=512)
        self.classifier = ClassificationHead(input_dim=512, hidden_dim=4096, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L] or [B, 1, L]
        x = self.cnn_extractor(x)

        # CNN output: [B, 512, reduced_length]
        # Mamba input: [B, reduced_length, 512]
        x = x.transpose(1, 2).contiguous()

        # Output after global temporal average: [B, 512]
        x = self.mamba_encoder(x)

        # [B, 512]
        x = self.mlp(x)

        # [B, 256]
        logits = self.classifier(x)

        return logits   
