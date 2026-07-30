import torch
import torch.nn as nn

class ASCADOriginalCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # CNN block 1: 700 -> 350
            nn.Conv1d(
                in_channels=1,
                out_channels=64,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            # CNN block 2: 350 -> 175
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            # CNN block 3: 175 -> 87
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            
            # CNN block 4: 87 -> 43
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            ),

            
            # CNN block 5: 43 -> 21
            nn.Conv1d(
                in_channels=512,
                out_channels=512,
                kernel_size=11,
                stride=1,
                padding=5
            ),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # MLP block 1
            nn.Linear(
                in_features=512*21,
                out_features=4096
            ),
            nn.ReLU(inplace=True),

            # MLP block 2
            nn.Linear(
                in_features=4096,
                out_features=4096
            ),
            nn.ReLU(inplace=True),

            # Classification Head
            nn.Linear(
                in_features=4096,
                out_features=256
            )
        )

        self._initialize_weights()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x