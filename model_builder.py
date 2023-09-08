import torch
from torch import nn

CLASSIFIER_INPUT_SIDE = 14


class LedgeriseLens(nn.Module):
    def __init__(self, input_channels, hidden_units, output_channels):
        super().__init__()

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                padding=1,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                padding=0,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=(3, 3),
                padding=0,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units
                * (CLASSIFIER_INPUT_SIDE * CLASSIFIER_INPUT_SIDE),
                out_features=hidden_units,
            ),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_layer_2(self.conv_layer_1(x)))
