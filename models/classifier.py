import torch.nn as nn


class FakeNewsClassifier(nn.Module):

    def __init__(self, input_dim=1537, num_classes=2, dropout=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(128, num_classes)
        )

    def forward(self, features):
        return self.net(features)