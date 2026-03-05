import torch.nn as nn


class FakeNewsClassifier(nn.Module):

    def __init__(self, input_dim=512, num_classes=2, dropout=0.3):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, features):

        x = self.dropout(features)

        logits = self.linear(x)

        return logits