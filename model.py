from torch import nn
from torchvision import models
import torch

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048):
        super(Model, self).__init__()

        base = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(base.children())[:-2])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 🔴 IMPORTANT: bias=False MUST MATCH TRAINING
        self.lstm = nn.LSTM(
            latent_dim,
            hidden_dim,
            lstm_layers,
            bias=False,
            bidirectional=False
        )

        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)

        # 🔴 SAME NAME AS TRAINING
        self.linear1 = nn.Linear(2048, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        fmap = self.model(x)
        x = self.avgpool(fmap)

        x = x.view(b, t, 2048)
        x_lstm, _ = self.lstm(x)

        out = self.linear1(torch.mean(x_lstm, dim=1))
        out = self.dp(out)

        return fmap, out
