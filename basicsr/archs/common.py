import torch
import torch.nn as nn
from basicsr.archs.arch_util import ResidualBlockNoBN


class CAB(nn.Module):
    """Channel Attention Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class PFA(nn.Module):
    """Progressive Feature Aggregation"""
    def __init__(self, num_feat=64, num_blocks=4):
        super(PFA, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList([ResidualBlockNoBN(num_feat=num_feat) for _ in range(num_blocks)])
        self.aggregation = nn.Conv2d(num_feat * (num_blocks + 1), num_feat, 1, 1, 0, bias=True)
        self.attention = CAB(num_feat, reduction=16)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            feat = block(features[-1])
            features.append(feat)
        aggregated = self.aggregation(torch.cat(features, dim=1))
        return self.attention(aggregated)