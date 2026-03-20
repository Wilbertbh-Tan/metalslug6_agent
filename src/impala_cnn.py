"""IMPALA-style ResNet feature extractor for Stable Baselines3.

Based on the IMPALA architecture (Espeholt et al., 2018) which uses a
15-layer residual network. Consistently outperforms the Nature CNN
(Mnih et al., 2015) on visual RL tasks, especially at higher resolutions.

Uses Global Average Pooling (Impoola variant) instead of flatten to be
resolution-independent and reduce parameter count.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.relu(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        return out + x


class ConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):
    """IMPALA ResNet with Global Average Pooling.

    Architecture: 3 ConvSequence blocks (16->32->32 channels),
    each with conv + maxpool + 2 residual blocks, followed by
    global average pooling and a linear projection.

    This is resolution-independent: works with any input size.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        channels: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__(observation_space, features_dim)
        # SB3's CnnPolicy uses VecTransposeImage which transposes both
        # observations AND observation_space from HWC to CHW format.
        # After VecFrameStack(4) on (H,W,1): obs_space=(H,W,4), then
        # VecTransposeImage changes it to (4,H,W). So shape[0] = channels.
        c_in = observation_space.shape[0]

        layers = []
        for c_out in channels:
            layers.append(ConvSequence(c_in, c_out))
            c_in = c_out
        self.conv_sequences = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc = nn.Linear(channels[-1], features_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations shape: (batch, C, H, W) after SB3 preprocessing
        x = observations.float() / 255.0
        x = self.conv_sequences(x)
        x = nn.functional.relu(x)
        x = self.gap(x)  # (batch, C, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, C)
        x = nn.functional.relu(self.fc(x))
        return x
