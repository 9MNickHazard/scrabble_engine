"""The afterstate value network.

Input: the position a move leaves behind, from the mover's side (see `features.py`).
Outputs: `value`, the expected spread the mover gains over the rest of the game, in units of
`value_scale` points, and `win_logit`, the log-odds that the mover wins.

Move selection uses Q(s, a) = score(a) + V(afterstate), where V in points is
`value * value_scale`, plus the static leave value when `leave_baseline` is on. In that mode the
network only has to learn what the board adds to (or takes from) the leave value, and an untrained
network plays exactly like the greedy+leave bot.

Shape: a residual CNN over the 15x15 board with no pooling and no BatchNorm. The non-board
features enter twice: added to the stem as a per-channel bias, so the trunk can relate the rack to
the board from the first layer, and again in the head.
"""

from __future__ import annotations

import torch
from torch import nn

from scrabble_ai.config import ModelConfig
from scrabble_ai.features import SQUARES, VECTOR, plane_count


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm: str):
        super().__init__()
        use_norm = norm == "group"
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_norm)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_norm)
        self.norm1 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        self.norm2 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        # Without normalization each block starts as the identity and learns how much to add.
        self.gate = None if use_norm else nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.gate is not None:
            return x + self.gate * y
        return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        channels = config.channels
        self.stem = nn.Conv2d(plane_count(config.cross_checks), channels, 3, padding=1)
        self.vector = nn.Sequential(
            nn.Linear(VECTOR, config.vector_hidden), nn.ReLU(), nn.Linear(config.vector_hidden, config.vector_hidden), nn.ReLU()
        )
        self.vector_to_stem = nn.Linear(config.vector_hidden, channels)
        self.blocks = nn.Sequential(*[ResidualBlock(channels, config.norm) for _ in range(config.blocks)])
        self.squares = nn.Conv2d(channels, config.value_channels, 1)
        head_inputs = config.value_channels * SQUARES + 2 * channels + config.vector_hidden
        self.head = nn.Sequential(nn.Linear(head_inputs, config.head_hidden), nn.ReLU())
        self.value = nn.Linear(config.head_hidden, 1)
        self.win = nn.Linear(config.head_hidden, 1)
        # Start both outputs at exactly zero: value 0 (the leave baseline alone) and even odds.
        for layer in (self.value, self.win):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, planes: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.vector(vector)
        x = torch.relu(self.stem(planes) + self.vector_to_stem(v)[:, :, None, None])
        x = torch.relu(self.blocks(x))
        per_square = torch.relu(self.squares(x)).flatten(1)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        h = self.head(torch.cat([per_square, pooled, v], dim=1))
        return self.value(h).squeeze(1), self.win(h).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
