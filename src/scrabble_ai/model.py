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

The body of the network is one of three kinds (`model.kind`), all turning the input planes into
`channels` features for each of the 225 squares, which every output then reads:

- `"cnn"`: residual blocks of 3 x 3 convolutions. Fast. Each layer sees one square further, so
  six blocks relate squares up to 13 apart.
- `"cnn-global"`: the same, with every third block also computing a summary of the whole board
  (the mean and maximum of some of its channels) and feeding it back to every square. This is
  KataGo's global pooling, which lets a convolutional network take board-wide facts into account
  from the early layers at almost no cost.
- `"transformer"`: one token per square, plus one for the rack and bag, and `blocks` layers of
  attention in which every square can look at every other and at that token. `channels` is the
  width of a token. This is the shape of the strongest published Scrabble value networks; it
  costs several times what the convolutional kinds do.

Three more choices, all in `ModelConfig`, were added for training from saved games (`offline.py`):

- `head = "pooled"`: the head sees the board only through each channel's mean and maximum, not
  through a layer with a weight for every square, which is what let the first networks learn
  particular boards by heart.
- `win_head = "wdl"`: three logits, for win, draw and loss. `forward` still returns one win
  logit, the log-odds of winning with a draw counted as half, so everything that plays a network
  works with either kind.
- `aux_heads`: outputs used only as extra training signal: the opponent's next score and whether
  they place all seven tiles, and for every square whether the opponent's next move, and the
  mover's own next move, puts a tile there.
"""

from __future__ import annotations

import torch
from torch import nn

from scrabble_ai.config import ModelConfig
from scrabble_ai.features import SQUARES, FeatureSet


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


class GlobalPoolBlock(nn.Module):
    """A residual block that also tells every square about the board as a whole. A quarter of the
    first convolution's channels are summarised over the board (mean and maximum) and the summary
    is added, through a small layer, to the rest of the channels at every square (KataGo's global
    pooling)."""

    def __init__(self, channels: int, norm: str):
        super().__init__()
        use_norm = norm == "group"
        self.summarised = max(8, channels // 4)
        local = channels - self.summarised
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=not use_norm)
        self.norm1 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        self.mix = nn.Linear(2 * self.summarised, local)
        self.conv2 = nn.Conv2d(local, channels, 3, padding=1, bias=not use_norm)
        self.norm2 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        self.gate = None if use_norm else nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.norm1(self.conv1(x)))
        summary, local = y[:, :self.summarised], y[:, self.summarised:]
        board = torch.cat([summary.mean(dim=(2, 3)), summary.amax(dim=(2, 3))], dim=1)
        y = self.norm2(self.conv2(torch.relu(local + self.mix(board)[:, :, None, None])))
        if self.gate is not None:
            return x + self.gate * y
        return torch.relu(x + y)


class TransformerBody(nn.Module):
    """One token per square and one for everything that is not on the board, then layers of
    attention. Returns the square tokens laid out as a board, `[B, channels, 15, 15]`."""

    def __init__(self, planes: int, channels: int, layers: int, vector_hidden: int):
        super().__init__()
        self.embed = nn.Conv2d(planes, channels, 1)
        self.rack = nn.Linear(vector_hidden, channels)
        self.position = nn.Parameter(torch.zeros(1, SQUARES + 1, channels))
        nn.init.normal_(self.position, std=0.02)
        # Attention heads of about 32 channels each, as many as divide the width evenly.
        heads = next(h for h in range(max(1, channels // 32), 0, -1) if channels % h == 0)
        layer = nn.TransformerEncoderLayer(channels, heads, 4 * channels, dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(channels)

    def forward(self, planes: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        b = planes.shape[0]
        squares = self.embed(planes).flatten(2).transpose(1, 2) + bias[:, None, :]  # [B, 225, C]
        tokens = torch.cat([self.rack(v)[:, None, :], squares], dim=1) + self.position
        tokens = self.norm(self.layers(tokens))
        return tokens[:, 1:].transpose(1, 2).reshape(b, -1, 15, 15)


class ValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        channels = config.channels
        self.features = FeatureSet.of(config)
        self.body = None
        if config.kind == "transformer":
            self.body = TransformerBody(self.features.planes, channels, config.blocks, config.vector_hidden)
        self.stem = None if self.body is not None else nn.Conv2d(self.features.planes, channels, 3, padding=1)
        self.vector = nn.Sequential(
            nn.Linear(self.features.vector, config.vector_hidden), nn.ReLU(),
            nn.Linear(config.vector_hidden, config.vector_hidden), nn.ReLU(),
        )
        self.vector_to_stem = nn.Linear(config.vector_hidden, channels)
        # With "cnn-global" every third block (the third, sixth, ...; the last if there are fewer
        # than three) is a global-pooling block.
        pooling = {i for i in range(config.blocks) if i % 3 == 2} or {config.blocks - 1}
        self.blocks = None if self.body is not None else nn.Sequential(*[
            GlobalPoolBlock(channels, config.norm) if config.kind == "cnn-global" and i in pooling else ResidualBlock(channels, config.norm)
            for i in range(config.blocks)])
        self.pooled = config.head == "pooled"
        self.wdl = config.win_head == "wdl"
        self.squares = None if self.pooled else nn.Conv2d(channels, config.value_channels, 1)
        head_inputs = (0 if self.pooled else config.value_channels * SQUARES) + 2 * channels + config.vector_hidden
        self.head = nn.Sequential(nn.Linear(head_inputs, config.head_hidden), nn.ReLU())
        self.value = nn.Linear(config.head_hidden, 1)
        self.win = nn.Linear(config.head_hidden, 3 if self.wdl else 1)
        # Start both outputs at exactly zero: value 0 (the leave baseline alone) and even odds.
        for layer in (self.value, self.win):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if config.aux_heads:
            self.next_turn = nn.Linear(config.head_hidden, 2)  # the opponent's next score, and their bingo logit
            self.next_squares = nn.Conv2d(channels, 2, 1)  # per square: the opponent's next move, the mover's own

    def outputs(self, planes: torch.Tensor, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every output, for training:

        - `value` [B]: spread still to gain, in units of the value scale (a correction to the
          leave value when `leave_baseline` is on).
        - `win`: [B] win logit, or [B, 3] logits for win, draw, loss with `win_head = "wdl"`.
        - with `aux_heads`: `next_turn` [B, 2] (the opponent's next score in units of the value
          scale, and the logit that they place all seven tiles) and `next_squares` [B, 2, 225]
          (logits that the opponent's next move, and the mover's own, puts a tile on each square).
        - `trunk`: the board features every head reads, for measuring how hard each output pulls.
        """
        v = self.vector(vector)
        if self.body is not None:
            x = self.body(planes, v, self.vector_to_stem(v))
        else:
            x = torch.relu(self.stem(planes) + self.vector_to_stem(v)[:, :, None, None])
            x = torch.relu(self.blocks(x))
        parts = [x.mean(dim=(2, 3)), x.amax(dim=(2, 3)), v]
        if self.squares is not None:
            parts.insert(0, torch.relu(self.squares(x)).flatten(1))
        h = self.head(torch.cat(parts, dim=1))
        out = {"value": self.value(h).squeeze(1), "win": self.win(h) if self.wdl else self.win(h).squeeze(1), "trunk": x}
        if self.config.aux_heads:
            out["next_turn"] = self.next_turn(h)
            out["next_squares"] = self.next_squares(x).flatten(2)
        return out

    def forward(self, planes: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.outputs(planes, vector)
        win = out["win"]
        if self.wdl:
            # One number for whoever picks a move: the log-odds of winning, a draw counting half.
            p = torch.softmax(win.float(), dim=1)
            share = (p[:, 0] + 0.5 * p[:, 1]).clamp(1e-6, 1.0 - 1e-6)
            win = torch.log(share) - torch.log1p(-share)
        return out["value"], win


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
