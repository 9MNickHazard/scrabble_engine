"""`model.ValueNet` with the network choices added for training from saved games: the three kinds
of body (`model.kind`), the two heads (`model.head`), the two win outputs (`model.win_head`) and
the helper outputs (`model.aux_heads`); and that a network described by a config from before
they existed is exactly the network it always was."""

from __future__ import annotations

import dataclasses
import itertools

import pytest
import torch

from scrabble_ai.config import BEFORE_IT_EXISTED, ConfigError, ModelConfig, config_from_dict, load_config
from scrabble_ai.features import SQUARES
from scrabble_ai.model import GlobalPoolBlock, ResidualBlock, TransformerBody, ValueNet, parameter_count

KINDS = ("cnn", "cnn-global", "transformer")
HEADS = ("flat", "pooled")
WIN_HEADS = ("binary", "wdl")
TINY = {"blocks": 3, "channels": 16, "vector_hidden": 16, "head_hidden": 16}


def tiny(**changes) -> ModelConfig:
    return ModelConfig(**{**TINY, **changes})


def inputs(net: ValueNet, batch: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    planes = torch.rand(batch, net.features.planes, 15, 15, generator=generator)
    vector = torch.randn(batch, net.features.vector, generator=generator)
    return planes, vector


def wake(net: ValueNet) -> None:
    """Move every parameter that starts at zero (the outputs, the residual gates) off it, so that
    gradients can flow through the whole network."""
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in net.parameters():
            if not parameter.any():
                parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator) + 0.1)


@pytest.mark.parametrize("kind,head,win_head", list(itertools.product(KINDS, HEADS, WIN_HEADS)))
def test_shapes_and_the_neutral_start(kind, head, win_head):
    net = ValueNet(tiny(kind=kind, head=head, win_head=win_head, aux_heads=True)).eval()
    planes, vector = inputs(net)
    out = net.outputs(planes, vector)
    assert set(out) == {"value", "win", "trunk", "next_turn", "next_squares"}
    assert out["value"].shape == (5,)
    assert out["win"].shape == ((5, 3) if win_head == "wdl" else (5,))
    assert out["trunk"].shape == (5, 16, 15, 15)
    assert out["next_turn"].shape == (5, 2) and out["next_squares"].shape == (5, 2, SQUARES)
    value, win = net(planes, vector)
    assert value.shape == (5,) and win.shape == (5,) and value.dtype == win.dtype == torch.float32
    # A new network says nothing: value exactly 0 (the leave baseline alone), even odds exactly.
    assert torch.equal(value, torch.zeros(5))
    assert torch.equal(torch.sigmoid(win), torch.full((5,), 0.5))
    without = ValueNet(tiny(kind=kind, head=head, win_head=win_head)).eval()
    assert set(without.outputs(*inputs(without))) == {"value", "win", "trunk"}


@pytest.mark.parametrize("kind,head,win_head", list(itertools.product(KINDS, HEADS, WIN_HEADS)))
def test_forward_is_what_outputs_imply(kind, head, win_head):
    net = ValueNet(tiny(kind=kind, head=head, win_head=win_head, aux_heads=True)).eval()
    wake(net)
    planes, vector = inputs(net, seed=3)
    with torch.no_grad():
        out = net.outputs(planes, vector)
        value, win = net(planes, vector)
    assert torch.equal(value, out["value"])
    if win_head == "wdl":
        p = torch.softmax(out["win"], dim=1)
        share = p[:, 0] + 0.5 * p[:, 1]  # win, draw counted as half
        assert torch.allclose(torch.sigmoid(win), share, atol=1e-6)
        assert torch.allclose(win, torch.log(share / (1 - share)), atol=1e-4)
    else:
        assert torch.equal(win, out["win"])
    assert not torch.equal(value, torch.zeros_like(value)), "once trained, the outputs move"


@pytest.mark.parametrize("kind,head", list(itertools.product(KINDS, HEADS)))
def test_gradients_reach_every_parameter(kind, head):
    net = ValueNet(tiny(kind=kind, head=head, win_head="wdl", aux_heads=True))
    wake(net)
    planes, vector = inputs(net, batch=6, seed=5)
    out = net.outputs(planes, vector)
    loss = (out["value"].square().mean() + out["win"].square().mean() + out["next_turn"].square().mean()
            + out["next_squares"].square().mean())
    loss.backward()
    for name, parameter in net.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.abs().sum() > 0, name


def test_at_the_start_the_win_and_value_outputs_pass_nothing_back():
    """Zero output layers and zero gates: the main outputs' gradient on the trunk starts at zero
    (which is why the trainer measures the helpers' pull only once training is under way)."""
    net = ValueNet(tiny(win_head="wdl", aux_heads=True))
    planes, vector = inputs(net)
    out = net.outputs(planes, vector)
    (gradient,) = torch.autograd.grad(torch.nn.functional.cross_entropy(out["win"], torch.zeros(5, dtype=torch.long)),
                                      out["trunk"], retain_graph=True)
    assert not gradient.any()
    (gradient,) = torch.autograd.grad(out["next_turn"].square().mean(), out["trunk"])
    assert gradient.any(), "the helpers start random and pull at once"


def test_parameter_counts_follow_the_choices():
    config = tiny()
    flat, pooled = ValueNet(config), ValueNet(dataclasses.replace(config, head="pooled"))
    # The pooled head has no per-square layer, and its first layer no per-square inputs.
    assert "squares.weight" in flat.state_dict() and pooled.squares is None
    assert not any(key.startswith("squares.") for key in pooled.state_dict())
    per_square = config.value_channels * (config.channels + 1) + config.value_channels * SQUARES * config.head_hidden
    assert parameter_count(flat) - parameter_count(pooled) == per_square

    # Three win logits instead of one.
    wdl = ValueNet(dataclasses.replace(config, win_head="wdl"))
    assert parameter_count(wdl) - parameter_count(flat) == 2 * (config.head_hidden + 1)
    assert wdl.win.out_features == 3 and flat.win.out_features == 1

    # Helper outputs: a two-number layer on the head and a two-map 1x1 convolution on the trunk.
    aux = ValueNet(dataclasses.replace(config, aux_heads=True))
    assert parameter_count(aux) - parameter_count(flat) == 2 * (config.head_hidden + 1) + 2 * (config.channels + 1)
    assert {"next_turn.weight", "next_squares.weight"} <= set(aux.state_dict())
    assert not hasattr(flat, "next_turn") and not hasattr(flat, "next_squares")


@pytest.mark.parametrize("blocks,pooling", [(1, {0}), (2, {1}), (3, {2}), (7, {2, 5}), (9, {2, 5, 8})])
def test_cnn_global_pools_in_every_third_block(blocks, pooling):
    net = ValueNet(tiny(blocks=blocks, kind="cnn-global"))
    kinds = [type(block) for block in net.blocks]
    assert [i for i, k in enumerate(kinds) if k is GlobalPoolBlock] == sorted(pooling)
    assert all(k is ResidualBlock for i, k in enumerate(kinds) if i not in pooling)
    keys = net.state_dict().keys()
    assert {int(k.split(".")[1]) for k in keys if ".mix." in k} == pooling
    # Each pooling block trades part of its second convolution for the board-wide summary layer.
    plain = ValueNet(tiny(blocks=blocks))
    channels = TINY["channels"]
    summarised = max(8, channels // 4)
    local = channels - summarised
    per_block = (2 * summarised * local + local) - (channels - local) * channels * 9
    assert parameter_count(net) - parameter_count(plain) == len(pooling) * per_block


def test_cnn_global_with_group_norm_and_wider_channels():
    net = ValueNet(tiny(blocks=3, channels=64, kind="cnn-global", norm="group"))
    block = net.blocks[2]
    assert isinstance(block, GlobalPoolBlock) and block.summarised == 16 and block.gate is None
    assert block.mix.in_features == 32 and block.mix.out_features == 48 and block.conv2.in_channels == 48
    value, win = net(*inputs(net))
    assert torch.isfinite(value).all() and torch.isfinite(win).all()


def test_the_transformer_body():
    net = ValueNet(tiny(blocks=2, channels=64, kind="transformer"))
    assert isinstance(net.body, TransformerBody) and net.stem is None and net.blocks is None
    assert len(net.body.layers.layers) == 2
    assert net.body.position.shape == (1, SQUARES + 1, 64)
    assert net.body.layers.layers[0].self_attn.num_heads == 2, "heads of about 32 channels"
    value, win = net(*inputs(net))
    assert value.shape == (5,) and win.shape == (5,)
    # Every square can see every other: a change on one corner reaches the opposite one.
    wake(net)
    planes, vector = inputs(net, batch=1)
    moved = planes.clone()
    moved[0, :, 0, 0] += 1.0
    with torch.no_grad():
        a, b = net.outputs(planes, vector)["trunk"], net.outputs(moved, vector)["trunk"]
    assert not torch.allclose(a[0, :, 14, 14], b[0, :, 14, 14])
    cnn = ValueNet(tiny(blocks=2, kind="cnn"))
    wake(cnn)
    planes, vector = inputs(cnn, batch=1)
    moved = planes.clone()
    moved[0, :, 0, 0] += 1.0
    with torch.no_grad():
        a, b = cnn.outputs(planes, vector)["trunk"], cnn.outputs(moved, vector)["trunk"]
    assert torch.equal(a[0, :, 14, 14], b[0, :, 14, 14]), "two convolutional blocks see a few squares, not the board"


def old_keys(blocks: int) -> list[str]:
    """The state-dict keys of the network as it was before these choices (no normalization)."""
    keys = ["stem.weight", "stem.bias", "vector.0.weight", "vector.0.bias", "vector.2.weight", "vector.2.bias",
            "vector_to_stem.weight", "vector_to_stem.bias"]
    for i in range(blocks):
        keys += [f"blocks.{i}.gate", f"blocks.{i}.conv1.weight", f"blocks.{i}.conv1.bias", f"blocks.{i}.conv2.weight",
                 f"blocks.{i}.conv2.bias"]
    return keys + ["squares.weight", "squares.bias", "head.0.weight", "head.0.bias", "value.weight", "value.bias",
                   "win.weight", "win.bias"]


def test_a_network_from_before_these_settings_is_unchanged():
    old = ModelConfig(**BEFORE_IT_EXISTED["model"])
    assert (old.kind, old.head, old.win_head, old.aux_heads) == ("cnn", "flat", "binary", False)
    net = ValueNet(old)
    assert parameter_count(net) == 1_055_056
    assert list(net.state_dict()) == old_keys(6)
    # Today's defaults keep the same body and head; only the inputs grew.
    assert list(ValueNet(ModelConfig()).state_dict()) == old_keys(6)
    # A checkpoint's config that does not mention the new settings reads back as the old network.
    saved = {"model": {k: v for k, v in dataclasses.asdict(old).items()
                       if k not in ("kind", "head", "win_head", "aux_heads")}}
    rebuilt = config_from_dict(saved).model
    assert (rebuilt.kind, rebuilt.head, rebuilt.win_head, rebuilt.aux_heads) == ("cnn", "flat", "binary", False)
    ValueNet(rebuilt).load_state_dict(net.state_dict())  # strict: the same keys and shapes


def test_the_old_forward_pass_is_unchanged():
    """What forward computed before the network got kinds and heads, written out by hand."""
    net = ValueNet(tiny(blocks=2))
    wake(net)
    planes, vector = inputs(net, seed=9)
    with torch.no_grad():
        v = net.vector(vector)
        x = torch.relu(net.stem(planes) + net.vector_to_stem(v)[:, :, None, None])
        x = torch.relu(net.blocks(x))
        per_square = torch.relu(net.squares(x)).flatten(1)
        pooled = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        h = net.head(torch.cat([per_square, pooled, v], dim=1))
        expected = (net.value(h).squeeze(1), net.win(h).squeeze(1))
        value, win = net(planes, vector)
    assert torch.equal(value, expected[0]) and torch.equal(win, expected[1])


def test_on_the_cpu_no_bfloat16_is_needed():
    for kind in KINDS:
        net = ValueNet(tiny(kind=kind, head="pooled", win_head="wdl", aux_heads=True))
        planes, vector = inputs(net)
        out = net.outputs(planes, vector)
        assert all(t.dtype == torch.float32 for t in out.values()), kind
        assert all(p.dtype == torch.float32 for p in net.parameters()), kind


def test_config_validation_of_the_new_choices():
    load_config(None, ["model.kind=cnn-global", "model.head=pooled", "model.win_head=wdl", "model.aux_heads=true"])
    load_config(None, ["model.kind=transformer", "train.target=q"])
    for item, message in (("model.kind=rnn", "model.kind"), ("model.kind=CNN", "model.kind"),
                          ("model.head=wide", "model.head"), ("model.win_head=ternary", "model.win_head"),
                          ("train.target=sarsa", "train.target")):
        with pytest.raises(ConfigError, match=message):
            load_config(None, [item])
    with pytest.raises(ConfigError):
        load_config(None, ["model.aux_heads=maybe"])


def test_before_it_existed_names_every_new_model_setting():
    before = BEFORE_IT_EXISTED["model"]
    assert {"kind": "cnn", "head": "flat", "win_head": "binary", "aux_heads": False}.items() <= before.items()
    assert set(before) <= {f.name for f in dataclasses.fields(ModelConfig)}
