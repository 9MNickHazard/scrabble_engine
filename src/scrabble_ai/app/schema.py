"""What the "new run" form needs to know about every training setting.

Defaults and types come from the dataclasses in `scrabble_ai.config`, so they cannot drift.
This file adds what a form needs on top: a label, help text, sensible limits, which settings
are everyday ones and which belong behind "all settings", and which cannot change on resume.
A test checks every config field is described here.
"""

from __future__ import annotations

import dataclasses

from scrabble_ai.config import Config

# key: (label, help, extra) where extra may hold min, max, step, options, advanced, locked, and
# multiple_of. `step` is only how far the arrow keys move; `multiple_of` is a rule.
_FIELDS: dict[str, tuple[str, str, dict]] = {
    # run
    "run.hours": ("Time limit", "Stop after this much wall-clock time. 0 means no time limit.",
                  {"min": 0, "max": 1000, "step": 0.5, "advanced": False}),
    "run.max_games": ("Game limit", "Stop after this many self-play games. 0 means no limit.",
                      {"min": 0, "max": 1_000_000_000, "step": 50_000, "advanced": False}),
    "run.seed": ("Random seed", "Fixes the self-play game seeds and all sampling. Change it to repeat an experiment with different luck.",
                 {"min": 0, "max": 1_000_000, "step": 1}),
    "run.lexicon": ("Word list", "A name from configs/lexicons.toml. Empty uses that file's default.", {"locked": True}),
    "run.leaves": ("Leave values file", "Empty uses the fitted file for the word list. \"zero\" trains without leave values.",
                   {"locked": True}),
    "run.device": ("Device", "Train on the GPU or, very slowly, the CPU.", {"options": ["cuda", "cpu"]}),
    "run.vram_fraction": ("GPU memory cap", "Share of GPU memory PyTorch may use. With the cap an oversized setting fails at once instead of silently spilling into system RAM.",
                          {"min": 0.1, "max": 1.0, "step": 0.05}),
    # model
    "model.blocks": ("Residual blocks", "Depth of the network. More blocks is slower per game.",
                     {"min": 1, "max": 40, "step": 1, "advanced": False, "locked": True}),
    "model.channels": ("Channels", "Width of the network (a multiple of 8). Cost grows with the square of this.",
                       {"min": 8, "max": 512, "step": 8, "multiple_of": 8, "advanced": False, "locked": True}),
    "model.norm": ("Normalization", "\"none\" is fastest: each block starts as the identity. \"group\" uses GroupNorm, about half the speed.",
                   {"options": ["none", "group"], "locked": True}),
    "model.vector_hidden": ("Rack features width", "Width of the layers that read the leave, unseen tiles and scalars.",
                            {"min": 8, "max": 1024, "step": 8, "locked": True}),
    "model.head_hidden": ("Head width", "Width of the last hidden layer.", {"min": 8, "max": 2048, "step": 8, "locked": True}),
    "model.value_channels": ("Per-square channels", "Channels kept per square for the head, so it can see where things are.",
                             {"min": 1, "max": 64, "step": 1, "locked": True}),
    "model.cross_checks": ("Cross-check planes", "Show the network which letters fit on each empty square. This is how it sees the dictionary; turn off only to measure what it is worth.",
                           {"locked": True}),
    "model.leave_baseline": ("Leave baseline", "The network predicts a correction to the static leave value, so an untrained network already plays like greedy + leave.",
                             {"locked": True}),
    # selfplay
    "selfplay.games": ("Games in lockstep", "Self-play games advanced together. Speed is flat from 256 to 512.",
                       {"min": 1, "max": 2048, "step": 32}),
    "selfplay.max_candidates": ("Candidates per move", "Moves per decision the network evaluates, the best by score + leave value. 32 is about 30% faster per game.",
                                {"min": 2, "max": 512, "step": 8}),
    "selfplay.max_exchanges": ("Exchange candidates", "Of those candidates, at most this many are exchanges.", {"min": 0, "max": 32, "step": 1}),
    "selfplay.temperature_start": ("Exploration at the start", "Moves are sampled from softmax(value / temperature), in points. At 4, a move 4 points worse is chosen about a third as often as the best.",
                                   {"min": 0, "max": 50, "step": 0.5}),
    "selfplay.temperature_end": ("Exploration at the end", "Temperature once the decay is complete.", {"min": 0, "max": 50, "step": 0.1}),
    "selfplay.temperature_decay_games": ("Exploration decay", "Self-play games over which the temperature falls from start to end. A good value is about a fifth of the games you expect to play.",
                                         {"min": 1, "max": 100_000_000, "step": 50_000}),
    "selfplay.win_weight": ("Play for wins", "0 chooses moves by expected spread, 1 by win probability, between blends the two.",
                            {"min": 0, "max": 1, "step": 0.05}),
    "selfplay.win_scale": ("Win value", "Points a certain win is worth in that blend.", {"min": 1, "max": 1000, "step": 10}),
    # warmstart
    "warmstart.games": ("Warm-start games", "Games played by the static evaluator before the network plays. 0 skips the warm start.",
                        {"min": 0, "max": 10_000_000, "step": 10_000}),
    "warmstart.temperature": ("Warm-start exploration", "Exploration in those games, so the network also sees poor moves.",
                              {"min": 0, "max": 50, "step": 0.5}),
    "warmstart.max_candidates": ("Warm-start candidates", "Candidates per move during the warm start.", {"min": 2, "max": 512, "step": 2}),
    "warmstart.steps": ("Pretraining steps", "Training steps on the warm-start games before self-play begins.",
                        {"min": 0, "max": 1_000_000, "step": 500}),
    # train
    "train.batch_size": ("Batch size", "Positions per training step.", {"min": 16, "max": 16384, "step": 256}),
    "train.learning_rate": ("Learning rate", "Step size. Lower it (for example 0.0003) if the loss jumps around or rises: the first thing to try when a run is unstable.",
                            {"min": 1e-6, "max": 1.0, "step": 0.0001}),
    "train.warmup_steps": ("Warm-up steps", "Steps over which the learning rate ramps up from zero.", {"min": 0, "max": 100_000, "step": 100}),
    "train.weight_decay": ("Weight decay", "Regularization on the weights.", {"min": 0, "max": 1, "step": 0.0001}),
    "train.grad_clip": ("Gradient clip", "Maximum gradient norm. 0 turns clipping off.", {"min": 0, "max": 100, "step": 0.5}),
    "train.value_scale": ("Value scale", "Points per unit of the value output.", {"min": 1, "max": 500, "step": 5, "locked": True}),
    "train.huber_delta": ("Huber delta", "The value loss is quadratic within this many units and linear beyond.", {"min": 0.01, "max": 100, "step": 0.5}),
    "train.win_loss_weight": ("Win loss weight", "Weight of the win-probability loss relative to the value loss.", {"min": 0, "max": 10, "step": 0.1}),
    "train.td_lambda": ("TD lambda", "1 trains on actual game outcomes. Lower mixes in the network's own estimate of the next position: less noise, some bias. Try 0.8 to 0.9 once a first run works.",
                        {"min": 0, "max": 1, "step": 0.05}),
    "train.replay_capacity": ("Replay capacity", "Positions kept in RAM, about 310 bytes each: 3 million is 0.9 GB.",
                              {"min": 1_000, "max": 100_000_000, "step": 500_000}),
    "train.min_replay": ("Replay needed to start", "Positions required before training begins.", {"min": 1, "max": 100_000_000, "step": 10_000}),
    "train.replay_ratio": ("Replay ratio", "Training samples drawn per new self-play position. Higher learns more from each game and slows self-play.",
                           {"min": 0.1, "max": 64, "step": 0.5}),
    "train.actor_sync_steps": ("Self-play sync", "Training steps between copies of the weights into the self-play network.",
                               {"min": 1, "max": 100_000, "step": 50}),
    "train.augment_transpose": ("Transpose augmentation", "Randomly swap rows and columns: the one exact symmetry of the board.", {}),
    "train.amp": ("Mixed precision", "bf16 autocast. Faster, and what the speed figures were measured with.", {}),
    "train.checkpoint_minutes": ("Checkpoint every", "How often the full training state is saved.", {"min": 0.5, "max": 600, "step": 1}),
    "train.eval_minutes": ("Evaluate every", "How often the network plays the baselines and is tested against the best so far.",
                           {"min": 0.5, "max": 600, "step": 1}),
    "train.eval_pairs": ("Evaluation pairs", "Seat-swapped pairs per baseline in each evaluation. More pairs narrow the interval and cost a few seconds.",
                         {"min": 2, "max": 100_000, "step": 50}),
    "train.gate_pairs": ("Gate pairs", "Pairs when testing a network against the current best.", {"min": 2, "max": 100_000, "step": 50}),
    "train.keep_checkpoints": ("Checkpoints kept", "Full checkpoints kept on disk. The best network is always kept.", {"min": 1, "max": 1000, "step": 1}),
}

# The form asks for the name itself, and the app decides where runs live.
HIDDEN = {"run.name", "run.out_dir"}

# What a number counts, shown beside its input.
_UNITS = {
    "run.hours": "hours",
    "run.max_games": "games",
    "selfplay.games": "games",
    "selfplay.temperature_start": "points",
    "selfplay.temperature_end": "points",
    "selfplay.temperature_decay_games": "games",
    "selfplay.win_scale": "points",
    "warmstart.games": "games",
    "warmstart.temperature": "points",
    "warmstart.steps": "steps",
    "train.batch_size": "positions",
    "train.warmup_steps": "steps",
    "train.value_scale": "points",
    "train.replay_capacity": "positions",
    "train.min_replay": "positions",
    "train.actor_sync_steps": "steps",
    "train.checkpoint_minutes": "minutes",
    "train.eval_minutes": "minutes",
    "train.eval_pairs": "pairs",
    "train.gate_pairs": "pairs",
}

_SECTION_TITLES = {"run": "Run", "model": "Network", "selfplay": "Self-play", "warmstart": "Warm start", "train": "Training"}

PRESETS = [
    {"name": "Default 6x64", "description": "About 55 games a second. A good first run.", "overrides": {}},
    {"name": "Larger 10x128", "description": "About 18 games a second. Try it once the default has clearly learned something.",
     "overrides": {"model.blocks": 10, "model.channels": 128}},
    {"name": "Wide 12x192", "description": "About 8.5 games a second: a weekend run. Fewer candidates and a shorter exploration decay to suit.",
     "overrides": {"model.blocks": 12, "model.channels": 192, "selfplay.max_candidates": 32,
                   "selfplay.temperature_decay_games": 100_000}},
    {"name": "Quick test (30 minutes)", "description": "A short run to check everything works. Too short to learn much.",
     "overrides": {"run.hours": 0.5, "warmstart.games": 20_000, "warmstart.steps": 600,
                   "selfplay.temperature_decay_games": 50_000, "train.eval_minutes": 10.0}},
]

# Benchmarks on the RTX 5070 Ti (docs/BENCHMARKS.md). Only 6x64 was measured with self-play and
# training running together; the rest are worked out from separately measured speeds.
SPEED = [
    {"network": "6x64", "parameters": 1_055_056, "games_per_second": 55, "measured": True, "vram_gb": 1.4},
    {"network": "10x128", "parameters": 3_655_316, "games_per_second": 18, "measured": False, "vram_gb": 4.1},
    {"network": "12x192", "parameters": 8_762_582, "games_per_second": 8.5, "measured": False, "vram_gb": 7.1},
    {"network": "16x192", "parameters": 11_418_330, "games_per_second": 6, "measured": False, "vram_gb": 9.1},
    {"network": "20x256", "parameters": 24_490_270, "games_per_second": 3.4, "measured": False, "vram_gb": 7.6},
]

_TYPE_NAMES = {"int": "int", "float": "float", "bool": "bool", "str": "str"}


def schema() -> dict:
    """The JSON of GET /api/config/schema."""
    defaults = Config()
    sections = []
    for section_field in dataclasses.fields(defaults):
        section = getattr(defaults, section_field.name)
        fields = []
        for field in dataclasses.fields(section):
            key = f"{section_field.name}.{field.name}"
            if key in HIDDEN:
                continue
            label, help_text, extra = _FIELDS[key]
            type_name = _TYPE_NAMES[field.type if isinstance(field.type, str) else field.type.__name__]
            options = extra.get("options")
            fields.append({
                "key": key,
                "label": label,
                "type": "choice" if options else type_name,
                "default": getattr(section, field.name),
                "min": extra.get("min"),
                "max": extra.get("max"),
                "step": extra.get("step"),
                "multiple_of": extra.get("multiple_of"),
                "unit": _UNITS.get(key),
                "options": options,
                "advanced": extra.get("advanced", True),
                "locked_on_resume": extra.get("locked", False),
                "help": help_text,
            })
        sections.append({"name": section_field.name, "title": _SECTION_TITLES[section_field.name], "fields": fields})
    return {"sections": sections, "presets": PRESETS, "speed": SPEED}


def described_keys() -> set[str]:
    return set(_FIELDS)
