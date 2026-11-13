"""What the "new run" form and the tuning guide need to know about every training setting.

Defaults and types come from the dataclasses in `scrabble_ai.config`, so they cannot drift.
This file adds what a person needs on top: a label, what the setting does, what turning it down
or up does, the range worth trying, sensible limits for the form, which settings are everyday
ones and which belong behind "all settings", and which cannot change on resume. A test checks
every config field is described here.
"""

from __future__ import annotations

import dataclasses

from scrabble_ai.config import Config


def _f(label: str, what: str, lower: str, higher: str, typical: str = "", **extra) -> dict:
    """One setting. `extra` may hold min, max, step, options, advanced, locked, multiple_of.
    `step` is only how far the arrow keys move; `multiple_of` is a rule."""
    return {"label": label, "help": what, "lower": lower, "higher": higher, "typical": typical, **extra}


_FIELDS: dict[str, dict] = {
    # ---- run ---------------------------------------------------------------------------------
    "run.hours": _f(
        "Time limit", "Stop after this much wall-clock time. 0 means no time limit.",
        "A shorter session: fewer games, an earlier answer.",
        "More games. Worth it only once a short run shows the network is learning.",
        "0.5 for a test, 4 to 12 for a real run", min=0, max=1000, step=0.5, advanced=False),
    "run.max_games": _f(
        "Game limit", "Stop after this many self-play games. 0 means no limit.",
        "Stops sooner.", "Runs longer. Whichever of the two limits is reached first ends the run.",
        "0 (use the time limit), or 200,000 to 2,000,000", min=0, max=1_000_000_000, step=50_000, advanced=False),
    "run.seed": _f(
        "Random seed", "Fixes the self-play game seeds and all sampling.",
        "Any other number gives the same experiment with different games.",
        "Run two seeds of one setting to see how much of a difference is luck.",
        "any whole number", min=0, max=1_000_000, step=1),
    "run.lexicon": _f(
        "Word list", "A name from configs/lexicons.toml. Empty uses that file's default.",
        "Empty: the default list.", "Another list is a different game: networks and leave values are tied to their list.",
        "leave empty", locked=True),
    "run.leaves": _f(
        "Leave values file", "Empty uses the fitted file for the word list. \"zero\" trains without leave values.",
        "Empty: the fitted values, so an untrained network already plays like greedy + leave.",
        "\"zero\": the network must learn what tiles are worth from nothing. Much harder; only to measure what leave values buy.",
        "leave empty", locked=True),
    "run.device": _f(
        "Device", "Train on the GPU or, very slowly, the CPU.",
        "cuda: the GPU.", "cpu: far slower; only for checking that things run.",
        "cuda", options=["cuda", "cpu"]),
    "run.vram_fraction": _f(
        "GPU memory cap", "Share of GPU memory PyTorch may use. With the cap an oversized setting fails at once "
        "instead of silently spilling into system RAM.",
        "Leaves room for other programs; a large network or batch fails sooner.",
        "Allows larger networks and batches; less room for anything else on the GPU.",
        "0.8 to 0.95", min=0.1, max=1.0, step=0.05),
    # ---- model -------------------------------------------------------------------------------
    "model.blocks": _f(
        "Residual blocks", "Depth of the network.",
        "Faster games, so more of them per hour; less capacity.",
        "Slower (cost grows with depth), more capacity, memorises more readily and needs more games to pay off.",
        "4 to 10", min=1, max=40, step=1, advanced=False, locked=True),
    "model.channels": _f(
        "Channels", "Width of the network (a multiple of 8).",
        "Much faster: cost grows with the square of the width.",
        "Much slower and more memory; more capacity. 10x128 ran at a third of the speed of 6x64.",
        "64 to 128", min=8, max=512, step=8, multiple_of=8, advanced=False, locked=True),
    "model.norm": _f(
        "Normalization", "\"none\": no normalization, each block starts as the identity. \"group\": GroupNorm.",
        "none: fastest, and what every speed figure was measured with.",
        "group: about half the speed and 2.7 times the training memory. Try it only if training is unstable at any learning rate.",
        "none", options=["none", "group"], locked=True),
    "model.vector_hidden": _f(
        "Rack features width", "Width of the layers that read the leave, the unseen tiles and the scalars.",
        "A little less capacity for rack and bag knowledge.", "A little more; hardly changes the speed.",
        "64 to 256", min=8, max=1024, step=8, locked=True),
    "model.head_hidden": _f(
        "Head width", "Width of the last hidden layer.",
        "A smaller head: less able to memorise particular boards.",
        "A larger head: more capacity, most of it in the layer that reads the whole board at once.",
        "128 to 512", min=8, max=2048, step=8, locked=True),
    "model.value_channels": _f(
        "Per-square channels", "Channels kept per square for the head, so it can see where things are.",
        "The head sees a coarser picture of the board; fewer weights, less memorising.",
        "More positional detail, and many more weights in the head (225 x this x head width).",
        "2 to 16", min=1, max=64, step=1, locked=True),
    "model.cross_checks": _f(
        "Cross-check planes", "Show the network which letters fit on each empty square. This is how it sees the dictionary.",
        "Off: the network is blind to the dictionary. Only to measure what the planes are worth.",
        "On: the default.", "on", locked=True),
    "model.leave_baseline": _f(
        "Leave baseline", "The network predicts a correction to the static leave value.",
        "Off: it must learn leave values itself and starts far weaker than greedy + leave.",
        "On: an untrained network already plays like greedy + leave, and training can only add to that.",
        "on", locked=True),
    # ---- selfplay ----------------------------------------------------------------------------
    "selfplay.games": _f(
        "Games in lockstep", "Self-play games advanced together.",
        "Smaller batches for the GPU: fewer games a second.",
        "Speed is flat from 256 to 512; beyond that only more memory.",
        "256 to 512", min=1, max=2048, step=32),
    "selfplay.max_candidates": _f(
        "Candidates per move", "Moves per decision the network evaluates: the best by score + leave value.",
        "Faster (32 is about 30% faster a game), but the network can never choose a move the static ranking puts below the cut.",
        "Slower; lets the network find moves the static ranking rates low.",
        "32 to 128", min=2, max=512, step=8),
    "selfplay.max_exchanges": _f(
        "Exchange candidates", "Of those candidates, at most this many are exchanges.",
        "Fewer exchanges considered; a little faster.", "More kinds of exchange considered.",
        "2 to 8", min=0, max=32, step=1),
    "selfplay.temperature_start": _f(
        "Exploration at the start", "Moves are sampled from softmax(value / temperature), in points. At 4, a move 4 points "
        "worse is chosen about a third as often as the best.",
        "Early games stay close to the best move: stronger, less varied play. 0 always takes the top move.",
        "More second-best moves early: more variety to learn from, weaker and noisier games.",
        "2 to 6", min=0, max=50, step=0.5),
    "selfplay.temperature_end": _f(
        "Exploration at the end", "Temperature once the decay is complete.",
        "Late games are nearly the network's best play, which is the play its values should describe.",
        "Keeps exploring: more variety, but the values describe a sloppier player.",
        "0.3 to 1", min=0, max=50, step=0.1),
    "selfplay.temperature_decay_games": _f(
        "Exploration decay", "Self-play games over which the temperature falls from start to end.",
        "Exploration ends sooner.", "Explores for longer.",
        "a fifth to a half of the games you expect to play", min=1, max=100_000_000, step=50_000),
    "selfplay.win_weight": _f(
        "Play for wins", "0 chooses moves by expected spread, 1 by win probability, between blends the two.",
        "0: plays for points. The right choice until the value network is clearly good.",
        "Leans on the win estimate, which matters near the end of close games. Needs a well-trained win head.",
        "0, later 0.1 to 0.3", min=0, max=1, step=0.05),
    "selfplay.win_scale": _f(
        "Win value", "Points a certain win is worth in that blend.",
        "Win probability counts for less.", "Win probability counts for more.",
        "50 to 150", min=1, max=1000, step=10),
    # ---- warmstart ---------------------------------------------------------------------------
    "warmstart.games": _f(
        "Warm-start games", "Games played by the static evaluator before the network plays. 0 skips the warm start.",
        "A quicker start with less data to pretrain on. 0: self-play starts from an untrained network (which, with "
        "the leave baseline, plays like greedy + leave).",
        "More pretraining data, at about two and a half minutes per 100,000 games. 100,000 games nearly fill a "
        "3-million-position replay buffer.",
        "50,000 to 200,000", min=0, max=10_000_000, step=10_000),
    "warmstart.temperature": _f(
        "Warm-start exploration", "Exploration in those games, so the network also sees poor moves.",
        "Games close to greedy + leave play.", "More poor moves in the data.",
        "3 to 8", min=0, max=50, step=0.5),
    "warmstart.max_candidates": _f(
        "Warm-start candidates", "Candidates per move during the warm start.",
        "Only the top static moves are ever played.", "A wider spread of moves; slower.",
        "8 to 32", min=2, max=512, step=2),
    "warmstart.steps": _f(
        "Pretraining steps", "Training steps on the warm-start games before self-play begins.",
        "Less pretraining.", "More. Past one pass over the data (warm-start positions / batch size, about 2,300 steps "
        "by default) positions repeat and the network starts memorising them.",
        "1,000 to 4,000", min=0, max=1_000_000, step=500),
    # ---- train -------------------------------------------------------------------------------
    "train.batch_size": _f(
        "Batch size", "Positions per training step.",
        "More, noisier updates for the same games.", "Fewer, smoother updates; more GPU memory.",
        "512 to 2,048", min=16, max=16384, step=256),
    "train.learning_rate": _f(
        "Learning rate", "Step size of each update.",
        "Slower, steadier learning that chases less noise. The first thing to lower (to 0.0001) if a run keeps getting worse.",
        "Faster, and more easily knocked about by noisy targets. 0.001 was too high with the first runs' settings.",
        "0.0001 to 0.001", min=1e-6, max=1.0, step=0.0001),
    "train.warmup_steps": _f(
        "Warm-up steps", "Steps over which the learning rate ramps up from zero.",
        "Full speed sooner; a rougher start.", "A gentler start.",
        "200 to 2,000", min=0, max=100_000, step=100),
    "train.weight_decay": _f(
        "Weight decay", "Pulls the weights towards zero at every step.",
        "Less restraint on the network.", "More restraint: less memorising, slower fitting; too much and it cannot fit anything.",
        "0.00001 to 0.001", min=0, max=1, step=0.0001),
    "train.grad_clip": _f(
        "Gradient clip", "Maximum gradient norm. 0 turns clipping off.",
        "Caps large updates harder: steadier, slower.", "Lets large updates through; 0 removes the cap.",
        "0.5 to 5", min=0, max=100, step=0.5),
    "train.value_scale": _f(
        "Value scale", "Points per unit of the value output.",
        "The network works in larger numbers.", "The network works in smaller numbers.",
        "leave at 50", min=1, max=500, step=5, locked=True),
    "train.huber_delta": _f(
        "Huber delta", "The value loss is quadratic within this many units (2 = 100 points) and linear beyond.",
        "Big misses count for less: more robust to lucky and unlucky games.",
        "Closer to squared error: outlying games pull harder.",
        "1 to 3", min=0.01, max=100, step=0.5),
    "train.win_loss_weight": _f(
        "Win loss weight", "Weight of the win-probability loss relative to the value loss.",
        "The win estimate gets less attention; 0 leaves it untrained.", "More of the network's effort goes to predicting wins.",
        "0.25 to 1", min=0, max=10, step=0.1),
    "train.td_lambda": _f(
        "TD lambda", "How far ahead the training targets look. At 1 the target is the spread actually gained by the "
        "end of the game; below 1 it leans on the estimate of the next position.",
        "Targets lean on the next position's estimate: little noise, but mistakes in the estimates feed back into the targets.",
        "Targets move towards actual final results: unbiased but noisy (about 75 points at 1.0), which invites "
        "memorising them. 1.0 was what the first two runs used.",
        "0.5 to 0.8", min=0, max=1, step=0.05),
    "train.replay_capacity": _f(
        "Replay capacity", "Positions kept in RAM, about 310 bytes each: 3 million is 0.9 GB.",
        "Only recent games are kept: less memory, less variety.", "More variety and more memory; old games stay longer.",
        "2,000,000 to 10,000,000", min=1_000, max=100_000_000, step=500_000),
    "train.min_replay": _f(
        "Replay needed to start", "Positions required before training begins.",
        "Training starts on very little data.", "A longer wait before the first update. A warm start already provides this.",
        "50,000 to 500,000", min=1, max=100_000_000, step=10_000),
    "train.replay_ratio": _f(
        "Replay ratio", "Training samples drawn per new self-play position: about how many times each position is trained on.",
        "Each position is used once or less: nothing to memorise, fewer updates per game, more of the GPU left for self-play.",
        "More updates per game, but the same games come round again and their outcomes get memorised. At 4 both "
        "of the first runs got much worse.",
        "0.5 to 1.5", min=0.1, max=64, step=0.5),
    "train.actor_sync_steps": _f(
        "Self-play sync", "Training steps between copies of the weights into the self-play network.",
        "Self-play uses fresher weights.", "A steadier but staler self-play player.",
        "50 to 500", min=1, max=100_000, step=50),
    "train.guard": _f(
        "Guard", "When a network is significantly behind the best one for several evaluations in a row, go back to "
        "the best network's weights and halve the learning rate.",
        "Off: nothing stops a run drifting far below its best (best.pt is still kept).",
        "On: a run can waste time but cannot end far worse than it started.",
        "on"),
    "train.guard_patience": _f(
        "Guard patience", "Evaluations in a row significantly behind the best before the guard acts.",
        "1 reacts at once, and to the occasional unlucky evaluation too.", "Waits longer before going back.",
        "2 to 3", min=1, max=20, step=1),
    "train.augment_transpose": _f(
        "Transpose augmentation", "Randomly swap rows and columns: the one exact symmetry of the board.",
        "Off: half the variety in the training data.", "On: the default.", "on"),
    "train.amp": _f(
        "Mixed precision", "bf16 autocast.",
        "Off: full precision; slower and more memory (how much has not been measured here).",
        "On: what every speed figure was measured with.", "on"),
    "train.checkpoint_minutes": _f(
        "Checkpoint every", "How often the full training state, replay buffer included, is saved.",
        "Less lost if the run dies; each save writes about a gigabyte.", "Fewer saves, more to lose.",
        "10 to 60", min=0.5, max=600, step=1),
    "train.eval_minutes": _f(
        "Evaluate every", "How often the network plays the baselines and is tested against the best so far.",
        "More readings and a quicker guard; each evaluation stops training for about 25 seconds (6x64) to a "
        "minute (10x128).",
        "Fewer readings; more time for training.",
        "10 to 30", min=0.5, max=600, step=1),
    "train.eval_pairs": _f(
        "Evaluation pairs", "Seat-swapped pairs per baseline in each evaluation.",
        "Quicker and vaguer: 250 pairs give about +/- 8 points.", "Slower and sharper: 1,000 pairs give about +/- 4.",
        "250 to 1,000", min=2, max=100_000, step=50),
    "train.gate_pairs": _f(
        "Gate pairs", "Pairs when testing a network against the current best.",
        "Promotions and the guard act on noisier results.", "Surer decisions, slower: 500 pairs give about +/- 5.5 points.",
        "300 to 1,000", min=2, max=100_000, step=50),
    "train.keep_checkpoints": _f(
        "Checkpoints kept", "Full checkpoints kept on disk. The best network is always kept.",
        "Less disk used.", "More history to go back to, at 20 to 75 MB each.",
        "3 to 10", min=1, max=1000, step=1),
}

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
    "train.guard_patience": "evaluations",
}

# The form asks for the name itself, and the app decides where runs live.
HIDDEN = {"run.name", "run.out_dir"}

_SECTION_TITLES = {"run": "Run", "model": "Network", "selfplay": "Self-play", "warmstart": "Warm start", "train": "Training"}

PRESETS = [
    {"name": "Default 6x64", "description": "About 57 games a second. The size to get working first.", "overrides": {}},
    {"name": "Larger 10x128", "description": "About 18 games a second. Try it once the default has clearly learned something.",
     "overrides": {"model.blocks": 10, "model.channels": 128}},
    {"name": "Wide 12x192", "description": "About 8.5 games a second: a weekend run. Fewer candidates and a shorter exploration decay to suit.",
     "overrides": {"model.blocks": 12, "model.channels": 192, "selfplay.max_candidates": 32,
                   "selfplay.temperature_decay_games": 100_000}},
    {"name": "Quick test (30 minutes)", "description": "A short run to check everything works. Too short to learn much.",
     "overrides": {"run.hours": 0.5, "warmstart.games": 20_000, "warmstart.steps": 400,
                   "selfplay.temperature_decay_games": 50_000, "train.eval_minutes": 10.0}},
]

# Speeds on the RTX 5070 Ti (docs/BENCHMARKS.md). 6x64 and 10x128 are whole-run averages from the
# first two real runs (2026-11-13: 450,005 games in 2.19 h and 441,805 games in 6.67 h, with a
# replay ratio of 4; the current default of 1 leaves more of the GPU for self-play, so expect
# somewhat more). The larger sizes are worked out from separately measured speeds.
SPEED = [
    {"network": "6x64", "parameters": 1_055_056, "games_per_second": 57, "measured": True, "vram_gb": 1.4},
    {"network": "10x128", "parameters": 3_655_316, "games_per_second": 18, "measured": True, "vram_gb": 4.1},
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
            described = _FIELDS[key]
            type_name = _TYPE_NAMES[field.type if isinstance(field.type, str) else field.type.__name__]
            options = described.get("options")
            fields.append({
                "key": key,
                "label": described["label"],
                "type": "choice" if options else type_name,
                "default": getattr(section, field.name),
                "min": described.get("min"),
                "max": described.get("max"),
                "step": described.get("step"),
                "multiple_of": described.get("multiple_of"),
                "unit": _UNITS.get(key),
                "options": options,
                "advanced": described.get("advanced", True),
                "locked_on_resume": described.get("locked", False),
                "help": described["help"],
                "effects": {"lower": described["lower"], "higher": described["higher"], "typical": described["typical"]},
            })
        sections.append({"name": section_field.name, "title": _SECTION_TITLES[section_field.name], "fields": fields})
    return {"sections": sections, "presets": PRESETS, "speed": SPEED}


def described_keys() -> set[str]:
    return set(_FIELDS)


def _shown(default) -> str:
    if default is True:
        return "on"
    if default is False:
        return "off"
    if default == "":
        return "(empty)"
    return f"{default:,}" if isinstance(default, int) else str(default)


def guide_markdown() -> str:
    """The tuning guide as one Markdown table per section. docs/TUNING.md holds its output, and a
    test keeps the two the same."""
    lines = []
    for section in schema()["sections"]:
        lines += [f"### {section['title']} (`[{section['name']}]`)", "",
                  "| Setting | Default | What it does | Lower | Higher | Worth trying |", "|---|---|---|---|---|---|"]
        for field in section["fields"]:
            cells = [f"**{field['label']}**<br>`{field['key']}`", _shown(field["default"]), field["help"],
                     field["effects"]["lower"], field["effects"]["higher"], field["effects"]["typical"]]
            lines.append("| " + " | ".join(cell.replace("|", "/") for cell in cells) + " |")
        lines.append("")
    return "\n".join(lines)


GUIDE_MARKER = "<!-- The tables below are generated: python -m scrabble_ai.app.schema --write -->"


def write_guide(path) -> None:
    """Replace everything after GUIDE_MARKER in docs/TUNING.md with the current tables."""
    text = path.read_text(encoding="utf-8")
    head, marker, _ = text.partition(GUIDE_MARKER)
    if not marker:
        raise SystemExit(f"{path} has no line reading {GUIDE_MARKER!r}")
    path.write_text(f"{head}{marker}\n\n{guide_markdown()}", encoding="utf-8", newline="\n")


if __name__ == "__main__":  # prints the tables; --write puts them into docs/TUNING.md
    import sys

    if "--write" in sys.argv[1:]:
        from scrabble_ai.lexicon import REPO_ROOT

        write_guide(REPO_ROOT / "docs" / "TUNING.md")
    else:
        print(guide_markdown())
