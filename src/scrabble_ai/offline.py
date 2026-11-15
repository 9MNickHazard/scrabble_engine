"""Training from saved games: `scrabble-ai train-games`.

The self-play trainer (`train.py`) has the network play itself and learn from its own games as
they are played. This one learns from a game set made beforehand (`games.py`): millions of games
the static evaluator played against itself. Each pass through the set takes **one** position from
every game, a different one each pass, and trains the network to say how the game ended for the
player who had just moved: win, draw or loss. A network that can tell which positions win then
picks, among the static evaluator's best moves, the one that leaves the position most likely to.

Why this shape (docs/PLAN.md has the sources):

- One position per game, because the positions of one game share its result and nearly the same
  board; a network shown all of them learns the games by heart. The first two self-play runs did.
- Win, draw or loss as the target, with the points still to gain as a helper output: the points
  of one game are mostly luck, and a network trained on them alone learned noise.
- Helper outputs (`model.aux_heads`): the opponent's next score and whether they place all seven
  tiles, and where the next two moves land. They are never used to choose a move; predicting them
  teaches the board-reading part of the network what a position invites. Each one's pull on the
  network is held to a fixed share of the main output's (`offline.aux_share`).
- The network that is evaluated and saved is a running average of the weights, which is steadier
  than the weights of any one step.

A run's folder looks like a self-play run's (config.toml, metrics.jsonl, checkpoints/ with
step files, best.pt and index.json), so everything that plays or lists networks works with it.
The best network is the one with the highest win rate against the static evaluator so far.
"""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scrabble_ai import _core
from scrabble_ai import engine as engine_module
from scrabble_ai import games as games_module
from scrabble_ai.config import Config
from scrabble_ai.evaluate import MatchResult, NetPlayer, eval_seeds, play_match
from scrabble_ai.features import N, SQUARES, FeatureBuilder, FeatureSet, engine_inputs, to_device
from scrabble_ai.lexicon import REPO_ROOT
from scrabble_ai.model import ValueNet, parameter_count
from scrabble_ai.selfplay import Scorer
from scrabble_ai.train import CHECKPOINT_FORMAT, atomic_write, config_toml, git_commit

EMA_DECAY = 0.999  # the averaged network follows the trained one over about a thousand steps
BALANCE_EVERY = 100  # steps between re-measurements of how hard each helper output pulls
BALANCE_AFTER = 200  # not before the main output has moved off its all-zero start
REPORT_SECONDS = 60.0
VALIDATION_POSITIONS = 20_000
HELPERS = ("spread", "next_score", "next_bingo", "next_squares")


class Prefetcher:
    """Builds the coming batches on another thread while the GPU works on the current one. The
    work is the engine's (reading games and replaying them), which runs without Python's lock."""

    def __init__(self, make, first: int, last: int, depth: int = 3):
        self.queue: queue.Queue = queue.Queue(depth)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(make, first, last), daemon=True)
        self.thread.start()

    def _run(self, make, first: int, last: int) -> None:
        try:
            for step in range(first, last):
                item = (step, make(step))
                while not self.stop.is_set():
                    try:
                        self.queue.put(item, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if self.stop.is_set():
                    return
        except BaseException as error:  # handed to the training thread, which raises it
            self.queue.put((None, error))

    def get(self) -> tuple[int, dict]:
        step, item = self.queue.get()
        if step is None:
            raise item
        return step, item

    def close(self) -> None:
        self.stop.set()
        while self.thread.is_alive():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                self.thread.join(timeout=0.05)


class GameTrainer:
    def __init__(self, config: Config, resume: Path | None = None, quiet: bool = False):
        self.config = config
        self.quiet = quiet
        run, model, off = config.run, config.model, config.offline
        self.run_dir = Path(run.out_dir) / run.name
        if not self.run_dir.is_absolute():
            self.run_dir = REPO_ROOT / self.run_dir
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(run.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("run.device is cuda but PyTorch cannot see a GPU")
            torch.cuda.set_per_process_memory_fraction(run.vram_fraction)
            torch.backends.cudnn.benchmark = True
        self.loaded = engine_module.load(run.lexicon, run.leaves)
        self.engine, self.leaves = self.loaded.engine, self.loaded.leaves
        if not off.games:
            raise games_module.GamesError("offline.games is not set: name the game set to train from")
        self.games_dir = games_module.set_dir(off.games)
        self.games, self.games_meta = games_module.open_games(self.games_dir, self.loaded)

        total = len(self.games)
        self.validation_games = min(off.validation_games, total // 10)
        self.train_games = total - self.validation_games
        self.batch = min(off.batch_size, self.train_games)
        self.steps_per_pass = max(1, self.train_games // self.batch)
        self.total_steps = max(1, math.ceil(off.passes * self.steps_per_pass))

        torch.manual_seed(run.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(run.seed)
        self.net = ValueNet(model).to(self.device).to(memory_format=torch.channels_last)
        self.average = copy.deepcopy(self.net).eval()
        for parameter in self.average.parameters():
            parameter.requires_grad_(False)
        self.amp = off.amp and self.device.type == "cuda"
        self.inputs = FeatureSet.of(model)
        self.features = FeatureBuilder(self.device, self.inputs, torch.bfloat16 if self.amp else torch.float32)
        decay = [p for p in self.net.parameters() if p.ndim > 1]
        no_decay = [p for p in self.net.parameters() if p.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": off.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=off.learning_rate,
        )
        self.weights = {name: (off.spread_weight if name == "spread" else off.aux_weight) for name in self._helpers()}
        self.step = 0
        self.elapsed = 0.0
        self.best_step: int | None = None
        self.best_win_rate: float | None = None
        self.losses: list[dict[str, float]] = []
        self._order: tuple[int, np.ndarray] | None = None
        self._validation: list[dict] | None = None
        self._validated_at: int | None = None  # the step of the last measurement on held-out games
        if resume is not None:
            self._load(resume)
        atomic_write(self.run_dir / "config.toml", lambda fh: fh.write(config_toml(config).encode("utf-8")))

    # ---- setup ---------------------------------------------------------------------------

    def _helpers(self) -> tuple[str, ...]:
        return HELPERS if self.config.model.aux_heads else ("spread",)

    def _say(self, text: str) -> None:
        if not self.quiet:
            print(text, flush=True)

    def _log(self, kind: str, values: dict) -> None:
        """Append a record to metrics.jsonl (docs/APP_API.md lists the kinds and fields).
        `games` and `positions` both count the positions trained on: one from a game each."""
        seen = self.step * self.batch
        record = {"kind": kind, "time": time.time(), "step": self.step, "games": seen, "positions": seen,
                  "hours": round(self.elapsed / 3600, 4), **values}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def _scorer(self, net: ValueNet) -> Scorer:
        sp = self.config.selfplay
        return Scorer(net, self.engine, self.config.train.value_scale, self.device, self.config.offline.amp,
                      win_weight=sp.win_weight, win_scale=sp.win_scale)

    # ---- data ----------------------------------------------------------------------------

    def _sample(self, games: np.ndarray, picks: np.ndarray) -> dict:
        """One position from each of `games`, as arrays, with what the engine works out from the
        boards alone (cross masks, openness)."""
        n = len(games)
        rows = {
            "board": np.empty((n, SQUARES), np.uint8), "leave": np.empty((n, _core.TILE_TYPES), np.uint8),
            "unseen": np.empty((n, _core.TILE_TYPES), np.uint8), "meta": np.empty((n, _core.META), np.int16),
            "leave_value": np.empty(n, np.float32), "targets": np.empty((n, _core.TARGETS), np.float32),
            "squares": np.empty((n, 2 * SQUARES), np.uint8), "turns": np.empty(n, np.int32),
        }
        self.games.sample(self.engine, self.leaves, games, picks, self.config.model.belief_samples, rows["board"], rows["leave"],
                          rows["unseen"], rows["meta"], rows["leave_value"], rows["targets"], rows["squares"], rows["turns"])
        rows["cross"], rows["open"] = engine_inputs(self.engine, self.inputs, rows["board"])
        return rows

    def _batch(self, step: int) -> dict:
        """The batch of training step `step`: a pure function of the step and the run's seed, so
        a resumed run carries on with exactly the batches it would have had."""
        number, index = divmod(step, self.steps_per_pass)
        if self._order is None or self._order[0] != number:
            order = np.random.default_rng([self.config.run.seed, number]).permutation(self.train_games)
            self._order = (number, order.astype(np.uint64))
        games = self._order[1][index * self.batch:(index + 1) * self.batch]
        picks = np.random.default_rng([self.config.run.seed, number, index, 1]).random(len(games), dtype=np.float32)
        return self._sample(games, picks)

    def _validation_batches(self) -> list[dict]:
        """Positions from the games held out of training: the same ones every time."""
        if self._validation is None:
            count = min(self.validation_games, VALIDATION_POSITIONS)
            games = (self.train_games + np.arange(count)).astype(np.uint64)
            picks = np.random.default_rng([self.config.run.seed, 77]).random(count, dtype=np.float32)
            size = max(1, min(self.batch, 2048))
            self._validation = [self._sample(games[i:i + size], picks[i:i + size]) for i in range(0, count, size)]
        return self._validation

    # ---- learning ------------------------------------------------------------------------

    def learning_rate(self) -> float:
        off = self.config.offline
        warm = min(1.0, (self.step + 1) / max(1, off.warmup_steps))
        progress = min(1.0, self.step / max(1, self.total_steps))
        return off.learning_rate * warm * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _forward(self, net: ValueNet, rows: dict, augment: bool) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(outputs, targets [B, TARGETS], squares [B, 2, 225], baseline [B])."""
        batch = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], rows["cross"],
                          self.device, rows["open"])
        targets = torch.from_numpy(rows["targets"]).to(self.device)
        squares = torch.from_numpy(rows["squares"]).to(self.device).view(-1, 2, N, N)
        if augment:
            flip = torch.rand(len(batch), device=self.device, generator=self.generator) < 0.5
            batch = self.features.transpose(batch, flip)
            squares = torch.where(flip.view(-1, 1, 1, 1), squares.transpose(2, 3), squares)
        planes, vector = self.features(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            out = net.outputs(planes, vector)
        baseline = batch.leave_value if self.config.model.leave_baseline else torch.zeros_like(batch.leave_value)
        return out, targets, squares.reshape(-1, 2, SQUARES).float(), baseline

    def _losses(self, out: dict, targets: torch.Tensor, squares: torch.Tensor, baseline: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every output's loss, each a mean over the positions it applies to."""
        scale = self.config.train.value_scale
        result = targets[:, _core.T_WIN]
        if self.net.wdl:
            classes = torch.where(result > 0.75, 0, torch.where(result > 0.25, 1, 2))
            win = F.cross_entropy(out["win"].float(), classes)
        else:
            win = F.binary_cross_entropy_with_logits(out["win"].float(), result)
        losses = {"win": win, "spread": F.huber_loss(out["value"].float(), (targets[:, _core.T_SPREAD] - baseline) / scale,
                                                    delta=self.config.train.huber_delta)}
        if self.config.model.aux_heads:
            def masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
                return (values * mask).sum() / mask.sum().clamp(min=1.0)

            has_opponent, has_own = targets[:, _core.T_HAS_OPP], targets[:, _core.T_HAS_OWN]
            turn = out["next_turn"].float()
            losses["next_score"] = masked(F.huber_loss(turn[:, 0], targets[:, _core.T_OPP_SCORE] / scale, reduction="none"), has_opponent)
            losses["next_bingo"] = masked(F.binary_cross_entropy_with_logits(turn[:, 1], targets[:, _core.T_OPP_BINGO], reduction="none"),
                                          has_opponent)
            per_map = F.binary_cross_entropy_with_logits(out["next_squares"].float(), squares, reduction="none").mean(dim=2)  # [B, 2]
            has = torch.stack([has_opponent, has_own], dim=1)
            losses["next_squares"] = masked(per_map, has)
        return losses

    def _rebalance(self, out: dict, losses: dict[str, torch.Tensor]) -> None:
        """Set each helper output's weight so that its gradient on the shared board features is
        `aux_share` of the main output's. Left to fixed weights, a helper that is easy to improve
        pulls the network harder than the output that matters."""
        share = self.config.offline.aux_share
        if share <= 0:
            return
        def pull(loss: torch.Tensor) -> float:
            (gradient,) = torch.autograd.grad(loss, out["trunk"], retain_graph=True, allow_unused=True)
            return 0.0 if gradient is None else float(gradient.float().norm())

        main = pull(losses["win"])
        if not main > 1e-12:
            return  # the main output has not moved yet: nothing to measure against
        for name in self._helpers():
            theirs = pull(losses[name])
            if theirs > 1e-12:
                target = min(100.0, max(0.01, share * main / theirs))
                self.weights[name] = 0.5 * self.weights[name] + 0.5 * target

    def train_step(self, rows: dict) -> None:
        off = self.config.offline
        self.net.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate()
        out, targets, squares, baseline = self._forward(self.net, rows, off.augment_transpose)
        losses = self._losses(out, targets, squares, baseline)
        if self.step >= BALANCE_AFTER and self.step % BALANCE_EVERY == 0:
            self._rebalance(out, losses)
        loss = losses["win"] + sum(self.weights[name] * losses[name] for name in self._helpers())
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if off.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), off.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            # The averaged network starts as a copy and then trails the trained one. Early on it
            # follows faster, so that it is not held back by the untrained start.
            decay = min(EMA_DECAY, (1.0 + self.step) / (10.0 + self.step))
            for averaged, trained in zip(self.average.parameters(), self.net.parameters()):
                averaged.lerp_(trained, 1.0 - decay)
        self.losses.append({"loss": loss.item(), **{name: value.item() for name, value in losses.items()}})

    def _report_training(self, seconds: float, steps: int) -> None:
        if not self.losses:
            return
        mean = {name: float(np.mean([entry[name] for entry in self.losses])) for name in self.losses[0]}
        self.losses.clear()
        rate = steps * self.batch / max(1e-9, seconds)
        self._log("train", {"loss": mean.pop("loss"), **{f"{name}_loss": value for name, value in mean.items()},
                            **{f"{name}_weight": round(weight, 4) for name, weight in self.weights.items()},
                            "learning_rate": self.learning_rate(), "positions_per_second": rate,
                            "pass": round(self.step / self.steps_per_pass, 3)})
        self._say(f"  step {self.step:,} of {self.total_steps:,} (pass {self.step / self.steps_per_pass:.2f}): "
                  f"win loss {mean['win']:.4f}, {rate:,.0f} positions/s")

    @torch.no_grad()
    def validate(self) -> dict | None:
        """The averaged network on positions from games it never trained on."""
        if self.validation_games == 0 or self._validated_at == self.step:
            return None
        self._validated_at = self.step
        self.average.eval()
        scale = self.config.train.value_scale
        sums: dict[str, float] = {}
        count = 0
        for rows in self._validation_batches():
            out, targets, squares, baseline = self._forward(self.average, rows, augment=False)
            n = len(targets)
            for name, value in self._losses(out, targets, squares, baseline).items():
                sums[f"{name}_loss"] = sums.get(f"{name}_loss", 0.0) + float(value) * n
            result = targets[:, _core.T_WIN]
            if self.average.wdl:
                p = torch.softmax(out["win"].float(), dim=1)
                share, right = p[:, 0] + 0.5 * p[:, 1], p.argmax(dim=1) == torch.where(result > 0.75, 0, torch.where(result > 0.25, 1, 2))
            else:
                share = torch.sigmoid(out["win"].float())
                right = (share > 0.5) == (result > 0.5)
            spread = targets[:, _core.T_SPREAD]
            extra = {"win_accuracy": right.float().sum(), "win_brier": (share - result).square().sum(),
                     "value_sq": (out["value"].float() * scale + baseline - spread).square().sum(),
                     "leave_sq": (baseline - spread).square().sum()}
            for name, value in extra.items():
                sums[name] = sums.get(name, 0.0) + float(value)
            count += n
        record = {name: (value / count) for name, value in sums.items() if name not in ("value_sq", "leave_sq")}
        record["value_rmse_points"] = math.sqrt(sums["value_sq"] / count)
        record["leave_only_rmse_points"] = math.sqrt(sums["leave_sq"] / count)
        record["positions_checked"] = count
        self._log("valid", record)
        self._say(f"  held-out games: win loss {record['win_loss']:.4f}, right {100 * record['win_accuracy']:.1f}% of the time, "
                  f"points error {record['value_rmse_points']:.1f} vs leave-only {record['leave_only_rmse_points']:.1f}")
        return record

    def evaluate(self, pairs: int) -> MatchResult:
        """The averaged network against the static evaluator, choosing as it is meant to."""
        sp = self.config.selfplay
        net = copy.deepcopy(self.average).eval()
        player = NetPlayer(f"net@{self.step}", self._scorer(net), sp.win_weight, sp.win_scale)
        result = play_match(self.engine, self.leaves, player, "greedy-leave", eval_seeds(pairs), self.device,
                            sp.max_candidates, sp.max_exchanges)
        self._say("eval: " + result.line())
        self._log("eval", {"opponent": "greedy-leave", "spread": result.spread, "spread_ci": result.spread_ci,
                           "win_rate": result.win_rate, "win_ci": result.win_ci, "score_for": result.score_a,
                           "score_against": result.score_b, "pairs": result.pairs})
        if self.best_win_rate is None or result.win_rate > self.best_win_rate:
            self.best_win_rate, self.best_step = result.win_rate, self.step
            atomic_write(self.checkpoint_dir / "best.pt", lambda fh: torch.save(self._model_file(net), fh))
            self._say(f"new best network at step {self.step}: wins {100 * result.win_rate:.1f}% against the static evaluator")
            self._log("best", {"win_rate": result.win_rate, "win_ci": result.win_ci, "pairs": result.pairs})
            self._write_index()
        return result

    # ---- checkpoints ---------------------------------------------------------------------

    def _model_file(self, net: ValueNet) -> dict:
        """A network on its own, with what is needed to load and trust it."""
        seen = self.step * self.batch
        return {"format": CHECKPOINT_FORMAT, "model": net.state_dict(), "config": self.config.to_dict(), "step": self.step,
                "games": seen, "lexicon_id": self.loaded.lexicon_id, "leaves_hash": self.loaded.leaves_hash,
                "git_commit": git_commit()}

    def _write_index(self) -> None:
        files = [{"file": p.name, "step": int(p.stem.split("_")[1]), "size_bytes": p.stat().st_size, "modified": p.stat().st_mtime}
                 for p in sorted(self.checkpoint_dir.glob("step_*.pt"))]
        best = self.checkpoint_dir / "best.pt"
        index = {
            "network": f"{self.config.model.blocks}x{self.config.model.channels}",
            "parameters": parameter_count(self.net),
            "lexicon_id": self.loaded.lexicon_id,
            "leaves_hash": self.loaded.leaves_hash,
            "value_scale": self.config.train.value_scale,
            "checkpoints": files,
            "best": None if not best.exists() else {"file": "best.pt", "step": self.best_step, "size_bytes": best.stat().st_size,
                                                    "modified": best.stat().st_mtime},
        }
        atomic_write(self.checkpoint_dir / "index.json", lambda fh: fh.write(json.dumps(index, indent=1).encode("utf-8")))

    def save(self) -> Path:
        """A step file holds the averaged network as `model`, which is the one to play, and the
        rest of what a resume needs."""
        state = {
            **self._model_file(self.average),
            "trained_model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "elapsed": self.elapsed,
            "weights": dict(self.weights),
            "best_step": self.best_step,
            "best_win_rate": self.best_win_rate,
            "games_set": {"name": self.games_meta["name"], "games": len(self.games), "first_seed": self.games_meta["first_seed"]},
            "torch_generator": self.generator.get_state(),
            "torch_version": torch.__version__,
        }
        path = self.checkpoint_dir / f"step_{self.step:09d}.pt"
        atomic_write(path, lambda fh: torch.save(state, fh))
        atomic_write(self.checkpoint_dir / "latest.txt", lambda fh: fh.write(path.name.encode("utf-8")))
        kept = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in kept[: -self.config.offline.keep_checkpoints]:
            old.unlink(missing_ok=True)
        self._log("checkpoint", {"file": path.name})
        self._write_index()
        self._say(f"checkpoint saved: {path}")
        return path

    def _load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT or "trained_model" not in state:
            raise RuntimeError(f"{path} is not a checkpoint of a run trained from games")
        if state["lexicon_id"] != self.loaded.lexicon_id:
            raise RuntimeError(f"checkpoint was trained with lexicon {state['lexicon_id']}, "
                               f"but the configured lexicon is {self.loaded.lexicon_id}")
        if state["leaves_hash"] != self.loaded.leaves_hash:
            raise RuntimeError("checkpoint was trained with different leave values than the configured file; "
                               "the network's baseline would no longer match")
        saved = state["games_set"]
        if (saved["first_seed"], saved["games"]) != (self.games_meta["first_seed"], len(self.games)):
            raise RuntimeError(f"checkpoint was trained on the game set {saved['name']} when it held {saved['games']:,} games; "
                               "the set has changed, so the run cannot carry on where it stopped")
        before = state["config"].get("offline", {})
        for key in ("batch_size", "validation_games"):
            if key in before and before[key] != getattr(self.config.offline, key):
                raise RuntimeError(f"offline.{key} was {before[key]:,} when this run began and cannot change on resume: "
                                   "it decides which games each step trains on and which are held out")
        self.net.load_state_dict(state["trained_model"])
        self.average.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.elapsed = state["step"], state["elapsed"]
        self.weights.update({name: weight for name, weight in state["weights"].items() if name in self.weights})
        self.best_step, self.best_win_rate = state["best_step"], state["best_win_rate"]
        try:
            self.generator.set_state(state["torch_generator"].cpu())
        except RuntimeError:
            self.generator.manual_seed(self.config.run.seed + self.step)  # saved on another device
        self._say(f"resumed {path.name}: step {self.step:,} of {self.total_steps:,}")

    # ---- main loop -----------------------------------------------------------------------

    def run(self) -> None:
        run, off = self.config.run, self.config.offline
        stop_file = self.run_dir / "STOP"
        session_start = time.time()
        limit = run.hours * 3600 if run.hours > 0 else None

        def stop_reason() -> str | None:
            if stop_file.exists():
                return "STOP file found"
            if limit is not None and time.time() - session_start >= limit:
                return f"time limit of {run.hours} h reached"
            return None

        from scrabble_ai.app.procs import clear_stale_stop

        clear_stale_stop(stop_file)  # one written while this process was starting up still counts
        first_step = self.step
        running = self.run_dir / "running.json"
        running.write_text(json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8")
        self._say(f"training from the game set {self.games_meta['name']}: {self.train_games:,} games to learn from, "
                  f"{self.validation_games:,} held out; network {self.config.model.blocks}x{self.config.model.channels}, "
                  f"{parameter_count(self.net):,} parameters; {self.total_steps:,} steps of {self.batch:,} "
                  f"({off.passes:g} passes) on {self.device}")
        self._log("session", {"event": "start", "mode": "games", "parameters": parameter_count(self.net),
                              "limit_hours": run.hours, "game_set": self.games_meta["name"], "train_games": self.train_games,
                              "validation_games": self.validation_games, "total_steps": self.total_steps,
                              "steps_per_pass": self.steps_per_pass, "batch": self.batch})
        reason, finished = None, False
        prefetch = Prefetcher(self._batch, self.step, self.total_steps)
        tick = time.time()
        last_report = last_eval = last_checkpoint = tick
        steps_since_report = 0
        try:
            while self.step < self.total_steps:
                reason = stop_reason()
                if reason is not None:
                    break
                _, rows = prefetch.get()
                self.train_step(rows)
                self.step += 1
                steps_since_report += 1
                now = time.time()
                self.elapsed += now - tick
                tick = now
                if now - last_report >= REPORT_SECONDS:
                    self._report_training(now - last_report, steps_since_report)
                    last_report, steps_since_report = now, 0
                if self.step % off.validate_steps == 0:
                    self.validate()
                if now - last_eval >= off.eval_minutes * 60:
                    self.evaluate(off.eval_pairs)
                    last_eval = time.time()
                if now - last_checkpoint >= off.checkpoint_minutes * 60:
                    self.save()
                    last_checkpoint = time.time()
                tick = time.time()  # validation, matches and saving are not training time
            else:
                reason, finished = f"all {self.total_steps:,} steps done", True
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            prefetch.close()
        try:
            self._say(f"stopping: {reason}")
            self._report_training(max(1e-9, time.time() - last_report), steps_since_report)
            # A session stopped before it trained a step has nothing new to measure.
            if reason != "interrupted" and (finished or self.step > first_step):
                self.validate()
                self.evaluate(off.final_pairs if finished else off.eval_pairs)
            self.save()
            self._log("session", {"event": "stop", "reason": reason, "finished": finished})
        finally:
            running.unlink(missing_ok=True)
            stop_file.unlink(missing_ok=True)
