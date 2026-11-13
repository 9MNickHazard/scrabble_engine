"""The training loop.

One process does everything: self-play (many games in lockstep, the network scoring every
candidate move in one batch), training on the replay buffer, periodic evaluation against the
baseline bots, and checkpoints that can be resumed.

A fresh run goes through three phases:
    1. warm start  the static evaluator (score + leave value) plays `warmstart.games` games, with
                   some randomness so poor moves are seen too
    2. pretrain    `warmstart.steps` training steps on those games
    3. self-play   the network plays itself; `train.replay_ratio` training samples are drawn for
                   every new position, until the time or game limit, a STOP file, or Ctrl+C

Files written to `<run.out_dir>/<run.name>/`:
    config.toml          the full configuration in force
    metrics.jsonl        one JSON object per log line, evaluation or checkpoint
    tb/                  TensorBoard event files
    checkpoints/         step_<n>.pt, plus latest.txt naming the newest and best.pt
    replay/              the replay buffer as .npy files (rewritten at each checkpoint)
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from scrabble_ai import engine as engine_module
from scrabble_ai.config import Config, config_from_dict
from scrabble_ai.evaluate import MatchResult, NetPlayer, eval_seeds, play_match
from scrabble_ai.features import FeatureBuilder, to_device
from scrabble_ai.lexicon import REPO_ROOT
from scrabble_ai.model import ValueNet, parameter_count
from scrabble_ai.replay import Replay
from scrabble_ai.selfplay import Actor, ActorStats, Scorer

CHECKPOINT_FORMAT = 1


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def config_toml(config: Config) -> str:
    lines = []
    for section, values in config.to_dict().items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            text = json.dumps(value) if isinstance(value, str) else str(value).lower() if isinstance(value, bool) else repr(value)
            lines.append(f"{key} = {text}")
        lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, write) -> None:
    """Write through a temporary file and rename, so a crash never leaves a half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as fh:
        write(fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def resolve_resume(path: str, run_dir: Path) -> Path:
    """`latest` means the checkpoint named in this run's latest.txt; otherwise a file path."""
    if path == "latest":
        pointer = run_dir / "checkpoints" / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"nothing to resume: {pointer} does not exist")
        return pointer.parent / pointer.read_text(encoding="utf-8").strip()
    return Path(path)


class Trainer:
    def __init__(self, config: Config, resume: Path | None = None, quiet: bool = False):
        self.config = config
        self.quiet = quiet
        run, train, model = config.run, config.train, config.model
        self.run_dir = Path(run.out_dir) / run.name
        if not self.run_dir.is_absolute():
            self.run_dir = REPO_ROOT / self.run_dir
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(run.device)
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("run.device is 'cuda' but PyTorch sees no GPU; run `uv run scrabble-ai gpu-check`")
            torch.cuda.set_per_process_memory_fraction(run.vram_fraction)
            torch.backends.cudnn.benchmark = True

        self.loaded = engine_module.load(run.lexicon, run.leaves)
        self.engine, self.leaves = self.loaded.engine, self.loaded.leaves

        torch.manual_seed(run.seed)
        self.rng = np.random.default_rng(run.seed)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(run.seed)

        self.net = ValueNet(model).to(self.device).to(memory_format=torch.channels_last)
        self.actor_net = copy.deepcopy(self.net).eval()
        self.best_net: ValueNet | None = None
        self.best_step = -1
        self.optimizer = self._new_optimizer()
        self.lr_scale = 1.0  # halved each time the guard goes back to the best network
        self.worse_streak = 0  # evaluations in a row significantly behind the best
        self.rollbacks = 0
        amp = train.amp and self.device.type == "cuda"
        self.features = FeatureBuilder(self.device, model.cross_checks, torch.bfloat16 if amp else torch.float32)
        self.replay = Replay(train.replay_capacity)

        self.step = 0  # training steps taken
        self.games = 0  # self-play games finished
        self.positions = 0  # self-play positions added to the replay
        self.elapsed = 0.0  # seconds of training wall-clock across all sessions
        self.phase = "warmstart" if config.warmstart.games > 0 else "selfplay"
        self.warm_games = 0  # warm-start games finished
        self.warm_next_seed = run.seed * 1_000_000_000  # seed of the next warm-start game
        self.samples_owed = 0.0
        self.window = ActorStats()
        self.losses: list[tuple[float, float, float, float, float]] = []

        # Self-play seeds for this run start well away from the evaluation seeds.
        self.seed_base = run.seed * 1_000_000_000
        self.actor = self._new_actor()
        if resume is not None:
            self._load(resume)

        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(self.run_dir / "tb"))
        except ImportError:
            self._say("tensorboard is not installed: metrics go to metrics.jsonl only")
        (self.run_dir / "config.toml").write_text(config_toml(config), encoding="utf-8")

    # ---- setup ---------------------------------------------------------------------------

    def _new_optimizer(self) -> torch.optim.Optimizer:
        train = self.config.train
        decay = [p for p in self.net.parameters() if p.ndim > 1]
        no_decay = [p for p in self.net.parameters() if p.ndim <= 1]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": train.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=train.learning_rate,
        )

    def _new_actor(self) -> Actor:
        c = self.config
        return Actor(self.engine, self.leaves, self.replay, c.selfplay.games, c.selfplay.max_candidates,
                     c.selfplay.max_exchanges, c.train.td_lambda, self.device,
                     first_seed=self.seed_base + 500_000_000, sampling_seed=c.run.seed + 1)

    def _scorer(self, net: ValueNet) -> Scorer:
        return Scorer(net, self.engine, self.config.train.value_scale, self.device, self.config.train.amp)

    def _say(self, text: str) -> None:
        if not self.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)

    def _log(self, kind: str, values: dict, tag: str | None = None) -> None:
        """Append a record to metrics.jsonl (docs/APP_API.md lists the kinds and fields) and
        mirror its numbers to TensorBoard under `tag` (default: the kind)."""
        record = {"kind": kind, "time": time.time(), "step": self.step, "games": self.games,
                  "positions": self.positions, "hours": round(self.elapsed / 3600, 4), **values}
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.writer.add_scalar(f"{tag or kind}/{key}", value, self.games if kind != "train" else self.step)

    # ---- learning ------------------------------------------------------------------------

    def learning_rate(self) -> float:
        train = self.config.train
        warm = min(1.0, (self.step + 1) / max(1, train.warmup_steps))
        return train.learning_rate * warm * self.lr_scale

    def train_step(self) -> None:
        train, model = self.config.train, self.config.model
        rows = self.replay.sample(train.batch_size, self.rng)
        cross = None
        if model.cross_checks:
            cross = np.empty((train.batch_size, 450), np.uint32)
            self.engine.cross_masks(rows["board"], cross)
        batch = to_device(rows["board"], rows["leave"], rows["unseen"], rows["meta"], rows["leave_value"], cross, self.device)
        if train.augment_transpose:
            flip = torch.rand(train.batch_size, device=self.device, generator=self.generator) < 0.5
            batch = self.features.transpose(batch, flip)
        planes, vector = self.features(batch)
        target = torch.from_numpy(rows["value_target"]).to(self.device)
        win_target = torch.from_numpy(rows["win_target"]).to(self.device)
        baseline = batch.leave_value if model.leave_baseline else torch.zeros_like(target)

        self.net.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train.amp and self.device.type == "cuda"):
            value, win_logit = self.net(planes, vector)
        value, win_logit = value.float(), win_logit.float()
        value_loss = torch.nn.functional.huber_loss(value, (target - baseline) / train.value_scale, delta=train.huber_delta)
        win_loss = torch.nn.functional.binary_cross_entropy_with_logits(win_logit, win_target)
        loss = value_loss + train.win_loss_weight * win_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), train.grad_clip)
        self.optimizer.step()
        self.step += 1

        with torch.no_grad():
            miss = value * train.value_scale + baseline - target
            error, squared = miss.abs().mean(), miss.square().mean()
        self.losses.append((loss.item(), value_loss.item(), win_loss.item(), error.item(), squared.item()))
        if self.step % train.actor_sync_steps == 0:
            self.actor_net.load_state_dict(self.net.state_dict())

    def temperature(self) -> float:
        s = self.config.selfplay
        progress = min(1.0, self.games / max(1, s.temperature_decay_games))
        return s.temperature_start + (s.temperature_end - s.temperature_start) * progress

    # ---- phases --------------------------------------------------------------------------

    def warm_start(self, should_stop) -> bool:
        """Fill the replay with games played by the static evaluator, then pretrain on them.

        Returns True when both parts are complete. If the run is stopped part-way, the progress
        counters (`warm_games`, `warm_next_seed`, `step`) say where a resumed run picks up, so
        nothing is skipped and no game seed is played twice.
        """
        warm = self.config.warmstart
        if self.warm_games < warm.games:
            slots = max(self.config.selfplay.games, 512)
            actor = Actor(self.engine, self.leaves, self.replay, slots, warm.max_candidates,
                          min(self.config.selfplay.max_exchanges, warm.max_candidates - 2),
                          self.config.train.td_lambda, self.device,
                          first_seed=self.warm_next_seed, sampling_seed=self.config.run.seed + 2 + self.warm_games)
            self._say(f"warm start: {warm.games - self.warm_games:,} games by the static evaluator "
                      f"(temperature {warm.temperature})")
            stats, start, last = ActorStats(), time.time(), time.time()
            while self.warm_games < warm.games and not should_stop():
                step = actor.step(None, warm.temperature)
                stats.merge(step)
                self.warm_games += step.games
                # Games still in progress are abandoned on a stop; their seeds are not reused.
                self.warm_next_seed = actor.next_seed
                if time.time() - last > 30:
                    last = time.time()
                    self._say(f"  {stats.games:,} games, {stats.games / (time.time() - start):.0f} games/s")
            seconds = time.time() - start
            self._say(f"warm start: {stats.games:,} games, {stats.positions:,} positions in {seconds:.0f} s "
                      f"({stats.games / max(seconds, 1e-9):.0f} games/s); replay holds {len(self.replay):,}")
            self._log("warmstart", {"games_played": stats.games, "positions_added": stats.positions, "seconds": seconds,
                                    "mean_score": stats.score_sum / max(1, 2 * stats.games)})
            if self.warm_games < warm.games:
                return False
        if len(self.replay) >= self.config.train.batch_size and self.step < warm.steps:
            self._say(f"pretraining: {warm.steps - self.step:,} steps")
            start = time.time()
            while self.step < warm.steps and not should_stop():
                self.train_step()
                if self.step % 500 == 0:
                    self._report_training()
            self._report_training()
            self._say(f"pretraining: reached step {self.step:,} in {time.time() - start:.0f} s")
            if self.step < warm.steps:
                return False
        self.actor_net.load_state_dict(self.net.state_dict())
        return True

    def _report_training(self) -> None:
        if not self.losses:
            return
        loss, value_loss, win_loss, error, squared = np.mean(self.losses, axis=0)
        self.losses.clear()
        # These errors are on positions drawn from the replay buffer, which the network may have
        # trained on before. Next to the self-play report's error on new games they show whether
        # it is learning or memorising.
        self._log("train", {"loss": float(loss), "value_loss": float(value_loss), "win_loss": float(win_loss),
                            "value_error_points": float(error), "value_rmse_points": float(squared) ** 0.5,
                            "learning_rate": self.learning_rate()})
        self._say(f"  step {self.step:,}: loss {loss:.4f} (value {value_loss:.4f}, win {win_loss:.4f}), "
                  f"mean value error {error:.1f} points")

    def _report_selfplay(self, seconds: float) -> None:
        w = self.window
        if w.games == 0:
            return
        n = max(1, w.positions)
        values = {
            "games_per_second": w.games / seconds,
            "positions_per_second": w.positions / seconds,
            "mean_score": w.score_sum / (2 * w.games),
            "mean_turns": w.turns / w.games,
            "mean_abs_spread": w.spread_abs_sum / w.games,
            "first_player_win_rate": w.first_player_wins / w.games,
            "temperature": self.temperature(),
            # Out-of-sample: predictions made during play against what then happened.
            "value_rmse_points": (w.value_sq_error / n) ** 0.5,
            "leave_only_rmse_points": (w.leave_sq_error / n) ** 0.5,
            "win_brier": w.win_sq_error / n,
            "replay_positions": len(self.replay),
            "replay_gb": self.replay.memory_bytes / 2**30,
            "bingos_per_game": w.bingos / w.games,
            "exchanges_per_game": w.exchanges / w.games,
            "passes_per_game": w.passes / w.games,
            "max_game_score": w.max_game_score,
            "max_play_score": w.max_play_score,
        }
        self._log("selfplay", values)
        self._say(f"games {self.games:,} ({values['games_per_second']:.1f}/s), step {self.step:,}, "
                  f"temp {values['temperature']:.2f}, score {values['mean_score']:.0f}, "
                  f"value RMSE {values['value_rmse_points']:.1f} vs leave-only {values['leave_only_rmse_points']:.1f}, "
                  f"replay {len(self.replay):,}")
        self.window = ActorStats()

    # ---- evaluation and gating -----------------------------------------------------------

    def evaluate(self) -> list[MatchResult]:
        """Play the current network against the baselines, then against the best so far."""
        train, sp = self.config.train, self.config.selfplay
        eval_net = copy.deepcopy(self.net).eval()
        current = NetPlayer(f"net@{self.step}", self._scorer(eval_net), sp.win_weight, sp.win_scale)
        results = []
        for bot in ("greedy-leave", "greedy"):
            result = play_match(self.engine, self.leaves, current, bot, eval_seeds(train.eval_pairs), self.device,
                                sp.max_candidates, sp.max_exchanges)
            results.append(result)
            self._say("eval: " + result.line())
            self._log("eval", {"opponent": bot, "spread": result.spread, "spread_ci": result.spread_ci,
                               "win_rate": result.win_rate, "win_ci": result.win_ci, "score_for": result.score_a,
                               "score_against": result.score_b, "pairs": result.pairs}, tag=f"eval_vs_{bot}")
        gate = None
        if self.best_net is None:
            promote, reason = True, "first evaluated network"
        else:
            best = NetPlayer(f"best@{self.best_step}", self._scorer(self.best_net), sp.win_weight, sp.win_scale)
            gate = play_match(self.engine, self.leaves, current, best, eval_seeds(train.gate_pairs, offset=1_000_000),
                              self.device, sp.max_candidates, sp.max_exchanges)
            results.append(gate)
            self._say("gate: " + gate.line())
            # Promote only on a win whose 95% interval excludes zero.
            promote, reason = gate.spread > gate.spread_ci, f"beat best@{self.best_step} by {gate.spread:+.1f} +/- {gate.spread_ci:.1f}"
            self._log("gate", {"spread": gate.spread, "spread_ci": gate.spread_ci, "win_rate": gate.win_rate,
                               "best_step": self.best_step, "promoted": bool(promote), "pairs": gate.pairs})
        if promote:
            self.best_net, self.best_step = eval_net, self.step
            atomic_write(self.checkpoint_dir / "best.pt", lambda fh: torch.save(self._model_file(eval_net), fh))
            self._say(f"new best network at step {self.step} ({reason})")
            if gate is None:
                self._log("gate", {"spread": None, "spread_ci": None, "win_rate": None, "best_step": self.step,
                                   "promoted": True, "pairs": 0})
            self._write_index()
            self.worse_streak = 0
        else:
            # Significantly behind the best: the whole interval is below zero.
            behind = gate is not None and gate.spread + gate.spread_ci < 0
            self.worse_streak = self.worse_streak + 1 if behind else 0
            if train.guard and self.worse_streak >= train.guard_patience:
                self._rollback(f"{self.worse_streak} evaluations in a row significantly behind best@{self.best_step} "
                               f"(latest {gate.spread:+.1f} +/- {gate.spread_ci:.1f})")
        return results

    def _rollback(self, reason: str) -> None:
        """The guard: training has made the network worse, so go back to the best one and carry on
        more carefully. The replay buffer is kept; the optimizer starts afresh."""
        self.net.load_state_dict(self.best_net.state_dict())
        self.actor_net.load_state_dict(self.best_net.state_dict())
        self.optimizer = self._new_optimizer()
        self.lr_scale = max(self.lr_scale * 0.5, 1 / 64)
        self.worse_streak = 0
        self.rollbacks += 1
        self.losses.clear()
        self._say(f"guard: back to the best network (step {self.best_step}), learning rate now "
                  f"{self.config.train.learning_rate * self.lr_scale:.2e}: {reason}")
        self._log("guard", {"action": "rollback", "to_step": self.best_step, "learning_rate_scale": self.lr_scale,
                            "reason": reason})

    def _write_index(self) -> None:
        """checkpoints/index.json: what the app needs to list this run's models without loading
        any weights."""
        files = [{"file": p.name, "step": int(p.stem.split("_")[1]), "size_bytes": p.stat().st_size,
                  "modified": p.stat().st_mtime} for p in sorted(self.checkpoint_dir.glob("step_*.pt"))]
        best = self.checkpoint_dir / "best.pt"
        index = {
            "network": f"{self.config.model.blocks}x{self.config.model.channels}",
            "parameters": parameter_count(self.net),
            "lexicon_id": self.loaded.lexicon_id,
            "leaves_hash": self.loaded.leaves_hash,
            "value_scale": self.config.train.value_scale,
            "checkpoints": files,
            "best": None if not best.exists() else {"file": "best.pt", "step": self.best_step,
                                                    "size_bytes": best.stat().st_size, "modified": best.stat().st_mtime},
        }
        atomic_write(self.checkpoint_dir / "index.json", lambda fh: fh.write(json.dumps(index, indent=1).encode("utf-8")))

    def _model_file(self, net: ValueNet) -> dict:
        """A network on its own, with what is needed to load and trust it."""
        return {"format": CHECKPOINT_FORMAT, "model": net.state_dict(), "config": self.config.to_dict(),
                "step": self.step, "games": self.games, "lexicon_id": self.loaded.lexicon_id,
                "leaves_hash": self.loaded.leaves_hash, "git_commit": git_commit()}

    # ---- checkpoints ---------------------------------------------------------------------

    def save(self) -> Path:
        replay_dir, staging = self.run_dir / "replay", self.run_dir / "replay.new"
        if staging.exists():
            shutil.rmtree(staging)
        self.replay.save(staging)
        if replay_dir.exists():
            shutil.rmtree(replay_dir)
        os.replace(staging, replay_dir)
        state = {
            **self._model_file(self.net),
            "actor_model": self.actor_net.state_dict(),
            "best_model": None if self.best_net is None else self.best_net.state_dict(),
            "best_step": self.best_step,
            "optimizer": self.optimizer.state_dict(),
            "positions": self.positions,
            "elapsed": self.elapsed,
            "phase": self.phase,
            "warm_games": self.warm_games,
            "warm_next_seed": self.warm_next_seed,
            "samples_owed": self.samples_owed,
            "lr_scale": self.lr_scale,
            "worse_streak": self.worse_streak,
            "rollbacks": self.rollbacks,
            "replay": self.replay.state(),
            "actor": self.actor.state(),
            "numpy_rng": self.rng.bit_generator.state,
            "torch_generator": self.generator.get_state(),
            "torch_version": torch.__version__,
        }
        path = self.checkpoint_dir / f"step_{self.step:09d}.pt"
        atomic_write(path, lambda fh: torch.save(state, fh))
        atomic_write(self.checkpoint_dir / "latest.txt", lambda fh: fh.write(path.name.encode("utf-8")))
        kept = sorted(self.checkpoint_dir.glob("step_*.pt"))
        for old in kept[: -self.config.train.keep_checkpoints]:
            old.unlink(missing_ok=True)
        self._log("checkpoint", {"file": path.name, "replay_positions": len(self.replay)})
        self._write_index()
        self._say(f"checkpoint saved: {path}")
        return path

    def _load(self, path: Path) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"{path} is not a checkpoint this version can read")
        if state["lexicon_id"] != self.loaded.lexicon_id:
            raise RuntimeError(f"checkpoint was trained with lexicon {state['lexicon_id']}, "
                               f"but the configured lexicon is {self.loaded.lexicon_id}")
        if state["leaves_hash"] != self.loaded.leaves_hash:
            raise RuntimeError("checkpoint was trained with different leave values than the configured file; "
                               "the network's baseline would no longer match")
        self.net.load_state_dict(state["model"])
        self.actor_net.load_state_dict(state["actor_model"])
        if state["best_model"] is not None:
            self.best_net = copy.deepcopy(self.net).eval()
            self.best_net.load_state_dict(state["best_model"])
        self.best_step = state["best_step"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.step, self.games, self.positions = state["step"], state["games"], state["positions"]
        self.elapsed, self.phase, self.samples_owed = state["elapsed"], state["phase"], state["samples_owed"]
        # Absent from checkpoints written before the guard existed.
        self.lr_scale, self.worse_streak = state.get("lr_scale", 1.0), state.get("worse_streak", 0)
        self.rollbacks = state.get("rollbacks", 0)
        self.warm_games, self.warm_next_seed = state["warm_games"], state["warm_next_seed"]
        self.rng.bit_generator.state = state["numpy_rng"]
        try:
            self.generator.set_state(state["torch_generator"].cpu())
        except RuntimeError:
            # Saved on another device (GPU vs CPU): start a new stream, fixed by seed and step.
            self.generator.manual_seed(self.config.run.seed + self.step)
        replay_dir = self.run_dir / "replay"
        if replay_dir.exists():
            self.replay.load(replay_dir, state["replay"])
        else:
            self._say("no saved replay buffer next to this checkpoint: starting with an empty one")
        self.actor.load_state(state["actor"])
        self._say(f"resumed {path.name}: step {self.step:,}, {self.games:,} games, replay {len(self.replay):,}")

    # ---- main loop -----------------------------------------------------------------------

    def run(self) -> None:
        run, train, sp = self.config.run, self.config.train, self.config.selfplay
        stop_file = self.run_dir / "STOP"
        stop_file.unlink(missing_ok=True)
        session_start = time.time()
        deadline = session_start + run.hours * 3600 if run.hours > 0 else float("inf")

        def should_stop() -> bool:
            return time.time() >= deadline or stop_file.exists()

        self._say(f"run '{run.name}' in {self.run_dir}")
        self._say(f"lexicon {self.loaded.lexicon_id}; leave values {self.loaded.leaves_path or 'all zero'}; "
                  f"network {self.config.model.blocks}x{self.config.model.channels}, {parameter_count(self.net):,} parameters; "
                  f"replay {self.replay.memory_bytes / 2**30:.1f} GB for {train.replay_capacity:,} positions")
        reason = "finished"
        last_tick = last_report = time.time()
        running_file = self.run_dir / "running.json"
        running_file.write_text(json.dumps({"pid": os.getpid(), "started": session_start}), encoding="utf-8")
        self._log("session", {"event": "start", "parameters": parameter_count(self.net),
                              "limit_hours": run.hours, "limit_games": run.max_games})
        try:
            if self.phase == "warmstart":
                finished = self.warm_start(should_stop)
                self.elapsed += time.time() - last_tick
                last_tick = time.time()
                if finished:
                    self.phase = "selfplay"
                    self.save()
            last_report = last_checkpoint = last_eval = time.time()
            scorer = self._scorer(self.actor_net)
            if self.phase == "selfplay" and not should_stop():
                self._log("session", {"event": "selfplay"})
            while True:
                if should_stop():
                    reason = "STOP file found" if stop_file.exists() else f"time limit of {run.hours} h reached"
                    break
                if run.max_games > 0 and self.games >= run.max_games:
                    reason = f"game limit of {run.max_games:,} reached"
                    break
                stats = self.actor.step(scorer, self.temperature(), sp.win_weight, sp.win_scale)
                self.games += stats.games
                self.positions += stats.positions
                self.window.merge(stats)
                if len(self.replay) >= train.min_replay:
                    self.samples_owed += stats.positions * train.replay_ratio
                    while self.samples_owed >= train.batch_size:
                        self.train_step()
                        self.samples_owed -= train.batch_size
                now = time.time()
                self.elapsed += now - last_tick
                last_tick = now
                if now - last_report >= 60:
                    self._report_training()
                    self._report_selfplay(now - last_report)
                    last_report = now
                if now - last_eval >= train.eval_minutes * 60:
                    self.evaluate()
                    last_eval = last_tick = time.time()
                if now - last_checkpoint >= train.checkpoint_minutes * 60:
                    self.save()
                    last_checkpoint = time.time()
        except KeyboardInterrupt:
            reason = "interrupted (Ctrl+C)"
        self._say(f"stopping: {reason}")
        self._report_training()
        self._report_selfplay(max(time.time() - last_report, 1e-9))
        if not reason.startswith("interrupted") and self.step > 0:
            # Close a normal stop with an evaluation, so every run ends with a measured result.
            self.actor_net.load_state_dict(self.net.state_dict())
            self.evaluate()
        self.save()
        finished = reason.startswith(("time limit", "game limit"))
        self._log("session", {"event": "stop", "reason": reason, "finished": finished})
        stop_file.unlink(missing_ok=True)
        running_file.unlink(missing_ok=True)
        if self.writer is not None:
            self.writer.close()
        self._say(f"done. Resume with: uv run scrabble-ai train --name {run.name} --resume latest")


def load_checkpoint_config(path: Path) -> Config:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return config_from_dict(state["config"])


def load_network(path: Path, device: torch.device) -> tuple[ValueNet, dict]:
    """Load a network from a full checkpoint or a best.pt file, for evaluation or analysis."""
    state = torch.load(path, map_location=device, weights_only=False)
    config = config_from_dict(state["config"])
    net = ValueNet(config.model).to(device).to(memory_format=torch.channels_last)
    net.load_state_dict(state["model"])
    return net.eval(), state
