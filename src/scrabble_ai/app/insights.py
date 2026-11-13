"""Numbers, headlines and notable events for the post-training report and the match page.

`run_report` turns a run's metrics records (the lines of its metrics.jsonl, in order) into the
JSON of GET /api/runs/<name>/report. `match_summary` turns the columnar games of a match into the
`summary` and `highlights` of GET /api/matches/<id>. docs/APP_API.md has both shapes.

Two rules hold throughout:

- Everything stated is a fact of the data. An evaluation is called significant only when its 95%
  interval excludes zero, and a run that has not beaten greedy-leave is never described as having
  done so. When the data is too thin to support a number or a sentence, it is null or left out.
- Odd input never raises. Records that are not dicts, missing fields, NaN, strings where numbers
  belong and the older key layouts are read where possible and skipped otherwise.

Pure Python and NumPy. The paired match statistics come from `scrabble_ai.evaluate.summarize`,
imported only when a match is summarized, because that module loads torch and the engine.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

BASELINE = "greedy-leave"

# ---- thresholds ------------------------------------------------------------------------------

# Regression: the drop from one evaluation against greedy-leave to the next must exceed
# REGRESSION_FACTOR x sqrt(ci_prev^2 + ci_cur^2). That root-sum-square is the 95% half-width of the
# difference of two independent estimates; 1.3 times it is about 2.5 standard errors (one-sided
# p ~ 0.005), so across the 25 to 50 evaluations of a long run ordinary noise almost never trips
# it. Evaluations reuse the same seeds, which correlates consecutive results and makes the test
# more conservative still.
REGRESSION_FACTOR = 1.3

# Plateau: at least this many evaluations against greedy-leave, and the best of them (the earliest
# on ties) lies before the last third. Fewer evaluations say too little about a trend.
PLATEAU_MIN_EVALS = 8

# Loss spike: each train record's loss against the median of the previous LOSS_WINDOW finite losses
# (at least LOSS_MIN_HISTORY of them). A spike is both LOSS_RATIO times that median and LOSS_MADS
# robust standard deviations (1.4826 x MAD) above it. The logged loss is an average over about a
# minute of steps, so it is smooth: ordinary wobble is a percent or two, while a 30% jump means
# something happened (an exploding update, a bad resume, a learning rate too high). The MAD
# condition stops a naturally noisy loss from tripping the ratio alone.
LOSS_WINDOW = 20
LOSS_MIN_HISTORY = 5
LOSS_RATIO = 1.3
LOSS_MADS = 6.0
MAX_SPIKE_EVENTS = 5

# Slowdown: self-play reports below SLOW_FRACTION of the run's median games/s for at least
# SLOW_MIN_RECORDS consecutive reports (about three minutes). A single slow report is routine: the
# window after an evaluation includes the evaluation's own time, and the last report of a session
# can cover a few seconds. SLOW_MIN_HISTORY reports are needed to know what is typical.
SLOW_FRACTION = 0.75
SLOW_MIN_RECORDS = 3
SLOW_MIN_HISTORY = 6
MAX_SLOWDOWN_EVENTS = 3

# Promotions: more than this many become first, one summary, last.
MAX_PROMOTION_EVENTS = 8

# Match highlights.
MIN_BINGOS_HIGHLIGHT = 2  # one bingo in a game is ordinary
MIN_EXCHANGES_HIGHLIGHT = 3  # one player exchanging three or more times in a game is unusual
MAX_HIGHLIGHTS = 8

_TROUBLE = re.compile(r"error|fail|crash|exception|kill", re.IGNORECASE)


# ---- reading values --------------------------------------------------------------------------


def _raw(value: Any) -> float | None:
    """A float that may be NaN or infinite, or None when the value is not a number at all."""
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            return float(value.strip())
    except (OverflowError, ValueError):
        return None
    return None


def _num(value: Any) -> float | None:
    """A finite float, or None for anything missing, non-numeric or not finite."""
    f = _raw(value)
    return f if f is not None and math.isfinite(f) else None


def _first(*values: Any) -> float | None:
    for value in values:
        f = _num(value)
        if f is not None:
            return f
    return None


def _flag(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    return None


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _cfg(config: Any, section: str, key: str) -> float | None:
    """A number from the run's config, nested ({"train": {"eval_minutes": 20}}) or flat."""
    if not isinstance(config, dict):
        return None
    part = config.get(section)
    if isinstance(part, dict) and key in part:
        return _num(part.get(key))
    return _num(config.get(f"{section}.{key}"))


# ---- formatting ------------------------------------------------------------------------------


def _round(value: Any, digits: int = 3) -> float | None:
    f = _num(value)
    return None if f is None else round(f, digits)


def _int(value: Any) -> int | None:
    f = _num(value)
    return None if f is None else int(round(f))


def _signed(x: float) -> str:
    return f"{round(x, 1) + 0.0:+.1f}"  # + 0.0 turns -0.0 into 0.0


def _pm(spread: float, ci: float | None) -> str:
    return f"{_signed(spread)} +/- {ci:.1f}" if ci is not None else f"{_signed(spread)} (no interval)"


def _count(x: float) -> str:
    return f"{int(round(x)):,}"


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def _duration(hours: float) -> str:
    minutes = hours * 60
    if minutes < 1:
        return "under a minute"
    if minutes < 59.5:
        return _plural(int(round(minutes)), "minute")
    return f"{hours:.1f} hours"


# ---- run records -----------------------------------------------------------------------------


@dataclass
class _Pos:
    """Where a record sits in the run. Missing values carry forward from earlier records."""

    games: float
    hours: float
    step: float | None


@dataclass
class _Eval:
    index: int
    pos: _Pos
    opponent: str
    spread: float
    ci: float | None
    win_rate: float | None
    pairs: float | None

    @property
    def above(self) -> bool:
        """The 95% interval lies entirely above zero."""
        return self.ci is not None and self.spread - self.ci > 0

    @property
    def below(self) -> bool:
        return self.ci is not None and self.spread + self.ci < 0

    def figure(self) -> str:
        return _pm(self.spread, self.ci)


@dataclass
class _Gate:
    index: int
    pos: _Pos
    spread: float | None
    ci: float | None
    pairs: float | None
    best_step: float | None
    promoted: bool | None

    @property
    def first(self) -> bool:
        """The run's first evaluated network: it becomes the best without playing a gate match
        (the trainer logs it with no spread and zero pairs)."""
        return bool(self.promoted) and self.spread is None and not self.pairs


def _parse_eval(index: int, record: dict, pos: _Pos) -> _Eval | None:
    """Both layouts: the contract's `spread`, `spread_ci`, `win_rate`, and the older
    `spread_vs_<opponent>`, `spread_ci_vs_<opponent>`, `win_rate_vs_<opponent>`."""
    opponent = _text(record.get("opponent"))
    if opponent is None:
        opponent = next((k[len("spread_vs_"):] for k in record
                         if isinstance(k, str) and k.startswith("spread_vs_") and k != "spread_vs_best"), None)
    if not opponent:
        return None
    spread = _first(record.get("spread"), record.get(f"spread_vs_{opponent}"))
    if spread is None:
        return None
    ci = _first(record.get("spread_ci"), record.get(f"spread_ci_vs_{opponent}"))
    if ci is not None and ci < 0:
        ci = None
    return _Eval(index, pos, opponent, spread, ci,
                 _first(record.get("win_rate"), record.get(f"win_rate_vs_{opponent}")), _num(record.get("pairs")))


def _parse_gate(index: int, record: dict, pos: _Pos) -> _Gate:
    ci = _num(record.get("spread_ci"))
    return _Gate(index, pos, _first(record.get("spread"), record.get("spread_vs_best")),
                 ci if ci is not None and ci >= 0 else None, _num(record.get("pairs")),
                 _num(record.get("best_step")), _flag(record.get("promoted")))


def _as_dicts(records: Any) -> list[dict]:
    out: list[dict] = []
    if records is None:
        return out
    try:
        for record in records:
            if isinstance(record, dict):
                out.append(record)
    except Exception:  # an unreadable iterable gives what was read before it failed
        pass
    return out


class _Run:
    """The records of one run, parsed once."""

    def __init__(self, records: Any, config: Any):
        self.config = config
        self.recs = _as_dicts(records)
        self.kinds = [r.get("kind") if isinstance(r.get("kind"), str) else None for r in self.recs]
        self.pos: list[_Pos] = []
        games, hours, step = 0.0, 0.0, None
        for r in self.recs:
            g, h, s = _num(r.get("games")), _num(r.get("hours")), _num(r.get("step"))
            games = g if g is not None else games
            hours = h if h is not None else hours
            step = s if s is not None else step
            self.pos.append(_Pos(games, hours, step))
        self.evals: list[_Eval] = []
        self.gates: list[_Gate] = []
        for i, (r, kind) in enumerate(zip(self.recs, self.kinds)):
            if kind == "eval":
                e = _parse_eval(i, r, self.pos[i])
                if e is not None:
                    self.evals.append(e)
            elif kind == "gate":
                self.gates.append(_parse_gate(i, r, self.pos[i]))
        self.baseline = [e for e in self.evals if e.opponent == BASELINE]
        self.warm_finished = self._warm_finished()
        self.selfplay_start = self._selfplay_start()

    def indices(self, kind: str) -> list[int]:
        return [i for i, k in enumerate(self.kinds) if k == kind]

    def peak(self, key: str) -> float | None:
        values = [v for v in (_num(r.get(key)) for r in self.recs) if v is not None]
        return max(values) if values else None

    def warm_games(self) -> float:
        return sum(_num(self.recs[i].get("games_played")) or 0.0 for i in self.indices("warmstart"))

    def _warm_finished(self) -> bool:
        """A warm start is finished when the config's game count is reached, or when training or
        self-play follows it (the trainer only goes on once the warm start is complete)."""
        warm = self.indices("warmstart")
        if not warm:
            return False
        target = _cfg(self.config, "warmstart", "games")
        if target is not None and self.warm_games() >= target:
            return True
        return any(k in ("train", "selfplay", "eval", "gate") for k in self.kinds[warm[-1] + 1:])

    def _selfplay_marks(self) -> list[int]:
        return [i for i in self.indices("session") if self.recs[i].get("event") == "selfplay"]

    def _selfplay_start(self) -> tuple[int, _Pos] | None:
        """(record index, position) where self-play began. The trainer marks it with a `session`
        record; logs written before that record existed get the latest record before the first
        self-play report with fewer games than it (the first report comes a minute in)."""
        marks = self._selfplay_marks()
        if marks:
            return marks[0], self.pos[marks[0]]
        sp = self.indices("selfplay")
        if not sp:
            return None
        first = sp[0]
        g0 = _num(self.recs[first].get("games"))
        if g0 is not None:
            for j in range(first - 1, -1, -1):
                g = _num(self.recs[j].get("games"))
                if g is not None and g < g0:
                    return j, self.pos[j]
        return first, self.pos[first]

    def phase(self) -> str | None:
        """None (no records), "started" (only bookkeeping records), "warmstart", "pretraining" or
        "selfplay"."""
        if not self.recs:
            return None
        if (self.indices("selfplay") or self._selfplay_marks() or self.evals or self.gates
                or (self.peak("games") or 0) > 0):
            return "selfplay"
        if self.warm_finished or self.indices("train"):
            return "pretraining"
        if self.indices("warmstart"):
            return "warmstart"
        return "started"


def _event(run: _Run, index: int, kind: str, tone: str, title: str, detail: str,
           pos: _Pos | None = None) -> dict:
    pos = pos if pos is not None else run.pos[index]
    return {"games": int(round(pos.games)), "hours": round(pos.hours, 4), "kind": kind, "tone": tone,
            "title": title, "detail": detail, "_order": index}


# ---- summary ---------------------------------------------------------------------------------


def _eval_object(e: _Eval | None) -> dict | None:
    if e is None:
        return None
    return {"step": _int(e.pos.step), "games": _int(e.pos.games), "hours": _round(e.pos.hours, 4),
            "spread": _round(e.spread), "ci": _round(e.ci), "win_rate": _round(e.win_rate, 4),
            "pairs": _int(e.pairs)}


def _best(evals: list[_Eval]) -> _Eval | None:
    best = None
    for e in evals:
        if best is None or e.spread > best.spread:  # strict: ties keep the earliest
            best = e
    return best


def _mean_games_per_second(run: _Run) -> float | None:
    """Total self-play games over total self-play time. Each report covers the games since the
    previous one at its own rate, so its time is games / rate; this weights every report by the
    stretch it covers (a few-second report at the end of a session counts for a few seconds)."""
    total_games = total_seconds = 0.0
    rates: list[float] = []
    previous = run.selfplay_start[1].games if run.selfplay_start else 0.0
    for i in run.indices("selfplay"):
        r = run.recs[i]
        rate, games = _num(r.get("games_per_second")), _num(r.get("games"))
        if rate is not None and rate > 0:
            rates.append(rate)
            if games is not None and games - previous > 0:
                total_games += games - previous
                total_seconds += (games - previous) / rate
        if games is not None:
            previous = games  # a forced kill and resume can step back: that report is skipped
    if total_seconds > 0:
        return total_games / total_seconds
    return float(np.mean(rates)) if rates else None


def _value_rmse_gain(run: _Run) -> float | None:
    gaps = []
    for i in run.indices("selfplay"):
        value = _num(run.recs[i].get("value_rmse_points"))
        leave = _num(run.recs[i].get("leave_only_rmse_points"))
        if value is not None and leave is not None:
            gaps.append(leave - value)
    if not gaps:
        return None
    last = gaps[-max(1, math.ceil(len(gaps) / 4)):]
    return float(np.mean(last))


def _summary(run: _Run) -> dict:
    base = run.baseline
    first = next((e for e in base if e.above), None)
    flags = [g.promoted for g in run.gates if g.promoted is not None and not g.first]
    return {
        "games": _int(run.peak("games")),
        "hours": _round(run.peak("hours"), 4),
        "steps": _int(run.peak("step")),
        "mean_games_per_second": _round(_mean_games_per_second(run), 2),
        "best": _eval_object(_best(base)),
        "final": _eval_object(base[-1] if base else None),
        "first_significant": None if first is None else {
            "games": _int(first.pos.games), "hours": _round(first.pos.hours, 4), "step": _int(first.pos.step),
            "spread": _round(first.spread), "ci": _round(first.ci)},
        "promotions": sum(flags) if flags else None,
        "value_rmse_gain": _round(_value_rmse_gain(run)),
        "evaluations": len(base),
    }


_EMPTY_SUMMARY = {"games": None, "hours": None, "steps": None, "mean_games_per_second": None, "best": None,
                  "final": None, "first_significant": None, "promotions": None, "value_rmse_gain": None,
                  "evaluations": 0}


# ---- headline --------------------------------------------------------------------------------


def _headline(name: str, run: _Run) -> str:
    base = run.baseline
    if not base:
        return _headline_without_baseline(name, run)
    best, final = _best(base), base[-1]
    first = next((e for e in base if e.above), None)
    # The figure that showed the earlier win: the best if its interval cleared zero, else the first that did.
    won = best if best is not None and best.above else first

    if final.above:
        text = (f"{name} beats greedy-leave by {final.spread:.1f} points a game at its latest evaluation "
                f"({final.figure()}), a significant margin.")
        if best is not None and best is not final:
            text += f" Its best was {best.figure()} at {_count(best.pos.games)} games."
        return text

    if final.below:
        if won is not None:
            return (f"{name} is now significantly worse than greedy-leave, {final.figure()} points a game at its "
                    f"latest evaluation, although it was significantly better earlier ({won.figure()} at "
                    f"{_count(won.pos.games)} games).")
        text = f"{name} is significantly worse than greedy-leave: {final.figure()} points a game at its latest evaluation."
        if best is not None and best is not final:
            if best.spread < 0:
                text += f" Every evaluation so far has been behind it; the best was {best.figure()}."
            else:
                text += f" It has not been shown to beat it at any evaluation; the best was {best.figure()}."
        return text

    if won is not None:
        latest = "no longer clears zero" if final.ci is not None else "has no interval to judge it by"
        return (f"{name} beat greedy-leave significantly earlier in the run ({won.figure()} points a game at "
                f"{_count(won.pos.games)} games), but its latest evaluation, {final.figure()}, {latest}.")

    if best is None:  # cannot happen with a non-empty list; keeps type checkers quiet
        return f"{name} has not been shown to beat greedy-leave."
    if best.spread < 0:
        text = (f"{name} has not been shown to beat greedy-leave: every evaluation so far has been behind it, "
                f"the best being {best.figure()} points a game.")
    elif best.ci is None:
        text = (f"{name} has not been shown to beat greedy-leave: its best evaluation was {_signed(best.spread)} "
                f"points a game, from too few pairs for an interval.")
    else:
        text = (f"{name} has not been shown to beat greedy-leave: its best evaluation was {best.figure()} points "
                f"a game, an interval that includes zero.")
    if final is not best:
        text += f" The latest is {final.figure()}."
    return text


def _headline_without_baseline(name: str, run: _Run) -> str:
    if run.evals:
        latest = run.evals[-1]
        text = (f"{name} has no evaluation against greedy-leave yet; its latest, against {latest.opponent}, "
                f"is {latest.figure()} points a game")
        if latest.above:
            return text + ", a significant margin."
        if latest.below:
            return text + ", significantly behind."
        return text + (", an interval that includes zero." if latest.ci is not None else ".")

    phase = run.phase()
    if phase is None:
        return f"{name} has not logged anything yet."
    if phase == "started":
        return f"{name} has started but has not reported any results yet."
    if phase == "warmstart":
        done, target = run.warm_games(), _cfg(run.config, "warmstart", "games")
        if target:
            progress = f" ({_count(done)} of {_count(target)} games)" if done > 0 else ""
            return f"{name} has not finished its warm start{progress}, so there is no evaluation yet."
        # Without the config a lone warm-start record may be the whole warm start, with pretraining
        # about to report: say only what is certain.
        progress = f" ({_count(done)} warm-start games so far)" if done > 0 else ""
        return f"{name} has not started self-play yet{progress}, so there is no evaluation yet."
    if phase == "pretraining":
        steps = [_num(run.recs[i].get("step")) for i in run.indices("train")]
        steps = [s for s in steps if s is not None]
        target = _cfg(run.config, "warmstart", "steps")
        where = ""
        if steps:
            where = f" (pretraining has reached step {_count(max(steps))}"
            where += f" of {_count(target)})" if target else ")"
        return (f"{name} has finished its warm start but has not started self-play yet{where}, "
                f"so there is no evaluation yet.")
    games, hours = run.peak("games") or 0.0, run.peak("hours")
    text = f"{name} has played {_count(games)} self-play games"
    if hours is not None and hours > 0:
        text += f" in {_duration(hours)}"
    text += " but has not been evaluated yet."
    minutes = _cfg(run.config, "train", "eval_minutes")
    if minutes:
        text += f" Evaluations run every {minutes:g} minutes of self-play."
    return text


# ---- events ----------------------------------------------------------------------------------


def _phase_events(run: _Run) -> list[dict]:
    out = []
    warm = run.indices("warmstart")
    if warm and run.warm_finished:
        records = [run.recs[i] for i in warm]
        games = run.warm_games()
        positions = sum(_num(r.get("positions_added")) or 0.0 for r in records)
        seconds = [s for s in (_num(r.get("seconds")) for r in records) if s is not None]
        parts = []
        if games > 0:
            parts.append(f"{_count(games)} games")
        if positions > 0:
            parts.append(f"{_count(positions)} positions")
        detail = " and ".join(parts) if parts else "The static evaluator's games"
        detail += f" in {sum(seconds):.0f} s." if seconds else " by the static evaluator."
        out.append(_event(run, warm[-1], "phase", "info", "Warm start finished", detail))

    if run.selfplay_start is not None:
        index, pos = run.selfplay_start
        reports = run.indices("selfplay")
        first_report = reports[0] if reports else len(run.recs)
        pretrain = [_num(run.recs[i].get("step")) for i in run.indices("train") if i < max(index, first_report)
                    and (_num(run.recs[i].get("games")) or 0.0) <= pos.games]
        pretrain = [s for s in pretrain if s is not None and s > 0]
        detail = (f"Pretraining ran to step {_count(max(pretrain))}; the network then began playing itself."
                  if pretrain else "The network began playing itself.")
        out.append(_event(run, index, "phase", "info", "Self-play began", detail, pos))

    for i in run.indices("session"):
        if run.recs[i].get("event") == "start" and i > 0:
            pos = run.pos[i]
            step = f" and step {_count(pos.step)}" if pos.step is not None else ""
            out.append(_event(run, i, "phase", "info", "Run resumed",
                              f"A new session began at {_count(pos.games)} games{step}."))
    return out


def _significance_events(run: _Run) -> list[dict]:
    out = []

    def pairs(e: _Eval) -> str:
        return f" over {_count(e.pairs)} pairs" if e.pairs else ""

    above = next((e for e in run.baseline if e.above), None)
    if above is not None:
        out.append(_event(run, above.index, "significant", "good", "First significant win over greedy-leave",
                          f"{above.figure()} points a game{pairs(above)}, the first interval to clear zero."))
    below = next((e for e in run.baseline if e.below), None)
    if below is not None:
        out.append(_event(run, below.index, "significant", "warning", "Significantly worse than greedy-leave",
                          f"{below.figure()} points a game{pairs(below)}: the whole interval is below zero."))
    return out


def _promotion_events(run: _Run) -> list[dict]:
    out = [_event(run, g.index, "promotion", "info", "First network to beat",
                  "The first evaluated network became the best so far; later networks have to beat "
                  "the current best head to head to replace it.") for g in run.gates if g.first][:1]
    return out + _gate_wins(run)


def _gate_wins(run: _Run) -> list[dict]:
    promoted = [g for g in run.gates if g.promoted and not g.first]
    if len(promoted) > MAX_PROMOTION_EVENTS:
        first, middle, last = promoted[0], promoted[1:-1], promoted[-1]
        summary = _event(run, middle[0].index, "promotion", "good", f"{len(middle)} more promotions",
                         f"{len(middle)} further new best networks between {_count(middle[0].pos.games)} and "
                         f"{_count(middle[-1].pos.games)} games.")
        return [_promotion_event(run, first), summary, _promotion_event(run, last)]
    return [_promotion_event(run, g) for g in promoted]


def _promotion_event(run: _Run, gate: _Gate) -> dict:
    against = f"the best so far (step {_count(gate.best_step)})" if gate.best_step is not None else "the best so far"
    if gate.spread is not None:
        pairs = f" over {_count(gate.pairs)} pairs" if gate.pairs else ""
        detail = f"Beat {against} by {_pm(gate.spread, gate.ci)}{pairs} and replaced it."
    else:
        detail = f"Beat {against} and replaced it."
    return _event(run, gate.index, "promotion", "good", "New best network", detail)


def _regression_events(run: _Run) -> list[dict]:
    out = []
    for prev, cur in zip(run.baseline, run.baseline[1:]):
        if prev.ci is None or cur.ci is None:
            continue
        combined = math.hypot(prev.ci, cur.ci)
        drop = prev.spread - cur.spread
        if drop > REGRESSION_FACTOR * combined:
            out.append(_event(run, cur.index, "regression", "warning", "Evaluation dropped",
                              f"Fell {drop:.1f} points against greedy-leave, from {prev.figure()} to {cur.figure()}, "
                              f"more than the {combined:.1f} that noise between two evaluations would explain."))
    return out


def _plateau_events(run: _Run) -> list[dict]:
    base = run.baseline
    n = len(base)
    if n < PLATEAU_MIN_EVALS:
        return []
    best = _best(base)
    at = next(k for k, e in enumerate(base) if e is best)
    since = n - 1 - at
    if since < math.ceil(n / 3):
        return []
    last = base[-1]
    ago = max(0.0, last.pos.hours - best.pos.hours)
    return [_event(run, last.index, "plateau", "info", f"No new best in {since} evaluations",
                   f"The best evaluation against greedy-leave, {best.figure()} at {_count(best.pos.games)} games, "
                   f"has not been beaten in the {since} evaluations since ({_duration(ago)} and "
                   f"{_count(max(0.0, last.pos.games - best.pos.games))} games ago).")]


def _loss_events(run: _Run) -> list[dict]:
    out = []
    history: list[float] = []
    nonfinite: list[tuple[int, float]] = []
    episodes: list[list[tuple[int, float, float]]] = []  # runs of consecutive spike reports
    in_spike = False
    for i in run.indices("train"):
        loss = _raw(run.recs[i].get("loss"))
        if loss is None:
            continue
        if not math.isfinite(loss):
            nonfinite.append((i, loss))
            continue
        spike = False
        if len(history) >= LOSS_MIN_HISTORY:
            window = np.array(history[-LOSS_WINDOW:])
            median = float(np.median(window))
            spread = 1.4826 * float(np.median(np.abs(window - median)))
            spike = median > 0 and loss >= LOSS_RATIO * median and loss - median >= LOSS_MADS * spread
            if spike:
                if not in_spike:
                    episodes.append([])
                episodes[-1].append((i, loss, median))
        in_spike = spike
        history.append(loss)

    def spike_event(episode: list[tuple[int, float, float]]) -> dict:
        start, _, median = episode[0]
        peak = max(loss for _, loss, _ in episode)
        step = run.pos[start].step
        at = f" at step {_count(step)}" if step is not None else ""
        lasting = f" over {len(episode)} reports" if len(episode) > 1 else ""
        return _event(run, start, "loss_spike", "warning", "Loss spike",
                      f"Training loss reached {peak:.3f}{at}{lasting}, {peak / median:.1f} times its recent "
                      f"median of {median:.3f}.")

    if len(episodes) > MAX_SPIKE_EVENTS:
        middle = episodes[1:-1]
        out.append(spike_event(episodes[0]))
        out.append(_event(run, middle[0][0][0], "loss_spike", "warning", f"{len(middle)} more loss spikes",
                          f"The loss jumped above {LOSS_RATIO:g} times its recent median {len(middle)} more times "
                          f"between {_count(run.pos[middle[0][0][0]].games)} and "
                          f"{_count(run.pos[middle[-1][0][0]].games)} games."))
        out.append(spike_event(episodes[-1]))
    else:
        out.extend(spike_event(e) for e in episodes)

    if nonfinite:
        i, loss = nonfinite[0]
        what = "NaN" if math.isnan(loss) else "infinite"
        step = run.pos[i].step
        at = f" at step {_count(step)}" if step is not None else ""
        detail = f"The training loss was {what}{at}"
        detail += f", the first of {len(nonfinite)} reports with a non-finite loss." if len(nonfinite) > 1 else "."
        out.append(_event(run, i, "loss_spike", "warning", "Loss is not a number", detail))
    return out


def _slowdown_events(run: _Run) -> list[dict]:
    reports = [(i, rate) for i in run.indices("selfplay")
               if (rate := _num(run.recs[i].get("games_per_second"))) is not None and rate > 0]
    if len(reports) < SLOW_MIN_HISTORY:
        return []
    typical = float(np.median([rate for _, rate in reports]))
    stretches: list[tuple[int, int]] = []  # [start, end) into reports
    start = None
    for k, (_, rate) in enumerate(reports + [(-1, math.inf)]):
        if rate < SLOW_FRACTION * typical:
            start = k if start is None else start
        else:
            if start is not None and k - start >= SLOW_MIN_RECORDS:
                stretches.append((start, k))
            start = None
    if not stretches:
        return []

    def describe(a: int, b: int) -> tuple[float, str]:
        mean = float(np.mean([rate for _, rate in reports[a:b]]))
        before = run.pos[reports[a - 1][0]].hours if a > 0 else run.pos[reports[a][0]].hours
        minutes = (run.pos[reports[b - 1][0]].hours - before) * 60
        span = f" (about {_plural(int(round(minutes)), 'minute')})" if minutes >= 1 else ""
        return mean, span

    if len(stretches) > MAX_SLOWDOWN_EVENTS:
        a, b = max(stretches, key=lambda s: s[1] - s[0])
        mean, _ = describe(a, b)
        first = reports[stretches[0][0]][0]
        return [_event(run, first, "slowdown", "info", f"Self-play slowed down {len(stretches)} times",
                       f"{len(stretches)} stretches of {SLOW_MIN_RECORDS} or more reports ran below "
                       f"{SLOW_FRACTION:.0%} of the typical {typical:.1f} games/s; the longest, {b - a} reports at "
                       f"{mean:.1f} games/s, began at {_count(run.pos[reports[a][0]].games)} games.")]
    out = []
    for a, b in stretches:
        mean, span = describe(a, b)
        tail = ", and had not recovered by the last report" if b == len(reports) else ""
        out.append(_event(run, reports[a][0], "slowdown", "info", "Self-play slowed down",
                          f"Self-play averaged {mean:.1f} games/s over {b - a} consecutive reports{span}, against "
                          f"the run's typical {typical:.1f}{tail}."))
    return out


def _record_events(run: _Run) -> list[dict]:
    """The run's overall highest game score and single play in self-play, each once, where it
    happened. Every per-report new high would mostly be noise early in a run."""
    out = []
    for key, title, sentence in (
        ("max_game_score", "Highest self-play game score", "A self-play game reached {} points, the highest of the run."),
        ("max_play_score", "Highest-scoring self-play move", "A single self-play move scored {} points, the most of the run."),
    ):
        best = None
        for i in run.indices("selfplay"):
            value = _num(run.recs[i].get(key))
            if value is not None and (best is None or value > best[0]):
                best = (value, i)
        if best is not None and best[0] > 0:
            out.append(_event(run, best[1], "record", "info", title, sentence.format(_count(best[0]))))
    return out


def _stop_events(run: _Run) -> list[dict]:
    out = []
    for i in run.indices("session"):
        r = run.recs[i]
        if r.get("event") != "stop":
            continue
        reason = _text(r.get("reason"))
        pos = run.pos[i]
        title = "Run finished" if _flag(r.get("finished")) else "Session stopped"
        tone = "warning" if reason and _TROUBLE.search(reason) else "info"
        detail = f"Stopped at {_count(pos.games)} games after {_duration(pos.hours)}"
        detail += f": {reason}." if reason else "."
        out.append(_event(run, i, "stop", tone, title, detail))
    return out


_DETECTORS = (_phase_events, _significance_events, _promotion_events, _regression_events, _plateau_events,
              _loss_events, _slowdown_events, _record_events, _stop_events)


# ---- public: run report ----------------------------------------------------------------------


def run_report(name: str, records: list[dict], config: dict | None = None) -> dict:
    """The JSON of GET /api/runs/<name>/report. `records` are the run's metrics.jsonl lines in order.
    `config` (the run's settings, nested by section or flat "section.key") only refines wording."""
    name = str(name) if name not in (None, "") else "This run"
    with np.errstate(all="ignore"):  # absurd values give inf, which the readers turn into None
        run = _Run(records, config)
        try:
            summary = _summary(run)
        except Exception:  # never fail the page over a number
            summary = dict(_EMPTY_SUMMARY)
        events: list[dict] = []
        for detector in _DETECTORS:
            try:
                events.extend(detector(run))
            except Exception:  # odd data one detector cannot read costs only that detector's events
                continue
        try:
            headline = _headline(name, run)
        except Exception:
            headline = f"{name}: its metrics could not be summarized."
    events.sort(key=lambda e: (e["games"], e["hours"], e["_order"]))
    for e in events:
        del e["_order"]
    return {"headline": headline, "summary": summary, "events": events}


# ---- matches ---------------------------------------------------------------------------------


class _Stats(NamedTuple):
    spread: float
    spread_ci: float
    win_rate: float
    win_ci: float
    score_a: float
    score_b: float


def _win(spread: np.ndarray) -> np.ndarray:
    return (spread > 0) + 0.5 * (spread == 0)


def _summarize_here(spread_first: np.ndarray, spread_second: np.ndarray, score_a: np.ndarray,
                    score_b: np.ndarray) -> _Stats:
    """The same arithmetic as evaluate.summarize, used only if that module cannot be imported
    (for example while the engine is being rebuilt). A test keeps the two identical."""
    n = len(spread_first)
    per_pair = (spread_first + spread_second) / 2.0

    def half_width(values: np.ndarray) -> float:
        return 1.96 * float(values.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")

    win_per_pair = (_win(spread_first) + _win(spread_second)) / 2.0
    return _Stats(float(per_pair.mean()), half_width(per_pair), float(win_per_pair.mean()), half_width(win_per_pair),
                  float(score_a.mean()), float(score_b.mean()))


def _paired_stats(spread_first: np.ndarray, spread_second: np.ndarray, score_a: np.ndarray,
                  score_b: np.ndarray) -> _Stats:
    try:
        from scrabble_ai.evaluate import summarize
    except Exception:
        return _summarize_here(spread_first, spread_second, score_a, score_b)
    r = summarize("A", "B", spread_first, spread_second, score_a, score_b, np.zeros(1))
    return _Stats(r.spread, r.spread_ci, r.win_rate, r.win_ci, r.score_a, r.score_b)


class _Games:
    """The columnar games of a match, read defensively."""

    KEYS = ("seed", "a_first", "score_a", "score_b", "turns", "bingos_a", "bingos_b", "exchanges_a", "exchanges_b",
            "passes_a", "passes_b", "best_play_a", "best_play_b", "best_word_a", "best_word_b", "ended_by")

    def __init__(self, games: Any):
        source = games if isinstance(games, dict) else {}
        self.cols: dict[str, list | None] = {}
        for key in self.KEYS:
            value = source.get(key)
            try:
                self.cols[key] = None if value is None or isinstance(value, (str, bytes, dict)) else list(value)
            except TypeError:
                self.cols[key] = None
        self.n = min(len(self.cols["score_a"] or []), len(self.cols["score_b"] or []))

    def value(self, key: str, i: int) -> Any:
        col = self.cols.get(key)
        return col[i] if col is not None and 0 <= i < len(col) else None

    def num(self, key: str, i: int) -> float | None:
        return _num(self.value(key, i))

    def scores(self, i: int) -> tuple[float, float] | None:
        a, b = self.num("score_a", i), self.num("score_b", i)
        return None if a is None or b is None else (a, b)

    def a_first(self, i: int) -> bool:
        flag = _flag(self.value("a_first", i))
        return flag if flag is not None else i % 2 == 0  # the match runner's order: A first in even games

    def seed(self, i: int) -> int | None:
        v = self.value("seed", i)
        if isinstance(v, (bool, np.bool_)):
            return None
        try:
            if isinstance(v, (float, np.floating)):
                return int(v) if math.isfinite(v) and float(v).is_integer() else None
            return int(v) if isinstance(v, (int, np.integer, str)) else None
        except ValueError:
            return None

    def pairs(self) -> list[tuple[int, int, int]]:
        """(pair number, A-first game, A-second game) for each complete, consistent pair: games 2k
        and 2k+1, both scored, the same seed when seeds are given, and the seats really swapped."""
        out = []
        for k in range(self.n // 2):
            i, j = 2 * k, 2 * k + 1
            if self.scores(i) is None or self.scores(j) is None:
                continue
            si, sj = self.seed(i), self.seed(j)
            if si is not None and sj is not None and si != sj:
                continue
            fi, fj = self.a_first(i), self.a_first(j)
            if fi == fj:
                continue
            out.append((k, i, j) if fi else (k, j, i))
        return out

    def spread(self, i: int) -> float:
        a, b = self.scores(i)  # type: ignore[misc]
        return a - b


def _match_stats(g: _Games) -> dict | None:
    pairs = g.pairs()
    if len(pairs) < 2:
        return None
    firsts = [p[1] for p in pairs]
    seconds = [p[2] for p in pairs]
    played = firsts + seconds
    spread_first = np.array([g.spread(i) for i in firsts], np.float64)
    spread_second = np.array([g.spread(i) for i in seconds], np.float64)
    score_a = np.array([g.scores(i)[0] for i in played], np.float64)  # type: ignore[index]
    score_b = np.array([g.scores(i)[1] for i in played], np.float64)  # type: ignore[index]
    stats = _paired_stats(spread_first, spread_second, score_a, score_b)

    def mean_of(key: str) -> float | None:
        values = [v for v in (g.num(key, i) for i in played) if v is not None]
        return float(np.mean(values)) if values else None

    endings = [e for e in (_text(g.value("ended_by", i)) for i in played) if e is not None]
    significant = math.isfinite(stats.spread_ci) and abs(stats.spread) > stats.spread_ci
    return {
        "spread": _round(stats.spread), "spread_ci": _round(stats.spread_ci),
        "win_rate": _round(stats.win_rate, 4), "win_ci": _round(stats.win_ci, 4),
        "significant": bool(significant),
        "score_a": _round(stats.score_a, 2), "score_b": _round(stats.score_b, 2),
        "turns": _round(mean_of("turns"), 2),
        "a_first": {"spread": _round(spread_first.mean()), "win_rate": _round(_win(spread_first).mean(), 4)},
        "a_second": {"spread": _round(spread_second.mean()), "win_rate": _round(_win(spread_second).mean(), 4)},
        "bingos_a": _round(mean_of("bingos_a")), "bingos_b": _round(mean_of("bingos_b")),
        "exchanges_a": _round(mean_of("exchanges_a")), "exchanges_b": _round(mean_of("exchanges_b")),
        "passes_a": _round(mean_of("passes_a")), "passes_b": _round(mean_of("passes_b")),
        "ended_by_going_out": _round(sum(e == "out" for e in endings) / len(endings), 4) if endings else None,
        "pairs": len(pairs),
    }


def _highlights(g: _Games, a: str, b: str) -> list[dict]:
    out: list[dict] = []
    scored = [i for i in range(g.n) if g.scores(i) is not None]
    names = {"a": a, "b": b}

    def result(i: int) -> str:
        sa, sb = g.scores(i)  # type: ignore[misc]
        if sa == sb:
            return f"tied at {_count(sa)}"
        winner, high, low = (a, sa, sb) if sa > sb else (b, sb, sa)
        return f"{winner} won {_count(high)}-{_count(low)}"

    def add(title: str, detail: str, game: int) -> None:
        out.append({"title": title, "detail": detail, "game": int(game)})

    # Highest-scoring single play, with the word.
    play = None
    for i in range(g.n):
        for side in ("a", "b"):
            v = g.num(f"best_play_{side}", i)
            if v is not None and (play is None or v > play[0]):
                play = (v, i, side)
    if play is not None and play[0] > 0:
        v, i, side = play
        word = _text(g.value(f"best_word_{side}", i))
        add("Highest-scoring play",
            f"{word} for {_count(v)} points by {names[side]}." if word else f"A {_count(v)}-point play by {names[side]}.", i)

    # Biggest win, and the highest score by one player (folded in when it is the same game).
    margins = {i: abs(g.spread(i)) for i in scored}
    win_game = max(scored, key=lambda i: (margins[i], -i)) if scored else None
    high = None
    for i in scored:
        for side, v in zip(("a", "b"), g.scores(i)):  # type: ignore[arg-type]
            if high is None or v > high[0]:
                high = (v, i, side)
    if win_game is not None and margins[win_game] > 0:
        detail = f"{result(win_game)}, a margin of {_count(margins[win_game])} points"
        if high is not None and high[1] == win_game and max(g.scores(win_game)) == high[0]:  # type: ignore[type-var]
            detail += f"; {_count(high[0])} is also the highest score in the match"
            high = None
        add("Biggest win", detail + ".", win_game)  # starts with a name: never re-case it
    if high is not None and high[0] > 0:
        v, i, side = high
        other = "b" if side == "a" else "a"
        other_score = g.scores(i)[1 if side == "a" else 0]  # type: ignore[index]
        add("Highest score", f"{names[side]} scored {_count(v)} against {names[other]}'s {_count(other_score)}.", i)

    # Closest game (a tie if there is one).
    if scored:
        close = min(scored, key=lambda i: (margins[i], i))
        if close != win_game or margins[close] == 0:
            ties = sum(margins[i] == 0 for i in scored)
            if margins[close] == 0:
                detail = f"Tied at {_count(g.scores(close)[0])}"  # type: ignore[index]
                detail += f", one of {ties} tied games." if ties > 1 else "."
            else:
                detail = f"{result(close)}, by {_plural(int(round(margins[close])), 'point')}."
            add("Closest game", detail, close)

    # Most bingos in one game.
    bingo = None
    for i in range(g.n):
        ba, bb = g.num("bingos_a", i), g.num("bingos_b", i)
        if ba is not None and bb is not None and (bingo is None or ba + bb > bingo[0]):
            bingo = (ba + bb, i, ba, bb)
    if bingo is not None and bingo[0] >= MIN_BINGOS_HIGHLIGHT:
        total, i, ba, bb = bingo
        add("Most bingos", f"{_count(total)} bingos in one game: {_count(ba)} by {a} and {_count(bb)} by {b}.", i)

    # The pair whose two games differ most: same seed, seats swapped.
    swing = None
    for _, first, second in g.pairs():
        sf, ss = g.spread(first), g.spread(second)
        if swing is None or abs(sf - ss) > swing[0]:
            swing = (abs(sf - ss), min(first, second), sf, ss)
    if swing is not None and swing[0] > 0:
        size, i, sf, ss = swing

        def went(x: float) -> str:
            return f"won by {_count(x)}" if x > 0 else f"lost by {_count(-x)}" if x < 0 else "tied"

        add("Biggest swing in a pair",
            f"With the same seed and the seats swapped, {a} {went(sf)} moving first and {went(ss)} moving second, "
            f"a swing of {_count(size)} points.", i)

    # Longest game.
    turns = {i: t for i in scored if (t := g.num("turns", i)) is not None}
    if len(turns) >= 2 and max(turns.values()) > min(turns.values()):
        longest = max(turns, key=lambda i: (turns[i], -i))
        add("Longest game", f"{_count(turns[longest])} turns against a match average of "
                            f"{np.mean(list(turns.values())):.1f}; {result(longest)}.", longest)

    # Games that ended on six scoreless turns rather than by going out.
    scoreless = [i for i in scored if (_text(g.value("ended_by", i)) or "").lower() == "scoreless"]
    if scoreless:
        i = scoreless[0]
        sa, sb = g.scores(i)  # type: ignore[misc]
        detail = f"Ended on six scoreless turns in a row, with {a} on {_count(sa)} and {b} on {_count(sb)}"
        detail += f"; {len(scoreless)} of {len(scored)} games ended this way." if len(scoreless) > 1 else "."
        add("Ended without going out", detail, i)

    # Most exchanges by one player in one game, when unusual.
    swap = None
    for i in range(g.n):
        for side in ("a", "b"):
            v = g.num(f"exchanges_{side}", i)
            if v is not None and (swap is None or v > swap[0]):
                swap = (v, i, side)
    if swap is not None and swap[0] >= MIN_EXCHANGES_HIGHLIGHT:
        v, i, side = swap
        add("Most exchanges", f"{names[side]} exchanged {_count(v)} times in one game; {result(i)}.", i)

    return out[:MAX_HIGHLIGHTS]


def match_summary(games: dict[str, list], a_name: str, b_name: str) -> tuple[dict | None, list[dict]]:
    """(summary, highlights) of GET /api/matches/<id> from the columnar `games`. Works on a
    partial match; summary is None with fewer than two complete pairs.

    The summary is over the games of complete pairs only (a trailing unpaired game is left out),
    so `score_a - score_b` equals `spread`. Highlights use every finished game; the pair swing
    uses complete pairs. Spread is always from A's side."""
    try:
        g = _Games(games)
    except Exception:
        return None, []
    a = str(a_name) if a_name not in (None, "") else "Player A"
    b = str(b_name) if b_name not in (None, "") else "Player B"
    if a == b:  # a network against itself: keep the two sides apart
        a, b = f"{a} (A)", f"{b} (B)"
    with np.errstate(all="ignore"):
        try:
            summary = _match_stats(g)
        except Exception:
            summary = None
        try:
            highlights = _highlights(g, a, b)
        except Exception:
            highlights = []
    return summary, highlights
