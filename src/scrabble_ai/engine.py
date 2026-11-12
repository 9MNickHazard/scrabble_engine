"""Loading the Rust engine: the lexicon's compiled GADDAG and the leave values."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from scrabble_ai import _core
from scrabble_ai.lexicon import DEFAULT_CONFIG, REPO_ROOT, LexiconError

DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class Loaded:
    engine: _core.Engine
    leaves: _core.LeaveModel
    lexicon_id: str
    leaves_path: Path | None  # None when the all-zero model is used
    leaves_hash: str


def lexicon_entry(name: str = "", config_path: Path = DEFAULT_CONFIG) -> tuple[str, Path]:
    """Resolve a lexicon name (empty = the configured default) to (name, word-list path)."""
    with open(config_path, "rb") as fh:
        config = tomllib.load(fh)
    name = name or config["default"]
    try:
        entry = config["lexicons"][name]
    except KeyError:
        known = ", ".join(sorted(config.get("lexicons", {})))
        raise LexiconError(f"unknown lexicon {name!r}; configured: {known}") from None
    path = Path(entry["path"])
    if not path.is_absolute():
        path = config_path.resolve().parent.parent / path
    if not path.exists():
        raise LexiconError(
            f"lexicon file for {name!r} not found at {path}. Word lists are not in the repo "
            f"(copyright); put the file there yourself."
        )
    return name, path


def load_engine(lexicon: str = "") -> _core.Engine:
    """The engine for a lexicon. The first call for a new word list compiles its GADDAG (about a
    second) and caches it next to the word list."""
    name, path = lexicon_entry(lexicon)
    return _core.Engine(name, path, path.parent)


def default_leaves_path(lexicon_id: str) -> Path:
    return DATA_DIR / "leaves" / f"{lexicon_id}.leaves"


def leaves_hash(leaves: _core.LeaveModel) -> str:
    return hashlib.sha256(leaves.weights().tobytes()).hexdigest()[:12]


def load(lexicon: str = "", leaves: str = "", allow_missing_leaves: bool = False) -> Loaded:
    """Engine plus leave values. `leaves` is a file path, "zero" for the all-zero model, or empty
    for the lexicon's default file (`scrabble-ai fit-leaves` writes it)."""
    engine = load_engine(lexicon)
    if leaves == "zero":
        model, path = _core.LeaveModel.zero(), None
    else:
        path = Path(leaves) if leaves else default_leaves_path(engine.lexicon_id)
        if path.exists():
            model = _core.LeaveModel.load(path)
        elif allow_missing_leaves and not leaves:
            model, path = _core.LeaveModel.zero(), None
        else:
            raise FileNotFoundError(
                f"no leave values at {path}. Run `uv run scrabble-ai fit-leaves` first "
                f"(a few minutes), or pass leaves = \"zero\" to do without them."
            )
    return Loaded(engine, model, engine.lexicon_id, path, leaves_hash(model))
