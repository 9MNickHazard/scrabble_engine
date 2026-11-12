"""Word lists.

The lexicon is a config option (configs/lexicons.toml) so NWL2023 or CSW can be swapped in.
Word-list files are copyrighted and live in data/lexicon/, which is gitignored.

File format: one entry per line, the word is the first whitespace-separated token; anything
after it (definitions, inflections) is ignored. This is the format of the NASPA .txt lists,
and a plain one-word-per-line file also works.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "lexicons.toml"

MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 15  # the board is 15 squares wide
_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class Lexicon:
    name: str
    path: Path
    words: tuple[str, ...]  # upper case, sorted, unique
    sha256: str  # of the normalized word list, so it is independent of file formatting

    @property
    def id(self) -> str:
        """Stable identifier stored in checkpoints: a model is only valid for its lexicon."""
        return f"{self.name}-{self.sha256[:12]}"


def read_word_list(path: Path) -> tuple[str, ...]:
    """Parse a word-list file. Reads text only: no CSV/NaN inference of any kind.

    (The old project's CSVs had lost NA, NAN and NULL to a pandas round trip.)
    """
    words: set[str] = set()
    with open(path, encoding="utf-8-sig") as fh:
        for line_number, line in enumerate(fh, 1):
            tokens = line.split()
            if not tokens:
                continue
            word = tokens[0].upper()
            if not (MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH) or not _LETTERS.issuperset(word):
                raise LexiconError(
                    f"{path}:{line_number}: {tokens[0]!r} is not a word of "
                    f"{MIN_WORD_LENGTH}-{MAX_WORD_LENGTH} letters A-Z"
                )
            words.add(word)
    if not words:
        raise LexiconError(f"{path}: no words found")
    return tuple(sorted(words))


def load_lexicon(name: str | None = None, config_path: Path = DEFAULT_CONFIG) -> Lexicon:
    """Load a lexicon by its name in the config (default: the config's `default` entry)."""
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
    words = read_word_list(path)
    expected = entry.get("expected_words")
    if expected is not None and len(words) != expected:
        raise LexiconError(f"{path}: expected {expected:,} words, found {len(words):,}")
    digest = hashlib.sha256("\n".join(words).encode("ascii")).hexdigest()
    return Lexicon(name=name, path=path, words=words, sha256=digest)
