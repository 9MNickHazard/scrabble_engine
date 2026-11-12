from pathlib import Path

import pytest

from scrabble_ai.lexicon import DEFAULT_CONFIG, LexiconError, load_lexicon, read_word_list


def write_config(tmp_path: Path, body: str) -> Path:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "lexicons.toml"
    config.write_text(body, encoding="utf-8")
    return config


def test_reads_first_token_and_ignores_definitions(tmp_path):
    path = tmp_path / "words.txt"
    path.write_text("AA rough, cindery lava [n AAS]\nqi the vital force\n\nZA\n", encoding="utf-8")
    assert read_word_list(path) == ("AA", "QI", "ZA")


def test_words_that_look_like_missing_values_survive(tmp_path):
    # NA, NAN and NULL are valid words; the old project's CSVs lost them to pandas.
    path = tmp_path / "words.txt"
    path.write_text("NA\nNAN\nNULL\nNONE\nTRUE\n", encoding="utf-8")
    assert read_word_list(path) == ("NA", "NAN", "NONE", "NULL", "TRUE")


@pytest.mark.parametrize("bad", ["A", "ABCDEFGHIJKLMNOP", "CAFÉ", "IT'S", "X2"])
def test_rejects_entries_that_are_not_playable_words(tmp_path, bad):
    path = tmp_path / "words.txt"
    path.write_text(f"AA\n{bad}\n", encoding="utf-8")
    with pytest.raises(LexiconError, match="words.txt:2"):
        read_word_list(path)


def test_load_by_name_resolves_paths_relative_to_repo_root(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiny.txt").write_text("AA\nAB\n", encoding="utf-8")
    config = write_config(tmp_path, 'default = "TINY"\n[lexicons.TINY]\npath = "data/tiny.txt"\nexpected_words = 2\n')
    lexicon = load_lexicon(config_path=config)
    assert lexicon.name == "TINY"
    assert lexicon.words == ("AA", "AB")
    assert lexicon.id.startswith("TINY-") and len(lexicon.id) == len("TINY-") + 12


def test_id_depends_on_words_not_on_file_formatting(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "a.txt").write_text("AB\nAA\n", encoding="utf-8")
    (tmp_path / "data" / "b.txt").write_text("aa first\r\nab second\r\n", encoding="utf-8")
    (tmp_path / "data" / "c.txt").write_text("AA\nAD\n", encoding="utf-8")
    config = write_config(
        tmp_path,
        'default = "A"\n'
        '[lexicons.A]\npath = "data/a.txt"\n'
        '[lexicons.B]\npath = "data/b.txt"\n'
        '[lexicons.C]\npath = "data/c.txt"\n',
    )
    a, b, c = (load_lexicon(name, config) for name in "ABC")
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_wrong_word_count_is_an_error(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tiny.txt").write_text("AA\nAB\n", encoding="utf-8")
    config = write_config(tmp_path, 'default = "TINY"\n[lexicons.TINY]\npath = "data/tiny.txt"\nexpected_words = 3\n')
    with pytest.raises(LexiconError, match="expected 3 words, found 2"):
        load_lexicon(config_path=config)


def test_unknown_name_and_missing_file_are_reported(tmp_path):
    config = write_config(tmp_path, 'default = "TINY"\n[lexicons.TINY]\npath = "data/missing.txt"\n')
    with pytest.raises(LexiconError, match="unknown lexicon 'NOPE'"):
        load_lexicon("NOPE", config)
    with pytest.raises(LexiconError, match="not found"):
        load_lexicon("TINY", config)


def test_default_lexicon_if_installed():
    """Runs only on a machine that has the (copyrighted, uncommitted) word list."""
    try:
        lexicon = load_lexicon(config_path=DEFAULT_CONFIG)
    except LexiconError as exc:
        if "not found" in str(exc):
            pytest.skip("default lexicon file is not installed")
        raise
    words = set(lexicon.words)
    assert len(words) == 195_747
    assert {"NA", "NAN", "NULL"} <= words
    assert max(map(len, words)) == 15  # the old engine's list stopped at 10 letters
    assert sum(1 for w in words if len(w) == 2) == 107
