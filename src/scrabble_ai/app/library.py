"""The model library: every checkpoint of every run, described without loading any weights."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from scrabble_ai import engine as engine_module
from scrabble_ai.app import players
from scrabble_ai.app.runs import RunManager, flatten, read_json


class Library:
    def __init__(self, runs: RunManager, loaded: engine_module.Loaded, app_dir: Path):
        self.runs = runs
        self.loaded = loaded
        self.meta_path = app_dir / "library.json"
        self.lock = threading.Lock()

    def _meta(self) -> dict:
        return read_json(self.meta_path) or {}

    def _entry(self, run: str, file: str, info: dict, index: dict, records: list[dict], meta: dict, flat: dict) -> dict:
        model_id = f"{run}/{file}"
        step = info.get("step")
        evaluation = {}
        games = None
        for record in records:
            if record.get("step") != step:
                continue
            if record.get("kind") == "eval" and record.get("spread") is not None:
                evaluation[record["opponent"]] = {"spread": record["spread"], "ci": record.get("spread_ci"),
                                                  "win_rate": record.get("win_rate")}
            games = record.get("games", games)
        reason = None
        if index.get("lexicon_id") and index.get("leaves_hash"):
            if index["lexicon_id"] != self.loaded.lexicon_id:
                reason = (f"Trained with the word list {index['lexicon_id']}; the one loaded now is "
                          f"{self.loaded.lexicon_id}. A network only knows the words it was trained with.")
            elif index["leaves_hash"] != self.loaded.leaves_hash:
                reason = ("Trained with leave values that have since been refitted, so its values no longer "
                          "mean the same thing.")
        saved = meta.get(model_id, {})
        return {
            "id": model_id,
            "spec": f"model:{model_id}",
            "name": players.model_label(run, file),
            "run": run,
            "file": file,
            "kind": "best" if file == "best.pt" else "checkpoint",
            "network": index.get("network") or f"{flat.get('model.blocks', '?')}x{flat.get('model.channels', '?')}",
            "parameters": index.get("parameters") or next(
                (r["parameters"] for r in reversed(records) if r.get("kind") == "session" and r.get("parameters")), None),
            "step": step,
            "games": games,
            "modified": info.get("modified"),
            "size_bytes": info.get("size_bytes"),
            "compatible": reason is None,
            "incompatible_reason": reason,
            "eval": evaluation or None,
            "starred": bool(saved.get("starred", False)),
            "note": saved.get("note", ""),
        }

    def models(self) -> dict:
        meta = self._meta()
        entries = []
        for run in self.runs.names():
            checkpoint_dir = self.runs.path(run) / "checkpoints"
            if not checkpoint_dir.is_dir():
                continue
            index = read_json(checkpoint_dir / "index.json") or {}
            records = self.runs.records(run)
            flat = flatten(self.runs.config(run))
            best_path = checkpoint_dir / "best.pt"
            if best_path.exists():
                info = index.get("best") or {"step": _best_step(records), "size_bytes": best_path.stat().st_size,
                                             "modified": best_path.stat().st_mtime}
                entries.append(self._entry(run, "best.pt", info, index, records, meta, flat))
            for path in sorted(checkpoint_dir.glob("step_*.pt")):
                info = {"step": int(path.stem.split("_")[1]), "size_bytes": path.stat().st_size, "modified": path.stat().st_mtime}
                entries.append(self._entry(run, path.name, info, index, records, meta, flat))
        entries.sort(key=lambda m: -(m["modified"] or 0))
        bots = [{"spec": spec, "name": label, "description": description} for spec, (label, description) in players.BOTS.items()]
        return {"bots": bots, "models": entries}

    def model(self, model_id: str) -> dict:
        for entry in self.models()["models"]:
            if entry["id"] == model_id:
                return entry
        raise players.NotFound(f"there is no model {model_id!r}")

    def forget_run(self, run: str) -> None:
        """Drop the stars and notes of a deleted run's models, so a new run of the same name
        does not inherit them."""
        with self.lock:
            meta = self._meta()
            # Windows folder names ignore case, so a run's stars and notes are found the same way.
            kept = {model_id: entry for model_id, entry in meta.items()
                    if model_id.partition("/")[0].lower() != run.lower()}
            if len(kept) != len(meta):
                self._write(kept)

    def rename_run(self, old: str, new: str) -> None:
        """Carry the stars and notes of a renamed run's models over to its new name: every
        `<old>/<file>` becomes `<new>/<file>` (the run compared without regard to case, as in
        `forget_run`). They replace any left behind under the new name by a run long gone."""
        with self.lock:
            meta = self._meta()
            moved = {f"{new}/{model_id.partition('/')[2]}": entry for model_id, entry in meta.items()
                     if model_id.partition("/")[0].lower() == old.lower()}
            if moved:
                kept = {model_id: entry for model_id, entry in meta.items()
                        if model_id.partition("/")[0].lower() != old.lower()}
                self._write({**kept, **moved})

    def _write(self, meta: dict) -> None:
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.meta_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        temporary.replace(self.meta_path)

    def set_meta(self, model_id: str, starred: bool | None = None, note: str | None = None) -> dict:
        players.split_model(model_id)
        if starred is not None and not isinstance(starred, bool):
            raise players.PlayerError("starred must be true or false")
        if note is not None and not isinstance(note, str):
            raise players.PlayerError("note must be text")
        self.model(model_id)  # nothing is written for a model that does not exist
        with self.lock:
            meta = self._meta()
            entry = meta.setdefault(model_id, {})
            if starred is not None:
                entry["starred"] = bool(starred)
            if note is not None:
                entry["note"] = note[:2000]
            self._write(meta)
        return self.model(model_id)


def _best_step(records: list[dict]) -> int | None:
    """For runs from before index.json existed: the step of the last promotion."""
    for record in reversed(records):
        if record.get("kind") == "gate" and record.get("promoted"):
            return record.get("step")
    return None
