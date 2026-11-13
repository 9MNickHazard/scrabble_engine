"""Run queued training runs one after another.

Started detached by the app (`python -m scrabble_ai.app.queue_runner <queue file>`), so the
queue keeps going after the app window is closed. "One after another" means one training run
on the GPU at a time: before starting each of its runs the queue waits for any run that is
training now (however it was started) and for runs waiting in a queue made before this one.
A run taken out of the queue (its `queued.json` removed, or its folder deleted) is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scrabble_ai.app.procs import marker_pid, marker_time, pid_alive, read_json, started_after

POLL_SECONDS = 5.0
STARTUP_GRACE_SECONDS = 30.0  # an older queue's runner may not have written its pid file yet


def waiting_for(runs_dir: Path, queue_dir: Path, queue_name: str) -> str | None:
    """The name of a run this queue has to wait for, or None when the GPU is free for it."""
    try:
        folders = sorted(p for p in runs_dir.iterdir() if p.is_dir())
    except OSError:
        return None
    for run_dir in folders:
        if _alive(run_dir / "running.json"):
            return run_dir.name
        queued = read_json(run_dir / "queued.json")
        if queued is not None and str(queued.get("queue", "")) < queue_name:
            try:
                fresh = time.time() - float(queued.get("created", 0) or 0) < STARTUP_GRACE_SECONDS
            except (TypeError, ValueError):
                fresh = False
            if fresh or _alive(queue_dir / f"{queued.get('queue', '')}.pid"):
                return run_dir.name
    return None


def _alive(marker_path: Path) -> bool:
    """Is the process this marker file names still running (and still that process)?"""
    marker = read_json(marker_path)
    pid = marker_pid(marker)
    return marker is not None and pid_alive(pid) and not started_after(pid, marker_time(marker, marker_path))


def main(queue_file: str, poll_seconds: float = POLL_SECONDS) -> int:
    queue_path = Path(queue_file)
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    runs_dir = Path(queue["runs_dir"])
    # The app reads this to tell a waiting queue from one that died.
    queue_path.with_suffix(".pid").write_text(json.dumps({"pid": os.getpid(), "started": time.time()}), encoding="utf-8")
    for entry in queue["runs"]:
        run_dir = runs_dir / entry["name"]
        marker = run_dir / "queued.json"
        told = None
        while marker.exists():
            blocker = waiting_for(runs_dir, queue_path.parent, queue_path.stem)
            if blocker is None:
                break
            if blocker != told:
                print(f"{entry['name']} is waiting for {blocker}", flush=True)
                told = blocker
            time.sleep(poll_seconds)
        if not marker.exists():
            print(f"skipping {entry['name']}: no longer queued", flush=True)
            continue
        print(f"starting {entry['name']}", flush=True)
        with open(entry["log"], "ab") as log:
            process = subprocess.Popen(
                entry["command"], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=0x08000000 if os.name == "nt" else 0,
                env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
            )
            # running.json first, then the marker: there is no moment when the run is neither.
            (run_dir / "running.json").write_text(json.dumps({"pid": process.pid, "started": time.time()}), encoding="utf-8")
            if not (run_dir / "config.toml").exists():
                # The settings stay on record until the trainer has written config.toml, so a run
                # that dies in its first seconds can still be resumed as itself.
                queued = read_json(marker) or {}
                overrides = queued.get("overrides") if isinstance(queued.get("overrides"), dict) else {}
                (run_dir / "unstarted.json").write_text(json.dumps({"overrides": overrides}), encoding="utf-8")
            marker.unlink(missing_ok=True)
            code = process.wait()
        print(f"{entry['name']} ended with exit code {code}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
