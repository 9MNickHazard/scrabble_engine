"""Process helpers shared by the app and the queue runner. Standard library only, so the queue
runner (which lives for as long as its queue) stays small and never loads the engine."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
from pathlib import Path

# A marker file is written just after its process starts; allow the clocks this much slack.
MARKER_SLACK_SECONDS = 5.0


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    # On Windows os.kill(pid, 0) would terminate the process, so ask the kernel instead.
    process_query_limited_information, still_active = 0x1000, 259
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def process_start_time(pid: int) -> float | None:
    """When the process was created, in Unix seconds, or None if that cannot be found out."""
    if pid <= 0 or os.name != "nt":
        return None
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]  # creation, exit, kernel, user (FILETIME)
        if not ctypes.windll.kernel32.GetProcessTimes(handle, *[ctypes.byref(t) for t in times]):
            return None
        return (times[0].value - 116_444_736_000_000_000) / 1e7  # 100 ns ticks since 1601 -> Unix seconds
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def clear_stale_stop(stop_file: Path) -> None:
    """Called by a process as it starts: remove a STOP file an earlier process left behind (it
    was killed before it could act on it), but keep one written since this process was created.
    That one is a request to stop this process, made while it was still starting up."""
    try:
        written = stop_file.stat().st_mtime
    except OSError:
        return
    created = process_start_time(os.getpid())
    if created is None or written < created:
        stop_file.unlink(missing_ok=True)


def started_after(pid: int, moment: float) -> bool:
    """True when process `pid` was created after `moment`: it cannot be the process a marker
    written at `moment` names, only a later one that was given the same number."""
    created = process_start_time(pid)
    return created is not None and moment > 0 and created > moment + MARKER_SLACK_SECONDS


def marker_pid(marker: dict | None) -> int:
    try:
        return int((marker or {}).get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def marker_time(marker: dict | None, path: Path) -> float:
    """When a marker was written: its `started` field, or the file's own time."""
    try:
        started = float((marker or {}).get("started") or 0)
    except (TypeError, ValueError):
        started = 0.0
    if started > 0:
        return started
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def kill_process(pid: int) -> None:
    """End a process and everything it started. The environment's python.exe is a small
    launcher that runs the real interpreter as a child, so the whole tree has to go."""
    if pid <= 0:
        return
    if os.name != "nt":
        os.kill(pid, 9)
        return
    try:
        done = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15,
                              creationflags=0x08000000)
        if done.returncode == 0:
            return
    except (OSError, subprocess.SubprocessError):
        pass
    handle = ctypes.windll.kernel32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if handle:
        ctypes.windll.kernel32.TerminateProcess(handle, 1)
        ctypes.windll.kernel32.CloseHandle(handle)


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None
