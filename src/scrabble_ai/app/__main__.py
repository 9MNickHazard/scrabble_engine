"""Scrabble Lab, the desktop app: `python -m scrabble_ai.app` (or `uv run scrabble-ai app`).

Starts the local server and opens the UI in its own window (Microsoft Edge in app mode, which
is on every Windows 11 machine; the default browser otherwise). Closing the window ends the
app. Training runs and matches started from it are separate processes and keep going; the app
finds them again the next time it opens.

    --install-shortcut   put a "Scrabble Lab" shortcut on the Desktop and exit
    --no-window          serve only and print the address (for development)
    --port N             use a fixed port instead of a free one
    --runs-dir PATH      look for training runs here instead of runs/ (for demos and tests)
    --data-dir PATH      keep the app's own files here instead of data/app/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from scrabble_ai.lexicon import REPO_ROOT

APP_DIR = REPO_ROOT / "data" / "app"
SERVER_FILE = APP_DIR / "server.json"


def _redirect_output() -> None:
    """Under pythonw there is no console and sys.stdout is None; send output to a log file."""
    if sys.stdout is None or sys.stderr is None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        log = open(APP_DIR / "app.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or log
        sys.stderr = sys.stderr or log


def find_edge() -> str | None:
    candidates = [
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return shutil.which("msedge")


def open_window(url: str) -> subprocess.Popen | None:
    """Open the UI in a chromeless window with its own profile, so the process we start is the
    one that owns the window and ends when it is closed."""
    edge = find_edge()
    if edge is None:
        webbrowser.open(url)
        return None
    profile = APP_DIR / "window-profile"
    profile.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([edge, f"--app={url}", f"--user-data-dir={profile}", "--window-size=1500,960",
                             "--no-first-run", "--no-default-browser-check", "--disable-features=Translate"])


def running_instance() -> str | None:
    """The address of an app that is already running, if there is one."""
    try:
        info = json.loads(SERVER_FILE.read_text(encoding="utf-8"))
        url = f"http://127.0.0.1:{int(info['port'])}"
        with urllib.request.urlopen(url + "/api/status", timeout=1.5) as response:
            return url if response.status == 200 else None
    except Exception:  # a missing, stale or damaged file all mean the same thing: start afresh
        return None


def make_icon(path: Path) -> bool:
    """Draw the app's icon (a letter tile) as an .ico file. Needs Pillow; returns False without it."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    size = 256
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((14, 14, size - 14, size - 14), radius=42, fill=(236, 220, 176, 255), outline=(201, 182, 132, 255), width=6)
    draw.rounded_rectangle((14, size - 46, size - 14, size - 14), radius=42, fill=(201, 182, 132, 255))
    draw.rectangle((14, size - 70, size - 14, size - 40), fill=(236, 220, 176, 255))
    try:
        big = ImageFont.truetype("segoeuib.ttf", 170)
        small = ImageFont.truetype("segoeuib.ttf", 54)
    except OSError:
        big = small = ImageFont.load_default()
    draw.text((size // 2 - 8, size // 2 - 8), "S", font=big, fill=(28, 26, 20, 255), anchor="mm")
    draw.text((size - 52, size - 62), "1", font=small, fill=(28, 26, 20, 255), anchor="mm")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    return True


def install_shortcut() -> Path:
    """Create "Scrabble Lab.lnk" on the Desktop, pointing at this environment's pythonw."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        pythonw = Path(sys.executable)
    icon = APP_DIR / "scrabble-lab.ico"
    has_icon = make_icon(icon)
    script = (
        "$desktop = [Environment]::GetFolderPath('Desktop'); "
        "$path = Join-Path $desktop 'Scrabble Lab.lnk'; "
        "$shell = New-Object -ComObject WScript.Shell; "
        "$link = $shell.CreateShortcut($path); "
        f"$link.TargetPath = '{pythonw}'; "
        "$link.Arguments = '-m scrabble_ai.app'; "
        f"$link.WorkingDirectory = '{REPO_ROOT}'; "
        "$link.Description = 'Scrabble Lab: train, watch and play the Scrabble models'; "
        + (f"$link.IconLocation = '{icon}'; " if has_icon else "")
        + "$link.Save(); Write-Output $path"
    )
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="scrabble-ai app", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--install-shortcut", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    _redirect_output()

    if args.install_shortcut:
        print(f"shortcut created: {install_shortcut()}")
        return 0

    custom = args.runs_dir is not None or args.data_dir is not None
    existing = None if custom else running_instance()
    if existing and not args.no_window:
        # Already running (perhaps with its window closed by accident): just show it again.
        window = open_window(existing)
        if window is not None:
            window.wait()
        return 0

    from scrabble_ai.app.runs import RUNS_DIR
    from scrabble_ai.app.server import AppState, create_server

    app_dir = (args.data_dir or APP_DIR).resolve()
    app_dir.mkdir(parents=True, exist_ok=True)
    state = AppState((args.runs_dir or RUNS_DIR).resolve(), app_dir)
    server = create_server(args.port, state=state, log_path=app_dir / "app.log")
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}"
    if not custom:
        SERVER_FILE.write_text(json.dumps({"port": port, "pid": os.getpid(), "started": time.time()}), encoding="utf-8")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Scrabble Lab is serving at {url}", flush=True)
    try:
        if args.no_window:
            thread.join()
        else:
            window = open_window(url)
            if window is None:
                thread.join()  # no app window to wait for: serve until the process is ended
            else:
                window.wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        if not custom:
            SERVER_FILE.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
