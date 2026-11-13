"""The app's HTTP server: the API in docs/APP_API.md plus the static UI.

Standard library only. It binds to 127.0.0.1 and nothing else: this is a local app, there is no
sign-in, and it must never be reachable from another machine.
"""

from __future__ import annotations

import json
import math
import mimetypes
import re
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from scrabble_ai import __version__
from scrabble_ai import engine as engine_module
from scrabble_ai.app import schema as schema_module
from scrabble_ai.app.library import Library
from scrabble_ai.app.matches import Matches, renamed_spec
from scrabble_ai.app.players import NotFound, PlayerError, Players
from scrabble_ai.app.runs import APP_DIR, RUNS_DIR, RunError, RunManager
from scrabble_ai.app.sessions import Sessions
from scrabble_ai.config import ConfigError

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _finite(value):
    """`value` with every NaN and infinity replaced by None. A diverged loss is logged as NaN,
    which Python would write into the reply as-is, and that is not JSON a browser accepts."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _integer(value, what: str, low: int, high: int) -> int:
    """A whole number from a request (a JSON number, or digits in a query string) within limits."""
    if isinstance(value, str) and re.fullmatch(r"-?\d{1,18}", value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlayerError(f"{what} must be a whole number")
    if not low <= value <= high:
        raise PlayerError(f"{what} must be between {low:,} and {high:,}")
    return value


class AppState:
    """Everything the handlers share."""

    def __init__(self, runs_dir: Path = RUNS_DIR, app_dir: Path = APP_DIR, loaded: engine_module.Loaded | None = None):
        """`runs_dir` and `app_dir` default to the repo's `runs/` and `data/app/`; tests and
        demos point them elsewhere. `loaded` is the engine and leave values to use."""
        self.loaded = loaded or engine_module.load(allow_missing_leaves=True)
        self.runs = RunManager(runs_dir, app_dir)
        self.players = Players(self.loaded, runs_dir)
        self.library = Library(self.runs, self.loaded, app_dir)
        self.sessions = Sessions(self.players)
        self.matches = Matches(app_dir, self.players)
        self._gpu: tuple[float, dict | None] = (0.0, None)
        self._gpu_lock = threading.Lock()

    def gpu(self) -> dict | None:
        """GPU load from nvidia-smi, refreshed at most every two seconds."""
        with self._gpu_lock:
            checked, value = self._gpu
            if time.time() - checked < 2.0:
                return value
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4,
                    creationflags=0x08000000 if hasattr(subprocess, "STARTUPINFO") else 0)
                name, utilization, used, total, temperature = [part.strip() for part in out.stdout.splitlines()[0].split(",")]
                value = {"name": name, "utilization": int(float(utilization)), "memory_used_mb": int(float(used)),
                         "memory_total_mb": int(float(total)), "temperature_c": int(float(temperature))}
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                value = None
            self._gpu = (time.time(), value)
            return value

    def status(self) -> dict:
        summaries = self.runs.list()
        active = [s["name"] for s in summaries if s["status"] in ("running", "stopping")]
        gpu = self.gpu()
        warnings = []
        if gpu and not active and gpu["memory_used_mb"] > 0.6 * gpu["memory_total_mb"]:
            # Windows does not refuse GPU memory when it runs out: it spills into system RAM and
            # everything slows to a crawl, so say so before a run is started into that.
            warnings.append({
                "kind": "gpu_memory",
                "message": (f"Other programs are using {gpu['memory_used_mb'] / 1024:.1f} of "
                            f"{gpu['memory_total_mb'] / 1024:.0f} GB of GPU memory. Training started now "
                            f"would be slow until they are closed."),
            })
        return {
            "version": __version__,
            "lexicon": {"id": self.loaded.lexicon_id, "words": self.loaded.engine.word_count},
            "leaves": {"ok": self.loaded.leaves_path is not None,
                       "file": self.loaded.leaves_path.name if self.loaded.leaves_path else None},
            "gpu": gpu,
            "warnings": warnings,
            "active_runs": active,
            "queued_runs": [s["name"] for s in summaries if s["status"] == "queued"],
            "active_matches": self.matches.active(),
        }


def _report(state: AppState, name: str) -> dict:
    from scrabble_ai.app import insights

    if not state.runs.exists(name):
        raise NotFound(f"there is no run named {name!r}")
    report = insights.run_report(name, state.runs.records(name), state.runs.config(name))
    summary = state.runs.summary(name)
    if summary["status"] == "failed":
        # The log has no stop record for a run that died, so the report cannot know: add it here.
        lines = [line.strip() for line in (summary.get("error") or "").splitlines() if line.strip()]
        reason = lines[-1] if lines else "The training process is gone and left no stop record."
        if reason[-1] not in ".!?":
            reason += "."
        advice = ("Resume continues it from its last checkpoint." if summary.get("checkpoints")
                  else "It had not saved a checkpoint; Resume starts it again from the beginning.")
        report["events"].append({
            "games": int(summary.get("games") or 0), "hours": round(float(summary.get("hours") or 0.0), 4),
            "kind": "stop", "tone": "warning", "title": "Run ended unexpectedly", "detail": f"{reason} {advice}",
        })
    return report


def _run_or_404(state: AppState, name: str) -> str:
    if not state.runs.exists(name):
        raise NotFound(f"there is no run named {name!r}")
    return name


def _leaves(state: AppState, query: dict) -> dict:
    rack = query.get("rack", "").upper()
    bag = _integer(query.get("bag", 60), "bag", 0, 100)
    if not re.fullmatch(r"[A-Z?]{0,7}", rack):
        raise PlayerError("a rack is up to seven letters, with ? for a blank")
    value = state.loaded.leaves.value(rack, bag)
    return {"rack": rack, "bag": bag, "value": round(value, 2), "relative": round(value - state.loaded.leaves.value("", bag), 2)}


def _start_runs(state: AppState, body: dict) -> dict:
    specs = body.get("runs")
    if not isinstance(specs, list) or not all(isinstance(s, dict) and isinstance(s.get("name"), str) for s in specs):
        raise RunError("runs must be a list of {name, overrides}")
    if state.loaded.leaves_path is None:
        raise RunError("leave values have not been fitted yet: run `uv run scrabble-ai fit-leaves` first")
    mode = body.get("mode", "parallel")
    if not isinstance(mode, str):
        raise RunError("mode must be 'parallel' or 'queue'")
    return state.runs.start(specs, mode)


def _stop_run(state: AppState, name: str, body: dict) -> dict:
    force = body.get("force", False)
    if not isinstance(force, bool):
        raise RunError("force must be true or false")
    state.runs.stop(_run_or_404(state, name), force)
    return {"ok": True}


def _delete_run(state: AppState, name: str) -> dict:
    state.runs.delete(_run_or_404(state, name))
    state.library.forget_run(name)
    return {"ok": True}


def _rename_run(state: AppState, name: str, body: dict) -> dict:
    _run_or_404(state, name)
    new = body.get("name")
    if not isinstance(new, str):
        raise RunError("name must be the run's new name")
    if state.matches.running_with(name):
        raise RunError(f"a match involving {name} is still running; rename the run once it has finished or been cancelled")
    state.runs.rename(name, new)
    # The run has its new name from here on, and what refers to it follows. Games in progress
    # look their networks up by spec at every move: point them at the new name too (the network
    # is then loaded again from its new path, since Players caches by path).
    with state.sessions.lock:
        games = list(state.sessions.sessions.values())
    for game in games:
        with game.lock:
            game.specs = [renamed_spec(spec, name, new) for spec in game.specs]
            game.advisor = renamed_spec(game.advisor, name, new)
    failed = []
    for what, follow in (("its stars and notes", state.library.rename_run), ("its matches", state.matches.rename_run)):
        try:
            follow(name, new)
        except OSError as error:  # one that cannot follow must not stop the other
            failed.append(f"{what} ({error})")
    if failed:
        raise RunError(f"{name} was renamed to {new}, but {' and '.join(failed)} could not be updated")
    return state.runs.summary(new)


def _create_game(state: AppState, body: dict) -> dict:
    players = body.get("players")
    if not isinstance(players, list) or len(players) != 2 or not all(isinstance(p, str) for p in players):
        raise PlayerError("players must be a list of two player specs")
    advisor = body.get("advisor")
    if advisor is not None and not isinstance(advisor, str):
        raise PlayerError("advisor must be a bot or model spec")
    return state.sessions.create(players, body.get("seed"), advisor).state()


def _start_match(state: AppState, body: dict) -> dict:
    a, b = body.get("a"), body.get("b")
    if not isinstance(a, str) or not isinstance(b, str):
        raise PlayerError("a match needs two players, a and b")
    pairs = _integer(body.get("pairs", 500), "pairs", 1, 100_000)
    candidates = _integer(body.get("candidates", 64), "candidates", 2, 512)
    return {"id": state.matches.start(a, b, pairs, candidates)}


# (method, path pattern, handler(state, match, body, query) -> JSON-able)
ROUTES = [
    ("GET", r"/api/status", lambda s, m, b, q: s.status()),
    ("GET", r"/api/config/schema", lambda s, m, b, q: schema_module.schema()),
    ("GET", r"/api/runs", lambda s, m, b, q: {"runs": s.runs.list()}),
    ("POST", r"/api/runs", lambda s, m, b, q: _start_runs(s, b)),
    ("GET", r"/api/runs/([^/]+)", lambda s, m, b, q: s.runs.detail(_run_or_404(s, m[1]))),
    ("DELETE", r"/api/runs/([^/]+)", lambda s, m, b, q: _delete_run(s, m[1])),
    ("GET", r"/api/runs/([^/]+)/metrics",
     lambda s, m, b, q: dict(zip(("records", "next"),
                                 s.runs.metrics(_run_or_404(s, m[1]), _integer(q.get("since", 0), "since", 0, 10**12))))),
    ("GET", r"/api/runs/([^/]+)/log",
     lambda s, m, b, q: {"lines": s.runs.log(_run_or_404(s, m[1]), _integer(q.get("tail", 200), "tail", 0, 100_000))}),
    ("GET", r"/api/runs/([^/]+)/report", lambda s, m, b, q: _report(s, m[1])),
    ("POST", r"/api/runs/([^/]+)/stop", lambda s, m, b, q: _stop_run(s, m[1], b)),
    ("POST", r"/api/runs/([^/]+)/rename", lambda s, m, b, q: _rename_run(s, m[1], b)),
    ("POST", r"/api/runs/([^/]+)/resume",
     lambda s, m, b, q: s.runs.resume(_run_or_404(s, m[1]), b.get("hours"), b.get("max_games"), b.get("overrides"))),
    ("GET", r"/api/models", lambda s, m, b, q: s.library.models()),
    ("POST", r"/api/models/([^/]+/[^/]+)/meta", lambda s, m, b, q: s.library.set_meta(m[1], b.get("starred"), b.get("note"))),
    ("GET", r"/api/matches", lambda s, m, b, q: {"matches": s.matches.list()}),
    ("POST", r"/api/matches", lambda s, m, b, q: _start_match(s, b)),
    ("GET", r"/api/matches/([^/]+)", lambda s, m, b, q: s.matches.get(m[1])),
    ("POST", r"/api/matches/([^/]+)/cancel", lambda s, m, b, q: (s.matches.cancel(m[1]), {"ok": True})[1]),
    ("POST", r"/api/games", lambda s, m, b, q: _create_game(s, b)),
    ("GET", r"/api/games/([^/]+)", lambda s, m, b, q: s.sessions.get(m[1]).state()),
    ("DELETE", r"/api/games/([^/]+)", lambda s, m, b, q: (s.sessions.delete(m[1]), {"ok": True})[1]),
    ("POST", r"/api/games/([^/]+)/move", lambda s, m, b, q: s.sessions.get(m[1]).human_move(b)),
    ("POST", r"/api/games/([^/]+)/ai", lambda s, m, b, q: s.sessions.get(m[1]).ai_move()),
    ("GET", r"/api/games/([^/]+)/hint",
     lambda s, m, b, q: s.sessions.get(m[1]).hint(_integer(q.get("top", 15), "top", 1, 200))),
    ("GET", r"/api/games/([^/]+)/analysis",
     lambda s, m, b, q: s.sessions.get(m[1]).analysis(_integer(q.get("turn"), "turn", 1, 10_000))),
    ("GET", r"/api/leaves", lambda s, m, b, q: _leaves(s, q)),
]
_COMPILED = [(method, re.compile(pattern + r"/?$"), handler) for method, pattern, handler in ROUTES]


def make_handler(state: AppState, log_path: Path | None = None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):  # noqa: A002 - signature fixed by the base class
            pass  # the app runs without a console; errors go to the log file instead

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            try:
                text = json.dumps(data, allow_nan=False)
            except ValueError:  # a NaN or infinity somewhere in it
                text = json.dumps(_finite(data), allow_nan=False)
            self._send(status, text.encode("utf-8"), "application/json; charset=utf-8")

        def _static(self, path: str) -> None:
            relative = "index.html" if path in ("", "/") else unquote(path).lstrip("/")
            try:
                target = (STATIC_DIR / relative).resolve()
                found = STATIC_DIR.resolve() in target.parents and target.is_file()
            except (ValueError, OSError):  # a NUL or other character no file name can hold
                found = False
            if not found:
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            if content_type.startswith("text/") or content_type in ("application/javascript", "image/svg+xml"):
                content_type += "; charset=utf-8"
            self._send(200, target.read_bytes(), content_type)

        def _handle(self, method: str) -> None:
            url = urlsplit(self.path)
            if not url.path.startswith("/api/"):
                if method == "GET":
                    self._static(url.path)
                else:
                    self._send(405, b"method not allowed", "text/plain; charset=utf-8")
                return
            query = {key: values[-1] for key, values in parse_qs(url.query).items()}
            body = {}
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._json(400, {"error": "the Content-Length header is not a number"})
                return
            if length < 0 or length > 16_000_000:
                self._json(400, {"error": "the request body is too large"})
                return
            if length:
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except ValueError:
                    self._json(400, {"error": "the request body is not valid JSON"})
                    return
                if not isinstance(body, dict):
                    self._json(400, {"error": "the request body must be a JSON object"})
                    return
            path = unquote(url.path)
            for route_method, pattern, handler in _COMPILED:
                match = pattern.match(path) if route_method == method else None
                if match:
                    try:
                        self._json(200, handler(state, match, body, query))
                    except NotFound as error:
                        self._json(404, {"error": str(error)})
                    except (RunError, PlayerError, ConfigError, ValueError, IndexError, FileNotFoundError) as error:
                        self._json(400, {"error": str(error)})
                    except Exception as error:  # a bug: say so, and keep the server up
                        if log_path is not None:
                            with open(log_path, "a", encoding="utf-8") as fh:
                                fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {method} {path}\n{traceback.format_exc()}\n")
                        self._json(500, {"error": f"{type(error).__name__}: {error}"})
                    return
            self._json(404, {"error": f"no such endpoint: {method} {path}"})

        def do_GET(self):  # noqa: N802
            self._handle("GET")

        def do_POST(self):  # noqa: N802
            self._handle("POST")

        def do_DELETE(self):  # noqa: N802
            self._handle("DELETE")

    return Handler


class _Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """A page that closes a connection it was keeping open (every reload and every closed
        window does) is not an error: keep those tracebacks out of the log so real ones show."""
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def create_server(port: int = 0, state: AppState | None = None, log_path: Path | None = None) -> ThreadingHTTPServer:
    """A server on 127.0.0.1 (port 0 picks a free port). Call `serve_forever()` on it."""
    state = state or AppState()
    server = _Server(("127.0.0.1", port), make_handler(state, log_path))
    server.daemon_threads = True
    server.state = state
    return server
