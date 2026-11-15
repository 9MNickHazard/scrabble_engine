"""Games with a board on screen: a person against a model, or two players being watched.

The engine's `Game` holds the position; this module adds who is playing, the move history, and
the analysis shown beside the board (what a player considered and how it valued each option).

The three bots choose their moves with the engine's own bot code (`Game.bot_choice`), the same
code and the same random stream that matches and evaluations use, so any game of a match can be
replayed here move for move from its seed and seating. Networks choose here exactly as they do
in a match: the highest score plus predicted value among the 64 best static candidates.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict

import numpy as np

from scrabble_ai import _core
from scrabble_ai.app import players as players_module
from scrabble_ai.app.players import NotFound, PlayerError, Players

NETWORK_CANDIDATES = 64
NETWORK_EXCHANGES = 4
# Greedy and random choose among every placement, so list as many as could matter on screen.
BOT_CANDIDATES = 4096
BOT_EXCHANGES = 127
MAX_SEED = 2**63 - 1


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class GameSession:
    def __init__(self, session_id: str, specs: list[str], advisor: str, seed: int, players: Players,
                 inference_lock: threading.Lock):
        self.id = session_id
        self.specs = specs
        self.advisor = advisor
        self.players = players
        self.inference_lock = inference_lock
        # Networks that take an inference about the opponent's rack as an input need the game to
        # make it, with as many sampled racks as the most demanding of them was trained with.
        models = {spec for spec in (*specs, advisor) if players_module.kind(spec) == "model"}
        belief_samples = max((getattr(players.scorer(spec), "belief_samples", 0) for spec in models), default=0)
        self.game = _core.Game(players.loaded.engine, players.loaded.leaves, seed, belief_samples)
        self.seed = seed
        self.history: list[dict] = []
        self.analyses: dict[int, dict] = {}  # turn number -> what the player of that turn considered
        self.last_placed: list[list[int]] = []
        self.result: dict | None = None
        self.lock = threading.Lock()
        self.touched = time.time()

    # ---- state -----------------------------------------------------------------------------

    def state(self) -> dict:
        raw = self.game.state()
        return {
            "id": self.id,
            "seed": self.seed,
            "players": [{"spec": spec, "name": players_module.name(spec)} for spec in self.specs],
            "advisor": self.advisor,
            "board": raw["board"],
            "last_placed": self.last_placed,
            "racks": raw["racks"],
            "scores": raw["scores"],
            "to_move": raw["to_move"],
            "turn": raw["turn"],
            "bag": raw["bag"],
            "scoreless_turns": raw["scoreless_turns"],
            "can_exchange": raw["can_exchange"],
            "unseen": raw["unseen"],
            "unseen_by_seat": raw["unseen_by_seat"],
            "over": raw["over"],
            "result": self.result,
            "history": self.history,
        }

    def _record(self, move: dict) -> None:
        """Add a played move (as returned by the engine) to the history."""
        raw = self.game.state()
        player = move["player"]
        self.history.append({
            "turn": len(self.history) + 1,
            "player": player,
            "kind": move["kind"],
            "text": move["text"],
            "word": move["word"],
            "position": move["position"],
            "score": move["score"],
            "total": raw["scores"][player],
            "tiles": [list(tile) for tile in move["tiles"]],
            "rack": move["rack"],
            "leave": move["leave"],
            "bingo": move["bingo"],
            "exchanged": move["exchanged"],
            "analysed": False,
        })
        self.last_placed = [[tile[0], tile[1]] for tile in move["tiles"]]
        if move["game_over"]:
            scores = raw["scores"]
            self.result = {
                "scores": scores,
                "winner": 0 if scores[0] > scores[1] else 1 if scores[1] > scores[0] else None,
                "adjustment": list(move["adjustment"]),
                "ended_by": raw["ended_by"],
            }

    # ---- choosing --------------------------------------------------------------------------

    @staticmethod
    def _option(index: int, d: dict, value: float, win: float | None) -> dict:
        """One candidate as the API shows it. `score` is the move's own points, as in the move
        history; whatever follows the move is `value`, which for a move that ends the game by
        going out is exactly the bonus for the tiles left on the opponent's rack."""
        bonus = d.get("bonus", 0)
        return {
            "index": index, "text": f"exchange {d['exchanged']}" if d["kind"] == "exchange" else d["text"],
            "kind": d["kind"], "word": d["word"], "position": d["position"],
            "score": d["score"] - bonus, "tiles": [list(tile) for tile in d["tiles"]], "leave": d["leave"],
            "exchanged": d["exchanged"], "leave_value": round(float(d["leave_value"]), 2),
            "value": round(float(value) + bonus, 2), "total": round(float(d["score"] + value), 2),
            "win": None if win is None else round(win, 4), "chosen": False,
        }

    def _options(self, spec: str) -> tuple[list[dict], int | None]:
        """Every option `spec` weighs for the side to move, each with `index` (its place in the
        engine's candidate list), and for a network the index of the one it plays. A bot's
        choice is made by the engine, in `ai_move`."""
        which = players_module.kind(spec)
        if which == "human":
            raise PlayerError("a person does not have an analysis")
        is_model = which == "model"
        uses_leave = is_model or spec == "bot:greedy-leave"
        limits = (NETWORK_CANDIDATES, NETWORK_EXCHANGES) if uses_leave else (BOT_CANDIDATES, BOT_EXCHANGES)
        descriptions, board, leave, unseen, meta, leave_value = self.game.candidates(*limits)
        if not descriptions:
            return [], None
        count = len(descriptions)
        win: list[float | None] = [None] * count
        chosen = None
        if is_model:
            scorer = self.players.scorer(spec)
            with self.inference_lock:
                value_t, win_t = scorer(board, leave, unseen, meta, leave_value)
                value, win_p = value_t.cpu().numpy().astype(np.float64), win_t.cpu().numpy().astype(np.float64)
            went_out = meta[:, _core.M_TERMINAL] == 1
            spread = meta[:, _core.M_SPREAD]
            value = np.where(went_out, 0.0, value)
            win_p = np.where(went_out, (spread > 0) + 0.5 * (spread == 0), win_p)
            win = [float(p) for p in win_p]
            score = np.array([d["score"] for d in descriptions], np.float64)
            # The same rule as in matches and evaluations (`selfplay.choose`): points, or for a
            # network trained to win, mostly its win chance with points breaking ties.
            weight, scale = getattr(scorer, "win_weight", 0.0), getattr(scorer, "win_scale", 100.0)
            utility = (1.0 - weight) * (score + value) + weight * scale * (2.0 * win_p - 1.0)
            chosen = int(np.argmax(utility))
        elif uses_leave:
            value = leave_value.astype(np.float64)
        else:
            value = np.zeros(count)  # greedy and random look at the score alone
        options = [self._option(i, d, value[i], win[i]) for i, d in enumerate(descriptions)]
        if is_model and weight > 0:
            for option, u in zip(options, utility):
                option["utility"] = round(float(u), 3)  # what this network ranks by
        return options, chosen

    @staticmethod
    def _public(options: list[dict], top: int, keep: int | None = None) -> list[dict]:
        """The best `top` options, plus the chosen one if it falls outside them."""
        ranked = sorted(options, key=lambda o: (-o.get("utility", o["total"]), o["index"]))
        shown = ranked[:top]
        if keep is not None and all(o["index"] != keep for o in shown):
            shown = shown + [o for o in ranked if o["index"] == keep]
        return [{k: v for k, v in option.items() if k != "index"} for option in shown]

    # ---- moves -----------------------------------------------------------------------------

    def human_move(self, body: dict) -> dict:
        with self.lock:
            self.touched = time.time()
            raw = self.game.state()
            if raw["over"]:
                raise PlayerError("the game is over")
            if players_module.kind(self.specs[raw["to_move"]]) != "human":
                raise PlayerError("it is not a person's turn")
            if body.get("pass"):
                move = self.game.pass_turn()
            elif "exchange" in body:
                if not isinstance(body["exchange"], str):
                    raise PlayerError('exchange is the letters to put back, such as "AEI" (? for a blank)')
                move = self.game.exchange(body["exchange"].upper())
            elif "tiles" in body:
                tiles = body["tiles"]
                if not isinstance(tiles, list):
                    raise PlayerError("tiles must be a list of {row, col, letter, blank}")
                placed = []
                for tile in tiles:
                    if (not isinstance(tile, dict) or not _is_int(tile.get("row")) or not _is_int(tile.get("col"))
                            or not isinstance(tile.get("letter"), str)):
                        raise PlayerError("each tile needs a row, a col and a letter")
                    if not (0 <= tile["row"] < 15 and 0 <= tile["col"] < 15):
                        raise PlayerError("a tile is off the board")
                    placed.append((tile["row"], tile["col"], tile["letter"], bool(tile.get("blank", False))))
                move = self.game.play_tiles(placed)
            else:
                raise PlayerError("a move needs tiles, exchange or pass")
            self._record(move)
            return self.state()

    def ai_move(self, top: int = 12) -> dict:
        with self.lock:
            self.touched = time.time()
            raw = self.game.state()
            if raw["over"]:
                raise PlayerError("the game is over")
            spec = self.specs[raw["to_move"]]
            if players_module.kind(spec) == "human":
                raise PlayerError("it is a person's turn")
            options, chosen = self._options(spec)
            if players_module.kind(spec) == "bot":
                # The engine's own bot decides, exactly as in a match with this seed.
                chosen, extra = self.game.bot_choice(players_module.bot_name(spec))
                if extra is not None:  # not among the candidates listed: show it too
                    value = float(extra["leave_value"]) if spec == "bot:greedy-leave" else 0.0
                    options.append(self._option(chosen, extra, value, None))
            for option in options:
                option["chosen"] = option["index"] == chosen
            move = self.game.play_candidate(chosen)
            self._record(move)
            analysis = {"player": players_module.name(spec), "turn": len(self.history), "considered": len(options),
                        "candidates": self._public(options, top, keep=chosen)}
            self.analyses[len(self.history)] = analysis
            self.history[-1]["analysed"] = True
            return {"state": self.state(), "analysis": analysis}

    def analysis(self, turn: int) -> dict:
        """What the player of `turn` considered, for moves an AI made in this session."""
        with self.lock:
            if turn not in self.analyses:
                raise NotFound(f"no analysis was recorded for turn {turn}")
            return self.analyses[turn]

    def hint(self, top: int = 15) -> dict:
        """The advisor's ranking of the moves available now. It changes nothing: in particular
        it never draws from the random bot's stream."""
        with self.lock:
            self.touched = time.time()
            if self.game.state()["over"]:
                raise PlayerError("the game is over")
            options, _ = self._options(self.advisor)
            return {"advisor": players_module.name(self.advisor), "candidates": self._public(options, top)}


class Sessions:
    def __init__(self, players: Players, limit: int = 40):
        self.players = players
        self.limit = limit
        self.sessions: OrderedDict[str, GameSession] = OrderedDict()
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()

    def create(self, specs: list[str], seed: int | None = None, advisor: str | None = None) -> GameSession:
        if not isinstance(specs, (list, tuple)) or len(specs) != 2:
            raise PlayerError("a game needs exactly two players")
        for spec in specs:
            if players_module.kind(spec) == "model":
                self.players.scorer(spec)  # fail now, with a clear message, if it cannot be loaded
        if advisor is None:
            models = [s for s in specs if players_module.kind(s) == "model"]
            advisor = models[0] if models else "bot:greedy-leave"
        elif players_module.kind(advisor) == "human":
            raise PlayerError("the advisor must be a bot or a model")
        elif players_module.kind(advisor) == "model":
            self.players.scorer(advisor)
        if seed is None:
            seed = secrets.randbelow(2**48)
        elif not _is_int(seed) or not 0 <= seed <= MAX_SEED:
            raise PlayerError("seed must be a whole number, 0 or more")
        session = GameSession("g_" + secrets.token_hex(4), list(specs), advisor, int(seed), self.players, self.inference_lock)
        with self.lock:
            self.sessions[session.id] = session
            while len(self.sessions) > self.limit:
                self.sessions.popitem(last=False)
        return session

    def get(self, session_id: str) -> GameSession:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                raise NotFound(f"there is no game {session_id}")
            self.sessions.move_to_end(session_id)
            return session

    def delete(self, session_id: str) -> None:
        with self.lock:
            self.sessions.pop(session_id, None)
