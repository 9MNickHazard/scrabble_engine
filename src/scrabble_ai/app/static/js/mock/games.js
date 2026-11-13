/* Mock API for interactive games (loaded only with ?mock=1).
 *
 * A small scripted engine: two complete games, generated offline from the real word list by a
 * greedy-plus-leave player (every word valid, every score worked out from the premium layout,
 * with the top 15 candidates of each move). `ai` steps through the script; `move` accepts the
 * scripted human move and rejects anything else with a realistic reason; `hint` returns the
 * scripted candidates. Seeds pick a script (seed % 2), so the same seed replays the same game.
 *
 * Mock-only extra: POST /api/games accepts "mock_turns": n to start n moves into the script, which
 * the views' ?demo= flags use to reach a state without clicking.
 */
(function () {
  "use strict";
  if (!window.App || !App.mock) return;

  const N = 15;
  const HEX = "0123456789abcde";
  const VALUES = { A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8, K: 5, L: 1, M: 3, N: 1, O: 1,
    P: 3, Q: 10, R: 1, S: 1, T: 1, U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10, "?": 0 };
  const COUNTS = { A: 9, B: 2, C: 2, D: 4, E: 12, F: 2, G: 3, H: 2, I: 9, J: 1, K: 1, L: 4, M: 2, N: 6, O: 8,
    P: 2, Q: 1, R: 6, S: 4, T: 6, U: 4, V: 2, W: 2, X: 1, Y: 2, Z: 1, "?": 2 };

  const BOTS = [
    { spec: "bot:greedy-leave", name: "Greedy + leave", description: "Highest score plus leave value." },
    { spec: "bot:greedy", name: "Greedy", description: "Highest score." },
    { spec: "bot:random", name: "Random", description: "A random legal placement." },
  ];
  const now = Date.now() / 1000;
  const MODELS = [
    { id: "night1/best.pt", spec: "model:night1/best.pt", name: "night1 (best)", run: "night1", file: "best.pt",
      kind: "best", network: "6x64", parameters: 1055056, step: 61000, games: 1400000, modified: now - 5400,
      size_bytes: 4300000, compatible: true,
      eval: { "greedy-leave": { spread: 6.1, ci: 3.9, win_rate: 0.53 }, greedy: { spread: 44.0, ci: 4.4, win_rate: 0.69 } },
      starred: true, note: "first to beat the baseline" },
    { id: "night1/step_000071000.pt", spec: "model:night1/step_000071000.pt", name: "night1 (step 71,000)",
      run: "night1", file: "step_000071000.pt", kind: "checkpoint", network: "6x64", parameters: 1055056,
      step: 71000, games: 1612000, modified: now - 1800, size_bytes: 4300000, compatible: true,
      eval: { "greedy-leave": { spread: 5.2, ci: 4.1, win_rate: 0.525 }, greedy: null }, starred: false, note: "" },
    { id: "wide2/best.pt", spec: "model:wide2/best.pt", name: "wide2 (best)", run: "wide2", file: "best.pt",
      kind: "best", network: "10x128", parameters: 3655316, step: 22000, games: 410000, modified: now - 86400 * 2,
      size_bytes: 14600000, compatible: true,
      eval: { "greedy-leave": { spread: 1.8, ci: 5.2, win_rate: 0.51 }, greedy: { spread: 37.5, ci: 5.0, win_rate: 0.66 } },
      starred: false, note: "" },
    { id: "old-csw/best.pt", spec: "model:old-csw/best.pt", name: "old-csw (best)", run: "old-csw", file: "best.pt",
      kind: "best", network: "6x64", parameters: 1055056, step: 30000, games: 700000, modified: now - 86400 * 9,
      size_bytes: 4300000, compatible: false, eval: null, starred: false, note: "trained on another word list" },
  ];

  if (!App.mock.has("GET", "/api/models")) {
    App.mock.on("GET", /^\/api\/models$/, () => ({ bots: BOTS, models: MODELS }));
  }
  if (!App.mock.has("GET", "/api/status")) {
    App.mock.on("GET", /^\/api\/status$/, () => ({
      version: "0.1.0", lexicon: { id: "NSWL2023-91a67230e1f6", words: 195747 },
      leaves: { ok: true, file: "NSWL2023-91a67230e1f6.leaves" }, gpu: null,
      active_runs: [], queued_runs: [], active_matches: [],
    }));
  }

  // ---- Games -----------------------------------------------------------------------------

  const STORE = "mock.games";
  const games = new Map();
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORE) || "{}");
    for (const [id, g] of Object.entries(saved)) games.set(id, g);
  } catch (error) { /* private mode */ }

  function save() {
    try { sessionStorage.setItem(STORE, JSON.stringify(Object.fromEntries(games))); } catch (error) { /* ignore */ }
  }

  function nameFor(spec) {
    if (spec === "human") return "You";
    const bot = BOTS.find((b) => b.spec === spec);
    if (bot) return bot.name;
    const model = MODELS.find((m) => m.spec === spec);
    if (model) return model.name;
    const id = spec.replace(/^model:/, "");
    const [run, file] = id.split("/");
    if (!file) return id;
    if (file === "best.pt") return run + " (best)";
    const step = /(\d+)/.exec(file);
    return run + " (" + (step ? "step " + Number(step[1]).toLocaleString("en-US") : file) + ")";
  }

  const isModel = (spec) => spec.startsWith("model:");

  function decode(text) {
    const tiles = [];
    for (let i = 0; i + 2 < text.length; i += 3) {
      const ch = text[i + 2];
      tiles.push([HEX.indexOf(text[i]), HEX.indexOf(text[i + 1]), ch.toUpperCase(), ch !== ch.toUpperCase()]);
    }
    return tiles;
  }

  function splitText(text) {
    if (text === "pass" || text.startsWith("exchange")) return { position: null, word: null };
    const [position, word] = text.split(" ");
    return { position, word: word.replace(/[()]/g, "") };
  }

  function candidate(row, opts) {
    const [text, score, tiles, leave, leaveValue, value, total, win] = row;
    const kind = text === "pass" ? "pass" : text.startsWith("exchange") ? "exchange" : "play";
    const out = Object.assign({ text, kind }, splitText(text), {
      score, tiles: decode(tiles), leave, leave_value: leaveValue, value, total,
      win: opts.model ? win : null,
    });
    if (opts.chosen !== undefined) out.chosen = opts.chosen;
    return out;
  }

  function applyTiles(grid, tiles) {
    for (const [r, c, L, b] of tiles) grid[r][c] = b ? L.toLowerCase() : L;
  }

  function stateOf(g) {
    const script = SCRIPTS[g.script];
    const grid = Array.from({ length: N }, () => Array(N).fill("."));
    const racks = script.racks.slice();
    const totals = [0, 0];
    const history = [];
    let scoreless = 0;
    let lastPlaced = [];
    for (let i = 0; i < g.step; i++) {
      const m = script.moves[i];
      const tiles = decode(m.x);
      applyTiles(grid, tiles);
      totals[m.p] += m.s;
      racks[m.p] = m.a;
      const kind = m.e ? "exchange" : m.t === "pass" ? "pass" : "play";
      scoreless = kind === "play" ? 0 : scoreless + 1;
      const entry = Object.assign({ turn: i + 1, player: m.p, kind, text: m.t }, splitText(m.t), {
        score: m.s, total: totals[m.p], tiles, rack: m.r, leave: m.l, bingo: tiles.length === 7 });
      if (m.e) entry.exchanged = m.e;
      history.push(entry);
      lastPlaced = tiles.map(([r, c]) => [r, c]);
    }
    const over = g.step >= script.moves.length;
    const bag = g.step ? script.moves[g.step - 1].b : 86;
    const toMove = g.step % 2;
    const board = grid.map((row) => row.join(""));
    const unseen = Object.assign({}, COUNTS);
    for (const row of board) for (const ch of row) if (ch !== ".") unseen[ch === ch.toUpperCase() ? ch : "?"]--;
    for (const ch of racks[toMove]) unseen[ch]--;
    for (const key of Object.keys(unseen)) if (unseen[key] <= 0) delete unseen[key];
    const scores = over ? script.result.scores.slice() : totals;
    return {
      id: g.id, seed: g.seed,
      players: g.players.map((spec) => ({ spec, name: nameFor(spec) })),
      board, last_placed: lastPlaced, racks, scores,
      to_move: toMove, turn: g.step + 1, bag, scoreless_turns: scoreless,
      can_exchange: !over && bag >= 7, unseen, over,
      result: over ? JSON.parse(JSON.stringify(script.result)) : null,
      history,
    };
  }

  function getGame(id) {
    const g = games.get(id);
    if (!g) return null;
    return g;
  }

  App.mock.on("POST", /^\/api\/games$/, (match, body) => {
    const players = (body && body.players) || [];
    if (players.length !== 2) return { error: "a game needs two players" };
    for (const spec of players) {
      if (spec !== "human" && !BOTS.some((b) => b.spec === spec) && !/^model:.+\/.+/.test(spec)) {
        return { error: "unknown player " + spec };
      }
      const model = MODELS.find((m) => m.spec === spec);
      if (model && !model.compatible) return { error: model.name + " was trained with a different word list" };
    }
    if (players[0] === "human" && players[1] === "human") return { error: "at least one player must be a bot or a model" };
    const seed = body.seed === undefined || body.seed === null || body.seed === ""
      ? Math.floor(Math.random() * 1e9) : Number(body.seed);
    const id = "g_" + Math.random().toString(16).slice(2, 8);
    const script = Math.abs(Math.floor(seed)) % SCRIPTS.length;
    const strongest = players.find((p) => isModel(p)) || "bot:greedy-leave";
    const g = { id, seed, players, advisor: body.advisor || strongest, script, step: 0 };
    if (body.mock_turns) g.step = Math.max(0, Math.min(SCRIPTS[script].moves.length, Number(body.mock_turns)));
    games.set(id, g);
    save();
    return stateOf(g);
  });

  App.mock.on("GET", /^\/api\/games\/([^/]+)$/, (match) => {
    const g = getGame(match[1]);
    return g ? stateOf(g) : { error: "no game " + match[1] };
  });

  App.mock.on("DELETE", /^\/api\/games\/([^/]+)$/, (match) => {
    games.delete(match[1]);
    save();
    return { ok: true };
  });

  App.mock.on("POST", /^\/api\/games\/([^/]+)\/ai$/, (match) => {
    const g = getGame(match[1]);
    if (!g) return { error: "no game " + match[1] };
    const script = SCRIPTS[g.script];
    if (g.step >= script.moves.length) return { error: "the game is over" };
    const spec = g.players[g.step % 2];
    if (spec === "human") return { error: "the side to move is human" };
    const m = script.moves[g.step];
    g.step++;
    save();
    const model = isModel(spec);
    const candidates = m.c.map((row) => {
      const chosen = m.e ? row[0] === "exchange " + m.e : row[0] === m.t;
      return candidate(row, { model, chosen });
    });
    return {
      state: stateOf(g),
      analysis: { player: nameFor(spec), considered: model ? Math.min(64, m.n) : m.n, candidates },
    };
  });

  App.mock.on("GET", /^\/api\/games\/([^/]+)\/hint$/, (match, body, query) => {
    const g = getGame(match[1]);
    if (!g) return { error: "no game " + match[1] };
    const script = SCRIPTS[g.script];
    if (g.step >= script.moves.length) return { error: "the game is over" };
    const top = Math.max(1, Math.min(50, Number(query.top) || 15));
    const m = script.moves[g.step];
    return {
      advisor: nameFor(g.advisor),
      candidates: m.c.slice(0, top).map((row) => candidate(row, { model: isModel(g.advisor) })),
    };
  });

  App.mock.on("POST", /^\/api\/games\/([^/]+)\/move$/, (match, body) => {
    const g = getGame(match[1]);
    if (!g) return { error: "no game " + match[1] };
    const script = SCRIPTS[g.script];
    if (g.step >= script.moves.length) return { error: "the game is over" };
    if (g.players[g.step % 2] !== "human") return { error: "it is not the human player's turn" };
    const state = stateOf(g);
    const rack = state.racks[g.step % 2];
    const m = script.moves[g.step];
    const scriptedText = "The scripted move here is " + (m.e ? "an exchange of " + m.e : m.t) + ".";
    const scripted = "This mock game only follows its script. " + scriptedText;
    const accept = () => { g.step++; save(); return stateOf(g); };
    if (body && body.pass) return m.t === "pass" ? accept() : { error: scripted };
    if (body && typeof body.exchange === "string") {
      if (!state.can_exchange) return { error: "an exchange needs at least 7 tiles in the bag" };
      const letters = body.exchange.toUpperCase();
      if (!letters.length) return { error: "choose at least one tile to exchange" };
      if (!holds(rack, letters.split(""))) return { error: "the rack does not hold those tiles" };
      if (m.e && sorted(letters) === sorted(m.e)) return accept();
      return { error: scripted };
    }
    const tiles = ((body && body.tiles) || []).map((t) => [t.row, t.col, String(t.letter).toUpperCase(), !!t.blank]);
    const reason = check(state.board, rack, tiles);
    if (reason) return { error: reason };
    const want = decode(m.x).map((t) => t.join(",")).sort().join(";");
    const got = tiles.map((t) => t.join(",")).sort().join(";");
    return want === got ? accept() : { error: scripted };
  });

  const sorted = (s) => s.split("").sort().join("");

  function holds(rack, tiles) {
    const left = rack.split("");
    for (const t of tiles) {
      const i = left.indexOf(t);
      if (i < 0) return false;
      left.splice(i, 1);
    }
    return true;
  }

  /** The reason a placement is illegal by its shape, its tiles or a known non-word, or null. */
  function check(board, rack, tiles) {
    if (!tiles.length) return "place at least one tile";
    if (!holds(rack, tiles.map((t) => (t[3] ? "?" : t[2])))) return "the rack does not hold the tiles for this play";
    const at = new Map();
    for (const [r, c, L] of tiles) {
      if (r < 0 || r >= N || c < 0 || c >= N) return "a tile is off the board";
      if (board[r][c] !== ".") return "square " + "ABCDEFGHIJKLMNO"[c] + (r + 1) + " is already taken";
      if (at.has(r * N + c)) return "two tiles are on the same square";
      at.set(r * N + c, L);
    }
    const letterAt = (r, c) => {
      if (r < 0 || r >= N || c < 0 || c >= N) return null;
      if (at.has(r * N + c)) return at.get(r * N + c);
      return board[r][c] === "." ? null : board[r][c].toUpperCase();
    };
    const rows = new Set(tiles.map((t) => t[0]));
    const cols = new Set(tiles.map((t) => t[1]));
    if (rows.size > 1 && cols.size > 1) return "the tiles must be in one row or one column";
    const empty = board.every((row) => row === ".".repeat(N));
    if (empty) {
      if (tiles.length < 2) return "the first play needs at least two tiles";
      if (!at.has(7 * N + 7)) return "the first play must cover the centre square, H8";
    }
    const down = tiles.length > 1 && cols.size === 1;
    const lineWord = (r, c, isDown) => {
      let r0 = r, c0 = c;
      while (letterAt(isDown ? r0 - 1 : r0, isDown ? c0 : c0 - 1)) { if (isDown) r0--; else c0--; }
      let word = "";
      let rr = r0, cc = c0;
      while (letterAt(rr, cc)) { word += letterAt(rr, cc); if (isDown) rr++; else cc++; }
      return word;
    };
    if (tiles.length > 1) {
      const span = tiles.map((t) => (down ? t[0] : t[1]));
      const lo = Math.min(...span), hi = Math.max(...span);
      for (let i = lo; i <= hi; i++) {
        if (!letterAt(down ? i : tiles[0][0], down ? tiles[0][1] : i)) return "the tiles must form one word with no gaps";
      }
    }
    if (!empty) {
      const touches = tiles.some(([r, c]) => [[r - 1, c], [r + 1, c], [r, c - 1], [r, c + 1]]
        .some(([y, x]) => y >= 0 && y < N && x >= 0 && x < N && board[y][x] !== "."));
      if (!touches) return "the play must connect to the tiles on the board";
    }
    const words = [];
    if (tiles.length > 1) {
      words.push(lineWord(tiles[0][0], tiles[0][1], down));
      for (const [r, c] of tiles) { const w = lineWord(r, c, !down); if (w.length > 1) words.push(w); }
    } else {
      for (const isDown of [false, true]) { const w = lineWord(tiles[0][0], tiles[0][1], isDown); if (w.length > 1) words.push(w); }
      if (!words.length) return "a single tile must form a word with a tile on the board";
    }
    // The mock has no dictionary (word lists are copyrighted and are never part of this repo):
    // it knows a few non-words, so the "not a word" message can be seen, and nothing more.
    for (const w of words) if (NOT_WORDS.has(w)) return w + " is not a word";
    return null;
  }

  // SCRIPTS: two games generated by a greedy-plus-leave player (seeds 11 and 19).
  const SCRIPTS = [{"seed":11,"racks":["IIUSEUA","IZEOABE"],"result":{"scores":[466,405],"winner":0,"adjustment":[16,0],"ended_by":"out"},"moves":[{"p":0,"t":"exchange 4","s":0,"x":"","r":"IIUSEUA","l":"AES","a":"SEAGGXT","b":86,"n":77,"c":[["exchange IIUU",0,"","AES",12,12,12,0.525],["exchange AIIUU",0,"","ES",11.5,11.5,11.5,0.523],["exchange AIUU",0,"","EIS",10,10,10,0.517],["exchange EIIUU",0,"","AS",8.5,8.5,8.5,0.511],["exchange AEIIUU",0,"","S",8,8,8,0.508],["exchange IUU",0,"","AEIS",8,8,8,0.508],["exchange AIIU",0,"","ESU",7.5,7.5,7.5,0.506],["exchange EIUU",0,"","AIS",7,7,7,0.504],["exchange AEIUU",0,"","IS",6.5,6.5,6.5,0.502],["exchange IIU",0,"","AESU",5.5,5.5,5.5,0.498],["exchange EIIU",0,"","ASU",4.5,4.5,4.5,0.494],["exchange AEIIU",0,"","SU",4,4,4,0.492],["exchange AIISUU",0,"","E",3.5,3.5,3.5,0.489],["exchange AIU",0,"","EISU",3.5,3.5,3.5,0.489],["exchange AEIU",0,"","ISU",2.5,2.5,2.5,0.485]],"e":"IIUU"},{"p":1,"t":"8D ZOEAE","s":48,"x":"73Z74O75E76A77E","r":"IZEOABE","l":"BI","a":"IBYTNOS","b":81,"n":151,"c":[["8D ZOEAE",48,"73Z74O75E76A77E","BI",-3.5,-3.5,44.5,0.658],["H4 ZOEAE",48,"37Z47O57E67A77E","BI",-3.5,-3.5,44.5,0.658],["8D BAIZE",38,"73B74A75I76Z77E","EO",-0.5,-0.5,37.5,0.63],["H4 BAIZE",38,"37B47A57I67Z77E","EO",-0.5,-0.5,37.5,0.63],["8H BAIZE",34,"77B78A79I7aZ7bE","EO",-0.5,-0.5,33.5,0.614],["H8 BAIZE",34,"77B87A97Ia7Zb7E","EO",-0.5,-0.5,33.5,0.614],["8E BAIZE",32,"74B75A76I77Z78E","EO",-0.5,-0.5,31.5,0.606],["8F BAIZE",32,"75B76A77I78Z79E","EO",-0.5,-0.5,31.5,0.606],["8G BAIZE",32,"76B77A78I79Z7aE","EO",-0.5,-0.5,31.5,0.606],["H5 BAIZE",32,"47B57A67I77Z87E","EO",-0.5,-0.5,31.5,0.606],["H6 BAIZE",32,"57B67A77I87Z97E","EO",-0.5,-0.5,31.5,0.606],["H7 BAIZE",32,"67B77A87I97Za7E","EO",-0.5,-0.5,31.5,0.606],["8E BIZE",30,"74B75I76Z77E","AEO",-2.5,-2.5,27.5,0.59],["8F BIZE",30,"75B76I77Z78E","AEO",-2.5,-2.5,27.5,0.59],["8G BIZE",30,"76B77I78Z79E","AEO",-2.5,-2.5,27.5,0.59]]},{"p":0,"t":"9C TAX","s":31,"x":"82T83A84X","r":"SEAGGXT","l":"EGGS","a":"SEGGIEL","b":78,"n":430,"c":[["9C TAX",31,"82T83A84X","EGGS",-1.6,-1.6,29.4,0.393],["D8 (Z)AXES",42,"83A93Xa3Eb3S","GGT",-15.1,-15.1,26.9,0.382],["9C GAST",22,"82G83A84S85T","EGX",4.5,4.5,26.5,0.381],["F7 T(E)GG",10,"65T85G95G","AESX",15.5,15.5,25.5,0.377],["9D AX",29,"83A84X","EGGST",-3.6,-3.6,25.4,0.376],["E6 GE(O)TAG",16,"54G64E84T94Aa4G","SX",9,9,25,0.374],["F8 (E)GG",9,"85G95G","AESTX",16,16,25,0.374],["G7 G(A)G",9,"66G86G","AESTX",16,16,25,0.374],["E7 G(O)XES",26,"64G84X94Ea4S","AGT",-1.5,-1.5,24.5,0.372],["9F TAX",25,"85T86A87X","EGGS",-1.6,-1.6,23.4,0.368],["7E GAT",13,"64G65A66T","EGSX",10,10,23,0.366],["G7 G(A)GA",10,"66G86G96A","ESTX",13,13,23,0.366],["7G TAX",24,"66T67A68X","EGGS",-1.6,-1.6,22.4,0.364],["7G TAG",12,"66T67A68G","EGSX",10,10,22,0.362],["9C SAX",31,"82S83A84X","EGGT",-9.1,-9.1,21.9,0.362]]},{"p":1,"t":"7E BOY","s":35,"x":"64B65O66Y","r":"IBYTNOS","l":"INST","a":"ITNSHFY","b":75,"n":535,"c":[["7E BOY",35,"64B65O66Y","INST",5,5,40,0.717],["10D SYBO",40,"93S94Y95B96O","INT",-0.5,-0.5,39.5,0.715],["7E BONY",32,"64B65O66N67Y","IST",7,7,39,0.713],["D6 BY(ZA)NTS",42,"53B63Y93Na3Tb3S","IO",-5.5,-5.5,36.5,0.703],["10B BISON",39,"91B92I93S94O95N","TY",-2.5,-2.5,36.5,0.703],["D8 (ZA)YIN",34,"93Ya3Ib3N","BOST",2.5,2.5,36.5,0.703],["7E SYBO",36,"64S65Y66B67O","INT",-0.5,-0.5,35.5,0.7],["10A NOISY",37,"90N91O92I93S94Y","BT",-4,-4,33,0.69],["10B NOSY",36,"91N92O93S94Y","BIT",-3,-3,33,0.69],["D8 (ZA)YINS",36,"93Ya3Ib3Nc3S","BOT",-3,-3,33,0.69],["E5 BOT(OX)",28,"44B54O64T","INSY",4,4,32,0.686],["E5 INB(OX)",28,"44I54N64B","OSTY",4,4,32,0.686],["10D SOY",36,"93S94O95Y","BINT",-5,-5,31,0.682],["7G YOB",26,"66Y67O68B","INST",5,5,31,0.682],["7E BY",24,"64B65Y","INOST",6,6,30,0.678]]},{"p":0,"t":"9H LEGGIES","s":63,"x":"87L88E89G8aG8bI8cE8dS","r":"SEGGIEL","l":"","a":"WRTAKLD","b":68,"n":103,"c":[["9H LEGGIES",63,"87L88E89G8aG8bI8cE8dS","",0,0,63,0.523],["10B GIG",24,"91G92I93G","EELS",11.5,11.5,35.5,0.396],["10A GLEG",23,"90G91L92E93G","EIS",10,10,33,0.385],["9G GLEG",20,"86G87L88E89G","EIS",10,10,30,0.372],["10B LEG",21,"91L92E93G","EGIS",7.5,7.5,28.5,0.366],["10A EGIS",23,"90E91G92I93S","EGL",1,1,24,0.346],["10B GIS",22,"91G92I93S","EEGL",1,1,23,0.342],["9G GLEE",19,"86G87L88E89E","GIS",4,4,23,0.342],["10A GIES",21,"90G91I92E93S","EGL",1,1,22,0.338],["10B SIG",21,"91S92I93G","EEGL",1,1,22,0.338],["D8 (ZA)G",13,"93G","EEGILS",7.5,7.5,20.5,0.332],["E7 (BOX)ES",28,"94Ea4S","EGGIL",-8.6,-8.6,19.4,0.327],["G7 (YA)GI",10,"86G96I","EEGLS",9,9,19,0.325],["10B GI",9,"91G92I","EEGLS",9,9,18,0.321],["G7 (YA)GE",10,"86G96E","EGILS",7.5,7.5,17.5,0.319]]},{"p":1,"t":"O5 SHIFTY","s":55,"x":"4eS5eH6eI7eF8eT9eY","r":"ITNSHFY","l":"N","a":"NNUACIW","b":62,"n":494,"c":[["O5 SHIFTY",55,"4eS5eH6eI7eF8eT9eY","N",0.5,0.5,55.5,0.684],["N9 (S)HIFTY",46,"9dHadIbdFcdTddY","NS",6,6,52,0.668],["O6 NIFTY",43,"5eN6eI7eF8eT9eY","HS",6.5,6.5,49.5,0.657],["O5 SHINTY",46,"4eS5eH6eI7eN8eT9eY","F",-2,-2,44,0.632],["N9 (S)HINTY",40,"9dHadIbdNcdTddY","FS",3.5,3.5,43.5,0.629],["O5 SHIFT",43,"4eS5eH6eI7eF8eT","NY",-2.5,-2.5,40.5,0.615],["D8 (ZA)NYISH",44,"93Na3Yb3Ic3Sd3H","FT",-4,-4,40,0.613],["E7 (BOX)FISH",44,"94Fa4Ib4Sc4H","NTY",-4.5,-4.5,39.5,0.611],["O6 SYNTH",43,"5eS6eY7eN8eT9eH","FI",-3.5,-3.5,39.5,0.611],["N9 (S)HIFTS",40,"9dHadIbdFcdTddS","NY",-2.5,-2.5,37.5,0.601],["L7 SH(I)FTY",38,"6bS7bH9bFabTbbY","IN",-1,-1,37,0.599],["N9 (S)YNTHS",40,"9dYadNbdTcdHddS","FI",-3.5,-3.5,36.5,0.596],["8L TYIN",34,"7bT7cY7dI7eN","FHS",2,2,36,0.594],["L8 F(I)SHY",36,"7bF9bSabHbbY","INT",-0.5,-0.5,35.5,0.591],["8L THIN",34,"7bT7cH7dI7eN","FSY",0.5,0.5,34.5,0.587]]},{"p":0,"t":"D8 (ZA)KAT","s":36,"x":"93Ka3Ab3T","r":"WRTAKLD","l":"DLRW","a":"WRLDANB","b":59,"n":393,"c":[["D8 (ZA)KAT",36,"93Ka3Ab3T","DLRW",-9.5,-9.5,26.5,0.378],["6F DRAWL",29,"55D56R57A58W59L","KT",-2.5,-2.5,26.5,0.378],["5K DAWK(S)",26,"4aD4bA4cW4dK","LRT",-3,-3,23,0.36],["5K WALK(S)",24,"4aW4bA4cL4dK","DRT",-2.5,-2.5,21.5,0.353],["6F DRAW",26,"55D56R57A58W","KLT",-5,-5,21,0.35],["6F TRAWL",23,"55T56R57A58W59L","DK",-2.5,-2.5,20.5,0.348],["5K WARK(S)",24,"4aW4bA4cR4dK","DLT",-4,-4,20,0.346],["K9 (G)AWK",24,"9aAaaWbaK","DLRT",-5,-5,19,0.341],["5J DRAWL(S)",20,"49D4aR4bA4cW4dL","KT",-2.5,-2.5,17.5,0.334],["E7 (BOX)LA",28,"94La4A","DKRTW",-12,-12,16,0.327],["5J TRAWL(S)",18,"49T4aR4bA4cW4dL","DK",-2.5,-2.5,15.5,0.324],["M4 TRAWL(E)D",15,"3cT4cR5cA6cW7cL9cD","K",-0.5,-0.5,14.5,0.32],["M5 WALK(E)D",15,"4cW5cA6cL7cK9cD","RT",-0.5,-0.5,14.5,0.32],["5K DAWT(S)",18,"4aD4bA4cW4dT","KLR",-4,-4,14,0.317],["M7 TW(E)RK",13,"6cT7cW9cRacK","ADL",1,1,14,0.317]]},{"p":1,"t":"E5 UN(BOX)","s":28,"x":"44U54N","r":"NNUACIW","l":"ACINW","a":"NACIWEU","b":57,"n":295,"c":[["E5 UN(BOX)",28,"44U54N","ACINW",-4,-4,24,0.639],["E11 WAIN",26,"a4Wb4Ac4Id4N","CNU",-3,-3,23,0.634],["E5 IN(BOX)",28,"44I54N","ACNUW",-6.5,-6.5,21.5,0.626],["E11 WIN",24,"a4Wb4Ic4N","ACNU",-2.5,-2.5,21.5,0.626],["N2 UNCIA",27,"1dU2dN3dC4dI5dA","NW",-6,-6,21,0.624],["K4 CAWIN(G)",24,"3aC4aA5aW6aI7aN","NU",-3.5,-3.5,20.5,0.621],["E11 WAN",24,"a4Wb4Ac4N","CINU",-4.5,-4.5,19.5,0.616],["J6 WAN(G)UN",20,"59W69A79N99Ua9N","CI",-1,-1,19,0.613],["K6 WAN(G)UN",20,"5aW6aA7aN9aUaaN","CI",-1,-1,19,0.613],["J4 CAWIN(G)",20,"39C49A59W69I79N","NU",-3.5,-3.5,16.5,0.6],["K4 AWNIN(G)",20,"3aA4aW5aN6aI7aN","CU",-3.5,-3.5,16.5,0.6],["K4 WANIN(G)",20,"3aW4aA5aN6aI7aN","CU",-3.5,-3.5,16.5,0.6],["6K WINC(H)",19,"5aW5bI5cN5dC","ANU",-3,-3,16,0.598],["K5 AWIN(G)",18,"4aA5aW6aI7aN","CNU",-3,-3,15,0.592],["K7 WI(G)AN",18,"6aW7aI9aAaaN","CNU",-3,-3,15,0.592]]},{"p":0,"t":"D1 BLAWN","s":28,"x":"03B13L23A33W43N","r":"WRLDANB","l":"DR","a":"RDTEITE","b":52,"n":785,"c":[["D1 BLAWN",28,"03B13L23A33W43N","DR",-0.5,-0.5,27.5,0.419],["E11 BAWD",30,"a4Bb4Ac4Wd4D","LNR",-3,-3,27,0.417],["11B BR(A)DAWL",26,"a1Ba2Ra4Da5Aa6Wa7L","N",0.5,0.5,26.5,0.414],["D1 BRAWN",28,"03B13R23A33W43N","DL",-2,-2,26,0.411],["E11 BAWL",28,"a4Bb4Ac4Wd4L","DNR",-2.5,-2.5,25.5,0.408],["E11 BAWN",28,"a4Bb4Ac4Wd4N","DLR",-3,-3,25,0.406],["E11 WAB",28,"a4Wb4Ac4B","DLNR",-5,-5,23,0.395],["C11 BAWD",26,"a2Bb2Ac2Wd2D","LNR",-3,-3,23,0.395],["E11 WAND",28,"a4Wb4Ac4Nd4D","BLR",-5.5,-5.5,22.5,0.392],["11B BL(A)WN",20,"a1Ba2La4Wa5N","ADR",2.5,2.5,22.5,0.392],["E11 WARD",28,"a4Wb4Ac4Rd4D","BLN",-6.5,-6.5,21.5,0.387],["C11 BAWL",24,"a2Bb2Ac2Wd2L","DNR",-2.5,-2.5,21.5,0.387],["11B BR(A)WL",20,"a1Ba2Ra4Wa5L","ADN",1.5,1.5,21.5,0.387],["C11 BAWN",24,"a2Bb2Ac2Wd2N","DLR",-3,-3,21,0.384],["11B BR(A)WN",20,"a1Ba2Ra4Wa5N","ADL",1,1,21,0.384]]},{"p":1,"t":"N2 UNCIA","s":27,"x":"1dU2dN3dC4dI5dA","r":"NACIWEU","l":"EW","a":"WEIAPR?","b":47,"n":655,"c":[["N2 UNCIA",27,"1dU2dN3dC4dI5dA","EW",-0.5,-0.5,26.5,0.663],["E11 WINCE",32,"a4Wb4Ic4Nd4Ce4E","AU",-6,-6,26,0.661],["2B AU(L)IC",26,"11A12U14I15C","ENW",0,0,26,0.661],["E11 WAIN",26,"a4Wb4Ac4Id4N","CEU",0,0,26,0.661],["N1 UNCIAE",29,"0dU1dN2dC3dI4dA5dE","W",-4,-4,25,0.655],["E11 WINE",26,"a4Wb4Ic4Nd4E","ACU",-3,-3,23,0.645],["2B UN(L)ACE",28,"11U12N14A15C16E","IW",-5.5,-5.5,22.5,0.642],["E11 WIN",24,"a4Wb4Ic4N","ACEU",-2,-2,22,0.639],["E11 WANE",26,"a4Wb4Ac4Nd4E","CIU",-5,-5,21,0.634],["E11 WEAN",26,"a4Wb4Ec4Ad4N","CIU",-5,-5,21,0.634],["K4 CAWIN(G)",24,"3aC4aA5aW6aI7aN","EU",-3,-3,21,0.634],["2B IN(L)ACE",28,"11I12N14A15C16E","UW",-8,-8,20,0.628],["1A NEW(B)",27,"00N01E02W","ACIU",-7,-7,20,0.628],["E11 WAN",24,"a4Wb4Ac4N","CEIU",-4,-4,20,0.628],["1A CEI(B)A",27,"00C01E02I04A","NUW",-7.5,-7.5,19.5,0.625]]},{"p":0,"t":"1D (B)ITTERED","s":83,"x":"04I05T06T07E08R09E0aD","r":"RDTEITE","l":"","a":"NAVIEIO","b":40,"n":759,"c":[["1D (B)ITTERED",83,"04I05T06T07E08R09E0aD","",0,0,83,0.743],["2D (L)ITTERED",63,"14I15T16T17E18R19E1aD","",0,0,63,0.636],["12A TIT(T)ERED",62,"b0Tb1Ib2Tb4Eb5Rb6Eb7D","",0,0,62,0.63],["12B TI(T)TERED",60,"b1Tb2Ib4Tb5Eb6Rb7Eb8D","",0,0,60,0.618],["12D (T)ITTERED",60,"b4Ib5Tb6Tb7Eb8Rb9EbaD","",0,0,60,0.618],["1D (B)ITTED",27,"04I05T06T07E08D","ER",5,5,32,0.443],["1D (B)IDET",24,"04I05D06E07T","ERT",5.5,5.5,29.5,0.428],["1D (B)ITTER",24,"04I05T06T07E08R","DE",4,4,28,0.418],["1D (B)ETTED",27,"04E05T06T07E08D","IR",0,0,27,0.412],["2A TIT(L)ED",22,"10T11I12T14E15D","ER",5,5,27,0.412],["1D (B)ETIDE",27,"04E05T06I07D08E","RT",-0.5,-0.5,26.5,0.409],["2A TIR(L)ED",22,"10T11I12R14E15D","ET",4,4,26,0.406],["1A TRI(B)E",21,"00T01R02I04E","DET",4.5,4.5,25.5,0.403],["1D (B)ITER",21,"04I05T06E07R","DET",4.5,4.5,25.5,0.403],["2B TI(L)ED",20,"11T12I14E15D","ERT",5.5,5.5,25.5,0.403]]},{"p":1,"t":"13B WAsPIER","s":92,"x":"c1Wc2Ac3sc4Pc5Ic6Ec7R","r":"WEIAPR?","l":"","a":"DQNRUJS","b":33,"n":10755,"c":[["13B WAsPIER",92,"c1Wc2Ac3sc4Pc5Ic6Ec7R","",0,0,92,0.742],["L4 PAIRW(I)sE",82,"3bP4bA5bI6bR7bW9bsabE","",0,0,82,0.685],["L7 PA(I)RWIsE",76,"6bP7bA9bRabWbbIcbsdbE","",0,0,76,0.648],["2J WAP",32,"19W1aA1bP","EIR?",27.5,27.5,59.5,0.538],["E11 WIPER",32,"a4Wb4Ic4Pd4Ee4R","A?",24.5,24.5,56.5,0.517],["E11 WIPE",30,"a4Wb4Ic4Pd4E","AR?",26,26,56,0.514],["2J WAR",30,"19W1aA1bR","EIP?",25.5,25.5,55.5,0.51],["E11 WAP",28,"a4Wb4Ac4P","EIR?",27.5,27.5,55.5,0.51],["2J WAE",30,"19W1aA1bE","IPR?",23.5,23.5,53.5,0.497],["E11 WARP",30,"a4Wb4Ac4Rd4P","EI?",23.5,23.5,53.5,0.497],["2J WE",29,"19W1aE","AIPR?",24,24,53,0.493],["E11 WAIR",26,"a4Wb4Ac4Id4R","EP?",27,27,53,0.493],["2I EW",28,"18E19W","AIPR?",24,24,52,0.486],["M5 PAW",23,"4cP5cA6cW","EIR?",27.5,27.5,50.5,0.476],["M5 WAP",23,"4cW5cA6cP","EIR?",27.5,27.5,50.5,0.476]]},{"p":0,"t":"4H AVIONI(C)","s":26,"x":"37A38V39I3aO3bN3cI","r":"NAVIEIO","l":"E","a":"EIMDEOO","b":27,"n":732,"c":[["4H AVIONI(C)",26,"37A38V39I3aO3bN3cI","E",3.5,3.5,29.5,0.347],["H11 VI(R)EO",27,"a7Vb7Id7Ee7O","AIN",-0.5,-0.5,26.5,0.327],["N9 (S)AVINE",29,"9dAadVbdIcdNddE","IO",-5.5,-5.5,23.5,0.307],["12H AVION",21,"b7Ab8Vb9IbaObbN","EI",-0.5,-0.5,20.5,0.287],["K7 VI(G)IA",18,"6aV7aI9aIaaA","ENO",2.5,2.5,20.5,0.287],["3J AVIO(N)",16,"29A2aV2bI2cO","EIN",2.5,2.5,18.5,0.275],["12H ENVOI",21,"b7Eb8Nb9VbaObbI","AI",-3.5,-3.5,17.5,0.269],["12H OVINE",21,"b7Ob8Vb9IbaNbbE","AI",-3.5,-3.5,17.5,0.269],["4J VINI(C)",20,"39V3aI3bN3cI","AEO",-2.5,-2.5,17.5,0.269],["K7 VO(G)IE",18,"6aV7aO9aIaaE","AIN",-0.5,-0.5,17.5,0.269],["K8 O(G)IVE",18,"7aO9aIaaVbaE","AIN",-0.5,-0.5,17.5,0.269],["H12 A(R)VO",24,"b7Ad7Ve7O","EIIN",-7.1,-7.1,16.9,0.265],["K1 (D)IVINE",20,"1aI2aV3aI4aN5aE","AO",-3.5,-3.5,16.5,0.263],["3K OVI(N)E",16,"2aO2bV2cI2eE","AIN",-0.5,-0.5,15.5,0.257],["14H ENVOI",18,"d7Ed8Nd9VdaOdbI","AI",-3.5,-3.5,14.5,0.251]]},{"p":1,"t":"J1 (E)QU(I)D","s":35,"x":"19Q29U49D","r":"DQNRUJS","l":"JNRS","a":"NRJSOOD","b":24,"n":525,"c":[["J1 (E)QU(I)D",35,"19Q29U49D","JNRS",0.5,0.5,35.5,0.833],["J1 (E)QU(I)DS",38,"19Q29U49D59S","JNR",-5,-5,33,0.821],["L8 J(I)RDS",42,"7bJ9bRabDbbS","NQU",-10.5,-10.5,31.5,0.814],["3B QU(A)D",28,"21Q22U24D","JNRS",0.5,0.5,28.5,0.798],["11B QU(A)D",28,"a1Qa2Ua4D","JNRS",0.5,0.5,28.5,0.798],["M7 QU(E)RN",24,"6cQ7cU9cRacN","DJS",1.5,1.5,25.5,0.781],["3A SQU(A)D",30,"20S21Q22U24D","JNR",-5,-5,25,0.778],["3B QU(A)DS",30,"21Q22U24D25S","JNR",-5,-5,25,0.778],["11A SQU(A)D",30,"a0Sa1Qa2Ua4D","JNR",-5,-5,25,0.778],["11B QU(A)DS",30,"a1Qa2Ua4Da5S","JNR",-5,-5,25,0.778],["3B QU(A)",24,"21Q22U","DJNRS",-1.5,-1.5,22.5,0.763],["M7 QU(E)RNS",25,"6cQ7cU9cRacNbcS","DJ",-4,-4,21,0.754],["F11 QU(I)D",18,"a5Qb5Ud5D","JNRS",0.5,0.5,18.5,0.738],["L8 Q(I)",21,"7bQ","DJNRSU",-3,-3,18,0.734],["F10 SQU(I)D",21,"95Sa5Qb5Ud5D","JNR",-5,-5,16,0.721]]},{"p":0,"t":"B10 MEO(W)ED","s":36,"x":"91Ma1Eb1Od1Ee1D","r":"EIMDEOO","l":"IO","a":"IOOUIHM","b":19,"n":622,"c":[["B10 MEO(W)ED",36,"91Ma1Eb1Od1Ee1D","IO",-5.5,-5.5,30.5,0.265],["E11 MO(P)ED",30,"a4Mb4Od4Ee4D","EIO",-4.5,-4.5,25.5,0.232],["E11 MO(P)E",26,"a4Mb4Od4E","DEIO",-1.5,-1.5,24.5,0.226],["C2 EMO",25,"12E22M32O","DEIO",-1.5,-1.5,23.5,0.22],["H11 OO(R)IE",18,"a7Ob7Od7Ie7E","DEM",4.5,4.5,22.5,0.214],["14A MOI",22,"d0Md1Od2I","DEEO",0,0,22,0.211],["K3 M(O)OED",26,"2aM4aO5aE6aD","EIO",-4.5,-4.5,21.5,0.208],["10I MOOD",24,"98M99O9aO9bD","EEI",-3,-3,21,0.205],["E11 DO(P)E",22,"a4Db4Od4E","EIMO",-1.5,-1.5,20.5,0.202],["B11 DO(W)IE",18,"a1Db1Od1Ie1E","EMO",2.5,2.5,20.5,0.202],["E11 MO(P)",24,"a4Mb4O","DEEIO",-4,-4,20,0.199],["K9 (G)OODIE",16,"9aOaaObaDcaIdaE","EM",4,4,20,0.199],["O1 EMO",21,"0eE1eM2eO","DEIO",-1.5,-1.5,19.5,0.196],["2C O(L)DIE",17,"12O14D15I16E","EMO",2.5,2.5,19.5,0.196],["N8 I(S)M",29,"7dI9dM","DEEOO",-9.6,-9.6,19.4,0.196]]},{"p":1,"t":"A7 DOJO","s":40,"x":"60D70O80J90O","r":"NRJSOOD","l":"NRS","a":"NRSRRCR","b":15,"n":775,"c":[["A7 DOJO",40,"60D70O80J90O","NRS",5,5,45,0.9],["L8 J(I)RDS",42,"7bJ9bRabDbbS","NOO",-8.1,-8.1,33.9,0.853],["A8 SNOOD",35,"70S80N90Oa0Ob0D","JR",-3,-3,32,0.844],["A7 DONORS",33,"60D70O80N90Oa0Rb0S","J",-2,-2,31,0.838],["6G RODS",31,"56R57O58D59S","JNO",-3,-3,28,0.822],["A7 DONOR",24,"60D70O80N90Oa0R","JS",3.5,3.5,27.5,0.819],["A8 ROODS",31,"70R80O90Oa0Db0S","JN",-4,-4,27,0.816],["A8 DOORS",30,"70D80O90Oa0Rb0S","JN",-4,-4,26,0.81],["A8 ODORS",30,"70O80D90Oa0Rb0S","JN",-4,-4,26,0.81],["A6 RONDO",22,"50R60O70N80D90O","JS",3.5,3.5,25.5,0.807],["6G ROOS",30,"56R57O58O59S","DJN",-6,-6,24,0.798],["12G DOJO",19,"b6Db7Ob8Jb9O","NRS",5,5,24,0.798],["A8 ROOD",22,"70R80O90Oa0D","JNS",1.5,1.5,23.5,0.795],["15B (D)OJO",20,"e2Oe3Je4O","DNRS",3,3,23,0.791],["A8 DOOR",21,"70D80O90Oa0R","JNS",1.5,1.5,22.5,0.788]]},{"p":0,"t":"15A O(D)IUM","s":27,"x":"e0Oe2Ie3Ue4M","r":"IOOUIHM","l":"HIO","a":"OIHLELE","b":11,"n":349,"c":[["15A O(D)IUM",27,"e0Oe2Ie3Ue4M","HIO",-2,-2,25,0.166],["15A I(D)IOM",27,"e0Ie2Ie3Oe4M","HOU",-4.5,-4.5,22.5,0.151],["8L HOO(F)",27,"7bH7cO7dO","IIMU",-14.6,-14.6,12.4,0.104],["3B MI(A)OU",14,"21M22I24O25U","HIO",-2,-2,12,0.102],["A14 OI",17,"d0Oe0I","HIMOU",-5.5,-5.5,11.5,0.101],["N8 I(S)M",29,"7dI9dM","HIOOU",-18.1,-18.1,10.9,0.098],["10I MOI",19,"98M99O9aI","HIOU",-8.5,-8.5,10.5,0.097],["A14 HI",29,"d0He0I","IMOOU",-18.6,-18.6,10.4,0.096],["A14 HO",29,"d0He0O","IIMOU",-18.6,-18.6,10.4,0.096],["O1 OHO",25,"0eO1eH2eO","IIMU",-14.6,-14.6,10.4,0.096],["O1 MHO",31,"0eM1eH2eO","IIOU",-21.6,-21.6,9.4,0.093],["E11 HI(P)",28,"a4Hb4I","IMOOU",-18.6,-18.6,9.4,0.093],["E11 HO(P)",28,"a4Hb4O","IIMOU",-18.6,-18.6,9.4,0.093],["B6 MOI",16,"51M61O71I","HIOU",-8.5,-8.5,7.5,0.086],["L2 IO(N)IUM",8,"1bI2bO4bI5bU6bM","HO",-0.5,-0.5,7.5,0.086]]},{"p":1,"t":"11A S(E)R(A)C","s":27,"x":"a0Sa2Ra4C","r":"NRSRRCR","l":"NRRR","a":"NRRRGAP","b":8,"n":154,"c":[["11A S(E)R(A)C",27,"a0Sa2Ra4C","NRRR",-9.5,-9.5,17.5,0.873],["3C C(A)RR",12,"22C24R25R","NRRS",0.5,0.5,12.5,0.845],["3C C(A)RRS",16,"22C24R25R26S","NRR",-5,-5,11,0.836],["3C C(A)RN",12,"22C24R25N","RRRS",-2,-2,10,0.829],["3C N(A)RC",12,"22N24R25C","RRRS",-2,-2,10,0.829],["3C C(A)RNS",16,"22C24R25N26S","RRR",-7.5,-7.5,8.5,0.819],["3C N(A)RCS",16,"22N24R25C26S","RRR",-7.5,-7.5,8.5,0.819],["exchange CNRRRR",0,"","S",8,8,8,0.816],["exchange CNRRR",0,"","RS",7,7,7,0.809],["3A SCR(A)N",14,"20S21C22R24N","RRR",-7.5,-7.5,6.5,0.805],["K9 (G)RR",8,"9aRaaR","CNRRS",-1.5,-1.5,6.5,0.805],["H3 C(A)RR",6,"27C47R57R","NRRS",0.5,0.5,6.5,0.805],["3C C(A)R",10,"22C24R","NRRRS",-4,-4,6,0.801],["11D (A)RC",10,"a4Ra5C","NRRRS",-4,-4,6,0.801],["exchange CRRRR",0,"","NS",6,6,6,0.801]]},{"p":0,"t":"10F HILLO","s":28,"x":"95H96I97L98L99O","r":"OIHLELE","l":"EE","a":"EEA?FST","b":3,"n":426,"c":[["10F HILLO",28,"95H96I97L98L99O","EE",1,1,29,0.154],["10F HELLO",28,"95H96E97L98L99O","EI",-0.5,-0.5,27.5,0.145],["N8 I(S)OHEL",23,"7dI9dOadHbdEcdL","EL",3.5,3.5,26.5,0.139],["10G ELHI",22,"96E97L98H99I","ELO",2,2,24,0.125],["C3 HO",25,"22H32O","EEILL",-1.5,-1.5,23.5,0.122],["H11 HI(R)EE",27,"a7Hb7Id7Ee7E","LLO",-5,-5,22,0.114],["C3 HE",25,"22H32E","EILLO",-3,-3,22,0.114],["K3 H(O)O",23,"2aH4aO","EEILL",-1.5,-1.5,21.5,0.112],["L8 H(I)LLO",24,"7bH9bLabLbbO","EEI",-3,-3,21,0.109],["12G HELIO",17,"b6Hb7Eb8Lb9IbaO","EL",3.5,3.5,20.5,0.107],["14G HELIO",17,"d6Hd7Ed8Ld9IdaO","EL",3.5,3.5,20.5,0.107],["K3 H(O)E",23,"2aH4aE","EILLO",-3,-3,20,0.104],["12H OLLIE",15,"b7Ob8Lb9LbaIbbE","EH",4.5,4.5,19.5,0.102],["C2 EH",22,"12E22H","EILLO",-3,-3,19,0.1],["14H ELHI",17,"d7Ed8Ld9HdaI","ELO",2,2,19,0.1]]},{"p":1,"t":"3A PAR(A)NG","s":18,"x":"20P21A22R24N25G","r":"NRRRGAP","l":"RR","a":"RRNEV","b":0,"n":430,"c":[["3A PAR(A)NG",18,"20P21A22R24N25G","RR",-3,-1.5,16.5,0.905],["12F PRANG",18,"b5Pb6Rb7Ab8Nb9G","RR",-3,-1.5,16.5,0.905],["3B PR(A)NG",16,"21P22R24N25G","ARR",0,0,16,0.903],["C3 NAG",21,"22N32A42G","PRRR",-10.5,-5.2,15.8,0.902],["H12 G(R)AN",21,"b7Gd7Ae7N","PRRR",-10.5,-5.2,15.8,0.902],["K9 (G)ANG",21,"9aAaaNbaG","PRRR",-10.5,-5.2,15.8,0.902],["M6 PAG(E)R",17,"5cP6cA7cG9cR","NRR",-5,-2.5,14.5,0.895],["3C P(A)GAN",18,"22P24G25A26N","RRR",-7.5,-3.7,14.3,0.894],["3C P(A)NGA",18,"22P24N25G26A","RRR",-7.5,-3.7,14.3,0.894],["12F GNARR",15,"b5Gb6Nb7Ab8Rb9R","PR",-1.5,-0.7,14.3,0.894],["C3 PA",21,"22P32A","GNRRR",-14.5,-7.2,13.8,0.891],["K9 (G)AP",21,"9aAaaP","GNRRR",-14.5,-7.2,13.8,0.891],["M6 PAR(E)R",16,"5cP6cA7cR9cR","GNR",-5.5,-2.7,13.3,0.888],["K9 (G)AG",19,"9aAaaG","NPRRR",-12.5,-6.2,12.8,0.886],["3B GN(A)RR",12,"21G22N24R25R","APR",1.5,0.8,12.8,0.886]]},{"p":0,"t":"14E AFrEETS","s":92,"x":"d4Ad5Fd6rd7Ed8Ed9TdaS","r":"EEA?FST","l":"","a":"","b":0,"n":8880,"c":[["14E AFrEETS",92,"d4Ad5Fd6rd7Ed8Ed9TdaS","",0,16,108,0.978],["12G FEASTEd",78,"b6Fb7Eb8Ab9SbaTbbEbcd","",0,16,94,0.95],["12G FEASTEr",78,"b6Fb7Eb8Ab9SbaTbbEbcr","",0,16,94,0.95],["12G FEATESt",78,"b6Fb7Eb8Ab9TbaEbbSbct","",0,16,94,0.95],["12G FEAtEST",78,"b6Fb7Eb8Ab9tbaEbbSbcT","",0,16,94,0.95],["12G dEAFEST",74,"b6db7Eb8Ab9FbaEbbSbcT","",0,16,90,0.936],["12G dEFEATS",74,"b6db7Eb8Fb9EbaAbbTbcS","",0,16,90,0.936],["12H AFrEETS",73,"b7Ab8Fb9rbaEbbEbcTbdS","",0,16,89,0.932],["L6 FET(I)AlES",72,"5bF6bE7bT9bAablbbEcbS","",0,16,88,0.928],["14G dEAFEST",70,"d6dd7Ed8Ad9FdaEdbSdcT","",0,16,86,0.92],["14G FEASTEd",68,"d6Fd7Ed8Ad9SdaTdbEdcd","",0,16,84,0.91],["14G FEASTEr",68,"d6Fd7Ed8Ad9SdaTdbEdcr","",0,16,84,0.91],["14G FEATESt",68,"d6Fd7Ed8Ad9TdaEdbSdct","",0,16,84,0.91],["14G FEAtEST",66,"d6Fd7Ed8Ad9tdaEdbSdcT","",0,16,82,0.899],["14G dEFEATS",64,"d6dd7Ed8Fd9EdaAdbTdcS","",0,16,80,0.887]]}]},{"seed":19,"racks":["ULATTTS","PDKNQTL"],"result":{"scores":[416,371],"winner":0,"adjustment":[12,0],"ended_by":"out"},"moves":[{"p":0,"t":"8F TUT","s":6,"x":"75T76U77T","r":"ULATTTS","l":"ALST","a":"LATSWGU","b":83,"n":281,"c":[["8F TUT",6,"75T76U77T","ALST",6.5,6.5,12.5,0.528],["8G TUT",6,"76T77U78T","ALST",6.5,6.5,12.5,0.528],["8H TUT",6,"77T78U79T","ALST",6.5,6.5,12.5,0.528],["H6 TUT",6,"57T67U77T","ALST",6.5,6.5,12.5,0.528],["H7 TUT",6,"67T77U87T","ALST",6.5,6.5,12.5,0.528],["H8 TUT",6,"77T87U97T","ALST",6.5,6.5,12.5,0.528],["8E TAUT",8,"74T75A76U77T","LST",3.5,3.5,11.5,0.523],["8F TAUT",8,"75T76A77U78T","LST",3.5,3.5,11.5,0.523],["8G TAUT",8,"76T77A78U79T","LST",3.5,3.5,11.5,0.523],["8H TAUT",8,"77T78A79U7aT","LST",3.5,3.5,11.5,0.523],["H5 TAUT",8,"47T57A67U77T","LST",3.5,3.5,11.5,0.523],["H6 TAUT",8,"57T67A77U87T","LST",3.5,3.5,11.5,0.523],["H7 TAUT",8,"67T77A87U97T","LST",3.5,3.5,11.5,0.523],["H8 TAUT",8,"77T87A97Ua7T","LST",3.5,3.5,11.5,0.523],["8D SUTTA",12,"73S74U75T76T77A","LT",-2,-2,10,0.517]]},{"p":1,"t":"exchange 6","s":0,"x":"","r":"PDKNQTL","l":"D","a":"DISPAEF","b":83,"n":24,"c":[["exchange KLNPQT",0,"","D",0.5,0.5,0.5,0.45],["exchange DKLPQT",0,"","N",0.5,0.5,0.5,0.45],["exchange DKLNPQ",0,"","T",0.5,0.5,0.5,0.45],["exchange DKLNPQT",0,"","",0,0,0,0.448],["exchange DKNPQT",0,"","L",0,0,0,0.448],["exchange DKLNQT",0,"","P",-0.5,-0.5,-0.5,0.446],["exchange DLNPQT",0,"","K",-0.5,-0.5,-0.5,0.446],["G6 PL(U)NK",13,"56P66L86N96K","DQT",-14,-14,-1,0.444],["exchange KLPQT",0,"","DN",-1.5,-1.5,-1.5,0.442],["exchange KLNPQ",0,"","DT",-1.5,-1.5,-1.5,0.442],["exchange DKLPQ",0,"","NT",-1.5,-1.5,-1.5,0.442],["G7 P(U)LK",14,"66P86L96K","DNQT",-16,-16,-2,0.439],["exchange KNPQT",0,"","DL",-2,-2,-2,0.439],["exchange DKPQT",0,"","LN",-2,-2,-2,0.439],["exchange DKNPQ",0,"","LT",-2,-2,-2,0.439]],"e":"KLNPQT"},{"p":0,"t":"I6 GLUT","s":11,"x":"58G68L78U88T","r":"LATSWGU","l":"ASW","a":"ASW?NEL","b":79,"n":356,"c":[["I6 GLUT",11,"58G68L78U88T","ASW",4.5,4.5,15.5,0.567],["9G TAW",16,"86T87A88W","GLSU",-1,-1,15,0.565],["F6 WA(T)T",15,"55W65A85T","GLSU",-1,-1,14,0.561],["I6 WAUL",13,"58W68A78U88L","GST",1,1,14,0.561],["I6 GAULT",12,"58G68A78U88L98T","SW",1.5,1.5,13.5,0.558],["I7 GUL",11,"68G78U88L","ASTW",2.5,2.5,13.5,0.558],["I7 LUG",11,"68L78U88G","ASTW",2.5,2.5,13.5,0.558],["I7 GUT",11,"68G78U88T","ALSW",2,2,13,0.556],["I7 TUG",11,"68T78U88G","ALSW",2,2,13,0.556],["I8 SWAG",16,"78S88W98Aa8G","LTU",-3.5,-3.5,12.5,0.554],["F6 WA(T)TS",18,"55W65A85T95S","GLU",-6.5,-6.5,11.5,0.55],["7H AWL",12,"67A68W69L","GSTU",-0.5,-0.5,11.5,0.55],["9H AWL",12,"87A88W89L","GSTU",-0.5,-0.5,11.5,0.55],["F6 WA(T)",14,"55W65A","GLSTU",-3,-3,11,0.548],["F8 (T)AW",14,"85A95W","GLSTU",-3,-3,11,0.548]]},{"p":1,"t":"10H DEFI","s":22,"x":"97D98E99F9aI","r":"DISPAEF","l":"APS","a":"SPAIBEA","b":75,"n":725,"c":[["10H DEFI",22,"97D98E99F9aI","APS",8,8,30,0.531],["F6 FE(T)ID",21,"55F65E85I95D","APS",8,8,29,0.527],["10F FADE",22,"95F96A97D98E","IPS",6,6,28,0.523],["6F FID(G)E",20,"55F56I57D59E","APS",8,8,28,0.523],["9E DIP",18,"84D85I86P","AEFS",10,10,28,0.523],["10H DEF",21,"97D98E99F","AIPS",6.5,6.5,27.5,0.52],["9G PI(T)",17,"86P87I","ADEFS",10.5,10.5,27.5,0.52],["F6 FA(T)ED",21,"55F65A85E95D","IPS",6,6,27,0.518],["J9 IF",15,"89I99F","ADEPS",12,12,27,0.518],["G6 PF(U)I",14,"56P66F86I","ADES",12.5,12.5,26.5,0.516],["6F FAD(G)E",20,"55F56A57D59E","IPS",6,6,26,0.514],["10I EF",19,"98E99F","ADIPS",7,7,26,0.514],["9E DAP",18,"84D85A86P","EFIS",8,8,26,0.514],["F6 FI(T)",14,"55F65I","ADEPS",12,12,26,0.514],["10F FIDS",22,"95F96I97D98S","AEP",3.5,3.5,25.5,0.511]]},{"p":0,"t":"K5 SWANL(I)kE","s":90,"x":"4aS5aW6aA7aN8aLaakbaE","r":"ASW?NEL","l":"","a":"OOAKMUO","b":68,"n":9927,"c":[["K5 SWANL(I)kE",90,"4aS5aW6aA7aN8aLaakbaE","",0,0,90,0.814],["L4 AWNLEsS",79,"3bA4bW5bN6bL7bE8bs9bS","",0,0,79,0.781],["L4 kNAWELS",79,"3bk4bN5bA6bW7bE8bL9bS","",0,0,79,0.781],["L4 LAWiNES",79,"3bL4bA5bW6bi7bN8bE9bS","",0,0,79,0.781],["L4 WANgLES",79,"3bW4bA5bN6bg7bL8bE9bS","",0,0,79,0.781],["11C LAWiNES",78,"a2La3Aa4Wa5ia6Na7Ea8S","",0,0,78,0.778],["11C WANgLES",78,"a2Wa3Aa4Na5ga6La7Ea8S","",0,0,78,0.778],["L4 AWNLESs",78,"3bA4bW5bN6bL7bE8bS9bs","",0,0,78,0.778],["6E tWAN(G)LES",71,"54t55W56A57N59L5aE5bS","",0,0,71,0.754],["6F WAN(G)LErS",71,"55W56A57N59L5aE5br5cS","",0,0,71,0.754],["F6 WE(T)LANdS",70,"55W65E85L95Aa5Nb5dc5S","",0,0,70,0.751],["K6 jAWL(I)NES",70,"5aj6aA7aW8aLaaNbaEcaS","",0,0,70,0.751],["K6 LAuW(I)NES",70,"5aL6aA7au8aWaaNbaEcaS","",0,0,70,0.751],["9A LAWiNES",69,"80L81A82W83i84N85E86S","",0,0,69,0.747],["L5 AWNLESs",69,"4bA5bW6bN7bL8bE9bSabs","",0,0,69,0.747]]},{"p":1,"t":"8K (N)ABIS","s":24,"x":"7bA7cB7dI7eS","r":"SPAIBEA","l":"AEP","a":"PEABRAE","b":64,"n":663,"c":[["8K (N)ABIS",24,"7bA7cB7dI7eS","AEP",3.5,3.5,27.5,0.219],["9G PA(T)",19,"86P87A","ABEIS",8.5,8.5,27.5,0.219],["9E BAP",19,"84B85A86P","AEIS",8,8,27,0.218],["L7 BAAS",25,"6bB7bA8bA9bS","EIP",1.5,1.5,26.5,0.216],["8K (N)IPAS",24,"7bI7cP7dA7eS","ABE",2,2,26,0.214],["J12 BAP",17,"b9Bc9Ad9P","AEIS",8,8,25,0.211],["L7 BAA",15,"6bB7bA8bA","EIPS",9.5,9.5,24.5,0.209],["8K (N)APAS",24,"7bA7cP7dA7eS","BEI",0,0,24,0.208],["8K (N)ABES",24,"7bA7cB7dE7eS","AIP",-1.5,-1.5,22.5,0.203],["F6 PI(T)A",12,"55P65I85A","ABES",10,10,22,0.201],["L7 BA",12,"6bB7bA","AEIPS",10,10,22,0.201],["11H AI",12,"a7Aa8I","ABEPS",9.5,9.5,21.5,0.199],["8K (N)APES",24,"7bA7cP7dE7eS","ABI",-3,-3,21,0.198],["8K (N)EAPS",24,"7bE7cA7dP7eS","ABI",-3,-3,21,0.198],["F6 BA(T)",11,"55B65A","AEIPS",10,10,21,0.198]]},{"p":0,"t":"L12 MOOK","s":38,"x":"bbMcbOdbOebK","r":"OOAKMUO","l":"AOU","a":"AUOQEET","b":60,"n":267,"c":[["L12 MOOK",38,"bbMcbOdbOebK","AOU",-10,-10,28,0.849],["N5 OOM(I)AK",24,"4dO5dO6dM8dA9dK","OU",-8,-8,16,0.814],["7C OAKUM",24,"62O63A64K65U66M","OO",-11.1,-11.1,12.9,0.804],["L12 MAKO",30,"bbMcbAdbKebO","OOU",-17.6,-17.6,12.4,0.803],["L11 AMOK",29,"abAbbMcbOdbK","OOU",-17.6,-17.6,11.4,0.8],["F6 KO(T)O",18,"55K65O85O","AMOU",-7,-7,11,0.798],["F7 O(T)AKU",19,"65O85A95Ka5U","MOO",-8.1,-8.1,10.9,0.798],["L12 MOO",18,"bbMcbOdbO","AKOU",-8,-8,10,0.795],["L4 MOO",17,"3bM4bO5bO","AKOU",-8,-8,9,0.791],["M6 KA(B)OOM",16,"5cK6cA8cO9cOacM","OU",-8,-8,8,0.788],["H10 (D)UOMO",9,"a7Ub7Oc7Md7O","AKO",-1.5,-1.5,7.5,0.786],["J12 MAKO",24,"b9Mc9Ad9Ke9O","OOU",-17.6,-17.6,6.4,0.782],["J12 MOOK",16,"b9Mc9Od9Oe9K","AOU",-10,-10,6,0.781],["11E MAKO",23,"a4Ma5Aa6Ka7O","OOU",-17.6,-17.6,5.4,0.779],["M6 KO(B)O",12,"5cK6cO8cO","AMOU",-7,-7,5,0.777]]},{"p":1,"t":"15G PARBA(K)E","s":45,"x":"e6Pe7Ae8Re9BeaAecE","r":"PEABRAE","l":"E","a":"EMFNRA?","b":54,"n":618,"c":[["15G PARBA(K)E",45,"e6Pe7Ae8Re9BeaAecE","E",3.5,3.5,48.5,0.226],["15G PREBA(K)E",45,"e6Pe7Re8Ee9BeaAecE","A",0.5,0.5,45.5,0.215],["15I BAR(K)EEP",45,"e8Be9AeaRecEedEeeP","A",0.5,0.5,45.5,0.215],["15H BREA(K)",33,"e7Be8Re9EeaA","AEP",3.5,3.5,36.5,0.184],["15H BERA(K)E",36,"e7Be8Ee9ReaAecE","AP",0,0,36,0.182],["15H APEA(K)",33,"e7Ae8Pe9EeaA","BER",3,3,36,0.182],["15H APEE(K)",33,"e7Ae8Pe9EeaE","ABR",0,0,33,0.173],["15L (K)APA",30,"ecAedPeeA","BEER",3,3,33,0.173],["15L (K)BAR",30,"ecBedAeeR","AEEP",1,1,31,0.167],["M11 PEREA",31,"acPbcEccRdcEecA","AB",-1.5,-1.5,29.5,0.162],["M13 PEA",26,"ccPdcEecA","ABER",3.5,3.5,29.5,0.162],["15L (K)ERB",30,"ecEedReeB","AAEP",-4.1,-4.1,25.9,0.152],["M10 BEEP",29,"9cBacEbcEccP","AAR",-3.1,-3.1,25.9,0.152],["M11 BEEP",29,"acBbcEccEdcP","AAR",-3.1,-3.1,25.9,0.152],["15L (K)EEP",30,"ecEedEeeP","AABR",-5.1,-5.1,24.9,0.149]]},{"p":0,"t":"F8 (T)OQUE","s":34,"x":"85O95Qa5Ub5E","r":"AUOQEET","l":"AET","a":"AETYNHO","b":50,"n":460,"c":[["F8 (T)OQUE",34,"85O95Qa5Ub5E","AET",4.5,4.5,38.5,0.882],["14F QUA",38,"d5Qd6Ud7A","EEOT",0,0,38,0.881],["F8 (T)OQUET",35,"85O95Qa5Ub5Ec5T","AE",1.5,1.5,36.5,0.877],["6F QUA(G)",34,"55Q56U57A","EEOT",0,0,34,0.871],["N6 QU(I)ET",36,"5dQ6dU8dE9dT","AEO",-2.5,-2.5,33.5,0.869],["N6 QU(I)TE",36,"5dQ6dU8dT9dE","AEO",-2.5,-2.5,33.5,0.869],["11D QUOTE",31,"a3Qa4Ua5Oa6Ta7E","AE",1.5,1.5,32.5,0.867],["11D TOQUE",31,"a3Ta4Oa5Qa6Ua7E","AE",1.5,1.5,32.5,0.867],["11D QUOTA",31,"a3Qa4Ua5Oa6Ta7A","EE",1,1,32,0.865],["11C EQUATE",33,"a2Ea3Qa4Ua5Aa6Ta7E","O",-1.5,-1.5,31.5,0.864],["11D QUATE",31,"a3Qa4Ua5Aa6Ta7E","EO",-0.5,-0.5,30.5,0.861],["N6 QU(I)T",33,"5dQ6dU8dT","AEEO",-5,-5,28,0.854],["F6 QA(T)",32,"55Q65A","EEOTU",-6.5,-6.5,25.5,0.847],["14E QUOTE",24,"d4Qd5Ud6Od7Td8E","AE",1.5,1.5,25.5,0.847],["H4 QUA",25,"37Q47U57A","EEOT",0,0,25,0.845]]},{"p":1,"t":"O1 ENFRAMe(S)","s":89,"x":"0eE1eN2eF3eR4eA5eM6ee","r":"EMFNRA?","l":"","a":"DTTLIOO","b":43,"n":8788,"c":[["O1 ENFRAMe(S)",89,"0eE1eN2eF3eR4eA5eM6ee","",0,0,89,0.47],["O1 eNFRAME(S)",89,"0ee1eN2eF3eR4eA5eM6eE","",0,0,89,0.47],["O1 FoRAMEN(S)",89,"0eF1eo2eR3eA4eM5eE6eN","",0,0,89,0.47],["O1 FRAENuM(S)",89,"0eF1eR2eA3eE4eN5eu6eM","",0,0,89,0.47],["14B FRAENuM",83,"d1Fd2Rd3Ad4Ed5Nd6ud7M","",0,0,83,0.433],["N7 R(I)FlEMAN",74,"6dR8dF9dladEbdMcdAddN","",0,0,74,0.38],["O4 RAFt(S)MEN",63,"3eR4eA5eF6et8eM9eEaeN","",0,0,63,0.319],["O5 FRE(S)hMAN",63,"4eF5eR6eE8eh9eMaeAbeN","",0,0,63,0.319],["14J AF(O)RE",33,"d9AdaFdcRddE","MN?",22.5,22.5,55.5,0.281],["14K F(O)RAM",27,"daFdcRddAdeM","EN?",28,28,55,0.279],["14K F(O)AM",25,"daFdcAddM","ENR?",29.5,29.5,54.5,0.276],["11E F(U)RAN",26,"a4Fa6Ra7Aa8N","EM?",28,28,54,0.274],["14K F(O)RM",25,"daFdcRddM","AEN?",28.5,28.5,53.5,0.271],["M8 (B)ARMEN",31,"8cA9cRacMbcEccN","F?",22,22,53,0.269],["M11 REF",27,"acRbcEccF","AMN?",25.5,25.5,52.5,0.267]]},{"p":0,"t":"N3 OATH","s":39,"x":"2dO3dA4dT5dH","r":"AETYNHO","l":"ENY","a":"EYNRSAD","b":39,"n":1029,"c":[["N3 OATH",39,"2dO3dA4dT5dH","ENY",3.5,3.5,42.5,0.72],["M10 AHOY",35,"9cAacHbcOccY","ENT",4.5,4.5,39.5,0.704],["M11 HOY",33,"acHbcOccY","AENT",5,5,38,0.696],["N3 EATH",39,"2dE3dA4dT5dH","NOY",-1.5,-1.5,37.5,0.693],["13C NEATH",39,"c2Nc3Ec4Ac5Tc6H","OY",-2,-2,37,0.69],["N6 HA(I)NT",34,"5dH6dA8dN9dT","EOY",1.5,1.5,35.5,0.681],["M10 THEY",35,"9cTacHbcEccY","ANO",-0.5,-0.5,34.5,0.676],["14F HOYA",29,"d5Hd6Od7Yd8A","ENT",4.5,4.5,33.5,0.67],["14K H(O)OTY",29,"daHdcOddTdeY","AEN",4.5,4.5,33.5,0.67],["M11 HEY",33,"acHbcEccY","ANOT",0,0,33,0.667],["M11 YEH",33,"acYbcEccH","ANOT",0,0,33,0.667],["14J AT(O)NY",30,"d9AdaTdcNddY","EHO",3,3,33,0.667],["14K H(O)YA",30,"daHdcYddA","ENOT",3,3,33,0.667],["13C YENTA",33,"c2Yc3Ec4Nc5Tc6A","HO",-0.5,-0.5,32.5,0.664],["13F THEY",33,"c5Tc6Hc7Ec8Y","ANO",-0.5,-0.5,32.5,0.664]]},{"p":1,"t":"13C DITTO","s":29,"x":"c2Dc3Ic4Tc5Tc6O","r":"DTTLIOO","l":"LO","a":"LOAHNOE","b":34,"n":438,"c":[["13C DITTO",29,"c2Dc3Ic4Tc5Tc6O","LO",-1.5,-1.5,27.5,0.386],["N11 OOTID",29,"adObdOcdTddIedD","LT",-2,-2,27,0.383],["13C LOTTO",27,"c2Lc3Oc4Tc5Tc6O","DI",-1,-1,26,0.376],["N12 TOLD",27,"bdTcdOddLedD","IOT",-2.5,-2.5,24.5,0.366],["13C DOIT",25,"c2Dc3Oc4Ic5T","LOT",-1,-1,24,0.363],["N12 LOID",27,"bdLcdOddIedD","OTT",-4,-4,23,0.356],["13C DOLT",25,"c2Dc3Oc4Lc5T","IOT",-2.5,-2.5,22.5,0.353],["N13 TOD",25,"cdTddOedD","ILOT",-2.5,-2.5,22.5,0.353],["13C LOOT",23,"c2Lc3Oc4Oc5T","DIT",-0.5,-0.5,22.5,0.353],["M1 LOOT",23,"0cL1cO2cO3cT","DIT",-0.5,-0.5,22.5,0.353],["M1 TOOL",23,"0cT1cO2cO3cL","DIT",-0.5,-0.5,22.5,0.353],["13C TOIT",23,"c2Tc3Oc4Ic5T","DLO",-1,-1,22,0.35],["13C TOOT",23,"c2Tc3Oc4Oc5T","DIL",-1,-1,22,0.35],["M1 TOOT",23,"0cT1cO2cO3cT","DIL",-1,-1,22,0.35],["M1 DIOL",25,"0cD1cI2cO3cL","OTT",-4,-4,21,0.343]]},{"p":0,"t":"N10 YARNED","s":53,"x":"9dYadAbdRcdNddEedD","r":"EYNRSAD","l":"S","a":"SRSNPGD","b":28,"n":1031,"c":[["N10 YARNED",53,"9dYadAbdRcdNddEedD","S",8,8,61,0.874],["N10 YEARNS",50,"9dYadEbdAcdRddNedS","D",0.5,0.5,50.5,0.835],["M8 (B)ARNEY",39,"8cA9cRacNbcEccY","DS",6,6,45,0.811],["N12 RYND",33,"bdRcdYddNedD","AES",12,12,45,0.811],["N12 YARD",33,"bdYcdAddRedD","ENS",12,12,45,0.811],["N10 DERAYS",44,"9dDadEbdRcdAddYedS","N",0.5,0.5,44.5,0.808],["M8 (B)RAYED",37,"8cR9cAacYbcEccD","NS",6,6,43,0.801],["M11 REDY(E)",34,"acRbcEccDdcY","ANS",9,9,43,0.801],["N11 RAYED",35,"adRbdAcdYddEedD","NS",6,6,41,0.791],["M12 EYN(E)",31,"bcEccYdcN","ADRS",8,8,39,0.781],["M13 DY(E)",25,"ccDdcY","AENRS",14,14,39,0.781],["N11 DRAYS",34,"adDbdRcdAddYedS","EN",4,4,38,0.776],["N11 DRYAS",34,"adDbdRcdYddAedS","EN",4,4,38,0.776],["N11 YARDS",34,"adYbdAcdRddDedS","EN",4,4,38,0.776],["M12 EYR(E)",31,"bcEccYdcR","ADNS",7,7,38,0.776]]},{"p":1,"t":"M3 OLEO","s":34,"x":"2cO3cL4cE5cO","r":"LOAHNOE","l":"AHN","a":"AHNOWZC","b":24,"n":623,"c":[["M3 OLEO",34,"2cO3cL4cE5cO","AHN",2,2,36,0.256],["M2 HOLE",32,"1cH2cO3cL4cE","ANO",-0.5,-0.5,31.5,0.229],["M2 HOO",27,"1cH2cO3cO","AELN",4.5,4.5,31.5,0.229],["12B OOH",25,"b1Ob2Ob3H","AELN",4.5,4.5,29.5,0.217],["M1 HOO",24,"0cH1cO2cO","AELN",4.5,4.5,28.5,0.211],["M1 OHO",24,"0cO1cH2cO","AELN",4.5,4.5,28.5,0.211],["M1 ANOLE",28,"0cA1cN2cO3cL4cE","HO",-0.5,-0.5,27.5,0.206],["M2 HOE",27,"1cH2cO3cE","ALNO",-0.5,-0.5,26.5,0.2],["M1 HAO",24,"0cH1cA2cO","ELNO",2.5,2.5,26.5,0.2],["L4 OHO",22,"3bO4bH5bO","AELN",4.5,4.5,26.5,0.2],["M2 LOO",21,"1cL2cO3cO","AEHN",5.5,5.5,26.5,0.2],["12C OH",23,"b2Ob3H","AELNO",3,3,26,0.198],["M2 NOO",21,"1cN2cO3cO","AEHL",5,5,26,0.198],["12B NOH",25,"b1Nb2Ob3H","AELO",0,0,25,0.193],["M2 HO",22,"1cH2cO","AELNO",3,3,25,0.193]]},{"p":0,"t":"D10 SPR(I)NG","s":22,"x":"93Sa3Pb3Rd3Ne3G","r":"SRSNPGD","l":"DS","a":"SDGRIYV","b":19,"n":243,"c":[["D10 SPR(I)NG",22,"93Sa3Pb3Rd3Ne3G","DS",6,6,28,0.856],["D11 GR(I)ND",18,"a3Gb3Rd3Ne3D","PSS",7.5,7.5,25.5,0.845],["D12 P(I)NG",18,"b3Pd3Ne3G","DRSS",7.5,7.5,25.5,0.845],["D12 D(I)NG",16,"b3Dd3Ne3G","PRSS",6.5,6.5,22.5,0.831],["D11 GR(I)PS",18,"a3Gb3Rd3Pe3S","DNS",4,4,22,0.828],["D11 PR(I)GS",18,"a3Pb3Rd3Ge3S","DNS",4,4,22,0.828],["D12 G(I)RD",16,"b3Gd3Re3D","NPSS",5.5,5.5,21.5,0.825],["G7 D(U)P",18,"66D86P","GNRSS",2.5,2.5,20.5,0.82],["9D DR(O)P",17,"83D84R86P","GNSS",3.5,3.5,20.5,0.82],["D11 GR(I)P",14,"a3Gb3Rd3P","DNSS",6.5,6.5,20.5,0.82],["D11 PR(I)G",14,"a3Pb3Rd3G","DNSS",6.5,6.5,20.5,0.82],["11C SPR(U)G",16,"a2Sa3Pa4Ra6G","DNS",4,4,20,0.818],["D10 SPR(I)G",16,"93Sa3Pb3Rd3G","DNS",4,4,20,0.818],["D12 R(I)NG",14,"b3Rd3Ne3G","DPSS",5.5,5.5,19.5,0.815],["D11 DR(I)PS",18,"a3Db3Rd3Pe3S","GNS",1,1,19,0.812]]},{"p":1,"t":"L2 WHA","s":36,"x":"1bW2bH3bA","r":"AHNOWZC","l":"CNOZ","a":"NOZCEIV","b":16,"n":320,"c":[["L2 WHA",36,"1bW2bH3bA","CNOZ",2,2,38,0.331],["J2 ZONA",35,"19Z29O39N49A","CHW",-7.5,-7.5,27.5,0.246],["L1 HOW",23,"0bH1bO2bW","ACNZ",4,4,27,0.243],["L1 COW",21,"0bC1bO2bW","AHNZ",4.5,4.5,25.5,0.232],["14A HWA(N)",23,"d0Hd1Wd2A","CNOZ",2,2,25,0.229],["L1 HAW",23,"0bH1bA2bW","CNOZ",2,2,25,0.229],["14K H(O)W(E)",20,"daHdcW","ACNOZ",5,5,25,0.229],["6E WHAN(G)",20,"54W55H56A57N","COZ",4,4,24,0.222],["14H ZOA",33,"d7Zd8Od9A","CHNW",-9.5,-9.5,23.5,0.218],["10B ZA(S)",32,"91Z92A","CHNOW",-8.5,-8.5,23.5,0.218],["L3 HA",28,"2bH3bA","CNOWZ",-4.5,-4.5,23.5,0.218],["15A HON(G)",24,"e0He1Oe2N","ACWZ",-0.5,-0.5,23.5,0.218],["L1 CAW",21,"0bC1bA2bW","HNOZ",2.5,2.5,23.5,0.218],["14J AZ(O)",31,"d9AdaZ","CHNOW",-8.5,-8.5,22.5,0.212],["14K Z(O)N(E)",26,"daZdcN","ACHOW",-3.5,-3.5,22.5,0.212]]},{"p":0,"t":"15A RID(G)Y","s":30,"x":"e0Re1Ie2De4Y","r":"SDGRIYV","l":"GSV","a":"SGVJRIE","b":12,"n":170,"c":[["15A RID(G)Y",30,"e0Re1Ie2De4Y","GSV",-5.5,-5.5,24.5,0.795],["G7 G(U)Y",22,"66G86Y","DIRSV",-2.5,-2.5,19.5,0.759],["12F (E)DGY",14,"b6Db7Gb8Y","IRSV",-0.5,-0.5,13.5,0.71],["15A GRI(G)",18,"e0Ge1Re2I","DSVY",-5.5,-5.5,12.5,0.701],["10A VIG(S)",10,"90V91I92G","DRSY",2,2,12,0.697],["J2 GYRI",14,"19G29Y39R49I","DSV",-2.5,-2.5,11.5,0.693],["10B VI(S)",14,"91V92I","DGRSY",-3,-3,11,0.688],["C7 GYRI",13,"62G72Y82R92I","DSV",-2.5,-2.5,10.5,0.684],["15A GRI(G)S",21,"e0Ge1Re2Ie4S","DVY",-11,-11,10,0.679],["10A VID(S)",10,"90V91I92D","GRSY",-1,-1,9,0.67],["exchange DGIRVY",0,"","S",8,8,8,0.661],["exchange DGVY",0,"","IRS",8,8,8,0.661],["14K Y(O)R(E)",14,"daYdcR","DGISV",-6.5,-6.5,7.5,0.656],["H10 (D)RYS",19,"a7Rb7Yc7S","DGIV",-12,-12,7,0.651],["G5 GYR(U)S",14,"46G56Y66R86S","DIV",-7,-7,7,0.651]]},{"p":1,"t":"6F ZIN(G)","s":34,"x":"55Z56I57N","r":"NOZCEIV","l":"CEOV","a":"OCEVIEN","b":9,"n":241,"c":[["6F ZIN(G)",34,"55Z56I57N","CEOV",-3.5,-3.5,30.5,0.292],["6D COVIN(G)",20,"53C54O55V56I57N","EZ",8.5,8.5,28.5,0.273],["J2 ZINE",35,"19Z29I39N49E","COV",-7,-7,28,0.269],["J2 ZONE",35,"19Z29O39N49E","CIV",-7,-7,28,0.269],["J2 VINO",17,"19V29I39N49O","CEZ",9,9,26,0.251],["C5 NOVICE",20,"42N52O62V72I82C92E","Z",5,5,25,0.243],["2J VO(W)",17,"19V1aO","CEINZ",8,8,25,0.243],["J2 VICE",19,"19V29I39C49E","NOZ",4,4,23,0.227],["C5 COVINE",18,"42C52O62V72I82N92E","Z",5,5,23,0.227],["C7 VINO",14,"62V72I82N92O","CEZ",9,9,23,0.227],["A11 VIZO(R)",18,"a0Vb0Ic0Zd0O","CEN",4.5,4.5,22.5,0.223],["C7 COVE",18,"62C72O82V92E","INZ",4,4,22,0.219],["C7 VICE",18,"62V72I82C92E","NOZ",4,4,22,0.219],["10B VI(S)",14,"91V92I","CENOZ",8,8,22,0.219],["14K Z(O)N(E)",26,"daZdcN","CEIOV",-5,-5,21,0.211]]},{"p":0,"t":"J2 JIVE","s":32,"x":"19J29I39V49E","r":"SGVJRIE","l":"GRS","a":"SGRALII","b":5,"n":248,"c":[["J2 JIVE",32,"19J29I39V49E","GRS",2,2,34,0.865],["C7 JIVE",28,"62J72I82V92E","GRS",2,2,30,0.84],["H1 VERSI(N)",30,"07V17E27R37S47I","GJ",-7,-7,23,0.789],["14B GI(N)",19,"d1Gd2I","EJRSV",0,0,19,0.754],["C7 GIVE",16,"62G72I82V92E","JRS",2.5,2.5,18.5,0.75],["10A JIG(S)",14,"90J91I92G","ERSV",4.5,4.5,18.5,0.75],["10A VIG(S)",10,"90V91I92G","EJRS",8.5,8.5,18.5,0.75],["A11 JIVE(R)",16,"a0Jb0Ic0Vd0E","GRS",2,2,18,0.745],["H4 JI(N)",18,"37J47I","EGRSV",-0.5,-0.5,17.5,0.741],["10B VI(S)",14,"91V92I","EGJRS",3.5,3.5,17.5,0.741],["J2 GIVE",14,"19G29I39V49E","JRS",2.5,2.5,16.5,0.731],["11M J(A)G",11,"acJaeG","EIRSV",5.5,5.5,16.5,0.731],["H2 GIVE(N)",13,"17G27I37V47E","JRS",2.5,2.5,15.5,0.721],["F3 GRI(Z)",14,"25G35R45I","EJSV",1,1,15,0.716],["A10 VIRGE(R)",11,"90Va0Ib0Rc0Gd0E","JS",3.5,3.5,14.5,0.711]]},{"p":1,"t":"C5 NOVICE","s":20,"x":"42N52O62V72I82C92E","r":"OCEVIEN","l":"E","a":"EEUCEX","b":0,"n":232,"c":[["C5 NOVICE",20,"42N52O62V72I82C92E","E",3.5,2.6,22.6,0.202],["C5 COVINE",18,"42C52O62V72I82N92E","E",3.5,2.6,20.6,0.185],["C7 COVE",18,"62C72O82V92E","EIN",2.5,1.9,19.9,0.18],["C7 VICE",18,"62V72I82C92E","ENO",2.5,1.9,19.9,0.18],["C6 VOICE",16,"52V62O72I82C92E","EN",4,3,19,0.173],["C6 OVINE",15,"52O62V72I82N92E","CE",4,3,18,0.165],["C7 VINO",14,"62V72I82N92O","CEE",4,3,17,0.158],["H2 COVI(N)",14,"17C27O37V47I","EEN",4,3,17,0.158],["10B VI(S)",14,"91V92I","CEENO",3,2.3,16.3,0.153],["12A VIE(R)",14,"b0Vb1Ib2E","CENO",3,2.3,16.3,0.153],["C5 EVINCE",17,"42E52V62I72N82C92E","O",-1.5,-1.1,15.9,0.15],["C7 NEVI",14,"62N72E82V92I","CEO",2.5,1.9,15.9,0.15],["C7 VINE",14,"62V72I82N92E","CEO",2.5,1.9,15.9,0.15],["H2 COVE(N)",14,"17C27O37V47E","EIN",2.5,1.9,15.9,0.15],["A10 VOICE(R)",12,"90Va0Ob0Ic0Cd0E","EN",4,3,15,0.144]]},{"p":0,"t":"11C S(P)R(U)G","s":28,"x":"a2Sa4Ra6G","r":"SGRALII","l":"AIIL","a":"ALII","b":0,"n":413,"c":[["11C S(P)R(U)G",28,"a2Sa4Ra6G","AIIL",-10.6,-10,18,0.924],["B2 ARGIL",22,"11A21R31G41I51L","IS",6.5,-7,15,0.91],["B2 GLIAS",22,"11G21L31I41A51S","IR",0,-7,15,0.91],["B2 GRAIL",22,"11G21R31A41I51L","IS",6.5,-7,15,0.91],["B2 RAGIS",22,"11R21A31G41I51S","IL",-1.5,-7,15,0.91],["B2 SIGIL",22,"11S21I31G41I51L","AR",2,-7,15,0.91],["14A AGI(N)",23,"d0Ad1Gd2I","ILRS",5.5,-10,13,0.899],["H1 RAISI(N)",21,"07R17A27I37S47I","GL",-5,-8.5,12.5,0.896],["B2 LARIS",20,"11L21A31R41I51S","GI",-4,-8.5,11.5,0.89],["B2 LIRAS",20,"11L21I31R41A51S","GI",-4,-8.5,11.5,0.89],["M8 (B)IALI",21,"8cI9cAacLbcI","GRS",2,-10,11,0.887],["B3 SLAG",18,"21S31L41A51G","IIR",-7.1,-8.5,9.5,0.877],["11A GAS(P)",19,"a0Ga1Aa2S","IILR",-7.1,-10,9,0.874],["D1 SIGLA",16,"03S13I23G33L43A","IR",0,-7,9,0.874],["D2 GLIAS",16,"13G23L33I43A53S","IR",0,-7,9,0.874]]},{"p":1,"t":"14A EX","s":38,"x":"d0Ed1X","r":"EEUCEX","l":"CEEU","a":"EUCE","b":0,"n":64,"c":[["14A EX",38,"d0Ed1X","CEEU",-2.5,-13,25,0.173],["10A EX(ES)",27,"90E91X","CEEU",-2.5,-13,14,0.095],["O10 EX",23,"9eEaeX","CEEU",-2.5,-13,10,0.076],["6B C(O)X",18,"51C53X","EEEU",-8,-10,8,0.068],["5G XU",20,"46X47U","CEEE",1.5,-13,7,0.064],["D4 XU",20,"33X43U","CEEE",1.5,-13,7,0.064],["D5 EX",20,"43E53X","CEEU",-2.5,-13,7,0.064],["O13 EX",20,"ceEdeX","CEEU",-2.5,-13,7,0.064],["B14 X(I)",18,"d1X","CEEEU",-5,-14.5,3.5,0.052],["3I X(I)",17,"28X","CEEEU",-5,-14.5,2.5,0.049],["7C (V)EX",13,"63E64X","CEEU",-2.5,-13,0,0.042],["B4 XU",11,"31X41U","CEEE",1.5,-13,-2,0.037],["D4 CEE",14,"33C43E53E","EUX",3,-19,-5,0.031],["D4 CUE",14,"33C43U53E","EEX",7,-19,-5,0.031],["6C (O)X",9,"53X","CEEEU",-5,-14.5,-5.5,0.03]]},{"p":0,"t":"M8 (B)IALI","s":21,"x":"8cI9cAacLbcI","r":"ALII","l":"","a":"","b":0,"n":93,"c":[["M8 (B)IALI",21,"8cI9cAacLbcI","",0,12,33,0.943],["B2 ILIA",10,"11I21L31I41A","",0,12,22,0.893],["D2 ILIA",10,"13I23L33I43A","",0,12,22,0.893],["8A IL(I)A",15,"70I71L73A","I",-1.5,-5.5,9.5,0.793],["K1 LA",15,"0aL1aA","II",-11.1,-7,8,0.777],["B4 AIL",11,"31A41I51L","I",-1.5,-5.5,5.5,0.749],["2J (J)A(W)",13,"1aA","IIL",-8.6,-8.5,4.5,0.737],["7H I(L)I(A)L",10,"67I69I6bL","A",0.5,-5.5,4.5,0.737],["9I (T)I(L)L",11,"89I8bL","AI",-3.5,-7,4,0.731],["B5 AL",10,"41A51L","II",-11.1,-7,3,0.719],["M11 AI",10,"acAbcI","IL",-1.5,-7,3,0.719],["M11 LI",10,"acLbcI","AI",-3.5,-7,3,0.719],["5B A(N)IL",8,"41A43I44L","I",-1.5,-5.5,2.5,0.712],["5B I(N)IA",8,"41I43I44A","L",0,-5.5,2.5,0.712],["M10 AL",9,"9cAacL","II",-11.1,-7,2,0.706]]}]}];
  const NOT_WORDS = new Set(["DANE", "ZZ", "XQ", "AEIO"]);
})();
