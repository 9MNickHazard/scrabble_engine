/* Mock API for training from saved games. Loaded only in mock mode (?mock=1), after
 * js/mock/training.js and js/mock/library.js.
 *
 * Answers the /api/gamesets endpoints (list, one set, log, start or carry on, stop, delete) with
 * three sets: twenty-million (complete), fifty-million (being built: it gains a block of 5,000
 * games every five seconds or so while the page polls) and trial (failed after a few blocks).
 *
 * It also simulates runs that train from games and registers them with js/mock/training.js
 * (window.MockTraining), so they share the list of runs and its endpoints: pool-6x64, a CNN with
 * board-wide pooling that finished its six passes and wins about 53.6% against the static
 * evaluator while its spread stays near zero, and transformer-8x192, running, a little past its
 * second pass and level so far. A simulated minute of training passes every five seconds. POST
 * /api/runs with `offline.games` makes more. The records, the summary and the report follow
 * docs/APP_API.md ("Training from saved games"); the models of these runs reach GET /api/models
 * through window.MockOffline.models().
 *
 * Mock-only flags: &empty=1 (nothing at all), &nosets=1 (no game sets and no runs from games),
 * &idle=1 (the transformer run is stopped, so nothing trains).
 */
(function () {
  "use strict";

  const params = new URLSearchParams(location.search);
  if (params.has("empty") || params.has("old")) return;
  const NOSETS = params.has("nosets");
  const IDLE = params.has("idle");
  const NOW = Date.now() / 1000;
  const HOUR = 3600;
  const BLOCK = 5000;
  const SCHEMA = window.MockSchema;
  const T = window.MockTraining;

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) {
    let u = 0;
    while (!u) u = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
  }
  const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
  const int = (v) => Math.round(v).toLocaleString("en-US");
  const pct = (v, ci) => `${(100 * v).toFixed(1)}%` + (ci === null || ci === undefined ? "" : ` +/- ${(100 * ci).toFixed(1)}`);
  const signed = (v) => (v >= 0 ? "+" : "-") + Math.abs(v).toFixed(1);
  const clock = (time) => new Date(time * 1000).toTimeString().slice(0, 8);
  const fail = (message) => ({ error: message });

  // ---- Game sets --------------------------------------------------------------------------

  const sets = [];

  function newSet(spec) {
    const set = Object.assign({ status: "complete", error: null, plies: 4, temperature: 6.0, startedAt: null, base: 0, stopAt: null,
      bytesPerGame: 286.3, log: [] }, spec);
    set.games = spec.games || 0;
    set.target = spec.target || set.games;
    sets.push(set);
    if (set.games) logBlocks(set, 0, set.games, set.created);
    return set;
  }

  function logBlocks(set, from, to, start) {
    const t0 = start || NOW;
    set.log.push(`[${clock(t0)}] game set ${set.name}: ${int(from)} of ${int(set.target)} games, word list NSWL2023-91a67230e1f6`);
    const every = Math.max(BLOCK, Math.round((to - from) / 8 / BLOCK) * BLOCK);
    for (let g = from + every; g <= to; g += every) {
      set.log.push(`[${clock(t0 + (g - from) / set.gps)}]   ${int(g)} games (${int(set.gps * (0.98 + 0.04 * Math.sin(g)))}/s), ` +
        `${int(g * set.bytesPerGame / 2 ** 20)} MB`);
    }
  }

  /** Bring a set that is being built up to date: whole blocks only, as the generator writes them. */
  function advanceSet(set) {
    if (set.status !== "building" && set.status !== "stopping") return;
    const now = Date.now() / 1000;
    const played = Math.floor(((now - set.startedAt) * set.gps) / BLOCK) * BLOCK;
    let games = Math.min(set.target, set.base + played);
    if (set.status === "stopping" && now >= set.stopAt) {
      games = Math.min(set.target, Math.max(games, set.stopGames + BLOCK));
      set.games = games;
      set.status = games >= set.target ? "complete" : "partial";
      set.log.push(`[${clock(now)}]   ${int(games)} games (${int(set.gps)}/s), ${int(games * set.bytesPerGame / 2 ** 20)} MB`);
      set.log.push(`[${clock(now)}] stopped: STOP file found`);
      set.updated = now;
      return;
    }
    if (games !== set.games) {
      set.games = games;
      set.updated = now;
      if (games % (BLOCK * 20) === 0) set.log.push(`[${clock(now)}]   ${int(games)} games (${int(set.gps)}/s), ${int(games * set.bytesPerGame / 2 ** 20)} MB`);
    }
    if (set.games >= set.target) {
      set.status = "complete";
      set.log.push(`[${clock(now)}] done: ${int(set.games)} games`);
    }
  }

  function setSummary(set) {
    advanceSet(set);
    const has = set.games > 0;
    const busy = set.status === "building" || set.status === "stopping";
    const seconds = has ? set.games / set.gps : 0;
    return {
      name: set.name, status: set.status, games: set.games, target: set.target,
      progress: set.target ? round(set.games / set.target, 4) : null,
      bytes: Math.round(set.games * set.bytesPerGame),
      lexicon_id: has ? "NSWL2023-91a67230e1f6" : null, leaves_hash: has ? "ab12f09c3e5d7781" : null,
      opening_plies: set.plies, opening_temperature: set.temperature,
      mean_turns: has ? set.turns : null, mean_score: has ? set.score : null,
      first_player_win_rate: has ? set.firstWins : null, bingos_per_game: has ? set.bingos : null,
      games_per_second: has ? set.gps : null, seconds: round(seconds, 1),
      created: set.created, updated: busy ? Date.now() / 1000 : set.updated,
      error: set.status === "failed" ? set.log.slice(-8).join("\n") : null,
    };
  }

  function listSets() {
    const list = sets.map(setSummary);
    list.sort((a, b) => ((a.status === "building" || a.status === "stopping" ? 0 : 1) - (b.status === "building" || b.status === "stopping" ? 0 : 1)) ||
      (b.updated || 0) - (a.updated || 0));
    const measured = list.filter((s) => s.games && s.bytes);
    const rates = list.filter((s) => s.games_per_second).map((s) => s.games_per_second);
    return { sets: list, estimate: {
      bytes_per_game: measured.length ? round(measured.reduce((a, s) => a + s.bytes, 0) / measured.reduce((a, s) => a + s.games, 0), 1) : 286,
      games_per_second: rates.length ? rates[0] : null,
    } };
  }

  const findSet = (name) => sets.find((s) => s.name === decodeURIComponent(name));

  if (!NOSETS) {
    newSet({ name: "twenty-million", games: 20e6, target: 20e6, gps: 1041.6, created: NOW - 52 * HOUR, updated: NOW - 46.6 * HOUR,
      turns: 22.98, score: 432.2, firstWins: 0.5541, bingos: 3.935 });
    const fifty = newSet({ name: "fifty-million", games: 0, target: 50e6, gps: 1036.2, created: NOW - 3.9 * HOUR, updated: NOW,
      turns: 22.97, score: 432.0, firstWins: 0.5538, bingos: 3.931, status: "building" });
    fifty.base = 0;
    fifty.startedAt = Date.now() / 1000 - 3.9 * HOUR;
    advanceSet(fifty);
    fifty.log.length = 0;
    logBlocks(fifty, 0, fifty.games, fifty.created);
    const trial = newSet({ name: "trial", games: 15000, target: 1e6, gps: 1012.4, created: NOW - 80 * HOUR, updated: NOW - 79.9 * HOUR,
      turns: 23.08, score: 431.4, firstWins: 0.5574, bingos: 3.902, status: "failed", plies: 2, temperature: 4.0 });
    trial.log.push(
      `[${clock(trial.updated)}] Traceback (most recent call last):`,
      "  File \"src/scrabble_ai/games.py\", line 158, in generate",
      "    write_block(directory, block)",
      "  File \"src/scrabble_ai/games.py\", line 74, in write_block",
      "    handle.write(payload)",
      "OSError: [Errno 28] No space left on device");
  }

  App.mock.on("GET", /^\/api\/gamesets$/, () => listSets());

  App.mock.on("GET", /^\/api\/gamesets\/([^/]+)$/, (match) => {
    const set = findSet(match[1]);
    return set ? setSummary(set) : fail(`there is no game set named '${decodeURIComponent(match[1])}'`);
  });

  App.mock.on("GET", /^\/api\/gamesets\/([^/]+)\/log$/, (match, body, query) => {
    const set = findSet(match[1]);
    if (!set) return fail(`there is no game set named '${decodeURIComponent(match[1])}'`);
    advanceSet(set);
    const tail = Math.max(0, parseInt(query.tail || "200", 10) || 200);
    return { lines: set.log.slice(-tail) };
  });

  App.mock.on("POST", /^\/api\/gamesets$/, (match, body) => {
    const name = String((body && body.name) || "");
    const games = body && body.games;
    if (!/^[A-Za-z0-9_-]+$/.test(name) || name.length > 64) return fail("a game set's name may only use letters, digits, - and _ (at most 64)");
    if (!Number.isInteger(games) || games < 1 || games > 500000000) return fail("games must be a whole number between 1 and 500,000,000");
    const plies = body.opening_plies;
    const temperature = body.opening_temperature;
    if (plies !== undefined && plies !== null && (!Number.isInteger(plies) || plies < 0 || plies > 20)) return fail("opening_plies must be a whole number between 0 and 20");
    if (temperature !== undefined && temperature !== null && (typeof temperature !== "number" || temperature < 0 || temperature > 100)) return fail("opening_temperature must be a number between 0 and 100");
    let set = sets.find((s) => s.name === name);
    if (!set && sets.some((s) => s.name.toLowerCase() === name.toLowerCase())) return fail(`a game set named '${name}' already exists`);
    if (set) {
      advanceSet(set);
      if (set.status === "building" || set.status === "stopping") return fail(`${name} is already being built`);
      if (games <= set.games) return fail(`${name} already has ${int(set.games)} games; ask for more than that to add to it`);
      if (plies !== undefined && plies !== null && plies !== set.plies) return fail(`${name} was made with opening plies ${set.plies}; a set keeps the settings it was made with`);
      if (temperature !== undefined && temperature !== null && temperature !== set.temperature) {
        return fail(`${name} was made with opening temperature ${set.temperature}; a set keeps the settings it was made with`);
      }
    } else {
      set = newSet({ name, games: 0, target: games, gps: 1038.5, created: Date.now() / 1000, updated: Date.now() / 1000,
        turns: 22.98, score: 432.1, firstWins: 0.554, bingos: 3.93, plies: plies ?? 4, temperature: temperature ?? 6.0 });
    }
    set.target = games;
    set.base = set.games;
    set.startedAt = Date.now() / 1000;
    set.status = "building";
    set.log.push(`[${clock(set.startedAt)}] game set ${name}: ${int(set.games)} of ${int(games)} games, word list NSWL2023-91a67230e1f6`);
    return setSummary(set);
  });

  App.mock.on("POST", /^\/api\/gamesets\/([^/]+)\/stop$/, (match, body) => {
    const set = findSet(match[1]);
    if (!set) return fail(`there is no game set named '${decodeURIComponent(match[1])}'`);
    advanceSet(set);
    if (set.status !== "building" && set.status !== "stopping") return fail(`${set.name} is not being built`);
    if (body && body.force) {
      set.status = set.games >= set.target ? "complete" : "partial";
      set.updated = Date.now() / 1000;
      set.log.push(`[${clock(set.updated)}] killed`);
    } else if (set.status === "building") {
      set.status = "stopping";
      set.stopAt = Date.now() / 1000 + 4;
      set.stopGames = set.games;
    }
    return { ok: true };
  });

  App.mock.on("DELETE", /^\/api\/gamesets\/([^/]+)$/, (match) => {
    const set = findSet(match[1]);
    if (!set) return fail(`there is no game set named '${decodeURIComponent(match[1])}'`);
    advanceSet(set);
    if (set.status === "building" || set.status === "stopping") return fail(`${set.name} is being built; stop it first`);
    const users = runs.filter((r) => r.setName === set.name && ["running", "stopping", "queued"].includes(r.status)).map((r) => r.name);
    if (users.length) return fail(`${set.name} is being trained from by ${users.join(", ")}; stop that first`);
    sets.splice(sets.indexOf(set), 1);
    return { ok: true };
  });

  // ---- Runs that train from games ----------------------------------------------------------

  const runs = [];
  const library = {};  // model id -> {starred, note}

  function defaultsConfig() {
    const config = {};
    for (const section of (SCHEMA && SCHEMA.sections) || []) {
      config[section.name] = {};
      for (const f of section.fields) config[section.name][f.key.split(".")[1]] = f.default;
    }
    return config;
  }

  const flatDefaults = () => {
    const out = {};
    for (const section of (SCHEMA && SCHEMA.sections) || []) for (const f of section.fields) out[f.key] = f.default;
    return out;
  };

  class GamesRun {
    /** spec: name, set, created, overrides, pps (positions a second), seed, gain and tau (how far
     *  and how fast its win rate rises, in steps), evalEvery (training minutes between matches). */
    constructor(spec) {
      this.spec = spec;
      this.name = spec.name;
      this.created = spec.created;
      this.status = "queued";
      this.r = rng(spec.seed || 77);
      this.records = [];
      this.error = null;
      const set = sets.find((s) => s.name === spec.set);
      this.setName = spec.set;
      this.setGames = set ? set.games : 0;
      // The settings: the defaults, then what training from games changes, then the run's own.
      const overrides = Object.assign({}, (SCHEMA && SCHEMA.games_defaults) || {}, spec.overrides || {},
        { "offline.games": `games/${spec.set}` });
      this.config = defaultsConfig();
      for (const [key, value] of Object.entries(overrides)) {
        const [section, field] = key.split(".");
        if (this.config[section]) this.config[section][field] = value;
      }
      this.config.run.name = spec.name;
      const defaults = flatDefaults();
      this.changed = {};
      for (const [key, value] of Object.entries(overrides)) {
        if (!["run.name", "run.out_dir", "run.hours", "run.max_games"].includes(key) && value !== defaults[key]) this.changed[key] = value;
      }
      const off = this.config.offline;
      this.heldOut = Math.min(off.validation_games, Math.floor(this.setGames / 10));
      this.trainGames = this.setGames - this.heldOut;
      this.batch = Math.min(off.batch_size, this.trainGames);
      this.stepsPerPass = Math.max(1, Math.floor(this.trainGames / this.batch));
      this.totalSteps = Math.max(1, Math.ceil(off.passes * this.stepsPerPass));
      const m = this.config.model;
      this.network = `${m.blocks}x${m.channels}`;
      this.parameters = m.kind === "transformer" ? Math.round(12.4 * m.channels * m.channels * m.blocks + 205000)
        : Math.round(18.5 * m.channels * m.channels * m.blocks + 157000);
      this.pps = spec.pps || (m.kind === "transformer" ? 6400 : 15500) * (64 / Math.max(32, m.channels)) ** (m.kind === "transformer" ? 0 : 0.5);
      this.sim = { step: 0, hours: 0, minutes: 0, lastEval: 0, lastCkpt: 0, nextValid: off.validate_steps, best: null, checkpoints: [],
        weights: { spread: off.spread_weight }, sessionMinutes: 0, limitMinutes: 0, wall: spec.created };
      if (m.aux_heads) Object.assign(this.sim.weights, { next_score: off.aux_weight, next_bingo: off.aux_weight, next_squares: off.aux_weight });
      this.limitHours = Number((spec.overrides || {})["run.hours"]) || 0;
    }

    get off() { return this.config.offline; }
    stamp() { return this.live ? Date.now() / 1000 : this.sim.wall + this.sim.hours * HOUR * 1.18; }

    push(kind, extra) {
      const s = this.sim;
      const seen = s.step * this.batch;
      const record = Object.assign({ kind, time: round(this.stamp(), 3), step: s.step, games: seen, positions: seen, hours: round(s.hours, 4) }, extra);
      this.records.push(record);
      return record;
    }

    quality() { return 1 - Math.exp(-this.sim.step / (this.spec.tau || 22000)); }
    trueWin() { return (this.spec.w0 || 0.497) + (this.spec.gain || 0.04) * this.quality(); }

    learningRate() {
      const off = this.off;
      const warm = Math.min(1, (this.sim.step + 1) / Math.max(1, off.warmup_steps));
      const progress = Math.min(1, this.sim.step / Math.max(1, this.totalSteps));
      return off.learning_rate * warm * (0.05 + 0.95 * 0.5 * (1 + Math.cos(Math.PI * progress)));
    }

    start() {
      this.status = "running";
      this.sim.sessionMinutes = 0;
      this.sim.limitMinutes = this.limitHours * 60;
      this.push("session", { event: "start", mode: "games", parameters: this.parameters, limit_hours: this.limitHours, game_set: this.setName,
        train_games: this.trainGames, validation_games: this.heldOut, total_steps: this.totalSteps, steps_per_pass: this.stepsPerPass, batch: this.batch });
    }

    losses(heldOut) {
      const q = this.quality();
      const r = this.r;
      const pass = this.sim.step / this.stepsPerPass;
      const gap = heldOut ? 0.0035 + 0.0012 * Math.max(0, pass - 2.5) : 0;
      const out = {
        win_loss: round(0.6905 - 0.061 * q + gap + 0.0025 * gauss(r), 4),
        spread_loss: round(0.418 - 0.036 * q + gap * 0.8 + 0.002 * gauss(r), 4),
      };
      if (this.config.model.aux_heads) {
        Object.assign(out, {
          next_score_loss: round(0.3125 - 0.047 * q + gap * 0.6 + 0.0021 * gauss(r), 4),
          next_bingo_loss: round(0.5216 - 0.058 * q + gap + 0.0024 * gauss(r), 4),
          next_squares_loss: round(0.1172 - 0.021 * q + gap * 0.3 + 0.0006 * gauss(r), 4),
        });
      }
      return out;
    }

    /** One simulated minute of training. Returns true when the run has just finished. */
    minute() {
      const s = this.sim;
      const r = this.r;
      const off = this.off;
      const rate = this.pps * (1 + 0.018 * gauss(r));
      const steps = Math.max(1, Math.round((rate * 60) / this.batch));
      s.step = Math.min(this.totalSteps, s.step + steps);
      s.hours += 1 / 60;
      s.minutes += 1;
      s.sessionMinutes += 1;
      if (s.step > 200) {
        // Every 100 steps the helper weights are set so that each pulls a fixed share of the main output.
        const target = { spread: 0.162, next_score: 0.318, next_bingo: 0.207, next_squares: 0.914 };
        for (const key of Object.keys(s.weights)) s.weights[key] = round(target[key] * (1 + 0.05 * Math.sin(s.step / 3100 + key.length)) + 0.004 * gauss(r), 4);
      }
      const losses = this.losses(false);
      const loss = losses.win_loss + Object.entries(s.weights).reduce((a, [k, w]) => a + w * (losses[k + "_loss"] || 0), 0);
      const weights = {};
      for (const [k, w] of Object.entries(s.weights)) weights[k + "_weight"] = w;
      this.push("train", Object.assign({ loss: round(loss, 4) }, losses, weights,
        { learning_rate: Number(this.learningRate().toPrecision(4)), positions_per_second: round(rate, 1), pass: round(s.step / this.stepsPerPass, 3) }));
      while (s.nextValid <= s.step) {
        this.validate();
        s.nextValid += off.validate_steps;
      }
      const done = s.step >= this.totalSteps;
      const limit = s.limitMinutes && s.sessionMinutes >= s.limitMinutes;
      if (!done && s.minutes - s.lastEval >= (this.spec.evalEvery || 22)) { this.evaluate(off.eval_pairs); s.lastEval = s.minutes; }
      if (!done && s.minutes - s.lastCkpt >= off.checkpoint_minutes) this.checkpoint();
      if (done) { this.finish(`all ${int(this.totalSteps)} steps done`, true); return true; }
      if (limit) { this.finish(`time limit of ${this.limitHours} h reached`, false); return true; }
      return false;
    }

    validate() {
      const q = this.quality();
      const r = this.r;
      this.push("valid", Object.assign(this.losses(true), {
        win_accuracy: round(0.5862 + 0.0331 * q + 0.0016 * gauss(r), 4),
        win_brier: round(0.2154 - 0.0127 * q + 0.0007 * gauss(r), 4),
        value_rmse_points: round(74.62 - 3.35 * q + 0.09 * gauss(r), 2),
        leave_only_rmse_points: 73.88,
        positions_checked: 20000,
      }));
    }

    evaluate(pairs) {
      const r = this.r;
      const win = this.trueWin();
      // A seat-swapped pair's games are correlated, which narrows the interval a little.
      const winCi = 1.96 * 0.86 * 0.5 / Math.sqrt(2 * pairs) * (1 + 0.03 * gauss(r));
      const rate = Math.min(0.99, Math.max(0.01, win + 0.48 * winCi * gauss(r)));
      const spreadTrue = -1.6 + 28 * (win - 0.5);
      const spreadCi = 1.96 * 96 / Math.sqrt(pairs) * (1 + 0.03 * gauss(r));
      const spread = spreadTrue + 0.48 * spreadCi * gauss(r);
      const base = 428.3 + 2 * gauss(r);
      this.push("eval", { opponent: "greedy-leave", spread: round(spread, 2), spread_ci: round(spreadCi, 3), win_rate: round(rate, 4),
        win_ci: round(winCi, 4), score_for: round(base + spread / 2, 1), score_against: round(base - spread / 2, 1), pairs });
      const s = this.sim;
      if (!s.best || rate > s.best.win_rate) {
        s.best = { step: s.step, win_rate: round(rate, 4), win_ci: round(winCi, 4), pairs };
        this.push("best", { win_rate: round(rate, 4), win_ci: round(winCi, 4), pairs });
      }
    }

    checkpoint() {
      const s = this.sim;
      s.lastCkpt = s.minutes;
      const file = `step_${String(s.step).padStart(9, "0")}.pt`;
      this.push("checkpoint", { file });
      s.checkpoints.push({ file, step: s.step, modified: this.stamp() });
      s.checkpoints = s.checkpoints.slice(-this.off.keep_checkpoints);
    }

    finish(reason, finished) {
      this.validate();
      this.evaluate(finished ? this.off.final_pairs : this.off.eval_pairs);
      this.checkpoint();
      this.push("session", { event: "stop", reason, finished });
      this.status = finished ? "finished" : "stopped";
    }

    simulate(minutes) { for (let i = 0; i < minutes && this.status === "running"; i++) this.minute(); }

    // -- what training.js asks of it --

    advance(minutes) {
      if (this.status === "stopping") {
        this.live = true;
        this.finish("STOP file found", false);
        return true;
      }
      if (this.status !== "running") return false;
      this.live = true;
      for (let i = 0; i < minutes; i++) if (this.minute()) return true;
      return false;
    }

    stop(force) {
      if (this.status === "queued") { this.status = "stopped"; return { ok: true }; }
      if (this.status !== "running" && this.status !== "stopping") return fail(`${this.name} is not running`);
      if (force) {
        this.status = "stopped";
        this.error = null;
      } else this.status = "stopping";
      return { ok: true };
    }

    resume(body) {
      if (["running", "stopping", "queued"].includes(this.status)) return fail(`${this.name} is ${this.status}`);
      if (!this.sim.checkpoints.length) return fail(`${this.name} has no checkpoint to resume from`);
      for (const key of Object.keys(body.overrides || {})) {
        if (["offline.games", "offline.batch_size", "offline.validation_games"].includes(key)) return fail(`${key} cannot be changed when resuming`);
        if (key.startsWith("model.")) return fail(`${key} cannot be changed when resuming`);
      }
      const set = sets.find((s) => s.name === this.setName);
      if (!set) return fail(`there is no game set named '${this.setName}'`);
      if (set.games !== this.setGames) {
        return fail(`checkpoint was trained on the game set ${this.setName} when it held ${int(this.setGames)} games; the set has changed, so the run cannot carry on where it stopped`);
      }
      if (this.sim.step >= this.totalSteps) return fail(`${this.name} has done all its ${int(this.totalSteps)} steps; raise offline.passes to train it further`);
      this.limitHours = Number(body.hours) || 0;
      this.live = true;
      const last = this.sim.checkpoints[this.sim.checkpoints.length - 1];
      this.sim.step = last.step;
      this.start();
      return this.summary();
    }

    rename(to) {
      this.name = to;
      this.config.run.name = to;
      for (const id of Object.keys(library)) {
        const [run, file] = id.split("/");
        if (run === this.spec.name) { library[`${to}/${file}`] = library[id]; delete library[id]; }
      }
      this.spec.name = to;
    }

    evals() { return this.records.filter((r) => r.kind === "eval" && typeof r.win_rate === "number"); }

    health() {
      const ms = this.evals();
      if (!ms.length) return null;
      const final = ms[ms.length - 1];
      const best = ms.reduce((a, m) => (m.win_rate > a.win_rate ? m : a), ms[0]);
      const above = final.win_rate - final.win_ci > 0.5;
      const below = final.win_rate + final.win_ci < 0.5;
      const fell = best !== final && best.win_rate - final.win_rate > 1.0 * Math.hypot(best.win_ci, final.win_ci);
      const at = "against the static evaluator at the latest match";
      if (above) {
        return { state: "ahead", edge: null, note: `Wins ${pct(final.win_rate, final.win_ci)} of games ${at}: more than half, beyond the margin of error.` +
          (fell ? ` It won more earlier (${pct(best.win_rate, best.win_ci)} of games at step ${int(best.step)}).` : "") };
      }
      if (fell) {
        return { state: "worse", edge: null, note: `It got worse as it trained: ${pct(final.win_rate, final.win_ci)} of games ${at}, down from ` +
          `${pct(best.win_rate, best.win_ci)} of games at step ${int(best.step)}. The best network is kept.` };
      }
      if (below) return { state: "behind", edge: null, note: `Wins ${pct(final.win_rate, final.win_ci)} of games ${at}: less than half, beyond the margin of error.` };
      return { state: "level", edge: null, note: `Wins ${pct(final.win_rate, final.win_ci)} of games ${at}: not distinguishable from half.` };
    }

    matchObject(m) {
      return m ? { games: m.games, step: m.step, hours: m.hours, win_rate: m.win_rate, win_ci: m.win_ci, spread: round(m.spread, 1),
        ci: round(m.spread_ci, 1), pairs: m.pairs } : null;
    }

    summary() {
      const s = this.sim;
      const last = this.records[this.records.length - 1] || {};
      const ms = this.evals();
      const final = ms[ms.length - 1];
      const best = ms.length ? ms.reduce((a, m) => (m.win_rate > a.win_rate ? m : a), ms[0]) : null;
      const trains = this.records.filter((r) => r.kind === "train");
      const started = this.records.some((r) => r.kind === "session" && r.event === "start");
      return {
        name: this.name, status: this.status,
        phase: started || this.status === "running" ? "training" : null,
        network: this.network, parameters: this.parameters,
        games: last.games || 0, step: last.step || 0, positions: last.positions || 0, hours: last.hours || 0,
        games_per_second: null, created: this.created, updated: last.time || this.created,
        limit: { hours: this.limitHours, max_games: 0 },
        progress: this.status === "finished" ? 1 : started ? round(Math.min(1, (last.step || 0) / this.totalSteps), 4) : null,
        latest_eval: final ? { games: final.games, "greedy-leave": { spread: final.spread, ci: final.spread_ci, win_rate: final.win_rate, win_ci: final.win_ci } } : null,
        best_step: s.best ? s.best.step : null, has_best: !!s.best, checkpoints: s.checkpoints.length,
        changed: Object.assign({}, this.changed), error: this.error,
        trend: { games: ms.map((m) => m.games), spread: ms.map((m) => round(m.spread, 1)), ci: ms.map((m) => round(m.spread_ci, 1)),
          step: ms.map((m) => m.step), win_rate: ms.map((m) => m.win_rate), win_ci: ms.map((m) => m.win_ci) },
        best_eval: this.matchObject(best),
        health: this.health(),
        mode: "games", game_set: this.setName, total_steps: started ? this.totalSteps : null, passes: this.off.passes,
        pass: started ? round((last.step || 0) / this.stepsPerPass, 3) : null,
        positions_per_second: trains.length ? trains[trains.length - 1].positions_per_second : null,
      };
    }

    detail() {
      const size = Math.round(this.parameters * 12.1 + 140000);
      return {
        run: this.summary(), config: JSON.parse(JSON.stringify(this.config)),
        checkpoints: this.sim.checkpoints.slice().reverse().map((c) => ({ file: c.file, step: c.step, size_bytes: size, modified: c.modified,
          model_id: `${this.name}/${c.file}` })),
        best: this.sim.best ? { file: "best.pt", step: this.sim.best.step, model_id: `${this.name}/best.pt`, size_bytes: Math.round(this.parameters * 4.1),
          modified: this.records.filter((r) => r.kind === "best").pop().time } : null,
      };
    }

    log() {
      const lines = [];
      for (const r of this.records) {
        const t = `[${clock(r.time)}] `;
        if (r.kind === "session" && r.event === "start") {
          lines.push(t + `training from the game set ${r.game_set}: ${int(r.train_games)} games to learn from, ${int(r.validation_games)} held out; ` +
            `network ${this.network}, ${int(this.parameters)} parameters; ${int(r.total_steps)} steps of ${int(r.batch)} (${this.off.passes} passes) on cuda`);
          if (r.step > 0) lines.push(t + `resumed step_${String(r.step).padStart(9, "0")}.pt: step ${int(r.step)} of ${int(r.total_steps)}`);
        } else if (r.kind === "session") lines.push(t + `stopping: ${r.reason}`);
        else if (r.kind === "train") lines.push(t + `  step ${int(r.step)} of ${int(this.totalSteps)} (pass ${r.pass.toFixed(2)}): win loss ${r.win_loss.toFixed(4)}, ${int(r.positions_per_second)} positions/s`);
        else if (r.kind === "valid") {
          lines.push(t + `  held-out games: win loss ${r.win_loss.toFixed(4)}, right ${(100 * r.win_accuracy).toFixed(1)}% of the time, ` +
            `points error ${r.value_rmse_points.toFixed(1)} vs leave-only ${r.leave_only_rmse_points.toFixed(1)}`);
        } else if (r.kind === "eval") {
          lines.push(t + `eval: net@${r.step} vs greedy-leave: win ${pct(r.win_rate, r.win_ci)}, spread ${signed(r.spread)} +/- ${r.spread_ci.toFixed(1)} (${int(r.pairs)} pairs)`);
        } else if (r.kind === "best") lines.push(t + `new best network at step ${r.step}: wins ${(100 * r.win_rate).toFixed(1)}% against the static evaluator`);
        else if (r.kind === "checkpoint") lines.push(t + `checkpoint saved: runs/${this.name}/checkpoints/${r.file}`);
      }
      return lines;
    }

    report() {
      const name = this.name;
      const ms = this.evals();
      const final = ms[ms.length - 1] || null;
      const best = ms.length ? ms.reduce((a, m) => (m.win_rate > a.win_rate ? m : a), ms[0]) : null;
      const first = ms.find((m) => m.win_rate - m.win_ci > 0.5) || null;
      const checks = this.records.filter((r) => r.kind === "valid");
      const rates = this.records.filter((r) => r.kind === "train").map((r) => r.positions_per_second);
      const last = this.records[this.records.length - 1] || {};
      const figure = (m) => pct(m.win_rate, m.win_ci) + " of games";
      let headline;
      if (!final) {
        headline = last.step ? `${name} has trained for ${int(last.step)} steps and has not played the static evaluator yet.` : `${name} has not reported any results yet.`;
      } else if (final.win_rate - final.win_ci > 0.5) {
        headline = `${name} wins ${figure(final)} against the static evaluator, more than half beyond the margin of error.`;
      } else if (final.win_rate + final.win_ci < 0.5) {
        headline = `${name} wins ${figure(final)} against the static evaluator: less than half.`;
      } else {
        headline = `${name} has not been shown to beat the static evaluator: it wins ${figure(final)}, which is not distinguishable from half.`;
      }
      if (final && best && best !== final) headline += ` Its best match was ${figure(best)} at step ${int(best.step)}.`;
      const events = [];
      this.records.forEach((r, i) => {
        if (r.kind === "session" && r.event === "start") {
          events.push({ games: r.games, hours: r.hours, kind: "phase", tone: "info", title: i === 0 ? "Training began" : "Run resumed",
            detail: `Learning from ${int(r.train_games)} games of the set ${r.game_set}, one position from each per pass.` });
        } else if (r.kind === "session" && r.event === "stop") {
          events.push({ games: r.games, hours: r.hours, kind: "stop", tone: "info", title: r.finished ? "Run finished" : "Session stopped",
            detail: `Stopped at step ${int(r.step)} after ${(r.hours).toFixed(1)} h: ${r.reason}.` });
        } else if (r.kind === "best") {
          events.push({ games: r.games, hours: r.hours, kind: "promotion", tone: "good", title: "New best network",
            detail: `Won ${pct(r.win_rate, r.win_ci)} of games against the static evaluator, the most so far.` });
        }
      });
      if (first) {
        events.push({ games: first.games, hours: first.hours, kind: "significant", tone: "good", title: "First win rate clearly above half",
          detail: `${figure(first)} against the static evaluator over ${int(first.pairs)} pairs.` });
      }
      events.sort((a, b) => a.hours - b.hours);
      const held = checks[checks.length - 1];
      return {
        headline,
        summary: {
          steps: last.step || 0, positions: last.positions || 0, hours: round(last.hours || 0, 4),
          passes: round((last.step || 0) / this.stepsPerPass, 2), game_set: this.setName, train_games: this.trainGames,
          mean_positions_per_second: rates.length ? round(rates.reduce((a, b) => a + b, 0) / rates.length, 1) : null,
          best: this.matchObject(best), final: this.matchObject(final), first_significant: this.matchObject(first),
          held_out: held ? { win_loss: held.win_loss, win_accuracy: held.win_accuracy, value_rmse_points: held.value_rmse_points,
            leave_only_rmse_points: held.leave_only_rmse_points } : null,
          evaluations: ms.length, health: this.health(),
        },
        events,
      };
    }

    models() {
      const out = [];
      const s = this.sim;
      const base = { run: this.name, network: this.network, parameters: this.parameters, compatible: true, incompatible_reason: null };
      const evalAt = (step) => {
        const e = this.evals().filter((m) => m.step === step).pop();
        return e ? { "greedy-leave": { spread: e.spread, ci: e.spread_ci, win_rate: e.win_rate } } : null;
      };
      const meta = (id) => library[id] || { starred: false, note: "" };
      if (s.best) {
        const id = `${this.name}/best.pt`;
        const at = this.records.filter((r) => r.kind === "best").pop();
        out.push(Object.assign({}, base, { id, spec: `model:${id}`, name: `${this.name} (best)`, file: "best.pt", kind: "best",
          step: s.best.step, games: s.best.step * this.batch, modified: at.time, size_bytes: Math.round(this.parameters * 4.1),
          eval: evalAt(s.best.step) }, meta(id)));
      }
      for (const c of s.checkpoints.slice().reverse()) {
        const id = `${this.name}/${c.file}`;
        out.push(Object.assign({}, base, { id, spec: `model:${id}`, name: `${this.name} (step ${int(c.step)})`, file: c.file, kind: "checkpoint",
          step: c.step, games: c.step * this.batch, modified: c.modified, size_bytes: Math.round(this.parameters * 12.1 + 140000),
          eval: evalAt(c.step) }, meta(id)));
      }
      return out;
    }
  }

  function addRun(spec, minutes, status) {
    const run = new GamesRun(spec);
    runs.push(run);
    T.add(run);
    if (status === "queued") return run;
    run.start();
    run.simulate(minutes);
    if (status === "stopped" && run.status === "running") run.finish("STOP file found", false);
    return run;
  }

  if (!NOSETS && SCHEMA && T) {
    // Six passes over twenty-million with board-wide pooling and a match every 15 minutes: it
    // finished at about 53.6% of games while its spread stayed close to zero.
    addRun({ name: "pool-6x64", set: "twenty-million", created: NOW - 30 * HOUR, seed: 6411, pps: 15400, tau: 19000, gain: 0.041, evalEvery: 11,
      overrides: { "model.kind": "cnn-global", "offline.eval_minutes": 15 } }, 100000);
    // The transformer, a little past its second pass, level with the static evaluator so far.
    const transformer = addRun({ name: "transformer-8x192", set: "twenty-million", created: NOW - 2.4 * HOUR, seed: 8192, pps: 6350, tau: 52000,
      gain: 0.055, w0: 0.494, evalEvery: 20, overrides: { "model.kind": "transformer", "model.blocks": 8, "model.channels": 192 } },
    IDLE ? 118 : 116, IDLE ? "stopped" : "running");
    transformer.live = !IDLE;
  }

  // POST /api/runs with offline.games, by way of js/mock/training.js.
  if (T) {
    T.onCreate((item, startNow, created) => {
      const name = item.overrides["offline.games"];
      const set = sets.find((s) => s.name === name);
      if (!set || !set.games) return fail(`there is no game set named '${name}' with games in it`);
      advanceSet(set);
      if (set.status === "building" || set.status === "stopping") return fail(`the game set ${name} is still being built; wait for it or stop it first`);
      const overrides = Object.assign({}, item.overrides);
      delete overrides["offline.games"];
      for (const key of Object.keys(overrides)) {
        const known = SCHEMA.sections.some((s) => s.fields.some((f) => f.key === key));
        if (!known) return fail(`unknown setting ${key}`);
      }
      if ("offline.eval_pairs" in overrides && overrides["offline.eval_pairs"] < 2) return fail("offline.eval_pairs must be at least 2");
      const run = new GamesRun({ name: item.name, set: name, created, seed: 9000 + runs.length, overrides, evalEvery: 20 });
      run.live = true;
      runs.push(run);
      T.add(run);
      if (startNow) run.start();
      return run;
    });
  }

  window.MockOffline = {
    models: () => runs.flatMap((r) => r.models()),
    meta(id, body) {
      const run = runs.find((r) => id.startsWith(r.name + "/"));
      if (!run) return null;
      const entry = library[id] || (library[id] = { starred: false, note: "" });
      if (body && "starred" in body) entry.starred = !!body.starred;
      if (body && "note" in body) entry.note = String(body.note || "");
      return run.models().find((m) => m.id === id) || null;
    },
  };
})();
