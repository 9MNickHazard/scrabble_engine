/* Mock API for training runs and matches. Loaded only in mock mode (?mock=1).
 *
 * Answers GET /api/status, the /api/runs endpoints (list, detail, metrics, log, report, start,
 * stop, resume, delete) and the /api/matches endpoints, with procedurally generated data from a
 * seeded random generator, so screenshots are repeatable. The running run gains one simulated
 * minute of training every 5 seconds and the running match about 45 games a second, so "live"
 * pages visibly move.
 *
 * Mock-only flags in the page URL: &theme=light or &theme=dark forces the theme, &empty=1 starts
 * with no runs and no matches (the first-launch state).
 *
 * GET /api/models belongs to js/mock/library.js; a small fallback here answers it only when no
 * other mock file does (see `fallback` below).
 */
(function () {
  "use strict";

  const params = new URLSearchParams(location.search);
  const forcedTheme = params.get("theme");
  if (forcedTheme === "light" || forcedTheme === "dark") document.documentElement.dataset.theme = forcedTheme;
  const EMPTY = params.has("empty");
  const NOW = Date.now() / 1000;
  const HOUR = 3600;

  // ---- Random numbers -------------------------------------------------------------------

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
  function poisson(r, lambda) {
    const limit = Math.exp(-lambda);
    let k = 0;
    let p = 1;
    do { k++; p *= r(); } while (p > limit);
    return k - 1;
  }
  const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
  const sd = (xs) => {
    const m = mean(xs);
    return xs.length > 1 ? Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1)) : 0;
  };
  const round = (v, d = 2) => Math.round(v * 10 ** d) / 10 ** d;
  const int = (v) => Math.round(v).toLocaleString("en-US");
  const signed = (v) => (v >= 0 ? "+" : "-") + Math.abs(v).toFixed(1);
  const spreadText = (v, ci) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} ± ${ci.toFixed(1)}`;
  const clone = (x) => JSON.parse(JSON.stringify(x));

  // ---- Settings -------------------------------------------------------------------------

  const DEFAULTS = {
    run: { name: "run", out_dir: "runs", lexicon: "", leaves: "", seed: 1, hours: 0.0, max_games: 0, device: "cuda", vram_fraction: 0.9 },
    model: { blocks: 6, channels: 64, norm: "none", vector_hidden: 128, head_hidden: 256, value_channels: 8, cross_checks: true, leave_baseline: true },
    selfplay: { games: 256, max_candidates: 64, max_exchanges: 4, temperature_start: 4.0, temperature_end: 0.5,
      temperature_decay_games: 300000, win_weight: 0.0, win_scale: 100.0 },
    warmstart: { games: 100000, temperature: 6.0, max_candidates: 16, steps: 3000 },
    train: { batch_size: 2048, learning_rate: 0.001, warmup_steps: 500, weight_decay: 0.0001, grad_clip: 1.0, value_scale: 50.0,
      huber_delta: 2.0, win_loss_weight: 0.5, td_lambda: 1.0, replay_capacity: 3000000, min_replay: 100000, replay_ratio: 4.0,
      actor_sync_steps: 250, augment_transpose: true, amp: true, checkpoint_minutes: 10.0, eval_minutes: 20.0, eval_pairs: 250,
      gate_pairs: 500, keep_checkpoints: 5 },
  };
  const PARAMETERS = { "6x64": 1055056, "10x128": 3655316, "8x96": 2093624, "20x256": 24318740 };

  function configOf(name, changed) {
    const config = clone(DEFAULTS);
    config.run.name = name;
    for (const [key, value] of Object.entries(changed)) {
      const [section, field] = key.split(".");
      if (config[section]) config[section][field] = value;
    }
    return config;
  }

  // ---- Runs: a small simulator that writes metrics records like train.py does ------------

  const runs = [];

  function newRun(spec) {
    const config = configOf(spec.name, spec.changed || {});
    const network = `${config.model.blocks}x${config.model.channels}`;
    const run = {
      spec, config, network, parameters: PARAMETERS[network] || 1000000,
      r: rng(spec.seed), records: [], status: spec.status || "running", error: null,
      sim: { h: 0, games: 0, step: 0, positions: 0, replay: 0, phase: null, lastEval: 0, lastCkpt: 0, lastLog: 0,
        best: null, bestTrue: -99, evalIndex: 0, sessionStartH: 0, wallOffset: 0, checkpoints: [], snapshot: null,
        maxGame: 640, maxPlay: 150 },
      limitHours: spec.limitHours || 0, maxGames: 0, live: false,
    };
    runs.push(run);
    return run;
  }

  function stamp(run) {
    return run.live ? Date.now() / 1000 : run.spec.created + run.sim.wallOffset + run.sim.h * HOUR;
  }

  function push(run, kind, extra) {
    const s = run.sim;
    const record = Object.assign({ kind, time: round(stamp(run), 3), step: s.step, games: s.games, positions: s.positions,
      hours: round(s.h, 4) }, extra);
    run.records.push(record);
    return record;
  }

  function quality(run, h) {
    return 1 - Math.exp(-h / run.spec.tau);
  }

  function evalCi(run) {
    const pairs = run.config.train.eval_pairs;
    return (run.spec.ciBase || 7.9) * Math.sqrt(250 / pairs) * (1 + 0.04 * gauss(run.r));
  }

  function startSession(run) {
    const s = run.sim;
    s.sessionStartH = s.h;
    push(run, "session", { event: "start" });
    if (s.phase === null) s.phase = "warmstart";
    run.status = "running";
  }

  /** Advance the run by one minute of training (or one phase at the start). */
  function minute(run) {
    const s = run.sim;
    const r = run.r;
    const cfg = run.config;
    if (s.phase === "warmstart") {
      s.h += 176 / HOUR;
      s.replay = Math.round(cfg.warmstart.games * 23.7);
      push(run, "warmstart", { games_played: cfg.warmstart.games, positions_added: s.replay, seconds: 176.2, mean_score: 402.6 });
      s.phase = "pretraining";
      return;
    }
    if (s.phase === "pretraining") {
      for (let step = 500; step <= cfg.warmstart.steps; step += 500) {
        s.h += 21 / HOUR;
        s.step = step;
        const vl = 0.962 - 0.012 * (step / 3000) + 0.004 * gauss(r);
        const wl = 0.468 - 0.014 * (step / 3000) + 0.003 * gauss(r);
        push(run, "train", { loss: round(vl + 0.5 * wl, 4), value_loss: round(vl, 4), win_loss: round(wl, 4),
          value_error_points: round(54.2 - 0.4 * (step / 3000) + 0.2 * gauss(r), 3), learning_rate: cfg.train.learning_rate });
      }
      s.phase = "selfplay";
      s.lastEval = s.h;
      checkpoint(run);
      return;
    }

    s.h += 1 / 60;
    const q = quality(run, s.h);
    const slow = run.spec.slowAt && s.h > run.spec.slowAt && s.h < run.spec.slowAt + 0.25 ? 0.74 : 1;
    const gps = run.spec.gps * slow * (1 + 0.013 * gauss(r));
    const games = Math.round(gps * 60);
    const positions = Math.round(games * (24.4 - 0.6 * q));
    s.games += games;
    s.positions += positions;
    s.step += Math.round((positions * cfg.train.replay_ratio) / cfg.train.batch_size);
    s.replay = Math.min(cfg.train.replay_capacity, s.replay + positions);
    const vl = 0.951 - 0.074 * q + 0.006 * gauss(r);
    const wl = 0.451 - 0.021 * q + 0.003 * gauss(r);
    push(run, "train", { loss: round(vl + 0.5 * wl, 4), value_loss: round(vl, 4), win_loss: round(wl, 4),
      value_error_points: round(53.9 - 3.1 * q + 0.25 * gauss(r), 3), learning_rate: cfg.train.learning_rate });
    const sp = cfg.selfplay;
    const temperature = sp.temperature_end + (sp.temperature_start - sp.temperature_end) *
      Math.max(0, 1 - s.games / sp.temperature_decay_games);
    const leaveOnly = 73.9 + 0.75 * gauss(r);
    const maxGame = 560 + Math.round(90 * r() + 40 * q);
    const maxPlay = 120 + Math.round(70 * r() ** 3);
    push(run, "selfplay", {
      games_per_second: round(gps, 3), positions_per_second: round(positions / 60, 2),
      mean_score: round(415.2 + 12.5 * q + 3.2 * (1 - temperature / 4) + 1.1 * gauss(r), 2),
      mean_turns: round(24.72 - 0.7 * q + 0.12 * gauss(r), 3), mean_abs_spread: round(86 - 4.2 * q + 0.9 * gauss(r), 2),
      first_player_win_rate: round(0.546 + 0.009 * gauss(r), 4), temperature: round(temperature, 4),
      value_rmse_points: round(leaveOnly - (run.spec.rmseGain || 2.1) * q + 0.16 * gauss(r), 3), leave_only_rmse_points: round(leaveOnly, 3),
      win_brier: round(0.1525 - 0.0125 * q + 0.0016 * gauss(r), 5), replay_positions: s.replay,
      replay_gb: round(s.replay * 310 / 1e9, 3), bingos_per_game: round(1.73 + 0.21 * q + 0.025 * gauss(r), 3),
      exchanges_per_game: round(0.46 - 0.15 * q + 0.012 * gauss(r), 3),
      passes_per_game: round(Math.max(0.004, 0.031 - 0.013 * q + 0.002 * gauss(r)), 4),
      max_game_score: maxGame, max_play_score: maxPlay,
    });
    if (s.h - s.lastCkpt >= cfg.train.checkpoint_minutes / 60 - 1e-6) checkpoint(run);
    if (s.h - s.lastEval >= cfg.train.eval_minutes / 60 - 1e-6) evaluate(run);
  }

  function checkpoint(run) {
    const s = run.sim;
    s.lastCkpt = s.h;
    const file = `step_${String(s.step).padStart(9, "0")}.pt`;
    push(run, "checkpoint", { file, replay_positions: s.replay });
    s.checkpoints.push({ file, step: s.step, games: s.games, modified: stamp(run) });
    s.checkpoints = s.checkpoints.slice(-run.config.train.keep_checkpoints);
    s.snapshot = { h: s.h, games: s.games, step: s.step, positions: s.positions, replay: s.replay };
  }

  function evaluate(run) {
    const s = run.sim;
    const r = run.r;
    s.lastEval = s.h;
    s.evalIndex++;
    const q = quality(run, s.h);
    let trueGl = run.spec.gl * q + (run.spec.gl0 || 0);
    if (run.spec.dipAt === s.evalIndex) trueGl -= 6.5;
    const ciGl = evalCi(run);
    const gl = trueGl + gauss(r) * (ciGl / 1.96) * 0.75;
    const trueG = 36 + 10.5 * q;
    const ciG = evalCi(run) * 1.2;
    const g = trueG + gauss(r) * (ciG / 1.96) * 0.75;
    const pairs = run.config.train.eval_pairs;
    const extra = (opponent, spread, ci, base) => ({
      opponent, spread: round(spread, 2), spread_ci: round(ci, 3), win_rate: round(0.5 + spread / 118 + 0.008 * gauss(r), 4),
      win_ci: round(0.98 / Math.sqrt(pairs) * 1.1, 4), score_for: round(base + spread / 2, 1), score_against: round(base - spread / 2, 1), pairs,
    });
    const glRecord = push(run, "eval", extra("greedy-leave", gl, ciGl, 428));
    push(run, "eval", extra("greedy", g, ciG, 421));
    // Gate against the best network so far: promoted only when the interval excludes zero.
    const ciGate = 5.3 + 0.3 * gauss(r);
    const gain = trueGl - s.bestTrue;
    const promoted = s.best === null || (gain > 0.85 && run.spec.dipAt !== s.evalIndex);
    const spread = s.best === null ? 9.5 + 3 * r() : promoted ? ciGate + 0.6 + 2.4 * r() : Math.min(ciGate - 0.4, gain + 2.2 * gauss(r));
    push(run, "gate", { spread: round(spread, 2), spread_ci: round(ciGate, 3), win_rate: round(0.5 + spread / 120, 4),
      best_step: s.best ? s.best.step : null, promoted, pairs: run.config.train.gate_pairs });
    if (promoted) {
      s.best = { step: s.step, games: s.games, hours: s.h, spread: glRecord.spread, ci: glRecord.spread_ci,
        win_rate: glRecord.win_rate, greedy: { spread: round(g, 2), ci: round(ciG, 3), win_rate: round(0.5 + g / 118, 4) } };
      s.bestTrue = trueGl;
    }
  }

  function stopSession(run, reason) {
    push(run, "session", { event: "stop", reason });
  }

  /** A clean end: evaluate and save unless that just happened, then close the session. */
  function finish(run, reason, status) {
    const s = run.sim;
    if (s.phase === "selfplay" && s.h - s.lastEval > 1e-6) evaluate(run);
    if (s.h - s.lastCkpt > 1e-6) checkpoint(run);
    stopSession(run, reason);
    run.status = status;
  }

  function simulateTo(run, hours) {
    while (run.sim.h < hours - 1e-9) minute(run);
  }

  /** The finished run's second session: resume from the last checkpoint after a gap. */
  function resumeFromSnapshot(run, gapHours, changed) {
    const s = run.sim;
    const snap = s.snapshot;
    s.wallOffset += (s.h - snap.h) * HOUR + gapHours * HOUR;
    Object.assign(s, { h: snap.h, games: snap.games, step: snap.step, positions: snap.positions, replay: snap.replay });
    s.lastEval = s.h;
    s.lastCkpt = s.h;
    for (const [key, value] of Object.entries(changed)) {
      const [section, field] = key.split(".");
      run.config[section][field] = value;
      run.spec.changed[key] = value;
    }
    startSession(run);
  }

  function buildRuns() {
    if (EMPTY) return;
    // A run that is training now: three hours into an eight-hour limit.
    const night3 = newRun({ name: "night3", seed: 3303, created: NOW - 3.25 * HOUR, limitHours: 8, gps: 55.4, tau: 2.6, gl: 6.6, gl0: -0.4,
      ciBase: 7.8, changed: { "run.hours": 8, "train.td_lambda": 0.9 } });
    startSession(night3);
    simulateTo(night3, 3.2);
    night3.live = true;

    // Waiting for night3 to finish.
    const night4 = newRun({ name: "night4", seed: 4404, created: NOW - 3.25 * HOUR + 2, limitHours: 8, gps: 18, tau: 2.6, gl: 7,
      status: "queued", changed: { "run.hours": 8, "model.blocks": 10, "model.channels": 128 } });
    void night4;

    // Last night's run: eight hours, interrupted once and resumed with more evaluation pairs.
    const night2 = newRun({ name: "night2", seed: 2202, created: NOW - 17 * HOUR, limitHours: 8, gps: 56.2, tau: 2.2, gl: 7.0, gl0: -0.6,
      ciBase: 8.1, dipAt: 15, slowAt: 5.6, rmseGain: 2.0, changed: { "run.hours": 8 } });
    startSession(night2);
    simulateTo(night2, 3.2);
    resumeFromSnapshot(night2, 1.45, { "run.hours": 5, "train.eval_pairs": 1000 });
    night2.limitHours = 5;
    simulateTo(night2, night2.sim.sessionStartH + 5);
    finish(night2, "time limit reached", "finished");

    // A larger network that ran out of GPU memory.
    const wide1 = newRun({ name: "wide1", seed: 1101, created: NOW - 30 * HOUR, limitHours: 6, gps: 17.6, tau: 3, gl: 5, gl0: -1,
      ciBase: 8.4, changed: { "run.hours": 6, "model.blocks": 10, "model.channels": 128, "selfplay.games": 1024, "train.batch_size": 8192 } });
    startSession(wide1);
    simulateTo(wide1, 0.93);
    wide1.status = "failed";
    wide1.error = [
      "  File \"src/scrabble_ai/train.py\", line 412, in train_step",
      "    loss.backward()",
      "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.25 GiB. GPU 0 has a total capacity of 15.92 GiB",
      "of which 1.31 GiB is free. Of the allocated memory 12.86 GiB is allocated by PyTorch. Lower train.batch_size",
      "or selfplay.games, or use a smaller network.",
    ].join("\n");

    // The first five-minute dry run.
    const dryrun = newRun({ name: "dryrun", seed: 909, created: NOW - 50 * HOUR, limitHours: 0.1, gps: 58.4, tau: 2.2, gl: 7, gl0: 3.6,
      changed: { "run.hours": 0.1, "warmstart.games": 20000, "train.eval_minutes": 2 } });
    startSession(dryrun);
    simulateTo(dryrun, 0.105);
    finish(dryrun, "time limit reached", "finished");
  }

  // ---- Live updates ---------------------------------------------------------------------

  let lastTick = Date.now();
  function tick() {
    const now = Date.now();
    const minutes = Math.floor((now - lastTick) / 5000);
    if (minutes <= 0) return;
    lastTick += minutes * 5000;
    for (const run of runs.slice()) {
      if (run.status === "stopping") {
        finish(run, "stopped from the app", "stopped");
        startNextQueued();
        continue;
      }
      if (run.status !== "running") continue;
      run.live = true;
      for (let i = 0; i < Math.min(minutes, 6); i++) {
        minute(run);
        const s = run.sim;
        if (run.limitHours && s.h - s.sessionStartH >= run.limitHours) {
          finish(run, "time limit reached", "finished");
          startNextQueued();
          break;
        }
      }
    }
  }

  function startNextQueued() {
    if (runs.some((r) => r.status === "running" || r.status === "stopping")) return;
    const next = runs.filter((r) => r.status === "queued").sort((a, b) => a.spec.created - b.spec.created)[0];
    if (next) {
      next.live = true;
      startSession(next);
    }
  }

  // ---- Shapes the API returns -----------------------------------------------------------

  function evalAt(run, opponent, step) {
    const evals = run.records.filter((r) => r.kind === "eval" && r.opponent === opponent && (step === undefined || r.step <= step));
    return evals[evals.length - 1] || null;
  }

  function summary(run) {
    const s = run.sim;
    const last = run.records[run.records.length - 1];
    const sp = run.records.filter((r) => r.kind === "selfplay");
    const gl = evalAt(run, "greedy-leave");
    const g = evalAt(run, "greedy");
    const sessionHours = s.h - s.sessionStartH;
    return {
      name: run.spec.name,
      status: run.status,
      phase: s.phase,
      network: run.network,
      parameters: run.parameters,
      games: s.games,
      step: s.step,
      positions: s.positions,
      hours: round(s.h, 4),
      games_per_second: sp.length ? sp[sp.length - 1].games_per_second : null,
      created: run.spec.created,
      updated: last ? last.time : run.spec.created,
      limit: { hours: run.limitHours, max_games: run.maxGames },
      progress: run.limitHours ? round(Math.min(1, sessionHours / run.limitHours), 4) : null,
      latest_eval: gl ? {
        games: gl.games,
        "greedy-leave": { spread: gl.spread, ci: gl.spread_ci, win_rate: gl.win_rate },
        greedy: g ? { spread: g.spread, ci: g.spread_ci, win_rate: g.win_rate } : null,
      } : null,
      best_step: s.best ? s.best.step : null,
      has_best: !!s.best,
      checkpoints: s.checkpoints.length,
      changed: Object.assign({}, run.spec.changed),
      error: run.error,
    };
  }

  function checkpointList(run) {
    const size = Math.round(run.parameters * 12.1 + 140000);
    return run.sim.checkpoints.slice().reverse().map((c) => ({
      file: c.file, step: c.step, size_bytes: size, modified: c.modified, model_id: `${run.spec.name}/${c.file}`,
    }));
  }

  function detail(run) {
    return {
      run: summary(run),
      config: run.config,
      checkpoints: checkpointList(run),
      best: run.sim.best ? { step: run.sim.best.step, model_id: `${run.spec.name}/best.pt` } : null,
    };
  }

  function clockTime(time) {
    return new Date(time * 1000).toTimeString().slice(0, 8);
  }

  function logLines(run) {
    const name = run.spec.name;
    const lines = [];
    for (const r of run.records) {
      const t = `[${clockTime(r.time)}] `;
      if (r.kind === "session" && r.event === "start") {
        lines.push(t + `run '${name}' in runs/${name}: ${run.network} network, ${int(run.parameters)} parameters`);
        if (r.games > 0) lines.push(t + `resumed from checkpoint at step ${int(r.step)} (${int(r.games)} games)`);
      } else if (r.kind === "session") lines.push(t + `stopped: ${r.reason}`);
      else if (r.kind === "warmstart") lines.push(t + `warm start: ${int(r.games_played)} games, ${int(r.positions_added)} positions in ${r.seconds.toFixed(0)} s`);
      else if (r.kind === "train" && r.games === 0) lines.push(t + `pretrain step ${int(r.step)}: loss ${r.loss.toFixed(4)}`);
      else if (r.kind === "selfplay") {
        lines.push(t + `games ${int(r.games)} (${r.games_per_second.toFixed(1)}/s), step ${int(r.step)}, temp ${r.temperature.toFixed(2)}, ` +
          `score ${Math.round(r.mean_score)}, value RMSE ${r.value_rmse_points.toFixed(1)} vs leave-only ${r.leave_only_rmse_points.toFixed(1)}, ` +
          `replay ${int(r.replay_positions)}`);
      } else if (r.kind === "eval") {
        lines.push(t + `net@${r.step} vs ${r.opponent}: spread ${signed(r.spread)} +/- ${r.spread_ci.toFixed(1)}, ` +
          `win ${(100 * r.win_rate).toFixed(1)}% (${r.pairs} pairs)`);
      } else if (r.kind === "gate") {
        lines.push(t + `gate: net@${r.step} vs best@${r.best_step === null ? "warm start" : r.best_step}: spread ${signed(r.spread)} +/- ` +
          `${r.spread_ci.toFixed(1)}, ${r.promoted ? "promoted to best.pt" : "kept best"}`);
      } else if (r.kind === "checkpoint") lines.push(t + `checkpoint ${r.file}`);
    }
    if (run.status === "failed" && run.error) {
      const last = run.records[run.records.length - 1];
      lines.push(`[${clockTime(last.time + 31)}] Traceback (most recent call last):`);
      for (const l of run.error.split("\n")) lines.push(l);
    }
    if (run.status === "stopping") lines.push(`[${clockTime(Date.now() / 1000)}] STOP requested: finishing the step, evaluating, saving`);
    return lines;
  }

  function report(run) {
    const name = run.spec.name;
    const s = run.sim;
    const recs = run.records;
    const evals = recs.filter((r) => r.kind === "eval" && r.opponent === "greedy-leave");
    const gates = recs.filter((r) => r.kind === "gate");
    const sps = recs.filter((r) => r.kind === "selfplay");
    const events = [];
    const first = evals.find((r) => r.spread - r.spread_ci > 0) || null;
    const final = evals[evals.length - 1] || null;
    const promoted = gates.filter((g) => g.promoted);
    const lastPromotion = promoted[promoted.length - 1];
    const bestEval = lastPromotion ? evals.filter((e) => e.step <= lastPromotion.step).pop() : null;
    const recent = sps.slice(-10);
    const rmseGain = recent.length ? mean(recent.map((r) => r.leave_only_rmse_points - r.value_rmse_points)) : null;

    for (const r of recs) {
      if (r.kind === "warmstart") {
        events.push({ games: r.games, hours: r.hours, kind: "phase", tone: "info", title: "Warm start finished",
          detail: `${int(r.games_played)} games in ${Math.round(r.seconds)} s.` });
      }
    }
    const firstSp = sps[0];
    if (firstSp) {
      events.push({ games: 0, hours: round(firstSp.hours - 1 / 60, 3), kind: "phase", tone: "info", title: "Self-play started",
        detail: `Pretraining finished after ${int(run.config.warmstart.steps)} steps.` });
    }
    for (const g of promoted) {
      events.push({ games: g.games, hours: g.hours, kind: "promotion", tone: "good", title: "New best network",
        detail: g.best_step === null ? `Step ${int(g.step)} became the first best network.`
          : `Step ${int(g.step)} beat the previous best by ${spreadText(g.spread, g.spread_ci)} over ${g.pairs} pairs.` });
    }
    if (first) {
      events.push({ games: first.games, hours: first.hours, kind: "significant", tone: "good", title: "First significant win over greedy-leave",
        detail: `${spreadText(first.spread, first.spread_ci)} over ${int(first.pairs)} pairs.` });
    }
    for (let i = 1; i < evals.length; i++) {
      const drop = evals[i - 1].spread - evals[i].spread;
      if (drop > 4.5) {
        events.push({ games: evals[i].games, hours: evals[i].hours, kind: "regression", tone: "warning", title: "Evaluation dropped",
          detail: `Fell ${drop.toFixed(1)} points from the previous evaluation.` });
      }
    }
    let record = 700;
    for (const r of sps) {
      if (r.max_game_score > record && r.hours > 1) {
        record = r.max_game_score;
        events.push({ games: r.games, hours: r.hours, kind: "record", tone: "info", title: "Highest game score so far",
          detail: `A self-play game of ${r.max_game_score} points.` });
      }
    }
    const median = sps.length ? sps.map((r) => r.games_per_second).sort((a, b) => a - b)[Math.floor(sps.length / 2)] : 0;
    let slowSince = null;
    for (const r of sps) {
      if (r.games_per_second < 0.85 * median && slowSince === null) {
        slowSince = r;
        events.push({ games: r.games, hours: r.hours, kind: "slowdown", tone: "warning", title: "Self-play slowed down",
          detail: `${r.games_per_second.toFixed(1)} games/s against a typical ${median.toFixed(1)}.` });
      } else if (r.games_per_second >= 0.9 * median) slowSince = null;
    }
    for (let i = 0; i < recs.length; i++) {
      const r = recs[i];
      if (r.kind !== "session") continue;
      const prev = recs[i - 1];
      if (r.event === "start" && prev && !(prev.kind === "session" && prev.event === "stop")) {
        events.push({ games: prev.games, hours: prev.hours, kind: "stop", tone: "warning", title: "Run interrupted",
          detail: `The process ended without a clean stop. It resumed ${Math.round((r.time - prev.time) / 60)} minutes later from the ` +
            `checkpoint at step ${int(r.step)}; ${int(prev.games - r.games)} self-play games after that checkpoint were lost.` });
      } else if (r.event === "start" && prev) {
        events.push({ games: r.games, hours: r.hours, kind: "phase", tone: "info", title: "Resumed",
          detail: `Resumed from the checkpoint at step ${int(r.step)}.` });
      } else if (r.event === "stop") {
        const limit = r.reason === "time limit reached";
        events.push({ games: r.games, hours: r.hours, kind: "stop", tone: "info", title: limit ? "Finished" : "Stopped",
          detail: limit ? "The session reached its time limit, evaluated and saved a final checkpoint." : `Stopped: ${r.reason}.` });
      }
    }
    events.sort((a, b) => a.hours - b.hours);

    let headline;
    const best = bestEval ? { step: lastPromotion.step, games: lastPromotion.games, spread: bestEval.spread, ci: bestEval.spread_ci,
      win_rate: bestEval.win_rate } : null;
    if (run.status === "queued") {
      headline = `${name} is waiting in the queue; there is nothing to report until it starts.`;
    } else if (!final) {
      headline = sps.length || run.status === "running"
        ? `${name} has not been evaluated yet. The first evaluation comes ${run.config.train.eval_minutes} minutes into self-play.`
        : `${name} ended before its first evaluation.`;
    } else if (best && best.spread - best.ci > 0) {
      headline = `${name} beat greedy-leave by ${best.spread.toFixed(1)} points a game at its best, and the gain is significant.`;
    } else {
      headline = `${name} is at ${spreadText(final.spread, final.spread_ci)} against greedy-leave so far: not yet a significant difference.`;
    }
    return {
      headline,
      summary: {
        games: s.games, hours: round(s.h, 3), steps: s.step,
        mean_games_per_second: sps.length ? round(mean(sps.map((r) => r.games_per_second)), 2) : null,
        best,
        final: final ? { step: final.step, games: final.games, spread: final.spread, ci: final.spread_ci, win_rate: final.win_rate } : null,
        first_significant: first ? { games: first.games, hours: first.hours } : null,
        promotions: promoted.length,
        value_rmse_gain: rmseGain === null ? null : round(rmseGain, 2),
      },
      events,
    };
  }

  // ---- Models (fallback) ----------------------------------------------------------------

  const BOTS = [
    { spec: "bot:greedy-leave", name: "Greedy + leave", description: "Highest score plus leave value." },
    { spec: "bot:greedy", name: "Greedy", description: "Highest score." },
    { spec: "bot:random", name: "Random", description: "A random legal placement." },
  ];

  function modelList() {
    const models = [];
    for (const run of runs) {
      const s = run.sim;
      const name = run.spec.name;
      const base = { run: name, network: run.network, parameters: run.parameters, compatible: true, starred: false, note: "" };
      const evalFor = (step) => {
        const gl = evalAt(run, "greedy-leave", step);
        const g = evalAt(run, "greedy", step);
        return gl ? { "greedy-leave": { spread: gl.spread, ci: gl.spread_ci, win_rate: gl.win_rate },
          greedy: g ? { spread: g.spread, ci: g.spread_ci, win_rate: g.win_rate } : null } : null;
      };
      if (s.best) {
        models.push(Object.assign({}, base, { id: `${name}/best.pt`, spec: `model:${name}/best.pt`, name: `${name} (best)`, file: "best.pt",
          kind: "best", step: s.best.step, games: s.best.games, modified: run.records[run.records.length - 1].time,
          size_bytes: Math.round(run.parameters * 4.1), eval: evalFor(s.best.step) }));
      }
      for (const c of s.checkpoints.slice().reverse()) {
        models.push(Object.assign({}, base, { id: `${name}/${c.file}`, spec: `model:${name}/${c.file}`,
          name: `${name} (step ${int(c.step)})`, file: c.file, kind: "checkpoint", step: c.step, games: c.games,
          modified: c.modified, size_bytes: Math.round(run.parameters * 12.1 + 140000), eval: evalFor(c.step) }));
      }
    }
    if (!EMPTY) {
      models.push({ id: "spring/best.pt", spec: "model:spring/best.pt", name: "spring (best)", run: "spring", file: "best.pt", kind: "best",
        network: "6x64", parameters: 1055056, step: 41200, games: 900000, modified: NOW - 40 * 86400, size_bytes: 4325000,
        compatible: false, eval: { "greedy-leave": { spread: 2.4, ci: 6.9, win_rate: 0.51 }, greedy: null }, starred: false,
        note: "Trained with the CSW word list" });
    }
    return { bots: BOTS, models };
  }

  function playerName(spec) {
    const bot = BOTS.find((b) => b.spec === spec);
    if (bot) return bot.name;
    if (spec === "human") return "You";
    const id = spec.replace(/^model:/, "");
    const [run, file] = id.split("/");
    if (file === "best.pt") return `${run} (best)`;
    const step = parseInt((file || "").replace(/\D/g, ""), 10);
    return Number.isFinite(step) ? `${run} (step ${int(step)})` : id;
  }

  /** Rough strength in points against greedy, to make simulated matches plausible. */
  function strength(spec) {
    if (spec === "bot:random") return -165;
    if (spec === "bot:greedy") return 0;
    if (spec === "bot:greedy-leave") return 37;
    const id = spec.replace(/^model:/, "");
    const [runName, file] = id.split("/");
    const run = runs.find((r) => r.spec.name === runName);
    if (!run) return 39;
    const step = file === "best.pt" && run.sim.best ? run.sim.best.step : parseInt(file.replace(/\D/g, ""), 10);
    const gl = evalAt(run, "greedy-leave", step);
    return 37 + (gl ? gl.spread : 0);
  }

  // ---- Matches --------------------------------------------------------------------------

  const matches = [];
  const BINGO_WORDS = ["RETAINS", "SALTINE", "ENTAILS", "TOENAILS", "REALIGNS", "ORIENTAL", "STEARIN", "AIRLINES", "DETAINS",
    "LATRINES", "OUTRAISE", "ANESTRI", "QUIXOTIC", "ZANIEST", "OUTJINX", "JOUSTERS", "WAXINESS", "BRONZIEST"];
  const SHORT_WORDS = ["ZAX", "QUA", "JEUX", "OXO", "ZOEA", "FAQIR", "QOPH", "JINX", "AZO", "ZEP", "COZY", "QUIZ", "JAMB", "WAXY"];

  function matchId(time) {
    const d = new Date(time * 1000);
    const p = (n) => String(n).padStart(2, "0");
    return `m_${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
  }

  function generateGames(seed, pairs, a, b) {
    const r = rng(seed);
    const adv = strength(a) - strength(b);
    const cols = { seed: [], a_first: [], score_a: [], score_b: [], turns: [], bingos_a: [], bingos_b: [], exchanges_a: [], exchanges_b: [],
      passes_a: [], passes_b: [], best_play_a: [], best_play_b: [], best_word_a: [], best_word_b: [], ended_by: [] };
    const sa0 = 1.86 + Math.max(-1.2, strength(a) / 400);
    const sb0 = 1.86 + Math.max(-1.2, strength(b) / 400);
    const word = (score, bingo) => (bingo && score >= 70 ? BINGO_WORDS : SHORT_WORDS)[Math.floor(r() * (bingo && score >= 70 ? BINGO_WORDS.length : SHORT_WORDS.length))];
    const bestPlay = (bingos) => {
      let score = 38 + Math.round(18 * Math.abs(gauss(r)));
      if (bingos > 0) score = 66 + Math.round(22 * Math.abs(gauss(r)));
      if (r() < 0.012) score = 140 + Math.round(80 * r());
      return score;
    };
    for (let k = 0; k < pairs; k++) {
      const seatLuck = gauss(r) * 70;
      const seedValue = 7000000000000 + seed * 10007 + k;
      for (const aFirst of [true, false]) {
        const spread = adv + (aFirst ? 11 : -11) + (aFirst ? seatLuck : -seatLuck) + gauss(r) * 50;
        const total = 862 + gauss(r) * 58 + Math.min(0, strength(a) + strength(b)) * 0.8;
        const sa = Math.round((total + spread) / 2);
        const sb = Math.round((total - spread) / 2);
        const ba = poisson(r, Math.max(0.1, sa0 + spread / 300));
        const bb = poisson(r, Math.max(0.1, sb0 - spread / 300));
        const pa = bestPlay(ba);
        const pb = bestPlay(bb);
        cols.seed.push(seedValue);
        cols.a_first.push(aFirst);
        cols.score_a.push(sa);
        cols.score_b.push(sb);
        cols.turns.push(Math.max(16, Math.round(23.1 + gauss(r) * 2.1)));
        cols.bingos_a.push(ba);
        cols.bingos_b.push(bb);
        cols.exchanges_a.push(poisson(r, 0.31));
        cols.exchanges_b.push(poisson(r, 0.28));
        cols.passes_a.push(r() < 0.015 ? 1 : 0);
        cols.passes_b.push(r() < 0.015 ? 1 : 0);
        cols.best_play_a.push(pa);
        cols.best_play_b.push(pb);
        cols.best_word_a.push(word(pa, ba > 0));
        cols.best_word_b.push(word(pb, bb > 0));
        cols.ended_by.push(r() < 0.997 ? "out" : "scoreless");
      }
    }
    return cols;
  }

  function newMatch({ id, a, b, pairs, created, seed, status, rate, done, seconds, candidates }) {
    const match = { id: id || matchId(created), a, b, a_name: playerName(a), b_name: playerName(b), pairs, created, status,
      rate: rate || 45, startedAt: Date.now() / 1000 - (done || 0) / (rate || 45), fixedDone: done, seconds, candidates: candidates || 64,
      cols: generateGames(seed, pairs, a, b) };
    matches.push(match);
    return match;
  }

  function gamesDone(match) {
    const total = match.pairs * 2;
    if (match.status === "finished") return total;
    if (match.status === "cancelled" || match.status === "failed") return match.fixedDone;
    const done = Math.min(total, Math.floor((Date.now() / 1000 - match.startedAt) * match.rate));
    if (done >= total) {
      match.status = "finished";
      match.seconds = round(total / match.rate, 1);
    }
    return done;
  }

  function sliceCols(cols, n) {
    const out = {};
    for (const [key, values] of Object.entries(cols)) out[key] = values.slice(0, n);
    return out;
  }

  function matchSummaryStats(cols) {
    const n = cols.score_a.length;
    const pairCount = Math.floor(n / 2);
    if (pairCount < 2) return null;
    const spreads = cols.score_a.map((s, i) => s - cols.score_b[i]);
    const pairSpreads = [];
    const pairWins = [];
    const win = (s) => (s > 0 ? 1 : s < 0 ? 0 : 0.5);
    for (let k = 0; k < pairCount; k++) {
      pairSpreads.push((spreads[2 * k] + spreads[2 * k + 1]) / 2);
      pairWins.push((win(spreads[2 * k]) + win(spreads[2 * k + 1])) / 2);
    }
    const spread = mean(pairSpreads);
    const ci = 1.96 * sd(pairSpreads) / Math.sqrt(pairCount);
    const winRate = mean(pairWins);
    const winCi = 1.96 * sd(pairWins) / Math.sqrt(pairCount);
    const side = (first) => {
      const idx = spreads.map((_, i) => i).filter((i) => cols.a_first[i] === first);
      return { spread: round(mean(idx.map((i) => spreads[i])), 2), win_rate: round(mean(idx.map((i) => win(spreads[i]))), 4) };
    };
    return {
      spread: round(spread, 2), spread_ci: round(ci, 2), win_rate: round(winRate, 4), win_ci: round(winCi, 4),
      significant: Math.abs(spread) > ci,
      score_a: round(mean(cols.score_a), 1), score_b: round(mean(cols.score_b), 1), turns: round(mean(cols.turns), 1),
      a_first: side(true), a_second: side(false),
      bingos_a: round(mean(cols.bingos_a), 2), bingos_b: round(mean(cols.bingos_b), 2),
      exchanges_a: round(mean(cols.exchanges_a), 2), exchanges_b: round(mean(cols.exchanges_b), 2),
      passes_a: round(mean(cols.passes_a), 3), passes_b: round(mean(cols.passes_b), 3),
      ended_by_going_out: round(mean(cols.ended_by.map((e) => (e === "out" ? 1 : 0))), 4),
    };
  }

  function matchSummary(match) {
    const done = gamesDone(match);
    const stats = matchSummaryStats(sliceCols(match.cols, done - (done % 2)));
    return {
      id: match.id, a: match.a, b: match.b, a_name: match.a_name, b_name: match.b_name, pairs: match.pairs,
      status: match.status, progress: round(done / (match.pairs * 2), 4), games_done: done, created: match.created,
      seconds: match.status === "running" ? round(Date.now() / 1000 - match.startedAt, 1) : match.seconds,
      spread: stats ? stats.spread : null, spread_ci: stats ? stats.spread_ci : null, win_rate: stats ? stats.win_rate : null,
    };
  }

  function highlights(match, cols) {
    const n = cols.score_a.length;
    if (!n) return [];
    const spreads = cols.score_a.map((s, i) => s - cols.score_b[i]);
    let top = 0, topSide = "a", topScore = -1, bigA = 0, bigB = 0, close = 0;
    for (let i = 0; i < n; i++) {
      if (cols.best_play_a[i] > topScore) { topScore = cols.best_play_a[i]; top = i; topSide = "a"; }
      if (cols.best_play_b[i] > topScore) { topScore = cols.best_play_b[i]; top = i; topSide = "b"; }
      if (spreads[i] > spreads[bigA]) bigA = i;
      if (spreads[i] < spreads[bigB]) bigB = i;
      if (Math.abs(spreads[i]) < Math.abs(spreads[close])) close = i;
    }
    const out = [{ title: "Highest-scoring play",
      detail: `${topSide === "a" ? cols.best_word_a[top] : cols.best_word_b[top]} for ${topScore} by ${topSide === "a" ? match.a_name : match.b_name}`, game: top }];
    if (spreads[bigA] > 0) out.push({ title: `Biggest win for ${match.a_name}`, detail: `${match.a_name} won ${cols.score_a[bigA]}–${cols.score_b[bigA]}`, game: bigA });
    if (spreads[bigB] < 0) out.push({ title: `Biggest win for ${match.b_name}`, detail: `${match.b_name} won ${cols.score_b[bigB]}–${cols.score_a[bigB]}`, game: bigB });
    out.push({ title: "Closest game", game: close,
      detail: spreads[close] === 0 ? `Tied at ${cols.score_a[close]}` : `${spreads[close] > 0 ? match.a_name : match.b_name} won ` +
        `${Math.max(cols.score_a[close], cols.score_b[close])}–${Math.min(cols.score_a[close], cols.score_b[close])}` });
    return out;
  }

  function matchDetail(match) {
    const done = gamesDone(match);
    const cols = sliceCols(match.cols, done);
    const stats = matchSummaryStats(sliceCols(match.cols, done - (done % 2)));
    return { match: matchSummary(match), summary: stats, games: cols, highlights: highlights(match, cols) };
  }

  function buildMatches() {
    if (EMPTY) return;
    // Fixed ids so screenshots and links are repeatable; the times are relative to now.
    newMatch({ id: "m_20261112_064512", a: "model:night2/best.pt", b: "bot:greedy-leave", pairs: 500, created: NOW - 6.2 * HOUR, seed: 57,
      status: "finished", seconds: 21.4 });
    newMatch({ id: "m_20261112_173005", a: "model:night3/best.pt", b: "bot:greedy", pairs: 2000, created: NOW - 20, seed: 12,
      status: "running", rate: 46, done: 920 });
    newMatch({ id: "m_20261112_102240", a: "model:night3/best.pt", b: "model:night2/best.pt", pairs: 250, created: NOW - 2.4 * HOUR, seed: 13,
      status: "finished", seconds: 13.1 });
    newMatch({ id: "m_20261112_061508", a: "model:night2/best.pt", b: "bot:greedy", pairs: 500, created: NOW - 6.6 * HOUR, seed: 14,
      status: "cancelled", done: 374, seconds: 8.2 });
    newMatch({ id: "m_20261111_183320", a: "bot:greedy-leave", b: "bot:greedy", pairs: 2000, created: NOW - 26 * HOUR, seed: 15,
      status: "finished", seconds: 9.6 });
    matches.sort((x, y) => y.created - x.created);
  }

  // ---- Fallback registration for endpoints owned by another mock file ------------------------

  /** Register `handler` for `regex` only for paths no other mock file answers. App loads this
   * file first, so instead of checking once now, the pattern asks App.mock.has at call time and
   * declines when someone else has registered the same path. Its own `test` says no, so it never
   * hides the path from that check. */
  function fallback(method, regex, handler) {
    let checking = false;
    const pattern = {
      test() { return false; },
      [Symbol.match](path) {
        if (checking) return null;
        checking = true;
        try {
          if (App.mock.has(method, path)) return null;
        } finally {
          checking = false;
        }
        return regex.exec(path);
      },
    };
    App.mock.on(method, pattern, handler);
  }

  // ---- Handlers -------------------------------------------------------------------------

  buildRuns();
  buildMatches();

  const findRun = (name) => runs.find((r) => r.spec.name === decodeURIComponent(name));
  const findMatch = (id) => matches.find((m) => m.id === decodeURIComponent(id));
  const noRun = (name) => ({ error: `No run called ${decodeURIComponent(name)}.` });

  App.mock.on("GET", /^\/api\/status$/, () => {
    tick();
    const active = runs.filter((r) => r.status === "running" || r.status === "stopping");
    return {
      version: "0.1.0",
      lexicon: { id: "NSWL2023-91a67230e1f6", words: 195747 },
      leaves: { ok: true, file: "NSWL2023-91a67230e1f6.leaves" },
      gpu: { name: "NVIDIA GeForce RTX 5070 Ti", utilization: active.length ? 84 + Math.round(6 * Math.random()) : 3,
        memory_used_mb: active.length ? 2140 : 610, memory_total_mb: 16303, temperature_c: active.length ? 61 : 38 },
      active_runs: active.map((r) => r.spec.name),
      queued_runs: runs.filter((r) => r.status === "queued").map((r) => r.spec.name),
      active_matches: matches.filter((m) => gamesDone(m) >= 0 && m.status === "running").map((m) => m.id),
    };
  });

  App.mock.on("GET", /^\/api\/runs$/, () => {
    tick();
    return { runs: runs.slice().sort((a, b) => b.spec.created - a.spec.created).map(summary) };
  });

  App.mock.on("POST", /^\/api\/runs$/, (match, body) => {
    const list = (body && body.runs) || [];
    if (!list.length) return { error: "No runs given." };
    for (const item of list) {
      if (!/^[A-Za-z0-9_-]+$/.test(item.name || "")) return { error: `"${item.name}" is not a valid run name: use letters, digits, - and _.` };
      if (findRun(item.name)) return { error: `A run called ${item.name} already exists.` };
    }
    const started = [];
    const queued = [];
    const busy = runs.some((r) => r.status === "running" || r.status === "stopping");
    list.forEach((item, i) => {
      const overrides = Object.assign({}, item.overrides || {});
      const run = newRun({ name: item.name, seed: 5000 + runs.length, created: Date.now() / 1000 + i, limitHours: Number(overrides["run.hours"]) || 0,
        gps: 55, tau: 2.4, gl: 6.5, gl0: -0.5, changed: overrides, status: "queued" });
      run.live = true;
      if (body.mode === "parallel" || (i === 0 && !busy)) {
        startSession(run);
        started.push(item.name);
      } else queued.push(item.name);
    });
    return { started, queued };
  });

  App.mock.on("GET", /^\/api\/runs\/([^/]+)$/, (match) => {
    tick();
    const run = findRun(match[1]);
    return run ? detail(run) : noRun(match[1]);
  });

  App.mock.on("GET", /^\/api\/runs\/([^/]+)\/metrics$/, (match, body, query) => {
    tick();
    const run = findRun(match[1]);
    if (!run) return noRun(match[1]);
    const since = Math.max(0, parseInt(query.since || "0", 10) || 0);
    return { records: run.records.slice(since), next: run.records.length };
  });

  App.mock.on("GET", /^\/api\/runs\/([^/]+)\/log$/, (match, body, query) => {
    tick();
    const run = findRun(match[1]);
    if (!run) return noRun(match[1]);
    const tail = Math.max(1, parseInt(query.tail || "200", 10) || 200);
    return { lines: logLines(run).slice(-tail) };
  });

  App.mock.on("GET", /^\/api\/runs\/([^/]+)\/report$/, (match) => {
    tick();
    const run = findRun(match[1]);
    return run ? report(run) : noRun(match[1]);
  });

  App.mock.on("POST", /^\/api\/runs\/([^/]+)\/stop$/, (match, body) => {
    const run = findRun(match[1]);
    if (!run) return noRun(match[1]);
    if (run.status === "queued") {
      run.status = "stopped";
      return { ok: true };
    }
    if (run.status !== "running" && run.status !== "stopping") return { error: `${run.spec.name} is not running.` };
    if (body && body.force) {
      stopSession(run, "killed from the app");
      run.status = "stopped";
      startNextQueued();
    } else run.status = "stopping";
    return { ok: true };
  });

  App.mock.on("POST", /^\/api\/runs\/([^/]+)\/resume$/, (match, body) => {
    const run = findRun(match[1]);
    if (!run) return noRun(match[1]);
    if (run.status === "running" || run.status === "stopping") return { error: `${run.spec.name} is already running.` };
    if (!run.sim.checkpoints.length) return { error: `${run.spec.name} has no checkpoint to resume from.` };
    const overrides = (body && body.overrides) || {};
    for (const [key, value] of Object.entries(overrides)) {
      const [section, field] = key.split(".");
      if (!DEFAULTS[section] || !(field in DEFAULTS[section])) return { error: `Unknown setting ${key}.` };
      if (section === "model") return { error: `${key} cannot be changed when resuming.` };
    }
    for (const [key, value] of Object.entries(overrides)) {
      const [section, field] = key.split(".");
      run.config[section][field] = value;
      run.spec.changed[key] = value;
    }
    run.limitHours = Number(body && body.hours) || 0;
    run.maxGames = Number(body && body.max_games) || 0;
    if (run.limitHours) { run.config.run.hours = run.limitHours; run.spec.changed["run.hours"] = run.limitHours; }
    run.error = null;
    run.live = true;
    const snap = run.sim.snapshot;
    if (snap) Object.assign(run.sim, { h: snap.h, games: snap.games, step: snap.step, positions: snap.positions, replay: snap.replay });
    run.sim.lastEval = run.sim.h;
    run.sim.lastCkpt = run.sim.h;
    startSession(run);
    return summary(run);
  });

  App.mock.on("DELETE", /^\/api\/runs\/([^/]+)$/, (match) => {
    const run = findRun(match[1]);
    if (!run) return noRun(match[1]);
    if (run.status === "running" || run.status === "stopping") return { error: `Stop ${run.spec.name} before deleting it.` };
    runs.splice(runs.indexOf(run), 1);
    return { ok: true };
  });

  App.mock.on("GET", /^\/api\/matches$/, () => ({ matches: matches.map(matchSummary) }));

  App.mock.on("GET", /^\/api\/matches\/([^/]+)$/, (match) => {
    const m = findMatch(match[1]);
    return m ? matchDetail(m) : { error: `No match ${decodeURIComponent(match[1])}.` };
  });

  App.mock.on("POST", /^\/api\/matches$/, (match, body) => {
    const a = body && body.a;
    const b = body && body.b;
    const pairs = Math.round(Number(body && body.pairs) || 0);
    if (!a || !b) return { error: "Choose both players." };
    if (pairs < 2 || pairs > 100000) return { error: "Seed pairs must be between 2 and 100,000." };
    const bots = a.startsWith("bot:") && b.startsWith("bot:");
    const m = newMatch({ a, b, pairs, created: Date.now() / 1000, seed: 100 + matches.length, status: "running",
      rate: bots ? 420 : 46, done: 0, candidates: body.candidates });
    matches.sort((x, y) => y.created - x.created);
    return { id: m.id };
  });

  App.mock.on("POST", /^\/api\/matches\/([^/]+)\/cancel$/, (match) => {
    const m = findMatch(match[1]);
    if (!m) return { error: "No such match." };
    if (m.status === "running") {
      m.fixedDone = gamesDone(m);
      m.fixedDone -= m.fixedDone % 2;
      m.seconds = round(Date.now() / 1000 - m.startedAt, 1);
      m.status = "cancelled";
    }
    return { ok: true };
  });

  fallback("GET", /^\/api\/models$/, () => modelList());
})();
