/* Run page: #/run/<name>[?tab=report|settings|log|checkpoints].
 *
 * One training run: header with actions (Rename among them) and its verdict, a row of stat
 * tiles, then tabs. Progress opens with a verdict banner (the report's headline, and a warning
 * with a way to the tuning guide when the run fell behind or got worse), then the strength chart
 * and three groups of small charts: "Is it learning?", "Training" and "Self-play games", each
 * with a line on how to read it. It is live while the run is running (metrics are fetched with
 * ?since= every 5 s and appended); Report is the server's analysis with a clickable event
 * timeline; Settings, Log and Checkpoints are what they say. Not in the sidebar: the Training
 * page links here. Fields an older server does not send (health, the seen error, guard records)
 * are simply left out.
 *
 * A run that trains from saved games (run.mode "games", docs/APP_API.md) has the same tabs in
 * its own terms: it is judged by its win rate against the static evaluator (50% is level), its
 * charts are drawn against steps or hours, and nothing of self-play (temperature, the gate,
 * self-play games) shows. Its report is the server's games report.
 */
(function () {
  "use strict";

  const { h, fmt } = App;
  const K = window.LabKit;

  const STATUS = {
    running: ["Running", "good"], stopping: ["Stopping", "warning"], queued: ["Queued", "info"],
    finished: ["Finished", "plain"], stopped: ["Stopped", "plain"], failed: ["Failed", "critical"],
  };
  const PHASE = { warmstart: "warm start", pretraining: "pretraining", selfplay: "self-play" };
  const TABS = [["progress", "Progress"], ["report", "Report"], ["settings", "Settings"], ["log", "Log"], ["checkpoints", "Checkpoints"]];
  const SECTION_TITLES = { run: "Run", model: "Network", selfplay: "Self-play", warmstart: "Warm start", train: "Training" };

  const ICON = {
    back: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3L5 8l5 5"/></svg>',
    stop: '<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><rect x="3.5" y="3.5" width="9" height="9" rx="1.5"/></svg>',
    play: '<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M5 3.2v9.6c0 .5.5.8.9.5l7.2-4.8a.6.6 0 000-1L5.9 2.7c-.4-.3-.9 0-.9.5z"/></svg>',
    trash: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M3 4.5h10M6.5 4.5V3h3v1.5M4.5 4.5l.6 8.5h5.8l.6-8.5"/></svg>',
    good: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.5"/><path d="M6.6 10.2l2.3 2.3 4.5-4.8"/></svg>',
    warning: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3l7.5 13.5h-15z"/><path d="M10 8.5v3.5M10 14.6v.1"/></svg>',
    info: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="10" cy="10" r="7.5"/><path d="M10 9v5M10 6.3v.1"/></svg>',
    critical: '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="10" cy="10" r="7.5"/><path d="M7.3 7.3l5.4 5.4M12.7 7.3l-5.4 5.4"/></svg>',
  };
  const icon = (svg, cls) => h("span", { class: "dash-icon" + (cls ? " " + cls : ""), html: svg, "aria-hidden": "true" });

  const X_MODES = {
    games: { label: "games", of: (r) => r.games, format: (v) => fmt.int(v), tooltipFormat: (v) => fmt.int(v) + " games" },
    hours: { label: "hours trained", of: (r) => r.hours, format: (v) => fmt.duration(v), tooltipFormat: (v) => fmt.duration(v) + " trained",
      tickFormat: (v) => (Number.isInteger(v) ? v : v.toFixed(1)) + " h" },
    steps: { label: "training steps", of: (r) => r.step, format: (v) => fmt.int(v), tooltipFormat: (v) => "Step " + fmt.int(v) },
  };

  const isLive = (run) => run && (run.status === "running" || run.status === "stopping");
  /** The run trains from saved games (docs/APP_API.md): judged by its win rate, not its spread. */
  const gamesMode = (state) => !!(state.detail && K.isGames(state.detail.run));
  /** "8 h" rather than "8 h 00 m" for whole hours. */
  const hoursText = (hours) => fmt.duration(hours).replace(/ 00 m$/, "");
  const enc = encodeURIComponent;
  const spreadTick = (v) => (v === 0 ? "0" : fmt.signed(v, Number.isInteger(v) ? 0 : 1));

  function spreadValue(spread, ci) {
    if (spread === null || spread === undefined) return "–";
    return h("span", null, fmt.signed(spread), ci === null || ci === undefined ? null : h("span", { class: "stat-ci" }, " ± " + Number(ci).toFixed(1)));
  }

  function significance(spread, ci) {
    if (spread === null || spread === undefined || ci === null || ci === undefined) return null;
    if (spread - ci > 0) return "significant";
    if (spread + ci < 0) return "significantly behind";
    return "not yet significant";
  }

  App.registerView("run", { title: "Run", nav: false, order: 25, mount });

  function mount(container, route) {
    const name = route.args[0];
    if (!name) {
      App.go("#/training");
      return;
    }
    const tab = TABS.some(([key]) => key === route.query.tab) ? route.query.tab : "progress";
    const state = {
      name, tab, detail: null, records: [], next: 0, xmode: "games", loaded: false, schema: undefined,
      charts: {}, tabStops: [], report: null, reportAt: 0, lastFetch: 0, missing: false,
    };
    document.title = name + " · Scrabble Lab";

    const els = {
      head: h("div"),
      notices: h("div", { class: "stack-tight run-notices" }),
      tiles: h("div", { class: "run-tiles" }),
      tabs: h("div", { class: "tabs", role: "tablist", "aria-label": "Run sections" }),
      panel: h("div", { class: "run-panel", role: "tabpanel" }),
    };
    state.els = els;
    container.appendChild(h("div", { class: "run-page" }, els.head, els.notices, els.tiles, els.tabs, els.panel));

    App.every(5000, () => poll(state, els));
    App.onCleanup(() => stopTabTimers(state));
  }

  // ---- Data -----------------------------------------------------------------------------

  async function poll(state, els) {
    if (state.missing) return;
    let detail;
    try {
      detail = await App.get(`/api/runs/${enc(state.name)}`);
    } catch (error) {
      if (!state.detail) {
        state.missing = true;
        showMissing(state, els, error);
      }
      return;
    }
    const before = state.detail ? state.detail.run.status : null;
    state.detail = detail;
    // A run trained from games counts its positions as "games": its charts go by steps.
    if (!state.loaded && K.isGames(detail.run) && state.xmode === "games") state.xmode = "steps";
    const statusChanged = before !== null && before !== detail.run.status;
    let fresh = false;
    if (!state.loaded || isLive(detail.run) || statusChanged) fresh = await loadMetrics(state);
    state.lastFetch = Date.now();
    renderHead(state, els);
    renderNotices(state, els);
    renderTiles(state, els);
    if (!state.loaded) {
      state.loaded = true;
      renderTabs(state, els);
      renderPanel(state, els);
      return;
    }
    if (statusChanged) {
      renderTabs(state, els);
      renderPanel(state, els);
      return;
    }
    if (state.tab === "progress") {
      if (fresh) updateProgress(state);
      updateLiveNote(state);
      if (verdictStale(state)) loadVerdict(state);
    } else if (state.tab === "report" && isLive(detail.run) && Date.now() - state.reportAt > 15000) {
      refreshReport(state, els);
    } else if (state.tab === "checkpoints" && fresh) {
      renderPanel(state, els);
    }
    const count = (detail.checkpoints || []).length + (detail.best ? 1 : 0);
    if (count !== state.tabCount) renderTabs(state, els);
  }

  async function loadMetrics(state) {
    const res = await App.get(`/api/runs/${enc(state.name)}/metrics?since=${state.next}`);
    if (res.next < state.next) {
      state.records = [];
      state.next = 0;
      return loadMetrics(state);
    }
    for (const record of res.records) state.records.push(record);
    state.next = res.next;
    return res.records.length > 0;
  }

  async function loadSchema(state) {
    if (state.schema !== undefined) return state.schema;
    try {
      state.schema = await App.get("/api/config/schema");
    } catch (error) {
      state.schema = null;
    }
    return state.schema;
  }

  function schemaField(schema, key) {
    if (!schema) return null;
    for (const section of schema.sections || []) for (const field of section.fields || []) if (field.key === key) return field;
    return null;
  }

  /** Spread of an eval record, accepting the older key layout ("spread_vs_greedy") too. */
  function evalSpread(r) {
    const spread = r.spread !== undefined ? r.spread : r["spread_vs_" + r.opponent];
    const ci = r.spread_ci !== undefined ? r.spread_ci : r["spread_ci_vs_" + r.opponent];
    return [spread, ci];
  }

  /** Every series the Progress tab draws, against the chosen x. */
  function buildSeries(records, mode) {
    const X = X_MODES[mode];
    const out = {
      gl: [], glBand: [], g: [], gBand: [], markers: [], sp: {}, tr: {}, gap: [], evalCount: 0, spCount: 0,
      gate: [], gateBand: [], gateMarkers: [], winGl: [], winGlBand: [], winG: [], winGBand: [], scoreFor: [], scoreAgainst: [],
      guardMarkers: [], seenCount: 0, lrCount: 0,
    };
    const spKeys = ["games_per_second", "temperature", "mean_score", "mean_turns", "bingos_per_game", "exchanges_per_game",
      "passes_per_game", "win_brier", "replay_positions", "value_rmse_points", "leave_only_rmse_points",
      "first_player_win_rate", "mean_abs_spread"];
    const trKeys = ["value_loss", "win_loss", "value_rmse_points", "learning_rate"];
    for (const k of spKeys) out.sp[k] = [];
    for (const k of trKeys) out.tr[k] = [];
    const lines = () => [out.gl, out.g, out.gap, out.gate, out.winGl, out.winG, out.scoreFor, out.scoreAgainst,
      ...Object.values(out.sp), ...Object.values(out.tr)];
    const bands = () => [out.glBand, out.gBand, out.gateBand, out.winGlBand, out.winGBand];
    const num = (v) => typeof v === "number" && Number.isFinite(v);
    let started = false;
    for (const r of records) {
      const x = X.of(r);
      if (r.kind === "session") {
        if (r.event === "start" && started) {
          for (const list of lines()) list.push([x, null]);
          for (const list of bands()) list.push([x, null, null]);
          out.markers.push({ x, label: "Resumed", tone: "info", detail: `A new session started from the checkpoint at step ${fmt.int(r.step)}.` });
        }
        continue;
      }
      started = true;
      if (r.kind === "eval") {
        const [spread, ci] = evalSpread(r);
        if (typeof spread !== "number") continue;
        const half = typeof ci === "number" ? ci : 0;
        const win = num(r.win_rate) ? r.win_rate : null;
        const winHalf = num(r.win_ci) ? r.win_ci : null;
        if (r.opponent === "greedy-leave") {
          out.gl.push([x, spread]);
          out.glBand.push([x, spread - half, spread + half]);
          out.evalCount++;
          if (win !== null) {
            out.winGl.push([x, win]);
            if (winHalf !== null) out.winGlBand.push([x, win - winHalf, win + winHalf]);
          }
          if (num(r.score_for)) out.scoreFor.push([x, r.score_for]);
          if (num(r.score_against)) out.scoreAgainst.push([x, r.score_against]);
        } else if (r.opponent === "greedy") {
          out.g.push([x, spread]);
          out.gBand.push([x, spread - half, spread + half]);
          if (win !== null) {
            out.winG.push([x, win]);
            if (winHalf !== null) out.winGBand.push([x, win - winHalf, win + winHalf]);
          }
        }
      } else if (r.kind === "gate") {
        const first = r.pairs === 0 && typeof r.spread !== "number";
        if (r.promoted) {
          out.markers.push({ x, label: first ? "First network to beat" : "New best network", tone: "good", group: "New best network",
            detail: first ? `Step ${fmt.int(r.step)}, the first network evaluated, became the one later networks have to beat.`
              : `Step ${fmt.int(r.step)} was promoted to best.pt` + (typeof r.spread === "number"
                ? `, ${fmt.spread(r.spread, r.spread_ci)} against the previous best.` : ".") });
        }
        // The first gate has no match (pairs 0): the first network becomes the best without one.
        if (!first && r.pairs !== 0 && num(r.spread)) {
          out.gate.push([x, r.spread]);
          if (num(r.spread_ci)) out.gateBand.push([x, r.spread - r.spread_ci, r.spread + r.spread_ci]);
          if (r.promoted) {
            out.gateMarkers.push({ x, label: "New best network", tone: "good",
              detail: `Step ${fmt.int(r.step)} won its gate, ${fmt.spread(r.spread, r.spread_ci)}, and became the best network.` });
          }
        }
      } else if (r.kind === "guard") {
        const scale = num(r.learning_rate_scale) ? r.learning_rate_scale : null;
        const marker = { x, label: "Back to best", group: "Back to the best network", tone: "warning", emphasis: true,
          detail: `The trainer went back to the best network` + (num(r.to_step) ? ` (step ${fmt.int(r.to_step)})` : "") +
            (scale !== null ? ` and lowered the learning rate to ${lrText(scale)} times its setting` : "") +
            (r.reason ? `: ${fmt.prose(r.reason)}.` : ".") };
        out.markers.push(marker);
        out.guardMarkers.push(marker);
      } else if (r.kind === "selfplay") {
        out.spCount++;
        for (const k of spKeys) if (typeof r[k] === "number") out.sp[k].push([x, r[k]]);
        if (typeof r.value_rmse_points === "number" && typeof r.leave_only_rmse_points === "number") {
          out.gap.push([x, r.leave_only_rmse_points - r.value_rmse_points]);
        }
      } else if (r.kind === "train") {
        if (mode === "games" && !r.games) continue;
        for (const k of trKeys) if (typeof r[k] === "number") out.tr[k].push([x, r[k]]);
        if (num(r.value_rmse_points)) out.seenCount++;
        if (num(r.learning_rate)) out.lrCount++;
      }
    }
    return out;
  }

  /** A share as a tick: "50%", "52.5%" (never a rounded "53%" beside an exact "55%"). */
  function pctTick(v) {
    const p = 100 * v;
    return (Math.abs(p - Math.round(p)) < 1e-6 ? String(Math.round(p)) : p.toFixed(1)) + "%";
  }

  /** A learning rate (or a multiple of one) without float noise: 0.001, 0.0005, 0.00025. */
  function lrText(v) {
    if (typeof v !== "number" || !Number.isFinite(v)) return "–";
    if (v === 0) return "0";
    return String(Number(v.toPrecision(3)));
  }

  // ---- Header, notices, tiles -----------------------------------------------------------

  function showMissing(state, els, error) {
    els.head.replaceChildren(h("a", { class: "dash-back", href: "#/training" }, icon(ICON.back), "Back to training"));
    els.notices.replaceChildren(h("div", { class: "notice notice-critical" }, icon(ICON.critical, "dash-notice-icon dash-tone-critical"),
      h("div", null, h("strong", null, `There is no run called ${state.name}. `), "It may have been deleted. ",
        h("span", { class: "muted" }, String(error.message || error)))));
  }

  function badge(run) {
    const [label, tone] = STATUS[run.status] || [run.status, "plain"];
    // Phases are self-play's (warm start, pretraining, self-play); a run trained from games has none.
    const text = run.status === "running" && run.phase && !K.isGames(run) ? `${label} · ${PHASE[run.phase] || run.phase}` : label;
    return h("span", { class: "badge badge-" + tone }, text);
  }

  /** The sidebar marks Training for every run page; a run trained from games belongs to Train from games. */
  function markSidebar(games) {
    for (const item of document.querySelectorAll("#sidebar .nav-item")) {
      const href = item.getAttribute("href");
      if (href !== "#/training" && href !== "#/games") continue;
      const on = href === (games ? "#/games" : "#/training");
      item.classList.toggle("active", on);
      if (on) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    }
  }

  function renderHead(state, els) {
    const run = state.detail.run;
    const games = K.isGames(run);
    markSidebar(games);
    const actions = h("div", { class: "page-actions" });
    const blocked = K.renameBlocked(run);
    actions.appendChild(h("span", { class: "tm-tip", title: blocked || `Rename ${run.name}, and with it its models` },
      h("button", { class: "button button-ghost", type: "button", disabled: !!blocked, onclick: () => renameRun(state) },
        K.icon("pencil", 14), "Rename")));
    if (run.status === "running") {
      actions.appendChild(h("button", { class: "button", type: "button", onclick: () => stopRun(state, els, false) }, icon(ICON.stop), "Stop"));
    } else if (run.status === "queued") {
      actions.appendChild(h("button", { class: "button", type: "button", onclick: () => stopRun(state, els, false) }, "Remove from queue"));
    } else if (run.status === "stopping") {
      actions.appendChild(h("button", { class: "button", type: "button", onclick: () => stopRun(state, els, true) }, icon(ICON.stop), "Force stop"));
    }
    if (["stopped", "finished", "failed"].includes(run.status)) {
      actions.appendChild(h("button", { class: "button button-primary", type: "button", disabled: !run.checkpoints,
        title: run.checkpoints ? "Continue from the newest checkpoint" : "This run has no checkpoint to resume from",
        onclick: () => openResume(state, els) }, icon(ICON.play), "Resume"));
    }
    if (!isLive(run)) {
      actions.appendChild(h("button", { class: "button button-ghost", type: "button", onclick: () => deleteRun(state) },
        icon(ICON.trash), "Delete"));
    }
    const params = run.parameters ? ` · ${fmt.int(run.parameters)} parameters` : "";
    const from = games && run.game_set ? ` · trained from the game set ${run.game_set}` : "";
    els.head.replaceChildren(h("div", { class: "page-head run-head" },
      h("div", null,
        games ? h("a", { class: "dash-back", href: "#/games" }, icon(ICON.back), "Back to training from games")
          : h("a", { class: "dash-back", href: "#/training" }, icon(ICON.back), "Back to training"),
        h("div", { class: "run-title" }, h("h1", null, run.name), badge(run), K.healthChip(run.health, null, games)),
        h("p", { class: "lede" }, `${K.networkLabel(run)} network${params}${from} · ` + (run.status === "queued"
          ? `queued ${fmt.datetime(run.created)}`
          : `started ${fmt.datetime(run.created)} · ` + (isLive(run) ? "live" : `last update ${fmt.ago(run.updated)}`)))),
      actions));
  }

  /** The tail of a failed run's console, scrolled to its end, where the reason is. */
  function errorLog(text) {
    const pre = h("pre", { class: "log dash-notice-log" }, text);
    requestAnimationFrame(() => { pre.scrollTop = pre.scrollHeight; });
    return pre;
  }

  function renderNotices(state, els) {
    const run = state.detail.run;
    const nodes = [];
    if (run.status === "failed") {
      nodes.push(h("div", { class: "notice notice-critical" }, icon(ICON.critical, "dash-notice-icon dash-tone-critical"),
        h("div", { class: "dash-notice-main" }, h("strong", null, "This run failed. "),
          run.checkpoints ? "Its checkpoints are intact: change the setting that caused it and resume." : "It stopped before saving a checkpoint.",
          run.error ? errorLog(run.error) : null)));
    } else if (run.status === "queued") {
      const ahead = (App.status && App.status.active_runs) || [];
      nodes.push(h("div", { class: "notice" }, icon(ICON.info, "dash-notice-icon dash-tone-info"),
        h("div", null, h("strong", null, "Queued. "), ahead.length
          ? `It starts on its own when ${ahead.join(" and ")} ${ahead.length > 1 ? "finish" : "finishes"}.`
          : "It starts on its own when the run ahead of it finishes.")));
    } else if (run.status === "stopping") {
      nodes.push(h("div", { class: "notice notice-warning" }, icon(ICON.warning, "dash-notice-icon dash-tone-warning"),
        h("div", null, h("strong", null, "Stopping. "), gamesMode(state)
          ? "The run is finishing its current step; then it checks its held-out games, plays a match against the static evaluator and saves a checkpoint."
          : "The run is finishing its current step; then it evaluates and saves a checkpoint.")));
    } else if (run.status === "running" && gamesMode(state)) {
      if (!state.records.some((r) => r.kind === "train")) {
        nodes.push(h("div", { class: "notice" }, icon(ICON.info, "dash-notice-icon dash-tone-info"), h("div", null, h("strong", null, "Starting. "),
          `The run is opening the game set ${run.game_set || ""}. The charts begin with the first training report, about a minute in.`)));
      }
    } else if (run.status === "running" && !state.records.some((r) => r.kind === "selfplay")) {
      const cfg = state.detail.config || {};
      const text = run.phase === "pretraining"
        ? `Pretraining on the warm-start games (${fmt.int((cfg.warmstart || {}).steps || 3000)} steps, about 2 minutes). Self-play and most charts start after that.`
        : `Warm start: the static evaluator is playing ${fmt.int((cfg.warmstart || {}).games || 100000)} games to fill the replay buffer, about 3 minutes. The charts start once self-play does.`;
      nodes.push(h("div", { class: "notice" }, icon(ICON.info, "dash-notice-icon dash-tone-info"), h("div", null, h("strong", null, "Starting. "), text)));
    }
    els.notices.replaceChildren(...nodes);
    els.notices.hidden = !nodes.length;
  }

  /** A win rate as a tile's value: "53.4%" and, smaller, "± 1.3". */
  function winValue(rate, ci) {
    if (typeof rate !== "number") return "–";
    return h("span", null, fmt.pct(rate), typeof ci === "number" ? h("span", { class: "stat-ci" }, " ± " + (100 * ci).toFixed(1)) : null);
  }

  function renderTilesGames(state, els) {
    const run = state.detail.run;
    const le = run.latest_eval;
    const e = le && le["greedy-leave"];
    const best = run.best_eval;
    const starts = state.records.filter((r) => r.kind === "session" && r.event === "start");
    const batch = (starts.length && starts[starts.length - 1].batch) || (((state.detail.config || {}).offline || {}).batch_size);
    const timeHint = run.limit && run.limit.hours
      ? `limit ${hoursText(run.limit.hours)} this session` : "no time limit: until the passes are done";
    const rate = run.positions_per_second;
    const passes = typeof run.passes === "number" ? String(Number(run.passes.toPrecision(4))) : null;
    els.tiles.replaceChildren(
      Charts.statTile({ label: "Positions trained", value: fmt.compact(run.positions),
        hint: `one from each game · ${fmt.int(run.step)} steps` + (batch ? ` of ${fmt.int(batch)}` : "") }),
      Charts.statTile({ label: "Positions per second", value: typeof rate === "number" ? fmt.int(rate) : "–",
        hint: isLive(run) ? "latest minute" : typeof rate === "number" ? "last measured" : "not measured yet" }),
      Charts.statTile({ label: "Time trained", value: fmt.duration(run.hours), hint: timeHint }),
      Charts.statTile({ label: "Pass", value: typeof run.pass === "number" ? `${fmt.fixed(run.pass, 1)}${passes ? " of " + passes : ""}` : "–",
        hint: run.total_steps ? `step ${fmt.int(run.step)} of ${fmt.int(run.total_steps)}` : "not started yet" }),
      Charts.statTile({ label: "Against the static evaluator", value: e ? winValue(e.win_rate, e.win_ci) : "–",
        hint: e ? `${K.winVerdict(e.win_rate, e.win_ci)}` : "no match yet" }),
      Charts.statTile({ label: "Best network", value: run.has_best && best ? "step " + fmt.int(best.step) : run.has_best ? "step " + fmt.int(run.best_step) : "none yet",
        hint: best ? `won ${K.winText(best.win_rate, best.win_ci)} of games` : "the first match sets one" }),
    );
  }

  function renderTiles(state, els) {
    if (gamesMode(state)) { renderTilesGames(state, els); return; }
    const run = state.detail.run;
    const le = run.latest_eval;
    const gl = le && le["greedy-leave"];
    // Gate matches won. The first evaluated network becomes the best without one.
    const promotions = state.records.filter((r) => r.kind === "gate" && r.promoted && !(r.pairs === 0 && typeof r.spread !== "number")).length;
    const bestEval = run.best_step === null || run.best_step === undefined ? null
      : state.records.filter((r) => r.kind === "eval" && r.opponent === "greedy-leave" && r.step <= run.best_step).pop();
    const [bestSpread, bestCi] = bestEval ? evalSpread(bestEval) : [null, null];
    const batch = ((state.detail.config || {}).train || {}).batch_size;
    let timeHint = "no time limit";
    if (run.limit && run.limit.hours) {
      timeHint = `limit ${hoursText(run.limit.hours)} this session` +(run.progress !== null && run.progress !== undefined && isLive(run)
        ? ` · ${Math.round(run.progress * 100)}%` : "");
    } else if (run.limit && run.limit.max_games) timeHint = `limit ${fmt.compact(run.limit.max_games)} games`;
    els.tiles.replaceChildren(
      Charts.statTile({ label: "Games", value: fmt.compact(run.games), hint: `${fmt.compact(run.positions)} positions` }),
      Charts.statTile({ label: "Games per second", value: run.games_per_second ? fmt.fixed(run.games_per_second, 1) : "–",
        hint: isLive(run) ? "latest minute" : run.games_per_second ? "last measured" : "not measured yet" }),
      Charts.statTile({ label: "Time trained", value: fmt.duration(run.hours), hint: timeHint }),
      Charts.statTile({ label: "Training steps", value: fmt.int(run.step), hint: batch ? `batches of ${fmt.int(batch)}` : null }),
      Charts.statTile({ label: "Against greedy + leave", value: gl ? spreadValue(gl.spread, gl.ci) : "–",
        hint: gl ? `${significance(gl.spread, gl.ci)} · at ${fmt.compact(le.games)} games` : "no evaluation yet" }),
      Charts.statTile({ label: "Best network", value: run.has_best ? "step " + fmt.int(run.best_step) : "none yet",
        hint: run.has_best ? (bestSpread !== null ? `${fmt.spread(bestSpread, bestCi)} · ` : "") +
          `${promotions} promotion${promotions === 1 ? "" : "s"}` : "the first evaluation sets one" }),
    );
  }

  // ---- Tabs -----------------------------------------------------------------------------

  function renderTabs(state, els) {
    const count = state.detail ? (state.detail.checkpoints || []).length + (state.detail.best ? 1 : 0) : 0;
    state.tabCount = count;
    els.tabs.replaceChildren(...TABS.map(([key, label]) => h("button", {
      class: "tab", type: "button", role: "tab", "aria-selected": String(state.tab === key), id: "run-tab-" + key,
      onclick: () => selectTab(state, els, key),
      onkeydown: (event) => {
        const i = TABS.findIndex(([k]) => k === key);
        let j = null;
        if (event.key === "ArrowRight") j = (i + 1) % TABS.length;
        if (event.key === "ArrowLeft") j = (i - 1 + TABS.length) % TABS.length;
        if (j === null) return;
        event.preventDefault();
        selectTab(state, els, TABS[j][0]);
        document.getElementById("run-tab-" + TABS[j][0]).focus();
      },
    }, label, key === "checkpoints" && count ? h("span", { class: "run-tab-count" }, String(count)) : null)));
  }

  function selectTab(state, els, key) {
    if (state.tab === key) return;
    state.tab = key;
    const hash = `#/run/${enc(state.name)}` + (key === "progress" ? "" : `?tab=${key}`);
    history.replaceState(null, "", hash);
    renderTabs(state, els);
    renderPanel(state, els);
  }

  function stopTabTimers(state) {
    while (state.tabStops.length) state.tabStops.pop()();
    for (const chart of Object.values(state.charts)) chart.destroy();
    state.charts = {};
  }

  function renderPanel(state, els) {
    stopTabTimers(state);
    const panel = App.clear(els.panel);
    if (state.tab === "progress") renderProgress(state, panel);
    else if (state.tab === "report") renderReport(state, els, panel);
    else if (state.tab === "settings") renderSettings(state, panel);
    else if (state.tab === "log") renderLog(state, panel);
    else if (state.tab === "checkpoints") renderCheckpoints(state, panel);
  }

  // ---- Progress -------------------------------------------------------------------------

  function xAxis(mode) {
    const X = X_MODES[mode];
    return { label: X.label, format: X.format, tooltipFormat: X.tooltipFormat, tickFormat: X.tickFormat || "auto" };
  }

  function strengthEmpty(state) {
    const run = state.detail.run;
    const minutes = (((state.detail.config || {}).train || {}).eval_minutes) || 20;
    if (run.status === "queued") return "Evaluations start once the run is training.";
    if (isLive(run)) return `No evaluation yet. The first one comes ${minutes} minutes into self-play, and one every ${minutes} minutes after that.`;
    return "This run ended before its first evaluation.";
  }

  const SMOOTHING = { off: 0, light: 0.8, heavy: 0.95 };

  /** Exponential moving average, debiased at the start of each piece (as TensorBoard does). */
  function smooth(points, weight) {
    if (!weight) return points;
    const out = [];
    let value = 0;
    let n = 0;
    let prevX = -Infinity;
    for (const p of points) {
      if (typeof p[1] !== "number" || p[0] < prevX) { value = 0; n = 0; }
      prevX = typeof p[1] === "number" ? p[0] : -Infinity;
      if (typeof p[1] !== "number") { out.push(p); continue; }
      value = value * weight + (1 - weight) * p[1];
      n++;
      out.push([p[0], value / (1 - Math.pow(weight, n))]);
    }
    return out;
  }

  function progressOptions(state) {
    const s = buildSeries(state.records, state.xmode);
    const x = xAxis(state.xmode);
    const w = SMOOTHING[state.smoothing] || 0;
    const sm = (points) => smooth(points, w);
    const small = 176;
    const empty = state.detail.run.status === "queued" ? "Waiting in the queue." : "No self-play yet.";
    const opts = {};
    // The two baselines sit 40 points apart, so each gets its own panel (one y-axis each) on a
    // shared x range, rather than squeezing the greedy + leave story into a thin strip.
    const xs = [...s.gl, ...s.g].filter((p) => typeof p[1] === "number").map((p) => p[0]);
    const xShared = xs.length > 1 ? Object.assign({}, x, { min: Math.min(...xs), max: Math.max(...xs) }) : x;
    const spreadY = { label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick };
    const margin = { left: 46, right: 64 };
    opts.strength = {
      title: "Against greedy + leave",
      series: [{ name: "Greedy + leave", points: s.gl, band: s.glBand, color: "var(--series-1)" }],
      x: xShared, y: spreadY, referenceY: [{ y: 0, label: "even" }], markers: s.markers, height: 330, margin,
      empty: strengthEmpty(state),
    };
    opts.strengthGreedy = {
      title: "Against greedy",
      series: [{ name: "Greedy", points: s.g, band: s.gBand, color: "var(--series-2)" }],
      x: xShared, y: spreadY, markers: s.markers, height: 200, margin, legend: false,
      empty: s.gl.length ? "No evaluation against greedy." : "No evaluation yet.",
    };
    // Is it learning? The error on games it has trained on (seen) against new games (unseen):
    // the seen line falling while the unseen one rises is memorising. Older runs have no seen line.
    opts.seen = {
      title: "Seen against unseen error",
      series: [
        { name: "Unseen games", points: sm(s.sp.value_rmse_points), color: "var(--series-1)" },
        s.seenCount ? { name: "Seen positions", points: sm(s.tr.value_rmse_points), color: "var(--series-2)" } : null,
        { name: "Leave values only", points: sm(s.sp.leave_only_rmse_points), color: "var(--text-muted)", dashed: true },
      ].filter(Boolean),
      x, y: { label: "RMSE of the predicted spread, points", format: (v) => v.toFixed(2), tickFormat: "auto" }, height: 250, empty,
    };
    opts.gap = {
      title: "Network's edge over leave values",
      series: [{ name: "Edge", points: sm(s.gap), color: "var(--series-1)" }],
      x, y: { label: "points of RMSE", format: (v) => fmt.signed(v, 2), tickFormat: spreadTick }, referenceY: [{ y: 0, label: "no edge" }],
      height: 250, empty,
    };
    const evalEmpty = s.evalCount ? "Not recorded for this run." : strengthEmpty(state);
    opts.gate = {
      title: "Against the best so far",
      series: [{ name: "Against the best", points: s.gate, band: s.gateBand, color: "var(--series-1)" }],
      x, y: { label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick },
      referenceY: [{ y: 0, label: "level with the best" }], markers: s.gateMarkers, height: small,
      empty: s.evalCount ? "No gate match yet: the first network evaluated becomes the best without one." : strengthEmpty(state),
    };
    opts.winRate = {
      title: "Win rate against the bots",
      series: [
        { name: "Greedy + leave", points: s.winGl, band: s.winGlBand, color: "var(--series-1)" },
        { name: "Greedy", points: s.winG, band: s.winGBand, color: "var(--series-2)" },
      ],
      x, y: { label: "games won", format: (v) => fmt.pct(v, 1), tickFormat: pctTick },
      referenceY: [{ y: 0.5, label: "50%" }], height: small, empty: evalEmpty,
    };
    opts.scores = {
      title: "Scores in evaluation",
      series: [
        { name: "Network", points: s.scoreFor, color: "var(--series-1)" },
        { name: "Greedy + leave", points: s.scoreAgainst, color: "var(--series-2)" },
      ],
      x, y: { label: "points per game", format: (v) => v.toFixed(1), tickFormat: "auto" }, height: small, empty: evalEmpty,
    };
    const smallChart = (title, points, ylabel, format, extra) => Object.assign({
      title, series: [{ name: title, points, color: "var(--series-1)" }], x, y: { label: ylabel, format, tickFormat: "auto" },
      height: small, empty,
    }, extra || {});
    opts.brier = smallChart("Win-probability accuracy", sm(s.sp.win_brier), "Brier score", (v) => v.toFixed(4));
    opts.valueLoss = smallChart("Value loss", sm(s.tr.value_loss), "Huber loss", (v) => v.toFixed(4));
    opts.winLoss = smallChart("Win loss", sm(s.tr.win_loss), "cross-entropy", (v) => v.toFixed(4));
    opts.lr = smallChart("Learning rate", s.tr.learning_rate, "step size", lrText,
      { y: { label: "step size", format: lrText, tickFormat: lrText }, markers: s.guardMarkers,
        empty: s.spCount ? "Not recorded for this run." : empty });
    opts.replay = smallChart("Replay buffer", s.sp.replay_positions, "positions", (v) => fmt.compact(v));
    opts.throughput = smallChart("Self-play throughput", sm(s.sp.games_per_second), "games per second", (v) => v.toFixed(1));
    opts.temperature = smallChart("Exploration temperature", s.sp.temperature, "points", (v) => v.toFixed(2));
    opts.score = smallChart("Mean game score", sm(s.sp.mean_score), "points per player", (v) => v.toFixed(1));
    opts.margin = smallChart("Margin of victory", sm(s.sp.mean_abs_spread), "points", (v) => v.toFixed(1));
    opts.firstWin = smallChart("Who wins in self-play", sm(s.sp.first_player_win_rate), "games won by the first player",
      (v) => fmt.pct(v, 1), { y: { label: "won by the first player", format: (v) => fmt.pct(v, 1), tickFormat: pctTick },
        referenceY: [{ y: 0.5, label: "50%" }] });
    opts.turns = smallChart("Game length", sm(s.sp.mean_turns), "turns per game", (v) => v.toFixed(1));
    opts.events = {
      title: "Bingos, exchanges, passes",
      series: [
        { name: "Bingos", points: sm(s.sp.bingos_per_game), color: "var(--series-1)" },
        { name: "Exchanges", points: sm(s.sp.exchanges_per_game), color: "var(--series-2)" },
        { name: "Passes", points: sm(s.sp.passes_per_game), color: "var(--series-3)" },
      ],
      x, y: { label: "per game", format: (v) => v.toFixed(2), tickFormat: "auto" }, height: small, empty,
    };
    return { opts, s };
  }

  /** One line under each chart on how to read it. `seen` depends on whether the run recorded
   *  its error on the positions it trains on (runs made from 13 November 2026 do). */
  function captions(s) {
    return {
      seen: s.seenCount
        ? "When the seen line falls while the unseen line rises, the network is memorising its games. The unseen line should end below the leave-only line."
        : "The unseen line should end below the leave-only line. This run did not record its error on positions it has trained on, so there is no seen line.",
      gap: "Leave-only error minus the network's, on the same new games. Above zero, the network predicts games better than leave values alone.",
      gate: "Each new network head to head against the best so far. A flag marks a promotion: its band cleared zero and it became the best.",
      winRate: "Share of evaluation games won against each bot, with the 95% interval. Above 50%, it wins more often than it loses.",
      scores: "Average points a game against greedy + leave, the network's and the bot's. The gap between the lines is the spread.",
      brier: "Error of the win estimate on new games. Lower is better; 0.25 is no better than a coin flip.",
      valueLoss: "Error of the value output on the batches it trains on. It should drift down; jumps mean unstable training.",
      winLoss: "Error of the win-probability output on the batches it trains on. It should drift down slowly.",
      lr: "Size of each training step. It stays at its setting unless the trainer goes back to its best network, which lowers it (a flag marks each time).",
      replay: "Positions kept to train on. It fills to the replay capacity, then new games push out the oldest.",
      throughput: "Self-play games finished per second. Dips mean something else was using the GPU.",
      temperature: "How random self-play moves are. It falls from the start value to the end value over the decay.",
      score: "Average points per player in a self-play game. It rises as play improves and exploration falls.",
      margin: "Average points between the winner and the loser of a self-play game.",
      firstWin: "How often the player who moves first wins a self-play game. Moving first is worth a little, so slightly above 50% is normal.",
      events: "Bingos, exchanges and passes per self-play game.",
      turns: "Moves per self-play game.",
    };
  }

  function strengthCaption(state, s) {
    if (!s.gl.length) {
      const pairs = (((state.detail.config || {}).train || {}).eval_pairs) || 250;
      return `Each evaluation plays ${fmt.int(pairs)} seat-swapped pairs against each baseline. Until the greedy + leave band clears zero, ` +
        "the network has not been shown to beat that baseline.";
    }
    const valid = s.gl.filter((p) => typeof p[1] === "number");
    const bands = s.glBand.filter((b) => typeof b[1] === "number");
    const last = valid[valid.length - 1];
    const band = bands[bands.length - 1];
    const ci = band ? (band[2] - band[1]) / 2 : null;
    const head = `The latest evaluation against greedy + leave is ${fmt.spread(last[1], ci)}`;
    if (band && band[1] > 0) {
      return `${head}: its band clears zero, so the network is shown to beat that baseline. ` +
        "Green flags mark each new best network.";
    }
    if (band && band[2] < 0) return `${head}: its band is below zero, so the network is playing worse than that baseline.`;
    return `${head}. Until that band clears zero, the network has not been shown to beat the baseline.`;
  }

  function segmented(label, choices, current, choose) {
    return h("div", { class: "row run-control" }, h("span", { class: "dash-toolbar-label" }, label),
      h("div", { class: "segmented", role: "group", "aria-label": label }, choices.map(([key, text]) => h("button", {
        type: "button", "aria-pressed": String(current === key), onclick: () => { if (current !== key) choose(key); },
      }, text))));
  }

  function renderProgress(state, panel) {
    if (!state.smoothing) state.smoothing = "light";
    if (gamesMode(state)) { renderProgressGames(state, panel); return; }
    const live = h("span", { class: "dash-live" });
    state.liveNote = live;
    state.verdictEl = h("section", { class: "card vb", "aria-label": "Verdict" });
    panel.appendChild(state.verdictEl);
    renderVerdict(state);
    if (verdictStale(state)) loadVerdict(state);
    panel.appendChild(h("div", { class: "run-toolbar" },
      h("div", { class: "row run-controls" },
        segmented("Plot against", [["games", "Games"], ["hours", "Hours"], ["steps", "Steps"]], state.xmode,
          (key) => { state.xmode = key; renderPanel(state, { panel }); }),
        segmented("Smoothing", [["off", "Off"], ["light", "Light"], ["heavy", "Heavy"]], state.smoothing,
          (key) => { state.smoothing = key; renderPanel(state, { panel }); })),
      live));
    updateLiveNote(state);

    const { opts, s } = progressOptions(state);
    state.hadSeen = !!s.seenCount;
    const caption = h("p", { class: "dash-caption" }, strengthCaption(state, s));
    state.caption = caption;
    const slot = () => h("div");
    const slots = {};
    for (const key of Object.keys(opts)) slots[key] = slot();
    panel.appendChild(h("div", { class: "card run-strength" },
      h("div", { class: "run-strength-head" },
        h("h2", null, "Strength against the baselines"),
        h("p", { class: "card-sub" }, "Mean spread per game for the network over seat-swapped pairs, with its 95% interval shaded. " +
          "Evaluations are not smoothed.")),
      slots.strength, h("div", { class: "run-strength-second" }, slots.strengthGreedy), caption));

    state.captions = {};
    if (!s.spCount && !s.evalCount) {
      panel.appendChild(h("p", { class: "empty" }, h("strong", null, "The other charts appear once self-play starts."),
        "Value accuracy, loss, throughput and the self-play statistics are logged about once a minute."));
    } else {
      const caps = captions(s);
      const card = (key, small) => {
        const cap = h("p", { class: "run-cap" }, caps[key]);
        state.captions[key] = cap;
        return h("div", { class: "card" + (small ? " run-small-card" : " run-mid-card") }, slots[key], cap);
      };
      const group = (title, sub, ...grids) => h("section", { class: "run-group" },
        h("div", { class: "run-group-head" }, h("h2", null, title), h("p", null, sub)), ...grids);
      panel.appendChild(group("Is it learning?",
        "Whether the network predicts new games better than leave values alone, and whether each new network beats the last.",
        h("div", { class: "grid-2 run-grid" }, card("seen"), card("gap")),
        h("div", { class: "grid-4 run-grid" }, ["gate", "winRate", "scores", "brier"].map((key) => card(key, true)))));
      panel.appendChild(group("Training", "The optimiser's side: the losses on the batches it trains on, its step size, and what it trains on.",
        h("div", { class: "grid-4 run-grid" }, ["valueLoss", "winLoss", "lr", "replay"].map((key) => card(key, true)))));
      panel.appendChild(group("Self-play games", "The games the network plays against itself to learn from.",
        h("div", { class: "grid-4 run-grid" },
          ["throughput", "temperature", "score", "margin", "firstWin", "events", "turns"].map((key) => card(key, true)))));
    }
    for (const [key, options] of Object.entries(opts)) {
      if (slots[key].isConnected) state.charts[key] = Charts.line(slots[key], options);
    }
  }

  function updateProgress(state) {
    if (gamesMode(state)) { updateProgressGames(state); return; }
    if (!state.charts.strength) return;
    const { opts, s } = progressOptions(state);
    const hadSmall = !!state.charts.valueLoss;
    const seenChanged = state.captions && state.captions.seen && !!s.seenCount !== state.hadSeen;
    if ((!hadSmall && (s.spCount || s.evalCount)) || seenChanged) {
      // Self-play has just started (or the first seen-error record arrived): lay the tab out again.
      const panel = state.charts.strength.element.closest(".run-panel");
      if (panel) {
        renderPanel(state, { panel });
        return;
      }
    }
    for (const [key, chart] of Object.entries(state.charts)) if (opts[key]) chart.update(opts[key]);
    if (state.caption) state.caption.textContent = strengthCaption(state, s);
  }

  // ---- Progress of a run that trains from games ----------------------------------------------

  // The helper outputs a network trained from games may have, in the order the records list them.
  const HELPERS = [
    ["spread", "Final spread", "the points still to gain from the position"],
    ["next_score", "Opponent's next score", "the opponent's next score"],
    ["next_bingo", "Opponent's next bingo", "whether the opponent places all seven tiles next"],
    ["next_squares", "Where the next moves land", "where the next two moves put their tiles"],
  ];
  const winTick = (v) => pctTick(v);
  const winCiFormat = (v) => (100 * Math.abs(v)).toFixed(1);

  function buildGamesSeries(records, mode) {
    const X = X_MODES[mode];
    const out = { win: [], winBand: [], spread: [], spreadBand: [], markers: [], trainWin: [], validWin: [], accuracy: [],
      rmse: [], leaveOnly: [], lr: [], pps: [], help: {}, weights: {}, evalCount: 0, trainCount: 0, validCount: 0 };
    for (const [key] of HELPERS) { out.help[key] = { train: [], valid: [] }; out.weights[key] = []; }
    const lines = () => [out.win, out.spread, out.trainWin, out.validWin, out.accuracy, out.rmse, out.leaveOnly, out.lr, out.pps,
      ...HELPERS.flatMap(([k]) => [out.help[k].train, out.help[k].valid, out.weights[k]])];
    const num = (v) => typeof v === "number" && Number.isFinite(v);
    let started = false;
    for (const r of records) {
      const x = X.of(r);
      if (r.kind === "session") {
        if (r.event === "start" && started) {
          for (const list of lines()) list.push([x, null]);
          out.winBand.push([x, null, null]);
          out.spreadBand.push([x, null, null]);
          out.markers.push({ x, label: "Resumed", tone: "info", detail: `A new session started from the checkpoint at step ${fmt.int(r.step)}.` });
        }
        continue;
      }
      started = true;
      if (r.kind === "eval") {
        if (r.opponent && r.opponent !== "greedy-leave") continue;
        if (num(r.win_rate)) {
          out.win.push([x, r.win_rate]);
          if (num(r.win_ci)) out.winBand.push([x, r.win_rate - r.win_ci, r.win_rate + r.win_ci]);
          out.evalCount++;
        }
        if (num(r.spread)) {
          out.spread.push([x, r.spread]);
          if (num(r.spread_ci)) out.spreadBand.push([x, r.spread - r.spread_ci, r.spread + r.spread_ci]);
        }
      } else if (r.kind === "best") {
        out.markers.push({ x, label: "New best network", tone: "good", group: "New best network",
          detail: `Step ${fmt.int(r.step)} won ${K.winText(r.win_rate, r.win_ci)} of games, the most so far, and was saved as best.pt.` });
      } else if (r.kind === "train") {
        out.trainCount++;
        if (num(r.win_loss)) out.trainWin.push([x, r.win_loss]);
        if (num(r.learning_rate)) out.lr.push([x, r.learning_rate]);
        if (num(r.positions_per_second)) out.pps.push([x, r.positions_per_second]);
        for (const [key] of HELPERS) {
          if (num(r[key + "_loss"])) out.help[key].train.push([x, r[key + "_loss"]]);
          if (num(r[key + "_weight"])) out.weights[key].push([x, r[key + "_weight"]]);
        }
      } else if (r.kind === "valid") {
        out.validCount++;
        if (num(r.win_loss)) out.validWin.push([x, r.win_loss]);
        if (num(r.win_accuracy)) out.accuracy.push([x, r.win_accuracy]);
        if (num(r.value_rmse_points)) out.rmse.push([x, r.value_rmse_points]);
        if (num(r.leave_only_rmse_points)) out.leaveOnly.push([x, r.leave_only_rmse_points]);
        for (const [key] of HELPERS) if (num(r[key + "_loss"])) out.help[key].valid.push([x, r[key + "_loss"]]);
      }
    }
    const has = (points) => points.some((p) => typeof p[1] === "number");
    out.helpers = HELPERS.filter(([k]) => has(out.help[k].train) || has(out.help[k].valid)).map(([k]) => k);
    out.weighted = HELPERS.filter(([k]) => has(out.weights[k])).map(([k]) => k);
    return out;
  }

  function gamesEmpty(state) {
    const run = state.detail.run;
    if (run.status === "queued") return "Waiting in the queue.";
    return isLive(run) ? "No training report yet: they come about once a minute." : "This run ended before its first training report.";
  }

  function matchEmpty(state) {
    const run = state.detail.run;
    const minutes = (((state.detail.config || {}).offline || {}).eval_minutes);
    if (run.status === "queued") return "Matches start once the run is training.";
    if (isLive(run)) return minutes ? `No match yet. The first comes about ${fmt.int(minutes)} minutes into training (offline.eval_minutes).` : "No match yet.";
    return "This run ended before its first match.";
  }

  function progressOptionsGames(state) {
    const s = buildGamesSeries(state.records, state.xmode);
    const x = xAxis(state.xmode);
    const w = SMOOTHING[state.smoothing] || 0;
    const sm = (points) => smooth(points, w);
    const small = 176;
    const empty = gamesEmpty(state);
    const opts = {};
    const xs = [...s.win, ...s.spread].filter((p) => typeof p[1] === "number").map((p) => p[0]);
    const xShared = xs.length > 1 ? Object.assign({}, x, { min: Math.min(...xs), max: Math.max(...xs) }) : x;
    const margin = { left: 52, right: 64 };
    const winY = { label: "games won", format: (v) => fmt.pct(v, 1), tickFormat: winTick, ciFormat: winCiFormat };
    opts.win = {
      title: "Win rate",
      series: [{ name: "Win rate", points: s.win, band: s.winBand, color: "var(--series-1)" }],
      x: xShared, y: winY, referenceY: [{ y: 0.5, label: "50%: level" }], markers: s.markers, height: 330, margin,
      empty: matchEmpty(state),
    };
    opts.spread = {
      title: "Spread, secondary",
      series: [{ name: "Spread", points: s.spread, band: s.spreadBand, color: "var(--series-2)" }],
      x: xShared, y: { label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick }, referenceY: [{ y: 0, label: "even" }],
      markers: s.markers, height: 200, margin, legend: false, empty: s.evalCount ? "No spread recorded." : matchEmpty(state),
    };
    const lossY = { label: "cross-entropy", format: (v) => v.toFixed(4), tickFormat: "auto" };
    opts.winLoss = {
      title: "Win loss: training batches and held-out games",
      series: [
        { name: "Training batches", points: sm(s.trainWin), color: "var(--series-1)" },
        { name: "Held-out games", points: sm(s.validWin), color: "var(--series-2)" },
      ],
      x, y: lossY, height: 250, empty,
    };
    opts.accuracy = {
      title: "Results named right on held-out games",
      series: [{ name: "Named right", points: sm(s.accuracy), color: "var(--series-1)" }],
      x, y: { label: "share of positions", format: (v) => fmt.pct(v, 1), tickFormat: winTick }, height: 250, legend: false,
      empty: s.trainCount ? "No check on held-out games yet." : empty,
    };
    opts.rmse = {
      title: "Points error on held-out games",
      series: [
        { name: "Network", points: sm(s.rmse), color: "var(--series-1)" },
        { name: "Leave values only", points: s.leaveOnly, color: "var(--text-muted)", dashed: true },
      ],
      x, y: { label: "RMSE of the predicted spread, points", format: (v) => v.toFixed(2), tickFormat: "auto" }, height: 250,
      empty: s.trainCount ? "No check on held-out games yet." : empty,
    };
    for (const [key, label] of HELPERS) {
      if (!s.helpers.includes(key)) continue;
      opts["help_" + key] = {
        title: `${label} loss`,
        series: [
          { name: "Training", points: sm(s.help[key].train), color: "var(--series-1)" },
          { name: "Held-out", points: sm(s.help[key].valid), color: "var(--series-2)" },
        ],
        x, y: { label: "loss", format: (v) => v.toFixed(4), tickFormat: "auto" }, height: small, empty,
      };
    }
    if (s.weighted.length) {
      opts.weights = {
        title: "Helper weights",
        series: HELPERS.filter(([k]) => s.weighted.includes(k)).map(([k, label], i) => ({ name: label, points: s.weights[k], color: `var(--series-${i + 1})` })),
        x, y: { label: "weight beside the win output's 1", format: (v) => v.toFixed(3), tickFormat: "auto" }, height: small, empty, endLabels: false,
      };
    }
    opts.lr = {
      title: "Learning rate",
      series: [{ name: "Learning rate", points: s.lr, color: "var(--series-1)" }],
      x, y: { label: "step size", format: lrText, tickFormat: lrText }, height: small, legend: false, empty,
    };
    opts.pps = {
      title: "Positions per second",
      series: [{ name: "Positions per second", points: sm(s.pps), color: "var(--series-1)" }],
      x, y: { label: "positions trained a second", format: (v) => fmt.int(v), tickFormat: "auto" }, height: small, legend: false, empty,
    };
    return { opts, s };
  }

  function gamesCaptions() {
    const caps = {
      winLoss: "Error of the predicted result: win, draw or loss. Good: both lines fall and the held-out line stays close to the training line; held-out rising while training falls is memorising.",
      accuracy: "Share of held-out positions whose most likely result was the one that happened. Good: it rises, then levels off.",
      rmse: "Error of the predicted final spread on held-out games, beside leave values alone. Good: the network's line ends below the leave-only line.",
      weights: "How much each helper output counts in the loss, beside the win output's 1. They are re-set as training goes so that each pulls a fixed share (offline.aux_share); steady lines are normal.",
      lr: "Size of each training step. It rises over the warm-up, then falls smoothly to a small fraction of its peak by the last step. Good: a smooth curve.",
      pps: "Positions trained a second. Good: steady; dips mean a match or a check was running, or something else was using the GPU.",
    };
    for (const [key, , what] of HELPERS) {
      caps["help_" + key] = `Error predicting ${what}: practice only, never used to choose a move. Good: training and held-out fall together.`;
    }
    return caps;
  }

  function winCaption(state, s) {
    const pairs = (((state.detail.config || {}).offline || {}).eval_pairs);
    if (!s.win.length) {
      return (pairs ? `Each match plays ${fmt.int(pairs)} seat-swapped pairs against the static evaluator. ` : "") +
        "Until a match's band clears 50%, the network has not been shown to win more than it loses.";
    }
    const valid = s.win.filter((p) => typeof p[1] === "number");
    const bands = s.winBand.filter((b) => typeof b[1] === "number");
    const last = valid[valid.length - 1];
    const band = bands[bands.length - 1];
    const ci = band ? (band[2] - band[1]) / 2 : null;
    const head = `The latest match is ${K.winText(last[1], ci)}`;
    if (band && band[1] > 0.5) return `${head}: its band clears 50%, so the network is shown to win more than half its games. Green flags mark each new best network.`;
    if (band && band[2] < 0.5) return `${head}: its band is below 50%, so the network loses more games than it wins.`;
    return `${head}. Until that band clears 50%, the network has not been shown to win more than it loses.`;
  }

  function renderProgressGames(state, panel) {
    const live = h("span", { class: "dash-live" });
    state.liveNote = live;
    state.verdictEl = h("section", { class: "card vb", "aria-label": "Verdict" });
    panel.appendChild(state.verdictEl);
    renderVerdict(state);
    if (verdictStale(state)) loadVerdict(state);
    panel.appendChild(h("div", { class: "run-toolbar" },
      h("div", { class: "row run-controls" },
        segmented("Plot against", [["steps", "Steps"], ["hours", "Hours"]], state.xmode,
          (key) => { state.xmode = key; renderPanel(state, { panel }); }),
        segmented("Smoothing", [["off", "Off"], ["light", "Light"], ["heavy", "Heavy"]], state.smoothing,
          (key) => { state.smoothing = key; renderPanel(state, { panel }); })),
      live));
    updateLiveNote(state);

    const { opts, s } = progressOptionsGames(state);
    state.gamesLayout = layoutKey(s);
    const slots = {};
    for (const key of Object.keys(opts)) slots[key] = h("div");
    const caption = h("p", { class: "dash-caption" }, winCaption(state, s));
    state.caption = caption;
    panel.appendChild(h("div", { class: "card run-strength" },
      h("div", { class: "run-strength-head" },
        h("h2", null, "Strength against the static evaluator"),
        h("p", { class: "card-sub" }, "Share of games the network won over seat-swapped pairs against the bot that plays the highest score plus " +
          "leave value, with its 95% interval shaded. 50% is level. Matches are not smoothed.")),
      slots.win, caption,
      h("div", { class: "run-strength-second" }, slots.spread,
        h("p", { class: "dash-caption" }, "Points per game against the static evaluator, with its 95% interval. Secondary for these runs: a network " +
          "trained to win can be behind on points and ahead on wins."))));

    if (!s.trainCount) {
      panel.appendChild(h("p", { class: "empty" }, h("strong", null, "The other charts appear once training reports."),
        "The losses, the checks on held-out games and the speed are logged about once a minute."));
    } else {
      const caps = gamesCaptions();
      const card = (key, small) => h("div", { class: "card" + (small ? " run-small-card" : " run-mid-card") }, slots[key],
        h("p", { class: "run-cap" }, caps[key]));
      const group = (title, sub, ...grids) => h("section", { class: "run-group" },
        h("div", { class: "run-group-head" }, h("h2", null, title), h("p", null, sub)), ...grids);
      panel.appendChild(group("Is it learning?",
        "On positions from games it never trains on (the held-out games), beside the batches it trains on.",
        h("div", { class: "grid-3 run-grid" }, ["winLoss", "accuracy", "rmse"].map((key) => card(key)))));
      const helperKeys = [...s.helpers.map((k) => "help_" + k), ...(opts.weights ? ["weights"] : [])];
      if (helperKeys.length) {
        panel.appendChild(group("Helper outputs",
          "Extra outputs the network learns as practice, to read the board better: the final spread and, with helper outputs on, what the opponent does next.",
          h("div", { class: "grid-4 run-grid" }, helperKeys.map((key) => card(key, true)))));
      }
      panel.appendChild(group("Training", "The optimiser's side: its step size, and how fast it works through the positions.",
        h("div", { class: "grid-4 run-grid" }, ["lr", "pps"].map((key) => card(key, true)))));
    }
    for (const [key, options] of Object.entries(opts)) {
      if (slots[key].isConnected) state.charts[key] = Charts.line(slots[key], options);
    }
  }

  /** What decides the layout of the games Progress tab: laid out again when it changes. */
  function layoutKey(s) {
    return [s.trainCount > 0, s.helpers.join(","), s.weighted.length > 0].join("|");
  }

  function updateProgressGames(state) {
    if (!state.charts.win) return;
    const { opts, s } = progressOptionsGames(state);
    if (layoutKey(s) !== state.gamesLayout) {
      const panel = state.charts.win.element.closest(".run-panel");
      if (panel) {
        renderPanel(state, { panel });
        return;
      }
    }
    for (const [key, chart] of Object.entries(state.charts)) if (opts[key]) chart.update(opts[key]);
    if (state.caption) state.caption.textContent = winCaption(state, s);
  }

  // ---- Verdict banner: how the run went, at the top of Progress ----------------------------

  const guideHref = (name) => `#/guide?from=${enc("run/" + name)}`;

  /** The report is read again when the run's status changes, and every 30 s while it trains. */
  function verdictStale(state) {
    const run = state.detail && state.detail.run;
    if (!run) return false;
    return !state.report || state.reportStatus !== run.status || (isLive(run) && Date.now() - state.reportAt > 30000);
  }

  async function loadVerdict(state) {
    if (state.verdictBusy) return;
    state.verdictBusy = true;
    const status = state.detail.run.status;
    try {
      const report = await App.get(`/api/runs/${enc(state.name)}/report`);
      state.report = report;
      state.reportAt = Date.now();
      state.reportStatus = status;
    } catch (error) {
      console.warn(error);
      if (!state.report) state.reportFailed = true;
    } finally {
      state.verdictBusy = false;
    }
    renderVerdict(state);
  }

  /** The health object: the report's when it has one, else the run summary's (both may lack it). */
  function healthOf(state) {
    const sum = state.report && state.report.summary;
    if (sum && sum.health !== undefined) return sum.health;
    return state.detail && state.detail.run ? state.detail.run.health : undefined;
  }

  function renderVerdict(state) {
    const el = state.verdictEl;
    if (!el || !el.isConnected) return;
    const report = state.report;
    if (!report) {
      el.hidden = !!state.reportFailed;
      el.replaceChildren(h("p", { class: "muted vb-loading" }, "Reading the run…"));
      return;
    }
    el.hidden = false;
    const health = healthOf(state);
    const bad = health && (health.state === "worse" || health.state === "behind");
    el.classList.toggle("vb-bad", !!bad);
    App.append(App.clear(el), [
      h("div", { class: "vb-top" },
        h("div", { class: "vb-main" },
          h("div", { class: "vb-label" }, "How it went", K.healthChip(health, null, gamesMode(state))),
          h("p", { class: "vb-headline" }, fmt.prose(report.headline))),
        h("div", { class: "vb-actions" },
          h("button", { type: "button", class: "button button-small", onclick: () => selectTab(state, state.els, "report") },
            "Read the report"),
          h("a", { class: "button button-small", href: guideHref(state.name) }, K.icon("book", 14), "Tuning guide"))),
      bad ? h("div", { class: "notice notice-warning vb-notice", role: "note" }, icon(ICON.warning, "dash-notice-icon"),
        h("div", { class: "dash-notice-main" },
          fmt.prose(health.note), " ",
          h("a", { href: guideHref(state.name) }, "See what each setting does, and which way to turn it, in the tuning guide."))) : null]);
  }

  function updateLiveNote(state) {
    if (!state.liveNote || !state.detail) return;
    const run = state.detail.run;
    if (isLive(run)) {
      state.liveNote.replaceChildren(h("span", { class: "dot dot-live" }), "Live, refreshed every 5 seconds");
    } else {
      state.liveNote.replaceChildren(`${STATUS[run.status] ? STATUS[run.status][0] : run.status} · last record ${fmt.datetime(run.updated)}`);
    }
  }

  // ---- Report ---------------------------------------------------------------------------

  async function renderReport(state, els, panel) {
    panel.appendChild(h("p", { class: "muted" }, "Loading the report…"));
    let report;
    try {
      report = await App.get(`/api/runs/${enc(state.name)}/report`);
    } catch (error) {
      if (state.tab !== "report") return;
      panel.replaceChildren(h("div", { class: "notice notice-critical" }, h("strong", null, "The report could not be loaded. "), String(error.message || error)));
      return;
    }
    if (state.tab !== "report" || !panel.isConnected) return;
    state.report = report;
    state.reportAt = Date.now();
    state.reportStatus = state.detail.run.status;
    panel.replaceChildren();
    drawReport(state, panel, report);
  }

  async function refreshReport(state, els) {
    try {
      const report = await App.get(`/api/runs/${enc(state.name)}/report`);
      if (state.tab !== "report") return;
      state.report = report;
      state.reportAt = Date.now();
      state.reportStatus = state.detail.run.status;
      const keep = state.selectedEvent;
      renderPanelWith(state, els, report, keep);
    } catch (error) {
      console.warn(error);
    }
  }

  function renderPanelWith(state, els, report, selected) {
    stopTabTimers(state);
    const panel = App.clear(els.panel);
    drawReport(state, panel, report, selected);
  }

  /** The health sentence under the report's headline: a warning when the run fell behind or got
   *  worse, a plain note otherwise. Nothing when the server sends no verdict. */
  function healthNotice(state, health) {
    if (!health || !health.note) return null;
    const bad = health.state === "worse" || health.state === "behind";
    return h("div", { class: "notice report-health" + (bad ? " notice-warning" : "") },
      bad ? icon(ICON.warning, "dash-notice-icon") : icon(ICON.info, "dash-notice-icon dash-tone-info"),
      h("div", { class: "dash-notice-main" }, K.healthChip(health, "report-health-chip", gamesMode(state)), " ", fmt.prose(health.note),
        bad ? [" ", h("a", { href: guideHref(state.name) }, "What to change: the tuning guide.")] : null));
  }

  /** The clickable timeline of a report. `at(ev)` is where an event sits on the chart's x axis. */
  function eventTimeline(state, list, events, chart, at, meta, selected) {
    const items = events.map((ev, i) => {
      const tone = ["good", "warning", "info", "critical"].includes(ev.tone) ? ev.tone : "info";
      const item = h("li", { class: "report-event report-event-" + tone, tabindex: "0", role: "button", "aria-pressed": "false" },
        icon(ICON[tone], "report-event-icon dash-tone-" + tone),
        h("div", null,
          h("div", { class: "report-event-title" }, fmt.prose(ev.title)),
          ev.detail ? h("div", { class: "report-event-detail" }, fmt.prose(ev.detail)) : null,
          h("div", { class: "report-event-meta" }, meta(ev))));
      const choose = () => {
        const already = state.selectedEvent === i;
        state.selectedEvent = already ? null : i;
        items.forEach((node, j) => {
          node.classList.toggle("selected", j === state.selectedEvent);
          node.setAttribute("aria-pressed", String(j === state.selectedEvent));
        });
        chart.highlight(already ? null : at(ev), already ? null : fmt.prose(ev.title));
      };
      item.addEventListener("click", choose);
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
      });
      return item;
    });
    list.append(...items);
    if (selected !== undefined && selected !== null && items[selected]) {
      state.selectedEvent = null;
      items[selected].click();
    } else state.selectedEvent = null;
  }

  /** The report of a run that trains from games: the same three parts, in terms of win rate. */
  function drawReportGames(state, panel, report, selected) {
    const sum = report.summary || {};
    const best = sum.best;
    const final = sum.final;
    const fs = sum.first_significant;
    const held = sum.held_out;
    const health = healthNotice(state, sum.health !== undefined ? sum.health : state.detail.run.health);
    const events = report.events || [];
    if (!sum.steps && !events.length) {
      panel.append(h("p", { class: "report-headline" }, fmt.prose(report.headline)), ...(health ? [health] : []),
        h("div", { class: "card" }, h("p", { class: "empty" }, h("strong", null, "Nothing to report yet."),
          "The report fills in as the run trains: its matches against the static evaluator, new best networks and how each session ended.")));
      return;
    }
    const passes = typeof sum.passes === "number" ? String(Number(sum.passes.toPrecision(3))) : null;
    const matchHint = (m) => `step ${fmt.int(m.step)} · ${K.winVerdict(m.win_rate, m.win_ci)}`;
    const tiles = [
      Charts.statTile({ label: "Positions trained", value: fmt.compact(sum.positions),
        hint: [typeof sum.hours === "number" ? `in ${fmt.duration(sum.hours)}` : null, passes ? `${passes} passes` : null].filter(Boolean).join(" · ") || null }),
      Charts.statTile({ label: "Mean speed", value: sum.mean_positions_per_second ? fmt.int(sum.mean_positions_per_second) : "–", hint: "positions per second" }),
      Charts.statTile({ label: "Training steps", value: fmt.int(sum.steps),
        hint: sum.train_games ? `${fmt.compact(sum.train_games)} games of ${sum.game_set || "its set"} to learn from` : null }),
      Charts.statTile({ label: "Matches", value: sum.evaluations === null || sum.evaluations === undefined ? "–" : String(sum.evaluations),
        hint: "against the static evaluator" }),
      Charts.statTile({ label: "Best win rate", value: best ? winValue(best.win_rate, best.win_ci) : "–",
        hint: best ? matchHint(best) : "no match yet" }),
      Charts.statTile({ label: "Latest win rate", value: final ? winValue(final.win_rate, final.win_ci) : "–",
        hint: final ? `${matchHint(final)}` + (typeof final.spread === "number" ? ` · spread ${fmt.spread(final.spread, final.ci)}` : "") : "no match yet" }),
      Charts.statTile({ label: "First clearly above half", value: fs ? `step ${fmt.int(fs.step)}` : final ? "not yet" : "–",
        hint: fs ? `after ${fmt.duration(fs.hours)} of training · ${K.winText(fs.win_rate, fs.win_ci)}` : final ? "no match's interval has been above 50%" : "no match yet" }),
      Charts.statTile({ label: "Held-out games", value: held && typeof held.win_accuracy === "number" ? fmt.pct(held.win_accuracy) + " right" : "–",
        hint: held ? `points error ${fmt.fixed(held.value_rmse_points, 1)}, leave values alone ${fmt.fixed(held.leave_only_rmse_points, 1)}` : "not checked yet" }),
    ];
    const chartSlot = h("div");
    const list = h("ol", { class: "report-timeline" });
    panel.append(
      h("p", { class: "report-headline" }, fmt.prose(report.headline)),
      ...(health ? [health] : []),
      h("div", { class: "grid-4 report-tiles" }, tiles),
      h("div", { class: "split report-split" },
        h("div", { class: "card report-chart" }, chartSlot,
          h("p", { class: "dash-caption" }, "Select an event to see where it happened.")),
        h("div", { class: "card report-events" },
          h("div", { class: "card-head" }, h("div", { class: "card-title" }, "What happened"),
            h("div", { class: "card-sub" }, events.length ? `${events.length} event${events.length === 1 ? "" : "s"}, oldest first` : "")),
          events.length ? list : h("p", { class: "muted" }, "Nothing notable yet."))));
    const s = buildGamesSeries(state.records, "hours");
    const chart = Charts.line(chartSlot, {
      title: "Win rate against the static evaluator",
      subtitle: "Share of games won, with its 95% interval",
      series: [{ name: "Win rate", points: s.win, band: s.winBand, color: "var(--series-1)" }],
      x: xAxis("hours"), y: { label: "games won", format: (v) => fmt.pct(v, 1), tickFormat: winTick, ciFormat: winCiFormat },
      referenceY: [{ y: 0.5, label: "50%: level" }], height: 340, empty: matchEmpty(state),
    });
    state.charts.report = chart;
    eventTimeline(state, list, events, chart, (ev) => ev.hours,
      (ev) => `${fmt.duration(ev.hours)} of training · ${fmt.compact(ev.games)} positions`, selected);
  }

  function drawReport(state, panel, report, selected) {
    if (gamesMode(state)) { drawReportGames(state, panel, report, selected); return; }
    const sum = report.summary || {};
    const best = sum.best;
    const final = sum.final;
    const fs = sum.first_significant;
    const health = healthNotice(state, sum.health !== undefined ? sum.health : state.detail.run.health);
    if (!sum.games && !(report.events || []).length) {
      panel.append(h("p", { class: "report-headline" }, fmt.prose(report.headline)), ...(health ? [health] : []),
        h("div", { class: "card" }, h("p", { class: "empty" }, h("strong", null, "Nothing to report yet."),
          "The report fills in as the run trains: its evaluations, new best networks and anything unusual along the way.")));
      return;
    }
    const tiles = [
      Charts.statTile({ label: "Games trained", value: fmt.compact(sum.games), hint: sum.hours !== null && sum.hours !== undefined ? `in ${fmt.duration(sum.hours)}` : null }),
      Charts.statTile({ label: "Mean throughput", value: sum.mean_games_per_second ? fmt.fixed(sum.mean_games_per_second, 1) : "–", hint: "games per second" }),
      Charts.statTile({ label: "Training steps", value: fmt.int(sum.steps) }),
      Charts.statTile({ label: "Promotions", value: sum.promotions === null || sum.promotions === undefined ? "–" : String(sum.promotions),
        hint: "times a new best network was kept" }),
      Charts.statTile({ label: "Best against greedy + leave", value: best ? spreadValue(best.spread, best.ci) : "–",
        hint: best ? `step ${fmt.int(best.step)} · ${fmt.compact(best.games)} games · ${significance(best.spread, best.ci)}` : "no best network yet" }),
      Charts.statTile({ label: "Latest against greedy + leave", value: final ? spreadValue(final.spread, final.ci) : "–",
        hint: final ? `step ${fmt.int(final.step)} · win rate ${fmt.pct(final.win_rate)}` : "no evaluation yet" }),
      Charts.statTile({ label: "First significant win", value: fs ? fmt.compact(fs.games) + " games" : final ? "not yet" : "–",
        hint: fs ? `after ${fmt.duration(fs.hours)} of training` : final ? "no evaluation's interval has been above zero" : "no evaluation yet" }),
      Charts.statTile({ label: "Value error below leave-only", value: sum.value_rmse_gain === null || sum.value_rmse_gain === undefined ? "–"
        : fmt.signed(sum.value_rmse_gain, 2), hint: "points of RMSE, last 10 measurements" }),
    ];
    const chartSlot = h("div");
    const list = h("ol", { class: "report-timeline" });
    const events = report.events || [];
    panel.append(
      h("p", { class: "report-headline" }, fmt.prose(report.headline)),
      ...(health ? [health] : []),
      h("div", { class: "grid-4 report-tiles" }, tiles),
      h("div", { class: "split report-split" },
        h("div", { class: "card report-chart" }, chartSlot,
          h("p", { class: "dash-caption" }, "Select an event to see where it happened.")),
        h("div", { class: "card report-events" },
          h("div", { class: "card-head" }, h("div", { class: "card-title" }, "What happened"),
            h("div", { class: "card-sub" }, events.length ? `${events.length} event${events.length === 1 ? "" : "s"}, oldest first` : "")),
          events.length ? list : h("p", { class: "muted" }, "Nothing notable yet."))));
    const s = buildSeries(state.records, "games");
    const chart = Charts.line(chartSlot, {
      title: "Strength against greedy + leave",
      subtitle: "Spread per game with its 95% interval",
      series: [{ name: "Greedy + leave", points: s.gl, band: s.glBand, color: "var(--series-1)" }],
      x: xAxis("games"), y: { label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick },
      referenceY: [{ y: 0, label: "even" }], height: 340, empty: strengthEmpty(state),
    });
    state.charts.report = chart;
    const items = events.map((ev, i) => {
      const tone = ["good", "warning", "info", "critical"].includes(ev.tone) ? ev.tone : "info";
      const item = h("li", { class: "report-event report-event-" + tone, tabindex: "0", role: "button", "aria-pressed": "false" },
        icon(ICON[tone], "report-event-icon dash-tone-" + tone),
        h("div", null,
          h("div", { class: "report-event-title" }, fmt.prose(ev.title)),
          ev.detail ? h("div", { class: "report-event-detail" }, fmt.prose(ev.detail)) : null,
          h("div", { class: "report-event-meta" }, `${fmt.compact(ev.games)} games · ${fmt.duration(ev.hours)}`)));
      const choose = () => {
        const already = state.selectedEvent === i;
        state.selectedEvent = already ? null : i;
        items.forEach((node, j) => {
          node.classList.toggle("selected", j === state.selectedEvent);
          node.setAttribute("aria-pressed", String(j === state.selectedEvent));
        });
        chart.highlight(already ? null : ev.games, already ? null : fmt.prose(ev.title));
      };
      item.addEventListener("click", choose);
      item.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); }
      });
      return item;
    });
    list.append(...items);
    if (selected !== undefined && selected !== null && items[selected]) {
      state.selectedEvent = null;
      items[selected].click();
    } else state.selectedEvent = null;
  }

  // ---- Settings -------------------------------------------------------------------------

  function settingValue(value) {
    if (value === "") return h("span", { class: "muted" }, "empty");
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "string") return `"${value}"`;
    if (typeof value === "number") return Math.abs(value) >= 1000 && Number.isInteger(value) ? fmt.int(value) : String(value);
    return JSON.stringify(value);
  }

  async function renderSettings(state, panel) {
    const schema = await loadSchema(state);
    if (state.tab !== "settings" || !panel.isConnected) return;
    const config = state.detail.config || {};
    const games = gamesMode(state);
    const mode = games ? "games" : "selfplay";
    // Each kind of run shows only the settings that apply to it (`modes` in the schema; an
    // older server sends none, and then every setting shows).
    const applies = (key) => {
      const meta = schemaField(schema, key);
      return !meta || !Array.isArray(meta.modes) || meta.modes.includes(mode);
    };
    // A games run's defaults are the schema's with games_defaults on top, as the server applies them.
    const gamesDefaults = (games && schema && schema.games_defaults) || {};
    const defaultOf = (key, meta) => (Object.prototype.hasOwnProperty.call(gamesDefaults, key) ? gamesDefaults[key] : meta ? meta.default : undefined);
    const changed = state.detail.run.changed || {};
    let changedKeys = Object.keys(changed).filter((k) => k !== "run.name" && applies(k));
    if (games) {
      changedKeys = changedKeys.filter((k) => k !== "offline.games" &&
        !(Object.prototype.hasOwnProperty.call(gamesDefaults, k) && K.same(changed[k], gamesDefaults[k])));
    }
    const order = schema ? (schema.sections || []).map((s) => s.name) : ["run", "model", "selfplay", "warmstart", "train"];
    for (const key of Object.keys(config)) if (!order.includes(key)) order.push(key);
    const against = games ? "the defaults for training from games" : "the defaults";
    panel.appendChild(h("p", { class: "run-settings-lede" }, changedKeys.length
      ? [`${changedKeys.length} setting${changedKeys.length === 1 ? " differs" : "s differ"} from ${against}: `,
        changedKeys.map((k, i) => [i ? ", " : "", h("code", { class: "mono" }, k)]), "."]
      : `Every setting is at ${against === "the defaults" ? "its default" : against}.`,
    games ? ` Only the settings that apply to training from games are listed.${state.detail.run.game_set ? ` The game set is ${state.detail.run.game_set}.` : ""}` : null));
    const columns = h("div", { class: "run-settings-columns" });
    for (const sectionName of order) {
      const values = config[sectionName];
      if (!values) continue;
      const section = schema ? (schema.sections || []).find((s) => s.name === sectionName) : null;
      // The run's name and folder are the app's own business and are in the page header.
      const entries = Object.entries(values).filter(([field]) => !(sectionName === "run" && (field === "name" || field === "out_dir")) &&
        applies(`${sectionName}.${field}`));
      if (!entries.length) continue;
      const rows = entries.map(([field, value]) => {
        const key = `${sectionName}.${field}`;
        const meta = schemaField(schema, key);
        const isChanged = games
          ? key !== "offline.games" && (meta ? !K.same(value, defaultOf(key, meta)) : Object.prototype.hasOwnProperty.call(changed, key))
          : Object.prototype.hasOwnProperty.call(changed, key) && key !== "run.name";
        return h("tr", { class: isChanged ? "run-setting-changed" : null, title: meta && meta.help ? meta.help : null },
          h("td", null, h("div", { class: "run-setting-label" }, meta ? meta.label : field), h("div", { class: "run-setting-key" }, key)),
          h("td", { class: "run-setting-value" },
            h("div", { class: "run-setting-line" },
              isChanged ? h("span", { class: "badge badge-info" }, "changed") : null,
              h("span", { class: "mono" }, settingValue(value))),
            isChanged && meta && defaultOf(key, meta) !== undefined
              ? h("div", { class: "run-setting-default" }, "default ", h("span", { class: "mono" }, settingValue(defaultOf(key, meta)))) : null));
      });
      columns.appendChild(h("div", { class: "card card-flush run-settings-card" },
        h("div", { class: "run-settings-head" }, h("h3", null, section ? section.title : SECTION_TITLES[sectionName] || sectionName)),
        h("table", { class: "table run-settings-table" }, h("tbody", null, rows))));
    }
    panel.appendChild(columns);
  }

  // ---- Log ------------------------------------------------------------------------------

  function renderLog(state, panel) {
    const pre = h("pre", { class: "log run-log", tabindex: "0", "aria-label": "Console log" }, "Loading the log…");
    const note = h("span", { class: "dash-live" });
    panel.append(h("div", { class: "run-toolbar" }, h("span", { class: "muted" }, "The last 400 lines of the run's console"), note), pre);
    let first = true;
    const stop = App.every(3000, async () => {
      const live = isLive(state.detail.run);
      note.replaceChildren(...(live ? [h("span", { class: "dot dot-live" }), "Following the log"] : ["Not running"]));
      if (!first && !live) return;
      const res = await App.get(`/api/runs/${enc(state.name)}/log?tail=400`);
      const atBottom = first || pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 16;
      pre.textContent = res.lines.length ? res.lines.join("\n") : "The log is empty.";
      if (atBottom) pre.scrollTop = pre.scrollHeight;
      first = false;
    });
    state.tabStops.push(stop);
  }

  // ---- Checkpoints ----------------------------------------------------------------------

  function renderCheckpoints(state, panel) {
    const d = state.detail;
    const games = gamesMode(state);
    const evals = state.records.filter((r) => r.kind === "eval" && r.opponent === "greedy-leave");
    const evalAtStep = (step) => evals.filter((r) => r.step === step).pop();
    const rows = [];
    if (d.best) rows.push({ file: "best.pt", step: d.best.step, model_id: d.best.model_id, size_bytes: d.best.size_bytes, modified: d.best.modified, best: true });
    for (const c of d.checkpoints || []) rows.push(c);
    const settings = (d.config || {})[games ? "offline" : "train"] || {};
    if (!rows.length) {
      panel.appendChild(h("div", { class: "card" }, h("p", { class: "empty" }, h("strong", null, "No checkpoints yet."), games
        ? `The first is written ${settings.checkpoint_minutes || "a few"} minutes into training (offline.checkpoint_minutes), and best.pt after the first match against the static evaluator.`
        : `The first is written ${settings.checkpoint_minutes || 10} minutes into training, and best.pt after the first evaluation.`)));
      return;
    }
    const link = (href, text) => h("a", { class: "button button-small", href }, text);
    panel.appendChild(h("div", { class: "card card-flush" }, h("table", { class: "table run-ckpt-table" },
      h("thead", null, h("tr", null, h("th", null, "Checkpoint"), h("th", { class: "num" }, "Step"),
        h("th", { class: "num" }, games ? "Against the static evaluator" : "Against greedy + leave"), h("th", { class: "num" }, "Size"), h("th", null, "Saved"), h("th", null, ""))),
      h("tbody", null, rows.map((c) => {
        const ev = evalAtStep(c.step);
        const [spread, ci] = ev ? evalSpread(ev) : [null, null];
        const spec = encodeURIComponent("model:" + c.model_id);
        const result = !ev ? h("span", { class: "muted" }, games ? "no match at this step" : "not evaluated")
          : games ? (typeof ev.win_rate === "number" ? `${K.winText(ev.win_rate, ev.win_ci)} of games` : "–") : fmt.spread(spread, ci);
        return h("tr", { class: c.best ? "run-ckpt-best" : null },
          h("td", null, h("span", { class: "mono" }, c.file), c.best ? h("span", { class: "badge badge-good run-ckpt-badge" }, "Best") : null),
          h("td", { class: "num" }, fmt.int(c.step)),
          h("td", { class: "num" }, result),
          h("td", { class: "num" }, c.size_bytes ? fmt.bytes(c.size_bytes) : "–"),
          h("td", null, c.modified ? fmt.datetime(c.modified) : "–"),
          h("td", { class: "run-ckpt-actions" },
            link(`#/play?opponent=${spec}`, "Play against it"),
            link(`#/watch?a=${spec}`, "Watch it play"),
            link(`#/arena?a=${spec}`, "Run a match")));
      })))));
    if (games) {
      panel.appendChild(h("p", { class: "dash-caption" }, (settings.keep_checkpoints ? `The newest ${settings.keep_checkpoints} step checkpoints are kept (offline.keep_checkpoints). ` : "") +
        "best.pt is the network with the highest win rate against the static evaluator so far, kept separately. Each saved network is the running average of the weights, the one that plays."));
      return;
    }
    const keep = settings.keep_checkpoints || 5;
    panel.appendChild(h("p", { class: "dash-caption" }, `The newest ${keep} step checkpoints are kept (train.keep_checkpoints). ` +
      "best.pt is the network that won its last gate against the previous best, kept separately."));
  }

  // ---- Actions --------------------------------------------------------------------------

  async function stopRun(state, els, force) {
    const run = state.detail.run;
    const queued = run.status === "queued";
    const games = gamesMode(state);
    const minutes = (((state.detail.config || {})[games ? "offline" : "train"] || {}).checkpoint_minutes) || 10;
    const ok = await App.confirm(force ? {
      title: `Force stop ${run.name}?`,
      body: `This ends the process at once. Anything since the last checkpoint, up to ${fmt.int(minutes)} minutes of training, is lost.`,
      confirmLabel: "Force stop", danger: true,
    } : queued ? {
      title: `Take ${run.name} out of the queue?`, body: "It will not start. You can start it later with Resume once it has a checkpoint, or start a new run.",
      confirmLabel: "Remove from queue",
    } : {
      title: `Stop ${run.name}?`,
      body: games
        ? "The run finishes its current step, checks its held-out games, plays a match against the static evaluator and saves a checkpoint, then stops. You can resume it later from that checkpoint."
        : "The run finishes its current step, evaluates and saves a checkpoint, then stops. You can resume it later from that checkpoint.",
      confirmLabel: "Stop run",
    });
    if (!ok) return;
    try {
      await App.post(`/api/runs/${enc(run.name)}/stop`, force ? { force: true } : {});
      App.toast(force ? `${run.name} was stopped.` : queued ? `${run.name} left the queue.` : `Stopping ${run.name}. It saves a checkpoint first.`, "good");
      await poll(state, els);
    } catch (error) {
      App.toast(String(error.message || error), "critical");
    }
  }

  /** Rename the run, then show the same tab under its new name. */
  function renameRun(state) {
    const run = state.detail.run;
    K.renameDialog({
      name: run.name, run,
      onRenamed: (to) => {
        state.missing = true;  // the old name is gone: stop asking for it
        App.go(`#/run/${enc(to)}` + (state.tab === "progress" ? "" : `?tab=${state.tab}`));
      },
    });
  }

  async function deleteRun(state) {
    const run = state.detail.run;
    const n = (state.detail.checkpoints || []).length + (state.detail.best ? 1 : 0);
    const ok = await App.confirm({
      title: `Delete ${run.name}?`,
      body: `This deletes the folder runs/${run.name}: ${n} checkpoint${n === 1 ? "" : "s"}, the metrics and the log. It cannot be undone.`,
      confirmLabel: "Delete run", danger: true,
    });
    if (!ok) return;
    const back = gamesMode(state) ? "#/games" : "#/training";
    try {
      await App.del(`/api/runs/${enc(run.name)}`);
      App.toast(`${run.name} was deleted.`, "good");
      App.go(back);
    } catch (error) {
      App.toast(String(error.message || error), "critical");
    }
  }

  function parseOverrides(text, schema) {
    const out = {};
    const lines = text.split("\n").map((l) => l.trim()).filter((l) => l && !l.startsWith("#"));
    for (const line of lines) {
      const m = line.match(/^([a-z_]+\.[a-z_0-9]+)\s*=\s*(.+)$/i);
      if (!m) throw new Error(`"${line}" is not in the form section.key = value.`);
      const [, key, raw] = m;
      let value = raw.trim();
      if (/^(true|false)$/i.test(value)) value = value.toLowerCase() === "true";
      else if (/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(value)) value = Number(value);
      else value = value.replace(/^["']|["']$/g, "");
      if (schema) {
        const field = schemaField(schema, key);
        if (!field) throw new Error(`${key} is not a setting.`);
        if (Array.isArray(field.modes) && !field.modes.includes("selfplay")) throw new Error(`${key} is a setting for training from games, not for a self-play run.`);
        if (field.locked_on_resume) throw new Error(`${key} cannot be changed when resuming.`);
      } else if (key.startsWith("model.")) throw new Error(`${key} cannot be changed when resuming.`);
      out[key] = value;
    }
    return out;
  }

  /** Resume for a run that trains from games: a time limit only, with the settings it started with. */
  async function openResumeGames(state, els) {
    const schema = await loadSchema(state);
    const run = state.detail.run;
    const hours = h("input", { class: "input", type: "number", min: "0", step: "0.5", id: "resume-hours",
      value: String((run.limit && run.limit.hours) || 0) });
    const error = h("p", { class: "dash-error", role: "alert" });
    error.hidden = true;
    let close = null;
    const left = typeof run.pass === "number" && typeof run.passes === "number"
      ? ` It has done ${fmt.fixed(run.pass, 1)} of its ${String(Number(run.passes.toPrecision(4)))} passes.` : "";
    const submit = h("button", { class: "button button-primary", type: "submit" }, icon(ICON.play), "Resume");
    const form = h("form", { class: "stack", novalidate: "", onsubmit: async (event) => {
      event.preventDefault();
      error.hidden = true;
      const value = Number(hours.value);
      if (hours.value.trim() === "" || !Number.isFinite(value) || value < 0) {
        error.textContent = "Hours must be a number, 0 or more.";
        error.hidden = false;
        return;
      }
      submit.disabled = true;
      try {
        await App.post(`/api/runs/${enc(run.name)}/resume`, { hours: value });
        close();
        App.toast(`${run.name} resumed` + (value ? ` for ${fmt.duration(value)}.` : "."), "good");
        await poll(state, els);
      } catch (problem) {
        const text = String(problem.message || problem).trim();
        error.textContent = text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".");
        error.hidden = false;
        submit.disabled = false;
      }
    } },
    h("p", { class: "secondary" }, `The run continues from its newest checkpoint, with the settings and the game set it started with.${left}`),
    h("label", { class: "field", for: "resume-hours" }, h("span", { class: "field-label" }, "Hours"), hours,
      h("span", { class: "field-help" }, "0 means no time limit: it stops when its passes are done.")),
    schema ? K.lockedNote(K.indexSchema(schema, "games").sections, "Fixed for the life of the run") : null,
    error,
    h("div", { class: "modal-actions" }, h("button", { class: "button", type: "button", onclick: () => close() }, "Cancel"), submit));
    close = App.modal({ title: `Resume ${run.name}`, content: form });
    App.onCleanup(() => { if (form.isConnected) close(); });
    hours.focus();
  }

  async function openResume(state, els) {
    if (gamesMode(state)) { openResumeGames(state, els); return; }
    const schema = await loadSchema(state);
    const run = state.detail.run;
    const config = state.detail.config || {};
    const hours = h("input", { class: "input", type: "number", min: "0", step: "0.5", id: "resume-hours",
      value: String((config.run && config.run.hours) || (run.limit && run.limit.hours) || 8) });
    const games = h("input", { class: "input", type: "number", min: "0", step: "10000", id: "resume-games", value: "0" });
    const overrides = h("textarea", { class: "input mono", rows: "4", id: "resume-overrides", spellcheck: "false",
      placeholder: "train.learning_rate = 0.0003\ntrain.eval_pairs = 500" });
    const error = h("p", { class: "dash-error", role: "alert" });
    error.hidden = true;
    const field = (label, input, help) => h("label", { class: "field", for: input.id },
      h("span", { class: "field-label" }, label), input, help ? h("span", { class: "field-help" }, help) : null);
    let close = null;
    const submit = h("button", { class: "button button-primary", type: "submit" }, icon(ICON.play), "Resume");
    const form = h("form", { class: "stack", novalidate: "", onsubmit: async (event) => {
      event.preventDefault();
      error.hidden = true;
      let parsed;
      try {
        const hoursValue = Number(hours.value);
        const gamesValue = Number(games.value);
        if (hours.value.trim() === "" || !Number.isFinite(hoursValue) || hoursValue < 0) throw new Error("Hours must be a number, 0 or more.");
        if (games.value.trim() !== "" && (!Number.isInteger(gamesValue) || gamesValue < 0)) throw new Error("The game limit must be a whole number, 0 or more.");
        parsed = parseOverrides(overrides.value, schema);
      } catch (problem) {
        error.textContent = problem.message;
        error.hidden = false;
        return;
      }
      const body = { hours: Number(hours.value) || 0, max_games: Number(games.value) || 0 };
      if (Object.keys(parsed).length) body.overrides = parsed;
      submit.disabled = true;
      try {
        await App.post(`/api/runs/${enc(run.name)}/resume`, body);
        close();
        App.toast(`${run.name} resumed` + (body.hours ? ` for ${fmt.duration(body.hours)}.` : "."), "good");
        await poll(state, els);
      } catch (problem) {
        error.textContent = String(problem.message || problem);
        error.hidden = false;
        submit.disabled = false;
      }
    } },
    h("p", { class: "secondary" }, "The run continues from its newest checkpoint with the settings saved in it. " +
      "Settings you name below change; the network's size cannot."),
    h("div", { class: "grid-2" }, field("Hours", hours, "0 means no time limit"), field("Game limit", games, "0 means no limit")),
    field("Setting overrides (optional)", overrides, "One per line, as section.key = value."),
    error,
    h("div", { class: "modal-actions" }, h("button", { class: "button", type: "button", onclick: () => close() }, "Cancel"), submit));
    close = App.modal({ title: `Resume ${run.name}`, content: form });
    // The dialog lives outside the page, so close it if the page changes underneath it.
    App.onCleanup(() => { if (form.isConnected) close(); });
    hours.focus();
  }
})();
