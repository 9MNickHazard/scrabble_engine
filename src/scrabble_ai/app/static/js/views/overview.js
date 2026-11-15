/* Overview: #/overview, the landing page.
 *
 * What is training now (a card per active run with progress, speed, the latest result against
 * greedy + leave and a sparkline of that result), or when nothing is, the latest run: how it
 * went in a sentence, its strength chart with the interval, a few figures and a way to all its
 * charts. Then the queue, the GPU, the strongest network of each run, recent runs (each with its
 * verdict) and recent matches. Before anything has been trained it shows a short welcome that
 * points to the Training page.
 */
(function () {
  "use strict";

  const { h, fmt } = App;
  const kit = () => window.LabKit;  // views/training.js loads after this file

  const ICON_OVERVIEW = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6">' +
    '<rect x="3" y="3" width="6" height="6" rx="1.5"/><rect x="11" y="3" width="6" height="6" rx="1.5"/>' +
    '<rect x="3" y="11" width="6" height="6" rx="1.5"/><rect x="11" y="11" width="6" height="6" rx="1.5"/></svg>';
  const ICON_ARROW = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M6 3l5 5-5 5"/></svg>';

  const STATUS = {
    running: ["Running", "good"], stopping: ["Stopping", "warning"], queued: ["Queued", "info"],
    finished: ["Finished", "plain"], stopped: ["Stopped", "plain"], failed: ["Failed", "critical"],
    cancelled: ["Cancelled", "plain"],
  };
  const PHASE = { warmstart: "warm start", pretraining: "pretraining", selfplay: "self-play" };
  const isLive = (run) => run.status === "running" || run.status === "stopping";
  /** A run that trains from saved games: judged by its win rate against the static evaluator. */
  const isGames = (run) => !!run && run.mode === "games";
  /** "8 h" rather than "8 h 00 m" for whole hours. */
  const hoursText = (hours) => fmt.duration(hours).replace(/ 00 m$/, "");

  /** A win rate and its interval as a figure: "53.4%" and, smaller, "± 1.3". */
  function winNode(rate, ci, cls) {
    if (typeof rate !== "number") return h("span", { class: cls }, "–");
    return h("span", { class: cls }, fmt.pct(rate), typeof ci === "number" ? h("span", { class: "stat-ci" }, " ± " + (100 * ci).toFixed(1)) : null);
  }

  /** In a few words, what a win rate's interval shows. */
  function winVerdict(rate, ci) {
    if (typeof rate !== "number" || typeof ci !== "number") return "";
    if (rate - ci > 0.5) return "interval above 50%";
    if (rate + ci < 0.5) return "interval below 50%";
    return "interval includes 50%";
  }

  /** The matches of a games run from its `trend`: [{step, games, win_rate, win_ci}], oldest first. */
  function matchesOf(run) {
    const t = run.trend || {};
    const rates = Array.isArray(t.win_rate) ? t.win_rate : [];
    return rates.map((rate, i) => ({ step: Array.isArray(t.step) ? t.step[i] : null, games: Array.isArray(t.games) ? t.games[i] : null,
      win_rate: rate, win_ci: Array.isArray(t.win_ci) ? t.win_ci[i] : null })).filter((m) => typeof m.win_rate === "number");
  }

  App.registerView("overview", { title: "Overview", nav: true, order: 10, icon: ICON_OVERVIEW, mount });

  function mount(container) {
    const state = { runs: null, models: null, matches: null, metrics: {}, signatures: {}, headlines: {} };
    const els = {
      head: h("div"),
      top: h("div", { class: "ov-top" }),
      active: h("section", { class: "ov-active", "aria-label": "Training now" }),
      gpu: h("section", { class: "card ov-gpu", "aria-label": "GPU" }),
      bottom: h("div", { class: "ov-bottom" }),
      strongest: h("section", { class: "card ov-list-card", "aria-label": "Strongest networks" }),
      runs: h("section", { class: "card ov-list-card", "aria-label": "Recent runs" }),
      matches: h("section", { class: "card ov-list-card", "aria-label": "Recent matches" }),
      welcome: h("div"),
    };
    els.top.append(els.active, els.gpu);
    els.bottom.append(els.strongest, els.runs, els.matches);
    container.appendChild(h("div", { class: "overview" }, els.head, els.welcome, els.top, els.bottom));
    App.every(5000, () => refresh(state, els));
    const off = App.on("status", () => { if (state.runs) renderGpu(els); });
    App.onCleanup(off);
  }

  async function refresh(state, els) {
    const [runs, models, matches] = await Promise.allSettled([App.get("/api/runs"), App.get("/api/models"), App.get("/api/matches")]);
    if (runs.status === "rejected") {
      els.head.replaceChildren(h("div", { class: "notice notice-critical" }, h("strong", null, "Cannot reach the app server. "),
        String(runs.reason && runs.reason.message || runs.reason)));
      return;
    }
    state.runs = runs.value.runs || [];
    state.models = models.status === "fulfilled" ? models.value : null;
    state.matches = matches.status === "fulfilled" ? matches.value.matches || [] : null;
    const active = state.runs.filter(isLive);
    // A run trained from games carries its matches in `trend`; only self-play runs need their metrics.
    await Promise.all(active.filter((run) => !isGames(run)).map((run) => loadEvals(state, run.name)));
    const latest = active.length ? null : latestRun(state.runs);
    if (latest) {
      if (!isGames(latest)) await loadEvals(state, latest.name);
      // An older server sends no verdict: the report's headline says how the run went instead.
      const key = latest.name + "@" + latest.updated;
      if (latest.health === undefined && state.headlines.key !== key) {
        try {
          const report = await App.get(`/api/runs/${encodeURIComponent(latest.name)}/report`);
          state.headlines = { key, text: report.headline || "" };
        } catch (error) {
          state.headlines = { key, text: "" };
        }
      }
    }
    render(state, els);
  }

  /** The run updated most recently, of those that are not waiting in the queue. */
  function latestRun(runs) {
    return runs.filter((r) => r.status !== "queued").sort((a, b) => (b.updated || 0) - (a.updated || 0))[0] || null;
  }

  /** Keep each active run's evaluations against greedy + leave for its sparkline. */
  async function loadEvals(state, name) {
    const cache = state.metrics[name] || (state.metrics[name] = { next: 0, gl: [] });
    try {
      const res = await App.get(`/api/runs/${encodeURIComponent(name)}/metrics?since=${cache.next}`);
      for (const r of res.records) {
        if (r.kind !== "eval" || r.opponent !== "greedy-leave") continue;
        const spread = r.spread !== undefined ? r.spread : r["spread_vs_greedy-leave"];
        const ci = r.spread_ci !== undefined ? r.spread_ci : r["spread_ci_vs_greedy-leave"];
        if (typeof spread === "number") cache.gl.push({ games: r.games, spread, ci: typeof ci === "number" ? ci : 0 });
      }
      cache.next = res.next;
    } catch (error) {
      console.warn(error);
    }
  }

  /** Rebuild a section only when what it shows has changed. */
  function section(state, key, el, data, build) {
    const signature = JSON.stringify(data);
    if (state.signatures[key] === signature) return;
    state.signatures[key] = signature;
    build(el);
  }

  function spreadNode(spread, ci, cls) {
    if (spread === null || spread === undefined) return h("span", { class: cls }, "–");
    return h("span", { class: cls }, fmt.signed(spread), ci === null || ci === undefined ? null
      : h("span", { class: "stat-ci" }, " ± " + Number(ci).toFixed(1)));
  }

  function verdict(spread, ci) {
    if (spread === null || spread === undefined || ci === null || ci === undefined) return "";
    if (spread - ci > 0) return "significant";
    if (spread + ci < 0) return "significantly behind";
    return "not yet significant";
  }

  function statusBadge(status, extra) {
    const [label, tone] = STATUS[status] || [status, "plain"];
    return h("span", { class: "badge badge-" + tone }, extra ? `${label} · ${extra}` : label);
  }

  /** The names of the runs that train from games, so their models are judged by wins. */
  function gamesRunNames(state) {
    return new Set((state.runs || []).filter(isGames).map((r) => r.name));
  }

  /** Each self-play run's strongest model by spread against greedy + leave, best first. */
  function strongest(state) {
    if (!state.models) return [];
    const games = gamesRunNames(state);
    const byRun = new Map();
    for (const m of state.models.models || []) {
      const e = m.eval && m.eval["greedy-leave"];
      if (!m.compatible || !e || typeof e.spread !== "number" || games.has(m.run)) continue;
      const key = m.run || m.id;
      const current = byRun.get(key);
      if (!current || e.spread > current.eval["greedy-leave"].spread) byRun.set(key, m);
    }
    return Array.from(byRun.values()).sort((a, b) => b.eval["greedy-leave"].spread - a.eval["greedy-leave"].spread).slice(0, 5);
  }

  /** Each games run's strongest model by win rate against greedy + leave, with its interval
   *  (from the run's trend), best first: [{model, run, win_rate, win_ci}]. */
  function strongestByWins(state) {
    if (!state.models || !window.LabKit) return [];
    const runs = new Map((state.runs || []).filter(isGames).map((r) => [r.name, r]));
    const byRun = new Map();
    for (const m of state.models.models || []) {
      const run = runs.get(m.run);
      if (!run || !m.compatible) continue;
      const w = window.LabKit.modelWin(m, run);
      if (!w) continue;
      const current = byRun.get(m.run);
      if (!current || w.win_rate > current.win_rate) byRun.set(m.run, Object.assign({ model: m, run }, w));
    }
    return Array.from(byRun.values()).sort((a, b) => b.win_rate - a.win_rate).slice(0, 5);
  }

  // ---- Render ---------------------------------------------------------------------------

  function render(state, els) {
    const runs = state.runs;
    const active = runs.filter(isLive);
    const queued = runs.filter((r) => r.status === "queued");
    const top = strongest(state);
    const topWins = strongestByWins(state);
    const best = top[0];
    const bestWins = topWins[0];

    let lede;
    if (!runs.length) lede = "Train a network, watch it learn, then play against it.";
    else {
      lede = active.length ? `${active.length} run${active.length === 1 ? " is" : "s are"} training` : "Nothing is training right now";
      lede += queued.length ? `, ${queued.length} queued.` : ".";
      if (best) {
        const e = best.eval["greedy-leave"];
        const ci = typeof e.ci === "number" ? e.ci : 0;
        if (e.spread - ci > 0) lede += ` The strongest network so far is ${best.name}, ${fmt.spread(e.spread, e.ci)} points a game against greedy + leave.`;
        else if (e.spread + ci < 0) lede += ` No network has beaten greedy + leave yet: the closest, ${best.name}, is at ${fmt.spread(e.spread, e.ci)} points a game.`;
        else lede += ` No network has been shown to beat greedy + leave yet: the strongest, ${best.name}, is level with it at ${fmt.spread(e.spread, e.ci)}.`;
      }
      if (bestWins) {
        // Judged by wins: a win only when the whole interval is above one half.
        const w = window.LabKit.winText(bestWins.win_rate, bestWins.win_ci);
        const name = bestWins.model.name;
        if (typeof bestWins.win_ci === "number" && bestWins.win_rate - bestWins.win_ci > 0.5) {
          lede += ` Of the networks trained from games, ${name} wins the most: ${w} of games against greedy + leave.`;
        } else {
          lede += ` No network trained from games has been shown to win more than half its games against greedy + leave yet: the best, ${name}, wins ${w}.`;
        }
      }
    }
    section(state, "head", els.head, [lede, !runs.length], (el) => el.replaceChildren(h("div", { class: "page-head" },
      h("div", null, h("h1", null, "Overview"), h("p", { class: "lede" }, lede)),
      runs.length ? h("div", { class: "page-actions" }, h("a", { class: "button button-primary", href: "#/training" }, "New training run")) : null)));

    if (!runs.length) {
      els.top.classList.add("ov-top-welcome");
      section(state, "welcome", els.welcome, true, (el) => el.replaceChildren(welcome()));
      els.bottom.hidden = !(state.matches && state.matches.length);
      els.strongest.hidden = true;
      els.runs.hidden = true;
      els.active.hidden = true;
      renderGpu(els);
      renderMatches(state, els);
      return;
    }
    els.top.classList.remove("ov-top-welcome");
    els.welcome.replaceChildren();
    state.signatures.welcome = null;
    els.bottom.hidden = false;
    els.strongest.hidden = false;
    els.runs.hidden = false;
    els.active.hidden = false;

    renderActive(state, els, active, queued, runs);
    renderGpu(els);
    section(state, "strongest", els.strongest, [top.map((m) => [m.id, m.eval, m.games]), topWins.map((w) => [w.model.id, w.win_rate, w.win_ci])],
      (el) => renderStrongest(el, top, topWins));
    const recent = runs.filter((r) => !isLive(r) && r.status !== "queued")
      .sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, 5);
    section(state, "runs", els.runs, recent.map((r) => [r.name, r.status, r.latest_eval, r.health, Math.floor(Date.now() / 60000)]),
      (el) => renderRuns(el, recent));
    renderMatches(state, els);
  }

  function renderActive(state, els, active, queued, runs) {
    const nodes = active.map((run) => activeCard(state, run));
    if (!active.length) {
      const last = latestRun(runs);
      if (last) {
        // Rebuilt only when what it shows changes, so its chart keeps its hover between refreshes.
        const evals = (state.metrics[last.name] || { gl: [] }).gl;
        const sig = JSON.stringify([last.name, last.status, last.updated, last.health, last.best_eval, evals.length,
          state.headlines.text, Math.floor(Date.now() / 60000)]);
        if (state.latestSig !== sig || !state.latestCard) {
          if (state.latestChart) state.latestChart.destroy();
          const built = latestCard(state, last);
          state.latestCard = built.card;
          state.latestChart = built.chart;
          state.latestSig = sig;
        }
        nodes.push(state.latestCard);
      } else {
        nodes.push(h("div", { class: "card ov-idle" },
          h("div", null, h("h2", null, "No training running"),
            h("p", { class: "secondary" }, "Start a run on the Training page; it appears here while it trains.")),
          h("div", { class: "page-actions" }, h("a", { class: "button button-primary", href: "#/training" }, "Start a run"))));
      }
    } else {
      state.latestSig = null;
      state.latestCard = null;
    }
    if (queued.length) {
      nodes.push(h("div", { class: "ov-queue" },
        h("span", { class: "badge badge-info" }, "Queued"),
        h("span", null, queued.map((r, i) => [i ? ", " : "", h("a", { href: `#/run/${encodeURIComponent(r.name)}` }, r.name),
          h("span", { class: "muted" }, ` (${r.network}${r.limit && r.limit.hours ? `, ${hoursText(r.limit.hours)}` : ""})`)])),
        h("span", { class: "muted" }, active.length ? "starts when the current run ends" : "starts shortly")));
    }
    if (nodes.length !== els.active.childNodes.length || nodes.some((node, i) => els.active.childNodes[i] !== node)) {
      els.active.replaceChildren(...nodes);
    }
    els.active.classList.toggle("ov-active-multi", active.length > 1);
  }

  /** The latest run when it trained from games: judged by its win rate against greedy + leave. */
  function latestCardGames(state, run) {
    const K = kit();
    const href = `#/run/${encodeURIComponent(run.name)}`;
    const matches = matchesOf(run);
    const le = run.latest_eval && run.latest_eval["greedy-leave"];
    const best = run.best_eval || null;
    const note = run.health ? run.health.note : null;
    const when = { finished: "finished", stopped: "stopped", failed: "failed" }[run.status] || "updated";
    const slot = h("div", { class: "ov-latest-chart" });
    const pass = typeof run.pass === "number" ? `pass ${fmt.fixed(run.pass, 1)}${typeof run.passes === "number" ? " of " + Number(run.passes.toPrecision(4)) : ""}` : null;
    const card = h("section", { class: "card ov-latest", "aria-label": "Latest run" },
      h("div", { class: "ov-latest-head" },
        h("div", { class: "ov-latest-titles" },
          h("div", { class: "ov-latest-eyebrow" }, `Latest run · trained from games · ${when} ${fmt.ago(run.updated)} · nothing is training now`),
          h("div", { class: "ov-run-head" },
            h("h2", null, h("a", { class: "ov-latest-name", href }, run.name)),
            statusBadge(run.status), K ? K.healthChip(run.health, null, true) : null,
            h("span", { class: "muted" }, K ? K.networkLabel(run) : run.network))),
        h("div", { class: "page-actions" },
          h("a", { class: "button button-primary", href }, K ? K.icon("chart", 15) : null, "Open all charts"),
          h("a", { class: "button", href: "#/games" }, "Train from games"))),
      note ? h("p", { class: "ov-latest-note" + (run.health && (run.health.state === "worse" || run.health.state === "behind") ? " is-bad" : "") },
        fmt.prose(note)) : run.health === null ? h("p", { class: "ov-latest-note" }, "It never played a match against greedy + leave, so there is no verdict.") : null,
      h("div", { class: "ov-latest-body" },
        h("dl", { class: "ov-latest-figures" },
          figure("Latest match", winNode(le ? le.win_rate : null, le ? le.win_ci : null, "ov-latest-value"),
            le ? `of games · ${winVerdict(le.win_rate, le.win_ci) || "no interval"}` : "no match"),
          figure("Best match", winNode(best ? best.win_rate : null, best ? best.win_ci : null, "ov-latest-value"),
            best ? `at step ${fmt.int(best.step)}` : "none yet"),
          figure("Positions trained", h("span", { class: "ov-latest-value" }, fmt.compact(run.positions)),
            [`in ${fmt.duration(run.hours)}`, pass].filter(Boolean).join(" · ")),
          figure("Game set", h("span", { class: "ov-latest-value" }, run.game_set || "–"),
            run.parameters ? `${fmt.compact(run.parameters)} parameters` : null)),
        slot));
    const chart = Charts.line(slot, {
      title: "Win rate against greedy + leave, at each match",
      series: [{ name: "Win rate", points: matches.map((m) => [m.step, m.win_rate]),
        band: matches.filter((m) => typeof m.win_ci === "number").map((m) => [m.step, m.win_rate - m.win_ci, m.win_rate + m.win_ci]),
        color: "var(--series-1)" }],
      x: { label: "training steps", format: (v) => fmt.int(v), tooltipFormat: (v) => "Step " + fmt.int(v), tickFormat: "auto" },
      y: { label: "games won · 95% interval shaded", format: (v) => fmt.pct(v, 1), tickFormat: (v) => trimPct(100 * v),
        ciFormat: (v) => (100 * Math.abs(v)).toFixed(1) },
      referenceY: [{ y: 0.5, label: "50%" }], height: 172, legend: false,
      empty: run.status === "failed" ? "It failed before its first match." : "No match was recorded.",
    });
    return { card, chart };
  }

  /** The latest run, when nothing is training: how it went, its strength chart, a few figures. */
  function latestCard(state, run) {
    if (isGames(run)) return latestCardGames(state, run);
    const K = kit();
    const href = `#/run/${encodeURIComponent(run.name)}`;
    const evals = (state.metrics[run.name] || { gl: [] }).gl;
    const le = run.latest_eval && run.latest_eval["greedy-leave"];
    let best = run.best_eval || null;
    if (!best && evals.length) {
      const top = evals.reduce((a, e) => (e.spread > a.spread ? e : a), evals[0]);
      best = { games: top.games, spread: top.spread, ci: top.ci };
    }
    const note = run.health ? run.health.note : run.health === undefined ? state.headlines.text : null;
    const when = { finished: "finished", stopped: "stopped", failed: "failed" }[run.status] || "updated";
    const slot = h("div", { class: "ov-latest-chart" });
    const card = h("section", { class: "card ov-latest", "aria-label": "Latest run" },
      h("div", { class: "ov-latest-head" },
        h("div", { class: "ov-latest-titles" },
          h("div", { class: "ov-latest-eyebrow" }, `Latest run · ${when} ${fmt.ago(run.updated)} · nothing is training now`),
          h("div", { class: "ov-run-head" },
            h("h2", null, h("a", { class: "ov-latest-name", href }, run.name)),
            statusBadge(run.status), K ? K.healthChip(run.health) : null,
            h("span", { class: "muted" }, run.network))),
        h("div", { class: "page-actions" },
          h("a", { class: "button button-primary", href }, K ? K.icon("chart", 15) : null, "Open all charts"),
          h("a", { class: "button", href: "#/training" }, "Start a run"))),
      note ? h("p", { class: "ov-latest-note" + (run.health && (run.health.state === "worse" || run.health.state === "behind") ? " is-bad" : "") },
        fmt.prose(note)) : run.health === null ? h("p", { class: "ov-latest-note" }, "It was never evaluated, so there is no verdict.") : null,
      h("div", { class: "ov-latest-body" },
        h("dl", { class: "ov-latest-figures" },
          figure("Latest evaluation", spreadNode(le ? le.spread : null, le ? le.ci : null, "ov-latest-value"),
            le ? verdict(le.spread, le.ci) : "not evaluated"),
          figure("Best evaluation", spreadNode(best ? best.spread : null, best ? best.ci : null, "ov-latest-value"),
            best ? `at ${fmt.compact(best.games)} games` : "none yet"),
          figure("Trained", h("span", { class: "ov-latest-value" }, `${fmt.compact(run.games)} games`), `in ${fmt.duration(run.hours)}`),
          figure("Network", h("span", { class: "ov-latest-value" }, run.network || "–"),
            run.parameters ? `${fmt.compact(run.parameters)} parameters` : null)),
        slot));
    const chart = Charts.line(slot, {
      title: "Strength against greedy + leave, at each evaluation",
      series: [{ name: "Greedy + leave", points: evals.map((e) => [e.games, e.spread]),
        band: evals.map((e) => [e.games, e.spread - e.ci, e.spread + e.ci]), color: "var(--series-1)" }],
      x: { label: "games", format: (v) => fmt.int(v), tooltipFormat: (v) => fmt.int(v) + " games", tickFormat: "auto" },
      y: { label: "points per game · 95% interval shaded", format: (v) => fmt.signed(v),
        tickFormat: (v) => (v === 0 ? "0" : fmt.signed(v, Number.isInteger(v) ? 0 : 1)) },
      referenceY: [{ y: 0, label: "even" }], height: 172, legend: false,
      empty: run.status === "failed" ? "It failed before its first evaluation." : "No evaluation was recorded.",
    });
    return { card, chart };
  }

  /** A run training from games right now: how far through its passes, and its latest win rate. */
  function activeCardGames(state, run) {
    const K = kit();
    const href = `#/run/${encodeURIComponent(run.name)}`;
    const le = run.latest_eval && run.latest_eval["greedy-leave"];
    const matches = matchesOf(run);
    let progress = null;
    if (run.progress !== null && run.progress !== undefined && typeof run.pass === "number") {
      const passes = typeof run.passes === "number" ? Number(run.passes.toPrecision(4)) : null;
      progress = { value: run.progress, left: `pass ${fmt.fixed(run.pass, 1)}${passes !== null ? " of " + passes : ""}`,
        right: run.total_steps ? `step ${fmt.int(run.step)} of ${fmt.int(run.total_steps)}` : `${Math.round(run.progress * 100)}%` };
    }
    const spark = h("div", { class: "ov-spark" });
    const card = h("a", { class: "card ov-run", href, "aria-label": `${run.name}, open the run page` },
      h("div", { class: "ov-run-main" },
        h("div", { class: "ov-run-head" },
          h("h2", null, run.name),
          statusBadge(run.status),
          K ? K.healthChip(run.health, null, true) : null,
          h("span", { class: "muted" }, K ? K.networkLabel(run) : run.network)),
        progress ? h("div", { class: "ov-progress" },
          h("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
            "aria-valuenow": String(Math.round(progress.value * 100)) }, h("span", { style: { width: `${(progress.value * 100).toFixed(1)}%` } })),
          h("div", { class: "ov-progress-text" }, h("span", null, progress.left), h("span", { class: "muted" }, progress.right)))
          : h("div", { class: "ov-progress-text muted" }, "Starting: reading its game set."),
        h("dl", { class: "ov-figures" },
          figure("Positions", fmt.compact(run.positions)),
          figure("Positions per second", typeof run.positions_per_second === "number" ? fmt.int(run.positions_per_second) : "–"),
          figure("Game set", run.game_set || "–"),
          figure("Time trained", fmt.duration(run.hours)))),
      h("div", { class: "ov-run-eval" },
        h("div", { class: "stat-label" }, "Against greedy + leave"),
        winNode(le ? le.win_rate : null, le ? le.win_ci : null, "ov-eval-value"),
        h("div", { class: "stat-hint" }, le
          ? `of games · ${winVerdict(le.win_rate, le.win_ci) || "no interval"} · ${matches.length} match${matches.length === 1 ? "" : "es"}`
          : "the first match against greedy + leave is to come"),
        spark),
      h("span", { class: "ov-run-go", html: ICON_ARROW, "aria-hidden": "true" }));
    Charts.sparkline(spark, {
      points: matches.map((m) => [m.step, m.win_rate]),
      band: matches.filter((m) => typeof m.win_ci === "number").map((m) => [m.step, m.win_rate - m.win_ci, m.win_rate + m.win_ci]),
      referenceY: 0.5, width: 260, height: 56, color: "var(--series-1)", format: (v) => fmt.pct(v) + " of games",
      xFormat: (v) => "Step " + fmt.int(v), label: "Win rate against greedy + leave over the run",
    });
    return card;
  }

  function activeCard(state, run) {
    if (isGames(run)) return activeCardGames(state, run);
    const href = `#/run/${encodeURIComponent(run.name)}`;
    const le = run.latest_eval && run.latest_eval["greedy-leave"];
    const evals = (state.metrics[run.name] || { gl: [] }).gl;
    let progress = null;
    if (run.limit && run.limit.hours && run.progress !== null && run.progress !== undefined) {
      const done = run.progress * run.limit.hours;
      progress = { value: run.progress, left: `${fmt.duration(done)} of ${hoursText(run.limit.hours)}`,
        right: run.progress < 1 ? `about ${fmt.duration(run.limit.hours - done)} left` : "finishing" };
    } else if (run.limit && run.limit.max_games && run.progress !== null && run.progress !== undefined) {
      progress = { value: run.progress, left: `${fmt.compact(run.games)} of ${fmt.compact(run.limit.max_games)} games`, right: `${Math.round(run.progress * 100)}%` };
    }
    const spark = h("div", { class: "ov-spark" });
    const card = h("a", { class: "card ov-run", href, "aria-label": `${run.name}, open the run page` },
      h("div", { class: "ov-run-main" },
        h("div", { class: "ov-run-head" },
          h("h2", null, run.name),
          statusBadge(run.status, run.status === "running" && run.phase ? PHASE[run.phase] : null),
          kit() ? kit().healthChip(run.health) : null,
          h("span", { class: "muted" }, run.network)),
        progress ? h("div", { class: "ov-progress" },
          h("div", { class: "progress", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
            "aria-valuenow": String(Math.round(progress.value * 100)) }, h("span", { style: { width: `${(progress.value * 100).toFixed(1)}%` } })),
          h("div", { class: "ov-progress-text" }, h("span", null, progress.left), h("span", { class: "muted" }, progress.right)))
          : h("div", { class: "ov-progress-text muted" }, "No limit: it runs until stopped."),
        h("dl", { class: "ov-figures" },
          figure("Games", fmt.compact(run.games)),
          figure("Games per second", run.games_per_second ? fmt.fixed(run.games_per_second, 1) : "–"),
          figure("Training steps", fmt.int(run.step)),
          progress ? null : figure("Time trained", fmt.duration(run.hours)))),
      h("div", { class: "ov-run-eval" },
        h("div", { class: "stat-label" }, "Against greedy + leave"),
        spreadNode(le ? le.spread : null, le ? le.ci : null, "ov-eval-value"),
        h("div", { class: "stat-hint" }, le
          ? `${verdict(le.spread, le.ci)} · ${evals.length} evaluation${evals.length === 1 ? "" : "s"}`
          : "the first evaluation comes 20 minutes into self-play"),
        spark),
      h("span", { class: "ov-run-go", html: ICON_ARROW, "aria-hidden": "true" }));
    Charts.sparkline(spark, {
      points: evals.map((e) => [e.games, e.spread]), band: evals.map((e) => [e.games, e.spread - e.ci, e.spread + e.ci]),
      referenceY: 0, width: 260, height: 56, color: "var(--series-1)", format: (v) => fmt.signed(v),
      xFormat: (v) => fmt.compact(v) + " games", label: "Spread against greedy + leave over the run",
    });
    return card;
  }

  function figure(label, value, hint) {
    return h("div", { class: "ov-figure" }, h("dt", null, label), h("dd", null, value,
      hint ? h("span", { class: "ov-figure-hint" }, hint) : null));
  }

  function renderGpu(els) {
    const status = App.status;
    const gpu = status && status.gpu;
    const meter = (label, value, fraction) => h("div", { class: "ov-meter" },
      h("div", { class: "ov-meter-text" }, h("span", { class: "secondary" }, label), h("span", { class: "num" }, value)),
      h("div", { class: "progress", role: "meter", "aria-label": label, "aria-valuemin": "0", "aria-valuemax": "100",
        "aria-valuenow": String(Math.round(fraction * 100)) }, h("span", { style: { width: `${Math.min(100, fraction * 100).toFixed(1)}%` } })));
    const nodes = [h("div", { class: "card-head" }, h("div", { class: "card-title" }, "GPU"),
      gpu ? h("div", { class: "card-sub ov-gpu-name", title: gpu.name }, gpu.name.replace(/^NVIDIA (GeForce )?/, "")) : null)];
    if (!status) nodes.push(h("p", { class: "muted" }, "Connecting…"));
    else if (!gpu) nodes.push(h("p", { class: "muted" }, "No GPU found: nvidia-smi is not available."));
    else {
      nodes.push(meter("Utilization", `${gpu.utilization}%`, gpu.utilization / 100));
      nodes.push(meter("Memory", `${(gpu.memory_used_mb / 1024).toFixed(1)} of ${(gpu.memory_total_mb / 1024).toFixed(1)} GB`,
        gpu.memory_used_mb / gpu.memory_total_mb));
      if (gpu.temperature_c !== null && gpu.temperature_c !== undefined) {
        nodes.push(h("div", { class: "ov-meter-text ov-temp" }, h("span", { class: "secondary" }, "Temperature"), h("span", { class: "num" }, `${gpu.temperature_c} °C`)));
      }
    }
    if (status) {
      nodes.push(h("div", { class: "ov-gpu-foot" },
        h("div", null, `${status.lexicon.id.split("-")[0]} word list, ${fmt.int(status.lexicon.words)} words`),
        h("div", null, status.leaves.ok ? "Leave values fitted" : h("span", { class: "ov-warn" }, h("span", { class: "dot", style: { background: "var(--warning)" } }),
          "Leave values not fitted: run scrabble-ai fit-leaves"))));
    }
    els.gpu.replaceChildren(...nodes);
  }

  function listHead(title, sub, href, linkText) {
    return h("div", { class: "card-head" },
      h("div", null, h("div", { class: "card-title" }, title), sub ? h("div", { class: "card-sub" }, sub) : null),
      href ? h("a", { class: "ov-more", href }, linkText) : null);
  }

  function renderStrongest(el, top, topWins) {
    const slot = h("div");
    const winSlot = h("div");
    const both = top.length && topWins.length;
    App.append(App.clear(el), [listHead("Strongest networks", topWins.length && !top.length
      ? "Each run's best match against greedy + leave: share of games won, 95% interval"
      : "Each run's best evaluation against greedy + leave: points per game, 95% interval",
    "#/models", "All models"),
    top.length || !topWins.length ? slot : null,
    topWins.length ? h("div", { class: both ? "ov-strongest-wins" : null },
      both ? h("div", { class: "card-sub ov-strongest-sub" }, "Trained from games, judged by wins: share of games won, 95% interval") : null,
      winSlot) : null]);
    if (top.length || !topWins.length) {
      Charts.intervals(slot, {
        rows: top.map((m) => {
          const e = m.eval["greedy-leave"];
          return { label: m.name, sub: `${m.network} · ${fmt.compact(m.games)} games`, value: e.spread, ci: e.ci,
            href: m.run ? `#/run/${encodeURIComponent(m.run)}` : null,
            detail: `Win rate ${fmt.pct(e.win_rate)}. ${verdict(e.spread, e.ci).replace(/^./, (c) => c.toUpperCase())}.` };
        }),
        center: 0, valueLabel: "points per game",
        empty: "No network has been evaluated yet.",
      });
    }
    if (topWins.length) {
      // In percent, so the chart prints "53.4% ± 1.3" and centres on 50.
      Charts.intervals(winSlot, {
        rows: topWins.map((w) => ({ label: w.model.name, sub: `${window.LabKit.networkLabel(w.run)} · trained from games`,
          value: 100 * w.win_rate, ci: typeof w.win_ci === "number" ? 100 * w.win_ci : null,
          href: `#/run/${encodeURIComponent(w.run.name)}`,
          detail: (typeof w.win_ci === "number" ? (w.win_rate - w.win_ci > 0.5 ? "The whole interval is above 50%: it wins more than it loses."
            : w.win_rate + w.win_ci < 0.5 ? "The whole interval is below 50%: it loses more than it wins."
              : "The interval includes 50%: not shown to win more than it loses.") : "No interval recorded.") +
            " Networks trained from games are judged by wins, not points." })),
        center: 50, format: (v) => v.toFixed(1) + "%", tickFormat: (v) => trimPct(v), valueLabel: "share of games won",
      });
    }
  }

  const trimPct = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1)) + "%";

  function renderRuns(el, recent) {
    const K = kit();
    el.replaceChildren(listHead("Recent runs", null, "#/training", "All runs"),
      recent.length ? h("ul", { class: "ov-list" }, recent.map((run) => {
        if (isGames(run)) return recentGamesRow(run);
        const e = run.latest_eval && run.latest_eval["greedy-leave"];
        const chip = K ? K.healthChip(run.health, "ov-row-chip") : null;
        return h("li", null, h("a", { class: "ov-row ov-run-row", href: `#/run/${encodeURIComponent(run.name)}`,
          title: `Open ${run.name}: its charts, report, settings and log` },
          h("div", { class: "ov-row-main" },
            h("div", { class: "ov-row-title" }, h("span", { class: "ov-row-name ov-row-link" }, run.name)),
            h("div", { class: "ov-row-sub", title: `${(STATUS[run.status] || [run.status])[0]} ${fmt.datetime(run.updated)} · ${run.network} · ` +
              `${fmt.duration(run.hours)} · ${fmt.int(run.games)} games` },
            `${(STATUS[run.status] || [run.status])[0]} ${fmt.ago(run.updated)} · ${run.network} · ${fmt.compact(run.games)} games`)),
          h("div", { class: "ov-row-side" },
            spreadNode(e ? e.spread : null, e ? e.ci : null, "ov-row-value"),
            chip || h("div", { class: "ov-row-sub" }, e ? verdict(e.spread, e.ci) : run.status === "failed" ? "no result" : "not evaluated")),
          h("span", { class: "ov-row-go", html: ICON_ARROW, "aria-hidden": "true" })));
      })) : h("p", { class: "muted ov-none" }, "Finished runs appear here."));
  }

  /** A recent run that trained from games: its win rate, never its spread. */
  function recentGamesRow(run) {
    const K = kit();
    const e = run.latest_eval && run.latest_eval["greedy-leave"];
    const chip = K ? K.healthChip(run.health, "ov-row-chip", true) : null;
    const label = (STATUS[run.status] || [run.status])[0];
    const net = K ? K.networkLabel(run) : run.network;
    return h("li", null, h("a", { class: "ov-row ov-run-row", href: `#/run/${encodeURIComponent(run.name)}`,
      title: `Open ${run.name}: its charts, report, settings and log` },
    h("div", { class: "ov-row-main" },
      h("div", { class: "ov-row-title" }, h("span", { class: "ov-row-name ov-row-link" }, run.name)),
      h("div", { class: "ov-row-sub", title: `${label} ${fmt.datetime(run.updated)} · ${net} · ${fmt.duration(run.hours)} · ` +
        `${fmt.int(run.positions)} positions from ${run.game_set || "a game set"}` },
      `${label} ${fmt.ago(run.updated)} · ${net} · from games`)),
    h("div", { class: "ov-row-side" },
      e && typeof e.win_rate === "number"
        ? h("span", { class: "ov-row-value", title: "Share of games won against greedy + leave at the latest match, ± its 95% interval" },
          fmt.pct(e.win_rate), typeof e.win_ci === "number" ? h("span", { class: "stat-ci" }, " ± " + (100 * e.win_ci).toFixed(1)) : null,
          h("span", { class: "stat-ci" }, " of games"))
        : h("span", { class: "ov-row-value" }, "–"),
      chip || h("div", { class: "ov-row-sub" }, e ? winVerdict(e.win_rate, e.win_ci) : run.status === "failed" ? "no result" : "no match yet")),
    h("span", { class: "ov-row-go", html: ICON_ARROW, "aria-hidden": "true" })));
  }

  function renderMatches(state, els) {
    const matches = (state.matches || []).slice(0, 5);
    const sig = matches.map((m) => [m.id, m.status, m.games_done, m.spread, Math.floor(Date.now() / 60000)]);
    section(state, "matches", els.matches, sig, (el) => {
      el.replaceChildren(listHead("Recent matches", null, "#/arena", "Arena"),
        matches.length ? h("ul", { class: "ov-list" }, matches.map((m) => {
          const running = m.status === "running";
          return h("li", null, h("a", { class: "ov-row", href: `#/arena/${encodeURIComponent(m.id)}` },
            h("div", { class: "ov-row-main" },
              h("div", { class: "ov-row-title" }, h("span", { class: "ov-row-name" }, m.a_name), h("span", { class: "muted" }, "vs"),
                h("span", { class: "ov-row-name" }, m.b_name)),
              running
                ? h("div", { class: "ov-row-progress" }, h("div", { class: "progress" }, h("span", { style: { width: `${(m.progress * 100).toFixed(1)}%` } })),
                  h("span", { class: "ov-row-sub" }, `${fmt.int(m.games_done)} of ${fmt.int(m.pairs * 2)} games`))
                : h("div", { class: "ov-row-sub" }, `${fmt.int(m.games_done)} games · ${m.status === "finished" ? fmt.ago(m.created) : (STATUS[m.status] || [m.status])[0].toLowerCase()}`)),
            h("div", { class: "ov-row-side" },
              spreadNode(m.spread, m.spread_ci, "ov-row-value"),
              h("div", { class: "ov-row-sub" }, running ? h("span", { class: "ov-live" }, h("span", { class: "dot dot-live" }), "running")
                : m.spread === null ? "" : verdict(m.spread, m.spread_ci).replace("not yet significant", "not significant")))));
        })) : h("p", { class: "muted ov-none" }, "No matches yet. Pit two players against each other in the Arena."));
    });
  }

  function welcome() {
    return h("section", { class: "card ov-welcome" },
      h("div", { class: "ov-welcome-main" },
        h("span", { class: "brand-tile ov-welcome-tile", "aria-hidden": "true" }, "S"),
        h("h2", null, "Welcome to Scrabble Lab"),
        h("p", { class: "secondary" }, "Nothing has been trained yet. Start with a training run: the default settings are a good first experiment, " +
          "and an overnight run of about 8 hours is enough to see whether the network learns to beat the greedy + leave baseline."),
        h("div", { class: "row" },
          h("a", { class: "button button-primary", href: "#/training" }, "Start a training run"),
          h("a", { class: "button", href: "#/arena" }, "Compare the built-in bots"))),
      h("ol", { class: "ov-steps" },
        h("li", null, h("strong", null, "Start a run"), h("span", null, "on the Training page. It keeps going if you close the app.")),
        h("li", null, h("strong", null, "Watch it learn"), h("span", null, "on the run's page: its strength against the baselines, every 20 minutes.")),
        h("li", null, h("strong", null, "Test the result"), h("span", null, "by playing against it, or in a fast match in the Arena."))));
  }
})();
