/* Overview: #/overview, the landing page.
 *
 * What is training now (a card per active run with progress, speed, the latest result against
 * greedy + leave and a sparkline of that result), the queue, the GPU, the strongest network of
 * each run, recent runs and recent matches. Before anything has been trained it shows a short
 * welcome that points to the Training page.
 */
(function () {
  "use strict";

  const { h, fmt } = App;

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
  /** "8 h" rather than "8 h 00 m" for whole hours. */
  const hoursText = (hours) => fmt.duration(hours).replace(/ 00 m$/, "");

  App.registerView("overview", { title: "Overview", nav: true, order: 10, icon: ICON_OVERVIEW, mount });

  function mount(container) {
    const state = { runs: null, models: null, matches: null, metrics: {}, signatures: {} };
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
    await Promise.all(active.map((run) => loadEvals(state, run.name)));
    render(state, els);
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

  function strongest(state) {
    if (!state.models) return [];
    const byRun = new Map();
    for (const m of state.models.models || []) {
      const e = m.eval && m.eval["greedy-leave"];
      if (!m.compatible || !e || typeof e.spread !== "number") continue;
      const key = m.run || m.id;
      const current = byRun.get(key);
      if (!current || e.spread > current.eval["greedy-leave"].spread) byRun.set(key, m);
    }
    return Array.from(byRun.values()).sort((a, b) => b.eval["greedy-leave"].spread - a.eval["greedy-leave"].spread).slice(0, 5);
  }

  // ---- Render ---------------------------------------------------------------------------

  function render(state, els) {
    const runs = state.runs;
    const active = runs.filter(isLive);
    const queued = runs.filter((r) => r.status === "queued");
    const top = strongest(state);
    const best = top[0];

    let lede;
    if (!runs.length) lede = "Train a network, watch it learn, then play against it.";
    else {
      lede = active.length ? `${active.length} run${active.length === 1 ? " is" : "s are"} training` : "Nothing is training right now";
      lede += queued.length ? `, ${queued.length} queued.` : ".";
      if (best) {
        const e = best.eval["greedy-leave"];
        lede += ` The strongest network so far is ${best.name}, ${fmt.spread(e.spread, e.ci)} points a game against greedy + leave.`;
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
    section(state, "strongest", els.strongest, top.map((m) => [m.id, m.eval, m.games]), (el) => renderStrongest(el, top));
    const recent = runs.filter((r) => !isLive(r) && r.status !== "queued")
      .sort((a, b) => (b.updated || 0) - (a.updated || 0)).slice(0, 5);
    section(state, "runs", els.runs, recent.map((r) => [r.name, r.status, r.latest_eval, Math.floor(Date.now() / 60000)]),
      (el) => renderRuns(el, recent));
    renderMatches(state, els);
  }

  function renderActive(state, els, active, queued, runs) {
    const nodes = active.map((run) => activeCard(state, run));
    if (!active.length) {
      const last = runs.filter((r) => r.status === "finished" || r.status === "stopped")
        .sort((a, b) => (b.updated || 0) - (a.updated || 0))[0];
      const e = last && last.latest_eval && last.latest_eval["greedy-leave"];
      nodes.push(h("div", { class: "card ov-idle" },
        h("div", null,
          h("h2", null, "No training running"),
          h("p", { class: "secondary" }, last
            ? [`The last run, `, h("a", { href: `#/run/${encodeURIComponent(last.name)}` }, last.name), ` ${last.status === "finished" ? "finished" : "stopped"} ${fmt.ago(last.updated)}`,
              e ? `, at ${fmt.spread(e.spread, e.ci)} against greedy + leave (${verdict(e.spread, e.ci)}).` : "."]
            : "Start a run on the Training page; it appears here while it trains.")),
        h("div", { class: "page-actions" }, h("a", { class: "button button-primary", href: "#/training" }, "Start a run"))));
    }
    if (queued.length) {
      nodes.push(h("div", { class: "ov-queue" },
        h("span", { class: "badge badge-info" }, "Queued"),
        h("span", null, queued.map((r, i) => [i ? ", " : "", h("a", { href: `#/run/${encodeURIComponent(r.name)}` }, r.name),
          h("span", { class: "muted" }, ` (${r.network}${r.limit && r.limit.hours ? `, ${hoursText(r.limit.hours)}` : ""})`)])),
        h("span", { class: "muted" }, active.length ? "starts when the current run ends" : "starts shortly")));
    }
    els.active.replaceChildren(...nodes);
    els.active.classList.toggle("ov-active-multi", active.length > 1);
  }

  function activeCard(state, run) {
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

  function figure(label, value) {
    return h("div", { class: "ov-figure" }, h("dt", null, label), h("dd", null, value));
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

  function renderStrongest(el, top) {
    const slot = h("div");
    el.replaceChildren(listHead("Strongest networks", "Each run's best evaluation against greedy + leave: points per game, 95% interval",
      "#/models", "All models"), slot);
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

  function renderRuns(el, recent) {
    el.replaceChildren(listHead("Recent runs", null, "#/training", "All runs"),
      recent.length ? h("ul", { class: "ov-list" }, recent.map((run) => {
        const e = run.latest_eval && run.latest_eval["greedy-leave"];
        return h("li", null, h("a", { class: "ov-row", href: `#/run/${encodeURIComponent(run.name)}` },
          h("div", { class: "ov-row-main" },
            h("div", { class: "ov-row-title" }, h("span", { class: "ov-row-name" }, run.name), statusBadge(run.status)),
            h("div", { class: "ov-row-sub" }, `${run.network} · ${fmt.duration(run.hours)} · ${fmt.compact(run.games)} games · ${fmt.ago(run.updated)}`)),
          h("div", { class: "ov-row-side" },
            spreadNode(e ? e.spread : null, e ? e.ci : null, "ov-row-value"),
            h("div", { class: "ov-row-sub" }, e ? verdict(e.spread, e.ci) : run.status === "failed" ? "no result" : "not evaluated"))));
      })) : h("p", { class: "muted ov-none" }, "Finished runs appear here."));
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
