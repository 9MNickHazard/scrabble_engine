/* Arena: #/arena (start a match, list past ones) and #/arena/<id> (one match).
 *
 * Fast matches without a board: two players, many seeds, each seed played twice with the seats
 * swapped. The detail page leads with player A's spread per game and whether its interval
 * excludes zero, then the statistics, charts, highlights and every game, each replayable in the
 * viewer (#/watch?a=&b=&seed=&first=).
 */
(function () {
  "use strict";

  const { h, fmt } = App;

  const ICON_ARENA = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h11M11 4l3 3-3 3"/><path d="M17 13H6M9 10l-3 3 3 3"/></svg>';
  const ICON_SWAP = '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h11M11 4l3 3-3 3"/><path d="M17 13H6M9 10l-3 3 3 3"/></svg>';
  const ICON_BACK = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M10 3L5 8l5 5"/></svg>';
  const ICON_PLAY = '<svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M5 3.2v9.6c0 .5.5.8.9.5l7.2-4.8a.6.6 0 000-1L5.9 2.7c-.4-.3-.9 0-.9.5z"/></svg>';
  const ICON_CHECK = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="7.5"/><path d="M6.6 10.2l2.3 2.3 4.5-4.8"/></svg>';
  const ICON_EQUAL = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round"><circle cx="10" cy="10" r="7.5"/><path d="M6.8 8.5h6.4M6.8 11.5h6.4"/></svg>';
  const icon = (svg, cls) => h("span", { class: "dash-icon" + (cls ? " " + cls : ""), html: svg, "aria-hidden": "true" });

  const STATUS = { running: ["Running", "good"], finished: ["Finished", "plain"], cancelled: ["Cancelled", "plain"], failed: ["Failed", "critical"] };
  const PRESETS = [100, 500, 2000];
  const PAGE_SIZE = 20;
  const enc = encodeURIComponent;
  const spreadTick = (v) => (v === 0 ? "0" : fmt.signed(v, Number.isInteger(v) ? 0 : 1));

  App.registerView("arena", { title: "Arena", nav: true, order: 60, icon: ICON_ARENA, mount });

  function mount(container, route) {
    if (route.args[0]) return mountDetail(container, route.args[0]);
    return mountList(container, route.query);
  }

  function statusBadge(status) {
    const [label, tone] = STATUS[status] || [status, "plain"];
    return h("span", { class: "badge badge-" + tone }, label);
  }

  function spreadNode(spread, ci, cls) {
    if (spread === null || spread === undefined) return h("span", { class: cls }, "–");
    return h("span", { class: cls }, fmt.signed(spread), ci === null || ci === undefined ? null
      : h("span", { class: "stat-ci" }, " ± " + Number(ci).toFixed(1)));
  }

  function verdictShort(spread, ci, aName, bName) {
    if (spread === null || spread === undefined || ci === null || ci === undefined) return "";
    if (Math.abs(spread) > ci) return `significant, favours ${spread > 0 ? aName : bName}`;
    return "not significant";
  }

  function watchHref(a, b, seed, aFirst) {
    return `#/watch?a=${enc(a)}&b=${enc(b)}&seed=${seed}&first=${aFirst ? "a" : "b"}`;
  }

  // ---- #/arena: start a match, list past ones -------------------------------------------

  async function mountList(container, query) {
    const page = h("div", { class: "arena" });
    container.appendChild(page);
    page.appendChild(h("div", { class: "page-head" }, h("div", null, h("h1", null, "Arena"),
      h("p", { class: "lede" }, "Fast matches without a board. Each seed is played twice with the players swapping seats, " +
        "so the luck of the draw mostly cancels out."))));
    const formSlot = h("div");
    const listSlot = h("section", { class: "card card-flush arena-matches", "aria-label": "Matches" });
    page.append(formSlot, listSlot);

    // Every 2 s while a match is running, every 10 s otherwise. Started before any await so
    // the timer belongs to this page.
    let tickCount = 0;
    let anyRunning = true;
    App.every(2000, async () => {
      tickCount++;
      if (!anyRunning && tickCount % 5 !== 1) return;
      const res = await App.get("/api/matches");
      const matches = res.matches || [];
      anyRunning = matches.some((m) => m.status === "running");
      renderMatchList(listSlot, matches);
    });

    let models = null;
    try {
      models = await App.get("/api/models");
    } catch (error) {
      formSlot.appendChild(h("div", { class: "notice notice-critical" }, h("strong", null, "The list of players could not be loaded. "),
        String(error.message || error)));
    }
    if (models && formSlot.isConnected) formSlot.appendChild(matchForm(models, query));
  }

  function playerOptions(models, selected) {
    const groups = [];
    groups.push(h("optgroup", { label: "Bots" }, (models.bots || []).map((b) =>
      h("option", { value: b.spec, selected: b.spec === selected }, b.name))));
    const nets = (models.models || []).slice().sort((a, b) => {
      if (a.compatible !== b.compatible) return a.compatible ? -1 : 1;
      if ((a.run || "") !== (b.run || "")) return (b.modified || 0) - (a.modified || 0);
      if (a.kind !== b.kind) return a.kind === "best" ? -1 : 1;
      return (b.step || 0) - (a.step || 0);
    });
    if (nets.length) {
      groups.push(h("optgroup", { label: "Networks" }, nets.map((m) => {
        const e = m.eval && m.eval["greedy-leave"];
        const text = m.compatible
          ? `${m.name} · ${m.network}` + (e ? ` · ${fmt.spread(e.spread, e.ci)}` : "")
          : `${m.name}: cannot play, trained with a different word list or leave values`;
        return h("option", { value: m.spec, disabled: !m.compatible, selected: m.spec === selected }, text);
      })));
    }
    return groups;
  }

  function defaultPlayers(models, query) {
    const specs = new Set([...(models.bots || []).map((b) => b.spec), ...(models.models || []).filter((m) => m.compatible).map((m) => m.spec)]);
    const strongest = (models.models || []).filter((m) => m.compatible && m.eval && m.eval["greedy-leave"])
      .sort((x, y) => y.eval["greedy-leave"].spread - x.eval["greedy-leave"].spread)[0];
    let a = query.a && specs.has(query.a) ? query.a : strongest ? strongest.spec : "bot:greedy-leave";
    let b = query.b && specs.has(query.b) ? query.b : a === "bot:greedy-leave" ? "bot:greedy" : "bot:greedy-leave";
    if (!specs.has(a)) a = "bot:greedy-leave";
    if (!specs.has(b)) b = "bot:greedy";
    return { a, b };
  }

  function matchForm(models, query) {
    const players = defaultPlayers(models, query);
    const selectA = h("select", { class: "select", id: "arena-a" }, playerOptions(models, players.a));
    const selectB = h("select", { class: "select", id: "arena-b" }, playerOptions(models, players.b));
    const startPairs = Number(query.pairs) > 1 ? Math.round(Number(query.pairs)) : 500;
    let pairs = startPairs;
    const custom = h("input", { class: "input arena-custom", type: "number", min: "2", max: "100000", step: "1", id: "arena-pairs",
      "aria-label": "Seed pairs", value: String(pairs) });
    const gamesNote = h("p", { class: "field-help" });
    const presetButtons = PRESETS.map((n) => h("button", { type: "button", "aria-pressed": String(n === pairs),
      onclick: () => { pairs = n; custom.value = String(n); sync(); } }, fmt.int(n)));
    const candidates = h("input", { class: "input arena-candidates", type: "number", min: "1", max: "512", step: "1", value: "64", id: "arena-candidates" });
    const sameNote = h("p", { class: "field-help arena-same" });
    const error = h("p", { class: "dash-error", role: "alert" });
    error.hidden = true;
    const start = h("button", { class: "button button-primary", type: "submit" }, icon(ICON_PLAY), "Start match");

    function sync() {
      const n = Math.round(Number(custom.value));
      if (Number.isFinite(n) && n > 0) pairs = n;
      presetButtons.forEach((button, i) => button.setAttribute("aria-pressed", String(PRESETS[i] === pairs)));
      gamesNote.textContent = `${fmt.int(pairs * 2)} games: ${fmt.int(pairs)} seeds, each played twice with the seats swapped.`;
      const same = selectA.value === selectB.value;
      sameNote.textContent = same ? "Both seats have the same player, so the result should be close to zero. A useful check, nothing more." : "";
      sameNote.hidden = !same;
      const bots = selectA.value.startsWith("bot:") && selectB.value.startsWith("bot:");
      candidates.disabled = bots;
    }
    custom.addEventListener("input", sync);
    selectA.addEventListener("change", sync);
    selectB.addEventListener("change", sync);
    sync();

    const swap = h("button", { class: "button button-ghost arena-swap", type: "button", title: "Swap the players", "aria-label": "Swap the players",
      onclick: () => { const a = selectA.value; selectA.value = selectB.value; selectB.value = a; sync(); } }, icon(ICON_SWAP));

    const form = h("form", { class: "card arena-form", onsubmit: async (event) => {
      event.preventDefault();
      error.hidden = true;
      if (!(pairs >= 2 && pairs <= 100000)) {
        error.textContent = "Seed pairs must be a whole number from 2 to 100,000.";
        error.hidden = false;
        return;
      }
      start.disabled = true;
      try {
        const body = { a: selectA.value, b: selectB.value, pairs };
        if (!candidates.disabled) body.candidates = Math.max(1, Math.round(Number(candidates.value) || 64));
        const res = await App.post("/api/matches", body);
        App.toast("Match started.", "good");
        App.go(`#/arena/${enc(res.id)}`);
      } catch (problem) {
        error.textContent = String(problem.message || problem);
        error.hidden = false;
        start.disabled = false;
      }
    } },
    h("div", { class: "card-head" }, h("div", { class: "card-title" }, "New match")),
    h("div", { class: "arena-players" },
      h("label", { class: "field", for: "arena-a" }, h("span", { class: "field-label" }, "Player A"), selectA),
      swap,
      h("label", { class: "field", for: "arena-b" }, h("span", { class: "field-label" }, "Player B"), selectB)),
    sameNote,
    h("div", { class: "arena-options" },
      h("div", { class: "field" }, h("span", { class: "field-label" }, "Seed pairs"),
        h("div", { class: "row arena-pairs" }, h("div", { class: "segmented", role: "group", "aria-label": "Seed pair presets" }, presetButtons), custom),
        gamesNote),
      h("details", { class: "arena-advanced" }, h("summary", null, "Advanced"),
        h("label", { class: "field", for: "arena-candidates" }, h("span", { class: "field-label" }, "Candidates per decision"), candidates,
          h("span", { class: "field-help" }, "How many moves a network evaluates per decision, the best by score plus leave value. " +
            "Fewer is faster. Bots ignore it."))),
      h("div", { class: "arena-submit" }, start)),
    error);
    return form;
  }

  function renderMatchList(slot, matches) {
    if (!matches.length) {
      slot.replaceChildren(h("div", { class: "card-head arena-list-head" }, h("div", { class: "card-title" }, "Matches")),
        h("p", { class: "empty" }, h("strong", null, "No matches yet."), "Start one above. 500 seed pairs take well under a minute with the bots."));
      return;
    }
    slot.replaceChildren(
      h("div", { class: "card-head arena-list-head" }, h("div", { class: "card-title" }, "Matches"),
        h("div", { class: "card-sub" }, "Result is player A's spread per game, with its 95% interval")),
      h("table", { class: "table arena-table" },
        h("thead", null, h("tr", null, h("th", null, "Players"), h("th", { class: "num" }, "Result for A"), h("th", { class: "num" }, "A's win rate"),
          h("th", { class: "num" }, "Games"), h("th", null, "Status"), h("th", null, "Started"))),
        h("tbody", null, matches.map((m) => {
          const running = m.status === "running";
          const open = () => App.go(`#/arena/${enc(m.id)}`);
          return h("tr", { class: "clickable", tabindex: "0", onclick: open, onkeydown: (event) => { if (event.key === "Enter") open(); } },
            h("td", null, h("div", { class: "arena-players-cell" }, h("span", { class: "arena-name" }, m.a_name), h("span", { class: "muted" }, "vs"),
              h("span", { class: "arena-name" }, m.b_name))),
            h("td", { class: "num" }, spreadNode(m.spread, m.spread_ci, "arena-result"),
              h("div", { class: "arena-sub" }, verdictShort(m.spread, m.spread_ci, m.a_name, m.b_name))),
            h("td", { class: "num" }, m.win_rate === null || m.win_rate === undefined ? "–" : fmt.pct(m.win_rate)),
            h("td", { class: "num" }, `${fmt.int(m.games_done)}`, h("span", { class: "muted" }, ` / ${fmt.int(m.pairs * 2)}`)),
            h("td", null, running
              ? h("div", { class: "arena-progress-cell" }, h("div", { class: "progress" }, h("span", { style: { width: `${(m.progress * 100).toFixed(1)}%` } })),
                h("span", { class: "arena-sub" }, `${Math.round(m.progress * 100)}%`))
              : statusBadge(m.status)),
            h("td", { class: "nowrap secondary" }, fmt.ago(m.created)));
        }))));
  }

  // ---- #/arena/<id>: one match ----------------------------------------------------------

  function mountDetail(container, id) {
    const state = { id, data: null, charts: {}, sort: { key: "index", dir: 1 }, page: 0, done: false, built: false };
    const els = {
      head: h("div"), progress: h("div"), hero: h("div"), tiles: h("div", { class: "arena-tiles" }),
      charts: h("div", { class: "grid-2 arena-charts" }), highlights: h("div"), games: h("section", { class: "card card-flush arena-games" }),
    };
    els.games.hidden = true;
    container.appendChild(h("div", { class: "arena-detail" }, els.head, els.progress, els.hero, els.tiles, els.charts, els.highlights, els.games));
    const stop = App.every(2000, async () => {
      let data;
      try {
        data = await App.get(`/api/matches/${enc(id)}`);
      } catch (error) {
        if (!state.data) {
          els.head.replaceChildren(h("a", { class: "dash-back", href: "#/arena" }, icon(ICON_BACK), "Back to the arena"),
            h("div", { class: "notice notice-critical" }, h("strong", null, "This match could not be loaded. "), String(error.message || error)));
          stop();
        }
        return;
      }
      state.data = data;
      renderDetail(state, els);
      if (data.match.status !== "running") stop();
    });
  }

  function renderDetail(state, els) {
    const { match, summary } = state.data;
    document.title = `${match.a_name} vs ${match.b_name} · Scrabble Lab`;
    const running = match.status === "running";
    const actions = h("div", { class: "page-actions" });
    if (running) actions.appendChild(h("button", { class: "button", type: "button", onclick: () => cancel(state, els) }, "Cancel match"));
    actions.appendChild(h("a", { class: "button", href: `#/arena?a=${enc(match.a)}&b=${enc(match.b)}&pairs=${match.pairs}`,
      title: "Open the form with these players" }, "New match with these players"));
    const total = match.pairs * 2;
    els.head.replaceChildren(h("div", { class: "page-head" },
      h("div", null,
        h("a", { class: "dash-back", href: "#/arena" }, icon(ICON_BACK), "Back to the arena"),
        h("div", { class: "run-title" }, h("h1", null, match.a_name, h("span", { class: "arena-vs" }, " vs "), match.b_name), statusBadge(match.status)),
        h("p", { class: "lede" }, `${fmt.int(match.pairs)} seed pairs · ${fmt.int(total)} games · ` +
          (match.seconds ? `${match.seconds.toFixed(1)} s · ` : "") + fmt.datetime(match.created))),
      actions));

    if (running) {
      const rate = match.seconds ? match.games_done / match.seconds : null;
      els.progress.replaceChildren(h("div", { class: "card arena-running" },
        h("div", { class: "row row-between" }, h("span", null, h("span", { class: "dot dot-live" }), ` ${fmt.int(match.games_done)} of ${fmt.int(total)} games`),
          h("span", { class: "muted" }, rate ? `${rate.toFixed(0)} games a second · about ${Math.max(1, Math.round((total - match.games_done) / rate))} s left` : "")),
        h("div", { class: "progress" }, h("span", { style: { width: `${(match.progress * 100).toFixed(1)}%` } }))));
    } else if (match.status === "cancelled") {
      els.progress.replaceChildren(h("div", { class: "notice" }, h("strong", null, "Cancelled. "),
        `The ${fmt.int(match.games_done)} games played before it was cancelled are shown.`));
    } else els.progress.replaceChildren();

    renderHero(els, match, summary, running);
    renderTiles(els, match, summary);
    renderCharts(state, els);
    renderHighlights(state, els);
    renderGames(state, els);
  }

  function renderHero(els, match, s, running) {
    const A = match.a_name;
    const B = match.b_name;
    if (!s || s.spread === null || s.spread === undefined) {
      els.hero.replaceChildren(h("section", { class: "card arena-hero" }, h("div", { class: "arena-hero-figure" }, h("span", { class: "hero" }, "–")),
        h("div", { class: "arena-hero-text" }, h("div", { class: "arena-hero-label" }, `points per game for ${A}`),
          h("p", { class: "arena-verdict secondary" }, "Waiting for the first seed pairs to finish."))));
      return;
    }
    const lo = s.spread - s.spread_ci;
    const hi = s.spread + s.spread_ci;
    const significant = s.significant !== undefined && s.significant !== null ? s.significant : Math.abs(s.spread) > s.spread_ci;
    const range = `${fmt.signed(lo)} to ${fmt.signed(hi)}`;
    let lead;
    let text;
    if (significant) {
      const winner = s.spread > 0 ? A : B;
      lead = running ? "Significant so far." : "Statistically significant.";
      text = ` The 95% interval, ${range}, excludes zero, so ${winner} is the stronger player in these games.`;
    } else {
      lead = running ? "Not significant so far." : "Not statistically significant.";
      text = ` The 95% interval, ${range}, includes zero, so these games do not separate the two players.` +
        (running ? "" : " More seed pairs would narrow it.");
    }
    if (running) text += ` Based on the first ${fmt.int(match.games_done)} of ${fmt.int(match.pairs * 2)} games.`;
    els.hero.replaceChildren(h("section", { class: "card arena-hero" },
      h("div", { class: "arena-hero-figure" }, h("span", { class: "hero" }, fmt.signed(s.spread)), h("span", { class: "arena-hero-ci" }, "± " + s.spread_ci.toFixed(1))),
      h("div", { class: "arena-hero-text" },
        h("div", { class: "arena-hero-label" }, `points per game for ${A} against ${B}`),
        h("p", { class: "arena-verdict" }, icon(significant ? ICON_CHECK : ICON_EQUAL, "arena-verdict-icon"),
          h("span", null, h("strong", null, lead), text)))));
  }

  function pairValue(a, b, digits) {
    if (a === null || a === undefined) return "–";
    return h("span", null, fmt.fixed(a, digits), h("span", { class: "dash-vs" }, " vs "), fmt.fixed(b, digits));
  }

  function renderTiles(els, match, s) {
    const A = match.a_name;
    const B = match.b_name;
    const both = `${A} vs ${B}`;
    if (!s) {
      els.tiles.replaceChildren();
      return;
    }
    els.tiles.replaceChildren(
      Charts.statTile({ label: `Win rate for ${A}`, value: h("span", null, fmt.pct(s.win_rate), h("span", { class: "stat-ci" }, ` ± ${(100 * s.win_ci).toFixed(1)}`)),
        hint: "ties count as half a win" }),
      Charts.statTile({ label: "Mean score", value: pairValue(s.score_a, s.score_b, 1), hint: both }),
      Charts.statTile({ label: "Turns per game", value: fmt.fixed(s.turns, 1), hint: "both players together" }),
      Charts.statTile({ label: "Bingos per game", value: pairValue(s.bingos_a, s.bingos_b, 2), hint: both }),
      Charts.statTile({ label: "Exchanges per game", value: pairValue(s.exchanges_a, s.exchanges_b, 2), hint: both }),
      Charts.statTile({ label: "Games ended by going out", value: fmt.pct(s.ended_by_going_out), hint: "the rest by six scoreless turns" }));
  }

  function runningMean(games) {
    const n = Math.floor(games.score_a.length / 2);
    const points = [];
    const band = [];
    let sum = 0;
    let sumSq = 0;
    for (let k = 0; k < n; k++) {
      const v = ((games.score_a[2 * k] - games.score_b[2 * k]) + (games.score_a[2 * k + 1] - games.score_b[2 * k + 1])) / 2;
      sum += v;
      sumSq += v * v;
      const count = k + 1;
      if (count < 10) continue;
      const mean = sum / count;
      const variance = Math.max(0, (sumSq - count * mean * mean) / (count - 1));
      const ci = 1.96 * Math.sqrt(variance / count);
      points.push([2 * count, mean]);
      band.push([2 * count, mean - ci, mean + ci]);
    }
    return { points, band };
  }

  function chartOptions(state) {
    const { match, summary, games } = state.data;
    const A = match.a_name;
    const B = match.b_name;
    const spreads = games.score_a.map((v, i) => v - games.score_b[i]);
    const rm = runningMean(games);
    const last = rm.points[rm.points.length - 1];
    const lastBand = rm.band[rm.band.length - 1];
    let yRange = {};
    if (last) {
      const lim = Math.max(12, Math.abs(last[1]) + 4.5 * (lastBand[2] - lastBand[1]) / 2);
      yRange = { min: -Math.ceil(lim / 5) * 5, max: Math.ceil(lim / 5) * 5 };
    }
    const s = summary || {};
    return {
      spread: {
        title: "Spread per game",
        subtitle: `${A}'s score minus ${B}'s, one bar per range of spreads`,
        series: [{ name: "Games", values: spreads, color: "var(--series-1)" }],
        bins: 32, center: 0, centerLabels: [`${B} ahead`, `${A} ahead`],
        x: { label: "spread for A, points", format: (v) => (v === 0 ? "0" : fmt.signed(v, 0)) }, y: { label: "games" }, height: 270,
        empty: "No finished games yet.",
      },
      scores: {
        title: "Final scores",
        subtitle: "Each player's score across all games, overlaid",
        series: [{ name: A, values: games.score_a, color: "var(--series-1)" }, { name: B, values: games.score_b, color: "var(--series-2)" }],
        bins: 30, x: { label: "points", format: (v) => fmt.int(v) }, y: { label: "games" }, height: 270, empty: "No finished games yet.",
      },
      running: {
        title: "Running estimate",
        subtitle: "Mean spread for A after each seed pair, from the 10th on, with its 95% interval narrowing",
        series: [{ name: `Spread for ${A}`, points: rm.points, band: rm.band, color: "var(--series-1)" }],
        x: { label: "games", format: (v) => fmt.int(v), tickFormat: "auto", tooltipFormat: (v) => `After ${fmt.int(v)} games` },
        y: Object.assign({ label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick }, yRange),
        referenceY: [{ y: 0, label: "even" }], height: 270, empty: "The estimate appears after 10 seed pairs.",
      },
      seats: {
        title: "Moving first and moving second",
        subtitle: s.a_first ? `${A} won ${fmt.pct(s.a_first.win_rate, 0)} of games moving first and ${fmt.pct(s.a_second.win_rate, 0)} moving second`
          : "Spread for A by seat",
        categories: [`${A} moved first`, `${A} moved second`],
        series: [{ name: `Spread for ${A}`, values: s.a_first ? [s.a_first.spread, s.a_second.spread] : [], color: "var(--series-1)" }],
        y: { label: "points per game", format: (v) => fmt.signed(v), tickFormat: spreadTick }, height: 270, empty: "No finished pairs yet.",
      },
    };
  }

  function renderCharts(state, els) {
    if (!state.data.games) return;
    const opts = chartOptions(state);
    if (!state.built) {
      const slots = { spread: h("div"), scores: h("div"), running: h("div"), seats: h("div") };
      els.charts.replaceChildren(...["spread", "scores", "running", "seats"].map((key) => h("div", { class: "card" }, slots[key])));
      state.charts.spread = Charts.histogram(slots.spread, opts.spread);
      state.charts.scores = Charts.histogram(slots.scores, opts.scores);
      state.charts.running = Charts.line(slots.running, opts.running);
      state.charts.seats = Charts.bars(slots.seats, opts.seats);
      state.built = true;
      return;
    }
    for (const key of Object.keys(opts)) state.charts[key].update(opts[key]);
  }

  function renderHighlights(state, els) {
    const { match, games, highlights } = state.data;
    if (!highlights || !highlights.length) {
      els.highlights.replaceChildren();
      return;
    }
    els.highlights.replaceChildren(h("section", { class: "card arena-highlights" },
      h("div", { class: "card-head" }, h("div", { class: "card-title" }, "Highlights"),
        h("div", { class: "card-sub" }, "Replay any of them move by move in the viewer")),
      h("ul", { class: "arena-highlight-list" }, highlights.map((hl) => {
        const i = hl.game;
        const seed = games.seed[i];
        const aFirst = games.a_first[i];
        return h("li", { class: "arena-highlight" },
          h("div", null, h("div", { class: "arena-highlight-title" }, fmt.prose(hl.title)), h("div", { class: "arena-highlight-detail" }, fmt.prose(hl.detail)),
            h("div", { class: "arena-sub" }, `Game ${fmt.int(i + 1)} · ${aFirst ? match.a_name : match.b_name} moved first`)),
          h("a", { class: "button button-small", href: watchHref(match.a, match.b, seed, aFirst) }, icon(ICON_PLAY), "Replay"));
      }))));
  }

  const COLUMNS = [
    { key: "index", label: "Game", num: true, value: (g, i) => i },
    { key: "first", label: "First to move", value: (g, i) => (g.a_first[i] ? 0 : 1) },
    { key: "score_a", label: "Score A", num: true, value: (g, i) => g.score_a[i] },
    { key: "score_b", label: "Score B", num: true, value: (g, i) => g.score_b[i] },
    { key: "spread", label: "Spread", num: true, value: (g, i) => g.score_a[i] - g.score_b[i] },
    { key: "turns", label: "Turns", num: true, value: (g, i) => g.turns[i] },
    { key: "bingos_a", label: "Bingos A", num: true, value: (g, i) => g.bingos_a[i] },
    { key: "bingos_b", label: "Bingos B", num: true, value: (g, i) => g.bingos_b[i] },
    { key: "best_a", label: "Best play A", value: (g, i) => g.best_play_a[i] },
    { key: "best_b", label: "Best play B", value: (g, i) => g.best_play_b[i] },
    { key: "ended", label: "Ended", value: (g, i) => (g.ended_by[i] === "out" ? 0 : 1) },
  ];

  function renderGames(state, els) {
    const { match, games } = state.data;
    const n = games ? games.score_a.length : 0;
    els.games.hidden = false;
    if (!n) {
      els.games.replaceChildren(h("div", { class: "card-head arena-list-head" }, h("div", { class: "card-title" }, "Every game")),
        h("p", { class: "empty" }, "Games appear here as they finish."));
      return;
    }
    const column = COLUMNS.find((c) => c.key === state.sort.key) || COLUMNS[0];
    const order = Array.from({ length: n }, (_, i) => i).sort((x, y) =>
      (column.value(games, x) - column.value(games, y)) * state.sort.dir || x - y);
    const pages = Math.max(1, Math.ceil(n / PAGE_SIZE));
    state.page = Math.min(state.page, pages - 1);
    const slice = order.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE);
    const sortBy = (key) => {
      if (state.sort.key === key) state.sort.dir *= -1;
      else state.sort = { key, dir: key === "index" || key === "first" || key === "ended" ? 1 : -1 };
      state.page = 0;
      renderGames(state, els);
    };
    const head = h("tr", null, COLUMNS.map((c) => {
      const sorted = state.sort.key === c.key;
      return h("th", { class: "sortable" + (c.num ? " num" : ""), "aria-sort": sorted ? (state.sort.dir > 0 ? "ascending" : "descending") : "none",
        tabindex: "0", onclick: () => sortBy(c.key), onkeydown: (event) => { if (event.key === "Enter") sortBy(c.key); } },
      c.label, sorted ? h("span", { class: "sort-mark" }, state.sort.dir > 0 ? " ▲" : " ▼") : null);
    }), h("th", null, ""));
    const rows = slice.map((i) => {
      const spread = games.score_a[i] - games.score_b[i];
      return h("tr", { title: `Seed ${games.seed[i]}` },
        h("td", { class: "num" }, fmt.int(i + 1)),
        h("td", null, games.a_first[i] ? "A" : "B"),
        h("td", { class: "num" }, String(games.score_a[i])),
        h("td", { class: "num" }, String(games.score_b[i])),
        h("td", { class: "num" }, spread === 0 ? "0" : fmt.signed(spread, 0)),
        h("td", { class: "num" }, String(games.turns[i])),
        h("td", { class: "num" }, String(games.bingos_a[i])),
        h("td", { class: "num" }, String(games.bingos_b[i])),
        h("td", null, h("span", { class: "mono" }, games.best_word_a[i]), h("span", { class: "muted" }, " " + games.best_play_a[i])),
        h("td", null, h("span", { class: "mono" }, games.best_word_b[i]), h("span", { class: "muted" }, " " + games.best_play_b[i])),
        h("td", null, games.ended_by[i] === "out" ? "went out" : "scoreless"),
        h("td", { class: "arena-replay" }, h("a", { class: "button button-small button-ghost", href: watchHref(match.a, match.b, games.seed[i], games.a_first[i]) },
          icon(ICON_PLAY), "Replay")));
    });
    const pager = h("div", { class: "arena-pager" },
      h("span", { class: "muted" }, `${fmt.int(state.page * PAGE_SIZE + 1)}–${fmt.int(Math.min(n, (state.page + 1) * PAGE_SIZE))} of ${fmt.int(n)}`),
      h("button", { class: "button button-small", type: "button", disabled: state.page === 0,
        onclick: () => { state.page--; renderGames(state, els); } }, "Previous"),
      h("button", { class: "button button-small", type: "button", disabled: state.page >= pages - 1,
        onclick: () => { state.page++; renderGames(state, els); } }, "Next"));
    els.games.replaceChildren(
      h("div", { class: "card-head arena-list-head" },
        h("div", null, h("div", { class: "card-title" }, "Every game"),
          h("div", { class: "card-sub" }, `A is ${match.a_name}, B is ${match.b_name}. Games 1 and 2, 3 and 4, and so on share a seed with the seats swapped.`)),
        pager),
      h("div", { class: "arena-table-wrap" }, h("table", { class: "table arena-games-table" }, h("thead", null, head), h("tbody", null, rows))));
  }

  async function cancel(state, els) {
    const ok = await App.confirm({ title: "Cancel this match?", body: "The games played so far are kept and shown.", confirmLabel: "Cancel match" });
    if (!ok) return;
    try {
      await App.post(`/api/matches/${enc(state.id)}/cancel`);
      App.toast("Match cancelled.", "good");
      state.data = await App.get(`/api/matches/${enc(state.id)}`);
      renderDetail(state, els);
    } catch (error) {
      App.toast(String(error.message || error), "critical");
    }
  }
})();
