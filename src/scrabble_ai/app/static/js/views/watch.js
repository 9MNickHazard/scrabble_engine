/* Watch: two bots or models play one game at a speed a person can follow.
 *
 * Setup (players A and B, who moves first, an optional seed, speed), then the viewer: the board,
 * both racks beside the players' names and scores, playback controls, what the mover considered,
 * the score over the game and the move list. Clicking a move shows the board as it was after it.
 *
 * Query: a, b (player specs), seed, first ("a" or "b") preselect; with seed the game starts at once
 * (other pages link here to replay a game). Keys in the viewer: Space plays or pauses, the left and
 * right arrows step through the moves, Esc returns to the live position.
 *
 * Mock-only flags (with ?mock=1): demo=midgame | history | gameover, theme=light | dark.
 */
(function () {
  "use strict";

  const { h, fmt } = App;
  const U = Board.ui;
  const ICON = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M1.8 10S4.8 4.5 10 4.5 18.2 10 18.2 10 15.2 15.5 10 15.5 1.8 10 1.8 10z"/><circle cx="10" cy="10" r="2.6"/></svg>';
  const SPEEDS = [["slow", "Slow", 2500], ["normal", "Normal", 1200], ["fast", "Fast", 300], ["instant", "Instant", 0]];
  const FALLBACK = {
    bots: [
      { spec: "bot:greedy-leave", name: "Greedy + leave", description: "Highest score plus leave value." },
      { spec: "bot:greedy", name: "Greedy", description: "Highest score." },
      { spec: "bot:random", name: "Random", description: "A random legal placement." },
    ],
    models: [],
  };

  let inPlace = false; // re-rendering without leaving the page (theme switch)
  let live = null;     // the game being watched, kept across an in-place re-render
  let current = null;  // the viewer showing, retired when another screen replaces it

  function retire() {
    if (current) current.dispose();
    current = null;
  }

  App.registerView("watch", { title: "Watch", nav: true, order: 50, icon: ICON, mount });

  function loadSpeed() {
    try { const s = localStorage.getItem("watch.speed"); if (SPEEDS.some((x) => x[0] === s)) return s; } catch (error) { /* ignore */ }
    return "normal";
  }
  function saveSpeed(s) {
    try { localStorage.setItem("watch.speed", s); } catch (error) { /* ignore */ }
  }
  async function playerList() {
    try { return await App.get("/api/models"); } catch (error) { return Object.assign({ error: error.message }, FALLBACK); }
  }

  async function mount(container, route) {
    const q = route.query || {};
    if (App.mock.enabled && (q.theme === "light" || q.theme === "dark")) document.documentElement.dataset.theme = q.theme;
    const reentered = inPlace;
    inPlace = false;
    App.onCleanup(() => { inPlace = /^#\/watch(\?|$)/.test(location.hash); });
    const data = await playerList();

    if (reentered && live) {
      try {
        const state = await App.get("/api/games/" + encodeURIComponent(live.id));
        return viewer(container, state, data, live);
      } catch (error) { live = null; }
    }
    if (App.mock.enabled && q.demo) return demo(container, q, data);
    if (q.seed !== undefined && q.seed !== "") {
      const cfg = { a: q.a || "bot:greedy-leave", b: q.b || "bot:greedy", first: q.first === "b" ? "b" : "a", seed: Number(q.seed) };
      try {
        return await start(container, data, cfg, { autoplay: true });
      } catch (error) {
        setup(container, q, data, error.message);
        return;
      }
    }
    setup(container, q, data);
  }

  function segmented(choices, value, onChange) {
    const node = h("div", { class: "segmented", role: "group" });
    const buttons = choices.map(([key, label]) => h("button", {
      type: "button", "aria-pressed": String(key === value),
      onmousedown: (e) => e.preventDefault(),
      onclick: () => {
        buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(choices[i][0] === key)));
        onChange(key);
      },
    }, label));
    node.append(...buttons);
    return node;
  }

  // ---- Setup ----------------------------------------------------------------------------

  function setup(container, q, data, problem) {
    retire();
    App.clear(container);
    let first = ["a", "b", "random"].includes(q.first) ? q.first : "a";
    let speed = loadSpeed();
    const selA = U.playerSelect(data, q.a || U.defaultOpponent(data), { id: "watch-a" });
    const selB = U.playerSelect(data, q.b || "bot:greedy-leave", { id: "watch-b" });
    const descA = h("div", { class: "player-desc" });
    const descB = h("div", { class: "player-desc" });
    const describe = () => {
      descA.textContent = U.specInfo(selA.value, data).description || "";
      descB.textContent = U.specInfo(selB.value, data).description || "";
    };
    selA.addEventListener("change", describe);
    selB.addEventListener("change", describe);
    describe();
    const seedInput = h("input", { class: "input", type: "text", inputmode: "numeric", placeholder: "Random", value: q.seed || "" });
    const errorBox = h("div", null, problem ? h("div", { class: "notice notice-critical" }, U.sentence(problem)) : null);
    const startButton = h("button", { class: "button button-primary", type: "button", onclick: go }, "Start the game");
    seedInput.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });

    async function go() {
      const raw = seedInput.value.trim();
      if (raw && !/^\d+$/.test(raw)) {
        errorBox.replaceChildren(h("div", { class: "notice notice-warning" }, "A seed is a whole number, or leave it empty for a random game."));
        return;
      }
      startButton.disabled = true;
      saveSpeed(speed);
      try {
        await start(container, data, { a: selA.value, b: selB.value, first, seed: raw ? Number(raw) : null }, { autoplay: true });
      } catch (error) {
        startButton.disabled = false;
        errorBox.replaceChildren(h("div", { class: "notice notice-critical" }, U.sentence(error.message)));
      }
    }

    container.appendChild(h("div", { class: "watch-setup" },
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Watch"),
          h("p", { class: "lede" }, "Two players, one game, at a speed you can follow. Bots or models; a model can play itself."))),
      data.error ? h("div", { class: "notice notice-warning", style: { marginBottom: "var(--space-4)" } },
        "Could not load the model list (" + data.error + "). Only the bots are listed.") : null,
      h("div", { class: "setup" },
        h("div", { class: "card setup-card" },
          h("div", { class: "setup-pair" },
            h("label", { class: "field" }, h("span", { class: "field-label" }, h("span", { class: "who who-a" }), "Player A"), selA, descA),
            h("label", { class: "field" }, h("span", { class: "field-label" }, h("span", { class: "who who-b" }), "Player B"), selB, descB)),
          h("div", { class: "setup-pair" },
            h("div", { class: "field" }, h("span", { class: "field-label" }, "Who moves first"),
              h("div", null, segmented([["a", "A"], ["b", "B"], ["random", "Random"]], first, (v) => { first = v; }))),
            h("label", { class: "field" }, h("span", { class: "field-label" }, "Seed"), seedInput,
              h("span", { class: "field-help" }, "The same seed and players replay a game exactly."))),
          h("div", { class: "field" }, h("span", { class: "field-label" }, "Speed"),
            h("div", null, segmented(SPEEDS.map(([k, label]) => [k, label]), speed, (v) => { speed = v; }))),
          h("div", { class: "setup-actions" }, startButton),
          errorBox),
        h("div", { class: "setup-aside" },
          h("h3", null, "What you will see"),
          h("ul", null,
            h("li", null, "Both racks, beside each player's name and score."),
            h("li", null, "For every move, the options the mover weighed: score, leave, value, total and, for a model, its win estimate. Hover one to see it on the board."),
            h("li", null, "The score over the game, and the move list. Click a move to see the board as it was."),
            h("li", null, "Space plays or pauses; the arrow keys step through the moves."))))));
  }

  async function start(container, data, cfg, opts) {
    const aFirst = cfg.first === "a" || (cfg.first === "random" && Math.random() < 0.5);
    const body = { players: aFirst ? [cfg.a, cfg.b] : [cfg.b, cfg.a] };
    if (cfg.seed !== null && cfg.seed !== undefined && !Number.isNaN(cfg.seed)) body.seed = cfg.seed;
    if (opts && opts.extra) Object.assign(body, opts.extra);
    const state = await App.post("/api/games", body);
    const params = new URLSearchParams({ a: cfg.a, b: cfg.b, seed: String(state.seed), first: aFirst ? "a" : "b" });
    // Put the seed in the address (without a hashchange) so a reload or a copied link replays this game.
    try { history.replaceState(null, "", location.href.split("#")[0] + "#/watch?" + params.toString()); } catch (error) { /* ignore */ }
    return viewer(container, state, data, { id: state.id, aSeat: aFirst ? 0 : 1, cfg, analyses: new Map(),
      playing: !!(opts && opts.autoplay) });
  }

  // ---- The viewer -----------------------------------------------------------------------

  function viewer(container, initial, data, ctx) {
    retire();
    App.clear(container);
    let state = initial;
    const aSeat = ctx.aSeat;
    const bSeat = 1 - aSeat;
    const analyses = ctx.analyses || new Map(); // turn -> {analysis, prevBoard}
    const fetching = new Set(); // turns whose stored analysis has been asked for
    const cfg = ctx.cfg;
    live = { id: state.id, aSeat, cfg, analyses, playing: false };
    let speed = loadSpeed();
    let playing = false;
    let busy = false;
    let alive = true;
    let token = 0;
    let viewIndex = null;
    let tab = "moves";
    let overlay = null;
    let previewing = false;
    let failure = null;
    let chart = null;
    const chartBox = h("div", { class: "chart-box" });

    const letter = (seat) => (seat === aSeat ? "A" : "B");
    const whoClass = (seat) => (seat === aSeat ? "who-a" : "who-b");
    const nameOf = (seat) => state.players[seat].name;

    // -- layout
    const noFocus = (e) => e.preventDefault();
    const playButton = h("button", { class: "button button-primary play-pause icon-button", type: "button", onmousedown: noFocus,
      onclick: () => setPlaying(!playing), title: "Play or pause (Space)" });
    const stepButton = h("button", { class: "button icon-button", type: "button", onmousedown: noFocus, onclick: () => stepButtonPressed(),
      title: "One move (right arrow)" }, h("span", { html: U.ICONS.step, style: { display: "inline-grid" } }), "Step");
    const speedControl = segmented(SPEEDS.map(([k, label]) => [k, label]), speed, (v) => {
      speed = v;
      saveSpeed(v);
    });
    speedControl.setAttribute("aria-label", "Speed");
    const meta = h("span", { class: "watch-meta" });
    const restartButton = h("button", { class: "button button-ghost icon-button", type: "button", onmousedown: noFocus,
      onclick: restart, title: "Play this game again from the start: same players, same seed" },
    h("span", { html: U.ICONS.restart, style: { display: "inline-grid" } }), "Restart");
    const newButton = h("button", { class: "button button-ghost", type: "button", onmousedown: noFocus, onclick: watchAnother }, "New game");
    const bar = h("div", { class: "watch-bar" }, playButton, stepButton, speedControl, h("span", { class: "spacer" }),
      restartButton, newButton);
    const boardHost = h("div");
    const main = h("div", { class: "game-main" }, bar, boardHost);
    const scoreCard = h("section", { class: "card" });
    const consideredCard = h("section", { class: "card analysis-card considered-card" });
    const movesCard = h("section", { class: "card card-grow" });
    const side = h("aside", { class: "game-side" }, scoreCard, consideredCard, movesCard);
    container.appendChild(h("div", { class: "game watch-game" }, main, side));

    const board = Board.create(boardHost, { alignY: "start" });
    const rackEls = {};
    for (const seat of [0, 1]) {
      const holder = h("div");
      rackEls[seat] = { holder, rack: Board.rack(holder, { tiles: state.racks[seat], size: 28, emptyText: "No tiles" }) };
    }

    // -- keyboard
    function onKey(e) {
      if (!alive || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.target && e.target.closest && e.target.closest("input, textarea, select, [contenteditable]")) return;
      const modal = document.getElementById("modal-root");
      if (modal && modal.childElementCount) return;
      const onButton = e.target && e.target.closest && e.target.closest("button, a, summary");
      if (e.key === " " && !onButton) { e.preventDefault(); setPlaying(!playing); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); stepView(-1); }
      else if (e.key === "ArrowRight") { e.preventDefault(); stepView(1); }
      else if (e.key === "Escape" && viewIndex !== null) { e.preventDefault(); backToLive(); }
    }
    document.addEventListener("keydown", onKey);

    const self = live;
    function dispose() {
      if (!alive) return;
      alive = false;
      self.playing = playing;
      token++;
      board.destroy();
      document.removeEventListener("keydown", onKey);
    }
    /** Stop watching for good: forget the game on the server. */
    function forget() {
      dispose();
      App.del("/api/games/" + encodeURIComponent(state.id)).catch(() => {});
      if (live === self) live = null;
    }
    current = { dispose };
    App.onCleanup(() => {
      // Leaving the page forgets the game; a re-render in place (theme switch) keeps it.
      if (/^#\/watch(\?|$)/.test(location.hash) && live === self) dispose();
      else forget();
    });

    // -- snapshot: the live position, or the one after an earlier move
    function snapshot() {
      const hist = state.history;
      if (viewIndex === null) {
        return { live: true, index: hist.length - 1, scores: state.scores, racks: state.racks,
          toMove: state.over ? null : state.to_move };
      }
      const scores = [0, 0];
      for (let i = 0; i <= viewIndex; i++) scores[hist[i].player] = hist[i].total;
      const racks = [0, 1].map((s) => {
        const next = hist.find((e, i) => i > viewIndex && e.player === s);
        return next ? next.rack : state.racks[s];
      });
      return { live: false, index: viewIndex, scores, racks, toMove: 1 - hist[viewIndex].player };
    }

    // -- rendering
    function updateControls() {
      playButton.replaceChildren(h("span", { html: playing ? U.ICONS.pause : U.ICONS.play, style: { display: "inline-grid" } }),
        playing ? "Pause" : state.over ? "Finished" : "Play");
      playButton.disabled = state.over;
      stepButton.disabled = state.over && viewIndex === null;
      const bits = [];
      if (viewIndex !== null) bits.push("After turn " + fmt.int(state.history[viewIndex].turn) + " of " + fmt.int(state.history.length));
      else bits.push(state.over ? "Final · " + U.plural(state.history.length, "turn") : "Turn " + fmt.int(state.turn));
      if (!state.over && viewIndex === null) bits.push(U.plural(state.bag, "tile") + " in the bag");
      bits.push("seed " + state.seed);
      meta.textContent = bits.join(" · ");
    }

    function lastMoveOf(seat, index) {
      for (let i = index; i >= 0; i--) if (state.history[i].player === seat) return state.history[i];
      return null;
    }

    function renderScore() {
      const snap = snapshot();
      const row = (seat) => {
        const p = state.players[seat];
        const toMove = snap.toMove === seat;
        const last = lastMoveOf(seat, snap.index);
        rackEls[seat].rack.set(snap.racks[seat], { animate: snap.live });
        let flag = null;
        if (toMove && snap.live && (busy || playing)) flag = h("span", { class: "turn-flag" + (busy ? " is-thinking" : "") }, busy ? "Thinking" : "To move");
        else if (toMove && snap.live) flag = h("span", { class: "turn-flag" }, "To move");
        return h("div", { class: "player" + (toMove ? " is-to-move" : "") },
          h("div", { class: "player-id" },
            h("span", { class: "who " + whoClass(seat) }),
            h("span", { class: "player-letter" }, letter(seat)),
            h("span", { class: "player-name", title: p.spec }, p.name),
            h("span", { class: "player-kind" }, U.kindLabel(p.spec, data)),
            seat === 0 ? h("span", { class: "first-flag", title: "Moved first" }, "1st") : null,
            flag ? h("span", { class: "player-flag" }, flag) : null),
          h("div", { class: "player-score" }, fmt.int(snap.scores[seat])),
          h("div", { class: "player-status" }, rackEls[seat].holder,
            last ? h("span", { class: "player-last" }, last.kind === "play" ? [U.moveText(last.text), " ", h("b", null, "+" + last.score)]
              : U.moveText(last.text)) : h("span", { class: "player-last" }, "")));
      };
      scoreCard.replaceChildren(h("div", { class: "score-head" }, meta), row(aSeat), row(bSeat));
    }

    function previewCandidate(c, base) {
      if (!c) {
        if (previewing) {
          showPosition();
          previewing = false;
        }
        return;
      }
      previewing = true;
      board.setPosition(base, [], { animate: false });
      board.preview(c.kind === "play" ? c.tiles : null);
    }

    function renderConsidered() {
      const snap = snapshot();
      const entry = state.history[snap.index];
      if (!entry) {
        consideredCard.replaceChildren(h("div", { class: "card-title" }, "What the mover considered"),
          h("p", { class: "analysis-note" }, failure ? U.sentence(failure)
            : "The options each player weighs appear here as the game is played."));
        return;
      }
      const rec = analyses.get(entry.turn);
      const seat = entry.player;
      if (!rec && entry.analysed && !fetching.has(entry.turn)) {
        // Played before this page had the game (a reload, or a game opened part-way): the server
        // kept what the mover considered.
        fetching.add(entry.turn);
        App.get("/api/games/" + encodeURIComponent(state.id) + "/analysis?turn=" + entry.turn).then((analysis) => {
          if (!alive) return;
          const before = Board.applyTiles(Board.emptyRows(), state.history.filter((e) => e.turn < entry.turn).flatMap((e) => e.tiles));
          analyses.set(entry.turn, { analysis, prevBoard: before });
          renderConsidered();
        }).catch(() => {});
      }
      const head = h("div", { class: "analysis-head" },
        h("div", { class: "card-title" }, h("span", { class: "who " + whoClass(seat) }), "What " + letter(seat) + " considered"),
        h("span", { class: "card-sub" }, "Turn " + entry.turn));
      if (!rec) {
        consideredCard.replaceChildren(head,
          h("div", { class: "analysis-line" }, nameOf(seat), " ", entry.kind === "play" ? "played " : "", U.moveText(entry.text),
            entry.kind === "play" ? " for " + fmt.int(entry.score) : ""),
          h("p", { class: "analysis-note" }, entry.analysed ? "Loading what was considered…"
            : "Not recorded: this move was played before the viewer opened."));
        return;
      }
      const cands = rec.analysis.candidates;
      const rank = cands.findIndex((c) => c.chosen);
      const choice = rank === 0 ? "its top-ranked option" : rank > 0 ? "its option number " + (rank + 1) : "a move outside this list";
      consideredCard.replaceChildren(head,
        h("div", { class: "analysis-line" }, h("strong", null, nameOf(seat)), " ",
          entry.kind === "play" ? ["played ", U.moveText(entry.text), " for " + fmt.int(entry.score)]
            : entry.kind === "exchange" ? "exchanged " + U.plural((entry.exchanged || "").length || 0, "tile") : "passed"),
        h("div", { class: "analysis-sub" }, "Chose " + choice + " of " + U.plural(rec.analysis.considered, "move") + " considered" +
          (entry.leave ? " · kept " + entry.leave : "")),
        U.candidateTable(cands, { limit: window.innerHeight >= 900 ? 8 : 7, onHover: (c) => previewCandidate(c, rec.prevBoard) }));
    }

    function renderMoves() {
      const hist = state.history;
      const selected = viewIndex === null ? null : viewIndex;
      const tabs = h("div", { class: "card-tabs", role: "tablist" },
        [["moves", "Moves"], ["score", "Score over the game"]].map(([key, label]) => h("button", {
          class: "tab", type: "button", role: "tab", "aria-selected": String(tab === key), onmousedown: noFocus,
          onclick: () => {
            tab = key;
            renderMoves();
            if (tab === "score") movesCard.scrollIntoView({ block: "nearest" });
          } }, label)));
      let body;
      if (tab === "moves") {
        if (!hist.length) body = h("p", { class: "empty" }, "No moves yet.");
        else {
          const totals = [0, 0];
          const rows = hist.map((e, i) => {
            totals[e.player] = e.total;
            const tr = h("tr", { class: (selected === i ? "is-selected" : "") + (i === hist.length - 1 ? " is-latest" : "") },
              h("td", { class: "turn" }, String(e.turn)),
              h("td", { class: "left" }, h("span", { class: "who " + whoClass(e.player), title: nameOf(e.player) }), U.moveText(e.text),
                e.bingo ? h("span", { class: "bingo", title: "All seven tiles" }, "BINGO") : null),
              h("td", { class: "pts" }, fmt.int(e.score)),
              h("td", { class: "tot" + (e.player === aSeat ? " tot-mover" : "") }, fmt.int(totals[aSeat])),
              h("td", { class: "tot" + (e.player === bSeat ? " tot-mover" : "") }, fmt.int(totals[bSeat])));
            tr.addEventListener("click", () => (selected === i ? backToLive() : viewAt(i)));
            return tr;
          });
          body = h("table", { class: "history" },
            h("thead", null, h("tr", null, h("th", { class: "left" }, "#"), h("th", { class: "left" }, "Move"), h("th", null, "Score"),
              h("th", { title: nameOf(aSeat) }, "A"), h("th", { title: nameOf(bSeat) }, "B"))),
            h("tbody", null, rows));
        }
      } else {
        body = chartBox;
        drawChart(body);
      }
      const scroll = h("div", { class: "card-scroll" }, body);
      movesCard.classList.toggle("is-chart", tab === "score");
      movesCard.replaceChildren(tabs, scroll);
      if (tab === "moves") {
        requestAnimationFrame(() => {
          if (selected === null) scroll.scrollTop = scroll.scrollHeight;
          else {
            const row = scroll.querySelectorAll("tbody tr")[selected];
            if (row) row.scrollIntoView({ block: "nearest" });
          }
        });
      }
    }

    function drawChart(box) {
      const hist = state.history;
      const series = [aSeat, bSeat].map((seat) => {
        const points = [[0, 0]];
        for (const e of hist) if (e.player === seat) points.push([e.turn, e.total]);
        return { name: letter(seat) + " · " + nameOf(seat), points };
      });
      if (window.Charts && typeof Charts.line === "function") {
        try {
          const options = { series, x: { label: "Turn" }, y: { label: "Score", zero: true }, height: 180 };
          if (chart && chart.update) chart.update(options);
          else chart = Charts.line(box, options);
          if (chart && chart.highlight) {
            if (viewIndex !== null) chart.highlight(hist[viewIndex].turn, "Turn " + hist[viewIndex].turn);
            else chart.highlight(null);
          }
          return;
        } catch (error) {
          console.warn(error);
          chart = null;
          App.clear(box);
        }
      }
      App.clear(box);
      if (!hist.length) { box.appendChild(h("p", { class: "empty" }, "No moves yet.")); return; }
      const totals = [0, 0];
      box.appendChild(h("table", { class: "score-table" },
        h("thead", null, h("tr", null, h("th", null, "Turn"), h("th", null, "A · " + nameOf(aSeat)), h("th", null, "B · " + nameOf(bSeat)))),
        h("tbody", null, hist.map((e) => {
          totals[e.player] = e.total;
          return h("tr", null, h("td", null, String(e.turn)), h("td", null, fmt.int(totals[aSeat])), h("td", null, fmt.int(totals[bSeat])));
        }))));
    }

    function showPosition() {
      const snap = snapshot();
      board.preview(null);
      const banner = board.layer.querySelector(".board-banner");
      if (banner) banner.remove();
      board.el.classList.toggle("is-past", !snap.live);
      if (snap.live) {
        board.setPosition(state.board, state.last_placed, { animate: false });
        return;
      }
      const hist = state.history;
      const rows = Board.applyTiles(Board.emptyRows(), hist.slice(0, viewIndex + 1).flatMap((e) => e.tiles));
      board.setPosition(rows, U.squaresOf(hist[viewIndex].tiles), { animate: false });
      const e = hist[viewIndex];
      board.layer.appendChild(h("div", { class: "board-banner" },
        h("span", null, "After turn ", h("b", null, String(e.turn)), " of " + hist.length + " · " + letter(e.player) + " ",
          U.moveText(e.text)),
        h("button", { type: "button", onclick: backToLive, title: "Back to the live position (Esc)" },
          h("span", { html: U.ICONS.back, style: { display: "inline-grid" } }), "Back to live")));
    }

    function renderAll() {
      renderScore();
      renderConsidered();
      renderMoves();
      updateControls();
    }

    // -- history browsing
    function viewAt(index) {
      if (index === null || index < 0) return;
      if (index >= state.history.length - 1) return backToLive();
      setPlaying(false);
      viewIndex = index;
      previewing = false;
      if (overlay) overlay.hidden = true;
      showPosition();
      renderAll();
    }

    function backToLive() {
      viewIndex = null;
      previewing = false;
      if (overlay) overlay.hidden = false;
      showPosition();
      renderAll();
    }

    function stepView(dir) {
      const last = state.history.length - 1;
      if (viewIndex === null) {
        if (dir < 0 && last >= 1) viewAt(last - 1);
        else if (dir > 0) stepButtonPressed();
        return;
      }
      const next = viewIndex + dir;
      if (next < 0) return;
      if (next >= last) return backToLive();
      viewAt(next);
    }

    // -- playback
    const speedMs = () => (SPEEDS.find((s) => s[0] === speed) || SPEEDS[1])[2];

    function setPlaying(on) {
      const want = !!on && !state.over;
      if (want && viewIndex !== null) backToLive();
      playing = want;
      live.playing = playing;
      token++;
      if (playing) run(token);
      updateControls();
      renderScore();
    }

    async function run(myToken) {
      let first = true;
      while (alive && playing && myToken === token && !state.over) {
        // The first move after pressing Play comes quickly; later ones wait for the chosen speed,
        // re-read every 40 ms so a speed change applies at once.
        const cap = first ? 500 : Infinity;
        first = false;
        const began = Date.now();
        while (alive && playing && myToken === token && Date.now() - began < Math.min(cap, speedMs())) {
          await U.sleep(40);
        }
        if (!alive || !playing || myToken !== token) return;
        await step();
        if (speedMs() === 0) await U.sleep(16);
      }
      if (alive && state.over && myToken === token) {
        playing = false;
        updateControls();
      }
    }

    function stepButtonPressed() {
      if (viewIndex !== null) { stepView(1); return; }
      if (playing) setPlaying(false);
      step();
    }

    async function step() {
      if (busy || state.over || !alive) return;
      busy = true;
      failure = null;
      renderScore();
      const before = state.board;
      try {
        const res = await App.post("/api/games/" + encodeURIComponent(state.id) + "/ai");
        if (!alive) return;
        const entry = res.state.history[res.state.history.length - 1];
        analyses.set(entry.turn, { analysis: res.analysis, prevBoard: before });
        busy = false;
        apply(res.state);
      } catch (error) {
        busy = false;
        if (!alive) return;
        failure = error.message;
        playing = false;
        token++;
        App.toast(U.sentence(error.message), "critical");
        renderAll();
      }
    }

    function apply(next) {
      state = next;
      if (viewIndex === null) {
        previewing = false;
        board.preview(null);
        board.setPosition(state.board, state.last_placed);
      }
      renderAll();
      if (state.over) {
        playing = false;
        token++;
        updateControls();
        showResult();
      }
    }

    // -- end of game
    function facts() {
      const hist = state.history;
      const seats = [aSeat, bSeat];
      const bingos = seats.map((s) => hist.filter((e) => e.player === s && e.bingo));
      const exchanges = seats.map((s) => hist.filter((e) => e.player === s && e.kind === "exchange").length);
      const passes = seats.map((s) => hist.filter((e) => e.player === s && e.kind === "pass").length);
      const best = hist.filter((e) => e.kind === "play").sort((x, y) => y.score - x.score)[0];
      const words = (list) => list.map((e) => (e.word || e.text.split(" ")[1] || "").replace(/[()]/g, "")).join(", ");
      const per = (counts) => seats.map((s, i) => letter(s) + " " + counts[i]).join(", ");
      const items = [
        ["Bingos", seats.map((s, i) => letter(s) + " " + bingos[i].length + (bingos[i].length ? " (" + words(bingos[i]) + ")" : "")).join("; ")],
        ["Highest play", best ? [U.moveText(best.text), " for " + fmt.int(best.score) + " by " + letter(best.player)] : "–"],
        ["Exchanges", exchanges[0] + exchanges[1] ? per(exchanges) : "None"],
        ["Passes", passes[0] + passes[1] ? per(passes) : "None"],
        ["Turns", fmt.int(hist.length)],
      ];
      return h("dl", { class: "result-facts" }, items.map(([k, v]) => [h("dt", null, k), h("dd", null, v)]));
    }

    function showResult() {
      if (overlay) overlay.remove();
      const r = state.result || { scores: state.scores, winner: null, adjustment: [0, 0], ended_by: "out" };
      const margin = Math.abs(r.scores[0] - r.scores[1]);
      const headline = r.winner === null || r.winner === undefined ? "A tie at " + fmt.int(r.scores[0])
        : letter(r.winner) + " won by " + fmt.int(margin);
      const order = [aSeat, bSeat];
      const adj = r.adjustment || [0, 0];
      let note;
      if (r.ended_by === "out") {
        const out = adj[0] > 0 ? 0 : adj[1] > 0 ? 1 : null;
        note = out === null ? "The game ended when a player used their last tile."
          : letter(out) + " went out and gets twice the " + U.plural(adj[out] / 2, "point") + " left on " + letter(1 - out) + "'s rack: +" + fmt.int(adj[out]) + ".";
      } else {
        note = "Six scoreless turns in a row ended the game. Each player loses the value of their rack: A " +
          fmt.signed(adj[aSeat], 0) + ", B " + fmt.signed(adj[bSeat], 0) + ".";
      }
      overlay = h("div", { class: "board-overlay" },
        h("div", { class: "result-card", role: "dialog", "aria-label": "Result" },
          h("button", { class: "result-close", type: "button", title: "Hide the result", "aria-label": "Hide the result",
            onclick: hideResult, html: U.ICONS.clear }),
          h("div", { class: "result-kicker" }, "Game over · seed " + state.seed),
          h("div", { class: "result-headline" }, headline),
          h("div", { class: "result-scores" }, order.map((i) => [
            h("span", { class: "name " + (r.winner === i ? "is-winner" : "is-loser") },
              h("span", { class: "who " + whoClass(i) }), letter(i) + " · " + nameOf(i)),
            h("span", { class: "pts " + (r.winner === i ? "is-winner" : "is-loser") }, fmt.int(r.scores[i])),
          ])),
          h("p", { class: "result-note" }, note),
          facts(),
          h("div", { class: "result-actions" },
            h("button", { class: "button button-primary", type: "button", onclick: watchAnother }, "Watch another"),
            h("button", { class: "button", type: "button", onclick: restart, title: "Same players, same seed, same seats" }, "Replay this game"))));
      board.layer.appendChild(overlay);
      if (viewIndex !== null) overlay.hidden = true;
    }

    function hideResult() {
      if (overlay) overlay.remove();
      overlay = h("button", { class: "button board-pill", type: "button", onclick: showResult }, "Show the result");
      board.layer.appendChild(overlay);
    }

    async function restart() {
      setPlaying(false);
      try {
        await start(container, data, { a: cfg.a, b: cfg.b, first: aSeat === 0 ? "a" : "b", seed: state.seed }, { autoplay: true });
        App.del("/api/games/" + encodeURIComponent(state.id)).catch(() => {});
      } catch (error) {
        App.toast(U.sentence(error.message), "critical");
      }
    }

    function watchAnother() {
      forget();
      App.go("#/watch?" + new URLSearchParams({ a: cfg.a, b: cfg.b, first: cfg.first === "random" ? "random" : (aSeat === 0 ? "a" : "b") }).toString());
    }

    // -- start
    board.setPosition(state.board, state.last_placed, { animate: false });
    renderAll();
    if (state.over) showResult();
    else if (ctx.playing) setPlaying(true);

    return {
      get state() { return state; }, step, viewAt, setPlaying, previewCandidate, board,
      setTab(t) { tab = t; renderMoves(); },
    };
  }

  // ---- Demo states (mock mode only) -------------------------------------------------------

  async function demo(container, q, data) {
    const cfg = { a: q.a || U.defaultOpponent(data), b: q.b || "bot:greedy-leave", first: "a", seed: q.seed ? Number(q.seed) : 7 };
    const turns = { midgame: 13, history: 13, gameover: 22 }[q.demo] || 13;
    const ctl = await start(container, data, cfg, { extra: { mock_turns: turns } });
    await ctl.step();
    if (q.demo === "history") ctl.viewAt(6);
    if (q.tab === "score") ctl.setTab("score");
  }
})();
