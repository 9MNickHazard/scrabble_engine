/* Play: a person against a bot or a trained model, on the board.
 *
 * Setup, then the game screen: the board with your rack under it, the scoreboard, the opponent's
 * reasoning or your hints, the move list and the unseen tiles. The game id is kept in
 * sessionStorage ("play.game"), so leaving the page and coming back offers to resume the game.
 *
 * Mock-only flags (with ?mock=1), for screenshots: #/play?demo=midgame | placing | hint | error |
 * exchange | thinking | gameover | resume, and theme=light | dark.
 */
(function () {
  "use strict";

  const { h, fmt } = App;
  const U = Board.ui;
  const STORE = "play.game";
  const ICON = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="2.8" y="2.8" width="6.4" height="6.4" rx="1.3"/><rect x="10.8" y="2.8" width="6.4" height="6.4" rx="1.3"/><rect x="2.8" y="10.8" width="6.4" height="6.4" rx="1.3"/><path d="M14 11.4v5.2M11.4 14h5.2" stroke-linecap="round"/></svg>';
  const FALLBACK = {
    bots: [
      { spec: "bot:greedy-leave", name: "Greedy + leave", description: "Highest score plus leave value." },
      { spec: "bot:greedy", name: "Greedy", description: "Highest score." },
      { spec: "bot:random", name: "Random", description: "A random legal placement." },
    ],
    models: [],
  };

  let inPlace = false; // the page is re-rendering without being left (theme switch)
  let current = null;  // the game screen showing, retired when another screen replaces it

  function retire() {
    if (current) current.dispose();
    current = null;
  }

  App.registerView("play", { title: "Play", nav: true, order: 40, icon: ICON, mount });

  // ---- Storage --------------------------------------------------------------------------

  function load(key, fallback) {
    try {
      const value = localStorage.getItem(key);
      return value === null ? fallback : JSON.parse(value);
    } catch (error) { return fallback; }
  }
  function store(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (error) { /* private mode */ }
  }
  function saved() {
    try { return JSON.parse(sessionStorage.getItem(STORE) || "null"); } catch (error) { return null; }
  }
  function remember(value) {
    try {
      if (value) sessionStorage.setItem(STORE, JSON.stringify(value));
      else sessionStorage.removeItem(STORE);
    } catch (error) { /* private mode */ }
  }

  async function playerList() {
    try { return await App.get("/api/models"); } catch (error) { return Object.assign({ error: error.message }, FALLBACK); }
  }

  // ---- Mount ----------------------------------------------------------------------------

  async function mount(container, route) {
    const q = route.query || {};
    if (App.mock.enabled && (q.theme === "light" || q.theme === "dark")) document.documentElement.dataset.theme = q.theme;
    const reentered = inPlace;
    inPlace = false;
    App.onCleanup(() => { inPlace = /^#\/play(\?|$)/.test(location.hash); });

    const data = await playerList();
    if (App.mock.enabled && q.demo) return demo(container, q, data);

    const prior = saved();
    let resumable = null;
    if (prior && prior.id) {
      try {
        const state = await App.get("/api/games/" + encodeURIComponent(prior.id));
        if (reentered || q.resume === "1") return game(container, state, prior.setup || {}, data);
        if (!state.over) resumable = { state, setup: prior.setup || {} };
        else remember(null);
      } catch (error) {
        remember(null);
      }
    }
    setup(container, q, data, resumable);
  }

  // ---- Setup ----------------------------------------------------------------------------

  function segmented(choices, value, onChange) {
    const node = h("div", { class: "segmented", role: "group" });
    const buttons = choices.map(([key, label]) => h("button", {
      type: "button", "aria-pressed": String(key === value),
      onclick: () => {
        buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(choices[i][0] === key)));
        onChange(key);
      },
    }, label));
    node.append(...buttons);
    return node;
  }

  function setup(container, q, data, resumable) {
    retire();
    App.clear(container);
    const prefs = load("play.setup", {});
    let first = ["you", "opponent", "random"].includes(q.first) ? q.first : prefs.first || "you";
    let hints = prefs.hints !== false;
    let advisorTouched = false;

    const opponent = U.playerSelect(data, q.opponent || prefs.opponent || U.defaultOpponent(data), { id: "play-opponent" });
    const advisor = U.playerSelect(data, "bot:greedy-leave", { id: "play-advisor" });
    const desc = h("div", { class: "player-desc" });
    const syncAdvisor = () => {
      if (!advisorTouched) advisor.value = opponent.value.startsWith("model:") ? opponent.value : "bot:greedy-leave";
    };
    const describe = () => { desc.textContent = U.specInfo(opponent.value, data).description || ""; };
    opponent.addEventListener("change", () => { describe(); syncAdvisor(); });
    advisor.addEventListener("change", () => { advisorTouched = true; });
    syncAdvisor();
    describe();

    const advisorField = h("label", { class: "field" },
      h("span", { class: "field-label" }, "Hints from"), advisor,
      h("span", { class: "field-help" }, "Who ranks your options when you ask for a hint."));
    advisorField.hidden = !hints;
    const hintToggle = h("label", { class: "toggle" },
      h("input", { type: "checkbox", checked: hints, onchange: (e) => { hints = e.target.checked; advisorField.hidden = !hints; } }),
      "Allow hints");

    const errorBox = h("div");
    const startButton = h("button", { class: "button button-primary", type: "button", onclick: start }, "Start game");

    async function start() {
      startButton.disabled = true;
      errorBox.replaceChildren();
      const s = { opponent: opponent.value, first, hints, advisor: hints ? advisor.value : null };
      store("play.setup", { opponent: s.opponent, first, hints });
      try {
        await create(container, s, data);
      } catch (error) {
        startButton.disabled = false;
        errorBox.replaceChildren(h("div", { class: "notice notice-critical" }, U.sentence(error.message)));
      }
    }

    let resumeNotice = null;
    if (resumable) {
      const st = resumable.state;
      const seat = st.players.findIndex((p) => p.spec === "human");
      const them = st.players[1 - seat];
      resumeNotice = h("div", { class: "notice resume", style: { marginBottom: "var(--space-4)" } },
        h("div", { class: "grow" },
          h("strong", null, "A game against " + them.name + " is in progress. "),
          "Turn " + fmt.int(st.turn) + ", you " + fmt.int(st.scores[seat]) + " to " + fmt.int(st.scores[1 - seat]) + "."),
        h("button", { class: "button button-primary button-small", type: "button",
          onclick: () => game(container, st, resumable.setup, data) }, "Resume"),
        h("button", { class: "button button-ghost button-small", type: "button", onclick: () => {
          App.del("/api/games/" + encodeURIComponent(st.id)).catch(() => {});
          remember(null);
          resumeNotice.remove();
        } }, "Abandon"));
    }

    const kbd = (text) => h("span", { class: "kbd" }, text);
    container.appendChild(h("div", { class: "play-setup" },
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Play"),
          h("p", { class: "lede" }, "Play a game on the board against a bot or one of your trained models."))),
      resumeNotice,
      data.error ? h("div", { class: "notice notice-warning", style: { marginBottom: "var(--space-4)" } },
        "Could not load the model list (" + data.error + "). Only the bots are listed.") : null,
      h("div", { class: "setup" },
        h("div", { class: "card setup-card" },
          h("label", { class: "field" }, h("span", { class: "field-label" }, "Opponent"), opponent, desc),
          h("div", { class: "field" }, h("span", { class: "field-label" }, "Who moves first"),
            h("div", null, segmented([["you", "You"], ["opponent", "Opponent"], ["random", "Random"]], first, (v) => { first = v; }))),
          h("div", { class: "field" }, h("span", { class: "field-label" }, "Hints"), hintToggle),
          advisorField,
          h("div", { class: "setup-actions" }, startButton),
          errorBox),
        h("div", { class: "setup-aside" },
          h("h3", null, "Entering a play"),
          h("ul", null,
            h("li", null, "Click a square and type. Click the square again to switch between across and down."),
            h("li", null, "A letter you do not hold uses a blank if you have one. Hold ", kbd("Shift"), " to play a blank on purpose."),
            h("li", null, kbd("Backspace"), " takes back the last tile, ", kbd("Esc"), " clears them all, ", kbd("Enter"), " plays."),
            h("li", null, "Or drag tiles from the rack to the board and back. Arrow keys move the cursor.")),
          h("p", { class: "muted" }, "The opponent's rack is hidden unless you turn on \"Show rack\". This is a lab, so peeking is allowed.")))));
  }

  async function create(container, s, data, extra) {
    const humanFirst = s.first === "you" || (s.first === "random" && Math.random() < 0.5);
    const body = { players: humanFirst ? ["human", s.opponent] : [s.opponent, "human"] };
    if (s.hints && s.advisor) body.advisor = s.advisor;
    if (s.seed !== undefined && s.seed !== null) body.seed = s.seed;
    if (extra) Object.assign(body, extra);
    const state = await App.post("/api/games", body);
    const prior = saved();
    if (prior && prior.id && prior.id !== state.id) App.del("/api/games/" + encodeURIComponent(prior.id)).catch(() => {});
    const gameSetup = { opponent: s.opponent, first: s.first, hints: s.hints, advisor: s.advisor };
    remember({ id: state.id, setup: gameSetup });
    return game(container, state, gameSetup, data);
  }

  // ---- The game screen ------------------------------------------------------------------

  function game(container, initial, gameSetup, data, demoMode) {
    retire();
    App.clear(container);
    let state = initial;
    const seat = Math.max(0, state.players.findIndex((p) => p.spec === "human"));
    const opp = 1 - seat;
    const them = state.players[opp];
    const hintsAllowed = gameSetup.hints !== false;
    let alive = true;
    let busy = false;
    let thinking = false;
    let errorShown = false;
    let selectedTurn = null;
    let lastAnalysis = null; // {analysis, prevBoard, entry}
    let hints = null;        // {advisor, candidates}
    let previewing = false;
    let showOpp = load("play.showOpponent", false);
    // Open by default only when the window is tall enough to show it beside the move list.
    let considerOpen = demoMode === "midgame" || load("play.considerOpen", window.innerHeight >= 900);
    let overlay = null;

    remember({ id: state.id, setup: gameSetup });

    // -- layout
    const boardHost = h("div");
    const rackRow = h("div", { class: "play-rack-row" });
    const message = h("div", { class: "play-message", role: "status", "aria-live": "polite" });
    const noFocus = (e) => e.preventDefault();
    const button = (label, cls, onclick, attrs) => h("button", Object.assign({ class: "button " + (cls || ""), type: "button",
      onclick, onmousedown: noFocus }, attrs || {}), label);

    const playButton = button([h("span", null, "Play"), h("span", { class: "kbd" }, "Enter")], "button-primary", () => submit(),
      { title: "Play the tiles you placed (Enter)" });
    const hintButton = hintsAllowed ? button([h("span", { html: U.ICONS.hint, style: { display: "inline-grid" } }), "Hint"], "", () => toggleHints(),
      { title: "Ask for a ranked list of your options" }) : null;
    const exchangeButton = button("Exchange", "", () => openExchange());
    const exchangeWrap = h("span", { class: "action-wrap" }, exchangeButton);
    const passButton = button("Pass", "", () => pass());
    const clearButton = h("button", { class: "rack-button", type: "button", title: "Take the placed tiles back (Esc)",
      "aria-label": "Clear the placement", html: U.ICONS.clear, onmousedown: noFocus, onclick: () => board.clearPlacement() });

    const controls = h("div", { class: "play-controls" },
      rackRow,
      h("div", { class: "play-action-row" }, message, hintButton, exchangeWrap, passButton, playButton));
    const main = h("div", { class: "game-main" }, boardHost, controls);

    const scoreCard = h("section", { class: "card" });
    const analysisCard = h("section", { class: "card analysis-card" });
    const historyCard = h("section", { class: "card card-grow" });
    const unseenCard = h("section", { class: "card" });
    const side = h("aside", { class: "game-side" }, scoreCard, analysisCard, historyCard, unseenCard);
    container.appendChild(h("div", { class: "game play-game" }, main, side));

    const rack = Board.rack(rackRow, { tiles: state.racks[seat], ledge: true, shuffle: true });
    rackRow.appendChild(clearButton);
    const oppRack = { holder: h("div") };
    oppRack.rack = Board.rack(oppRack.holder, { tiles: state.racks[opp], size: 26, hidden: !showOpp });

    const board = Board.create(boardHost, {
      rack,
      alignY: "end",
      onChange: onPlacement,
      onSubmit: () => submit(),
      onMessage: (text, tone) => say(text, tone),
      onResize: (px) => main.style.setProperty("--board-px", px + "px"),
    });

    function dispose() {
      if (!alive) return;
      alive = false;
      board.destroy();
    }
    current = { dispose };
    App.onCleanup(dispose);

    // -- messages

    function say(text, tone, extra) {
      message.className = "play-message tone-" + (tone || "info");
      message.replaceChildren(h("span", null, text));
      if (extra) message.appendChild(extra);
      errorShown = tone === "critical";
    }

    function defaultMessage() {
      if (state.over) return say("The game is over.");
      if (state.to_move !== seat) return;
      const n = board.placement().length;
      if (!n) say("Your turn. Click a square and type.");
      else say(U.plural(n, "tile") + " placed. Enter to play, Esc to clear.");
    }

    function onPlacement() {
      if (errorShown && !busy) errorShown = false;
      if (!busy && state.to_move === seat && !state.over) defaultMessage();
      updateButtons();
    }

    function updateButtons() {
      const mine = !state.over && state.to_move === seat && !busy && !thinking;
      const placed = board.placement().length;
      playButton.disabled = !mine || !placed;
      clearButton.disabled = !placed;
      if (hintButton) {
        hintButton.disabled = !mine;
        hintButton.setAttribute("aria-pressed", String(!!hints));
      }
      passButton.disabled = !mine;
      exchangeButton.disabled = !mine || !state.can_exchange;
      exchangeWrap.title = state.can_exchange ? "Put tiles back in the bag and draw new ones"
        : "Exchanges need at least 7 tiles in the bag (" + U.plural(state.bag, "tile") + " left)";
    }

    // -- side cards

    function renderScore() {
      const row = (i) => {
        const p = state.players[i];
        const toMove = !state.over && state.to_move === i;
        const isYou = i === seat;
        let flag = null;
        if (toMove && isYou) flag = h("span", { class: "turn-flag" }, "Your turn");
        else if (toMove && thinking) flag = h("span", { class: "turn-flag is-thinking" }, "Thinking");
        let status = null;
        if (!isYou) {
          // Once the game is over both racks are public.
          oppRack.rack.setHidden(!showOpp && !state.over, state.racks[i].length);
          status = h("div", { class: "player-status" }, oppRack.holder,
            state.over ? null : h("label", { class: "toggle toggle-small", onmousedown: (e) => e.preventDefault() },
              h("input", { type: "checkbox", checked: showOpp, onchange: (e) => {
                showOpp = e.target.checked;
                store("play.showOpponent", showOpp);
                e.target.blur(); // keep typing on the board
                renderScore();
              } }), "Show rack"));
        }
        return h("div", { class: "player" + (toMove ? " is-to-move" : "") },
          h("div", { class: "player-id" },
            h("span", { class: "who " + (isYou ? "who-you" : "who-b") }),
            h("span", { class: "player-name" }, p.name),
            isYou ? null : h("span", { class: "player-kind" }, U.kindLabel(p.spec, data)),
            i === 0 ? h("span", { class: "first-flag", title: "Moved first" }, "1st") : null,
            flag ? h("span", { class: "player-flag" }, flag) : null),
          h("div", { class: "player-score" }, fmt.int(state.scores[i])),
          status);
      };
      scoreCard.replaceChildren(
        h("div", { class: "score-head" },
          h("span", { class: "score-meta" }, state.over ? "Final · " + U.plural(state.history.length, "turn")
            : "Turn " + fmt.int(state.turn) + " · " + U.plural(state.bag, "tile") + " in the bag"),
          h("button", { class: "button button-ghost button-small", type: "button", onclick: newGame }, "New game")),
        row(seat), row(opp));
    }

    function previewCandidate(c, base) {
      if (!c) {
        if (previewing) {
          board.setPosition(state.board, state.last_placed, { animate: false });
          board.preview(null);
          previewing = false;
        }
        return;
      }
      previewing = true;
      board.setPosition(base, base === state.board ? state.last_placed : [], { animate: false });
      board.preview(c.kind === "play" ? c.tiles : null);
    }

    function renderAnalysis() {
      if (hints) {
        analysisCard.hidden = false;
        analysisCard.replaceChildren(
          h("div", { class: "analysis-head" },
            h("div", null, h("div", { class: "card-title" }, "Hints"),
              h("div", { class: "analysis-sub" }, "Ranked by " + hints.advisor + ". Hover to preview, click to place.")),
            h("button", { class: "button button-ghost button-small", type: "button", onclick: () => toggleHints(false) }, "Hide")),
          U.candidateTable(hints.candidates, {
            onHover: (c) => previewCandidate(c, state.board),
            onClick: useHint,
          }));
        return;
      }
      if (!lastAnalysis) {
        analysisCard.hidden = true;
        return;
      }
      analysisCard.hidden = false;
      const { analysis, entry, prevBoard } = lastAnalysis;
      const details = h("details", { open: considerOpen, ontoggle: (e) => { considerOpen = e.target.open; store("play.considerOpen", considerOpen); } },
        h("summary", { onmousedown: (e) => e.preventDefault() }, "What it considered"),
        U.candidateTable(analysis.candidates, { onHover: (c) => previewCandidate(c, prevBoard) }));
      const verb = entry.kind === "play" ? [U.moveText(entry.text), " for " + fmt.int(entry.score)]
        : entry.kind === "exchange" ? ["exchanged " + U.plural(entry.exchanged ? entry.exchanged.length : Number(entry.text.split(" ")[1]) || 0, "tile")]
          : ["passed"];
      analysisCard.replaceChildren(
        h("div", { class: "analysis-line" }, h("strong", null, analysis.player), " ", entry.kind === "play" ? "played " : "", verb),
        h("div", { class: "analysis-sub" },
          (entry.leave ? "Kept " + entry.leave + " · " : "") + "considered " + U.plural(analysis.considered, "move")),
        details);
    }

    function renderHistory() {
      const rows = state.history.map((e, i) => {
        const isYou = e.player === seat;
        const tr = h("tr", { class: (selectedTurn === e.turn ? "is-selected" : "") + (i === state.history.length - 1 ? " is-latest" : "") },
          h("td", { class: "turn" }, String(e.turn)),
          h("td", { class: "left" }, h("span", { class: "who " + (isYou ? "who-you" : "who-b") }), isYou ? "You" : them.name),
          h("td", { class: "left" }, U.moveText(e.text), e.bingo ? h("span", { class: "bingo", title: "All seven tiles" }, "BINGO") : null),
          h("td", { class: "pts" }, fmt.int(e.score)),
          h("td", { class: "tot" }, fmt.int(e.total)));
        tr.addEventListener("click", () => {
          selectedTurn = selectedTurn === e.turn ? null : e.turn;
          board.highlight(selectedTurn === null ? null : U.squaresOf(e.tiles));
          renderHistory();
        });
        return tr;
      });
      const scroll = h("div", { class: "card-scroll" },
        state.history.length
          ? h("table", { class: "history" },
            h("thead", null, h("tr", null, h("th", { class: "left" }, "#"), h("th", { class: "left" }, "Player"),
              h("th", { class: "left" }, "Move"), h("th", null, "Score"), h("th", null, "Total"))),
            h("tbody", null, rows))
          : h("p", { class: "empty" }, "No moves yet."));
      historyCard.replaceChildren(
        h("div", { class: "card-head" }, h("span", { class: "card-title" }, "Moves"),
          h("span", { class: "card-sub" }, state.history.length ? "Click a move to mark its tiles" : "")),
        scroll);
      // After the cards below have laid out, so the latest move is in view.
      if (selectedTurn === null) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
    }

    function renderUnseen() {
      // What this player cannot see: the bag and the other rack. The server gives it per seat.
      const counts = state.unseen_by_seat ? state.unseen_by_seat[seat]
        : state.to_move === seat && state.unseen ? state.unseen : Board.unseen(state.board, state.racks[seat]);
      const theirs = state.racks[opp].length;
      unseenCard.replaceChildren(
        h("div", { class: "card-head" }, h("span", { class: "card-title" }, "Unseen tiles"),
          h("span", { class: "card-sub" }, "The bag and their rack")),
        U.pool(counts, "Bag " + fmt.int(state.bag) + ", their rack " + fmt.int(theirs)));
    }

    function renderAll() {
      renderScore();
      renderAnalysis();
      renderHistory();
      renderUnseen();
      updateButtons();
    }

    // -- turns

    function apply(next) {
      state = next;
      remember({ id: state.id, setup: gameSetup });
      previewing = false;
      board.preview(null);
      board.setPosition(state.board, state.last_placed);
      rack.set(state.racks[seat]);
      oppRack.rack.set(state.racks[opp], { animate: false });
      selectedTurn = null;
      board.highlight(null);
      renderAll();
      if (state.over) return finish();
      if (state.to_move === seat) return yourTurn();
      return theirTurn();
    }

    function yourTurn() {
      board.setInteractive(true);
      defaultMessage();
      updateButtons();
    }

    async function theirTurn() {
      board.setInteractive(false);
      thinking = true;
      renderScore();
      updateButtons();
      say(them.name + " is thinking…", "info");
      message.prepend(h("span", { class: "spinner" }));
      if (demoMode === "thinking") return;
      await U.sleep(650);
      if (!alive) return;
      const before = state.board;
      try {
        const res = await App.post("/api/games/" + encodeURIComponent(state.id) + "/ai");
        if (!alive) return;
        thinking = false;
        const entry = res.state.history[res.state.history.length - 1];
        lastAnalysis = { analysis: res.analysis, prevBoard: before, entry };
        hints = null;
        apply(res.state);
      } catch (error) {
        if (!alive) return;
        thinking = false;
        renderScore();
        say(U.sentence(error.message), "critical",
          h("button", { class: "button button-small", type: "button", onclick: () => theirTurn() }, "Try again"));
      }
    }

    async function submit() {
      if (busy || thinking || state.over || state.to_move !== seat) return;
      const tiles = board.placement();
      if (!tiles.length) {
        say("Place some tiles first: click a square and type, or drag tiles from your rack.", "warning");
        board.shake();
        return;
      }
      return send({ tiles });
    }

    async function send(body) {
      busy = true;
      board.setInteractive(false);
      updateButtons();
      try {
        const next = await App.post("/api/games/" + encodeURIComponent(state.id) + "/move", body);
        if (!alive) return;
        busy = false;
        hints = null;
        board.clearPlacement();
        apply(next);
      } catch (error) {
        if (!alive) return;
        busy = false;
        board.setInteractive(true);
        updateButtons();
        say(U.sentence(error.message), "critical");
      }
    }

    async function toggleHints(force) {
      const want = force === undefined ? !hints : force;
      if (!want) {
        hints = null;
        previewCandidate(null);
        renderAnalysis();
        updateButtons();
        return;
      }
      if (state.to_move !== seat || state.over) return;
      try {
        hints = await App.get("/api/games/" + encodeURIComponent(state.id) + "/hint?top=15");
        renderAnalysis();
        updateButtons();
      } catch (error) {
        say(U.sentence(error.message), "critical");
      }
    }

    function useHint(c) {
      previewCandidate(null);
      if (c.kind === "play") {
        board.setPlacement(c.tiles);
        say("Placed " + c.text.replace(/[()]/g, "") + ". Press Enter to play it.", "info");
      } else if (c.kind === "exchange") {
        openExchange(c.exchanged || c.text.replace(/^exchange\s*/, ""));
      } else {
        pass();
      }
    }

    function openExchange(preselect) {
      if (!state.can_exchange || state.to_move !== seat || busy) return;
      board.clearPlacement();
      const letters = rack.letters().split("");
      const want = (preselect || "").split("");
      const picked = letters.map((L) => {
        const i = want.indexOf(L);
        if (i >= 0) { want.splice(i, 1); return true; }
        return false;
      });
      const confirmButton = h("button", { class: "button button-primary", type: "button", onclick: confirm });
      const note = h("p", { class: "exchange-note" });
      const buttons = letters.map((L, i) => {
        const b = h("button", { type: "button", "aria-pressed": String(picked[i]), "aria-label": L === "?" ? "blank" : L,
          onclick: () => { picked[i] = !picked[i]; b.setAttribute("aria-pressed", String(picked[i])); refresh(); } },
        Board.tile(L));
        return b;
      });
      function refresh() {
        const n = picked.filter(Boolean).length;
        confirmButton.textContent = n ? "Exchange " + U.plural(n, "tile") : "Exchange";
        confirmButton.disabled = !n;
        note.textContent = n ? "You keep " + (letters.filter((_, i) => !picked[i]).join("") || "nothing") + "."
          : "Choose the tiles to put back. Type a letter to pick it.";
      }
      const onKey = (e) => {
        if (e.ctrlKey || e.metaKey || e.altKey) return;
        if (/^[a-z?]$/i.test(e.key)) {
          const L = e.key === "?" ? "?" : e.key.toUpperCase();
          const i = letters.findIndex((x, j) => x === L && !picked[j]);
          const j = i >= 0 ? i : letters.findIndex((x, k) => x === L && picked[k]);
          if (j >= 0) { picked[j] = !picked[j]; buttons[j].setAttribute("aria-pressed", String(picked[j])); refresh(); }
          e.preventDefault();
        } else if (e.key === "Enter" && !confirmButton.disabled) {
          e.preventDefault();
          confirm();
        }
      };
      document.addEventListener("keydown", onKey);
      const close = App.modal({
        title: "Exchange tiles",
        content: h("div", null,
          h("p", null, "You draw the same number of new tiles, then the ones you put back go into the bag. An exchange scores nothing."),
          h("div", { class: "exchange-rack" }, buttons),
          note,
          h("div", { class: "modal-actions", style: { marginTop: "var(--space-4)" } },
            h("button", { class: "button", type: "button", onclick: () => done() }, "Cancel"), confirmButton)),
      });
      const observer = new MutationObserver(() => {
        if (!document.getElementById("modal-root").childElementCount) { document.removeEventListener("keydown", onKey); observer.disconnect(); }
      });
      observer.observe(document.getElementById("modal-root"), { childList: true });
      function done() { close(); }
      function confirm() {
        const chosen = letters.filter((_, i) => picked[i]).join("");
        if (!chosen) return;
        done();
        send({ exchange: chosen });
      }
      refresh();
      return { buttons, picked, refresh };
    }

    async function pass() {
      if (state.to_move !== seat || busy || state.over) return;
      const ok = await App.confirm({
        title: "Pass this turn?",
        body: "You score nothing this turn. Six scoreless turns in a row, counting both players, end the game.",
        confirmLabel: "Pass",
      });
      if (ok) send({ pass: true });
    }

    async function newGame() {
      if (!state.over) {
        const ok = await App.confirm({
          title: "Leave this game?",
          body: "The game against " + them.name + " will be abandoned.",
          confirmLabel: "Leave the game", danger: true,
        });
        if (!ok) return;
        App.del("/api/games/" + encodeURIComponent(state.id)).catch(() => {});
      }
      remember(null);
      setup(container, {}, data, null);
    }

    // -- end of game

    function finish() {
      board.setInteractive(false);
      thinking = false;
      updateButtons();
      say("The game is over.");
      showResult();
    }

    function showResult() {
      if (overlay) overlay.remove();
      const r = state.result || { scores: state.scores, winner: null, adjustment: [0, 0], ended_by: "out" };
      const name = (i) => (i === seat ? "You" : them.name);
      const margin = Math.abs(r.scores[0] - r.scores[1]);
      const headline = r.winner === null || r.winner === undefined ? "A tie at " + fmt.int(r.scores[0])
        : r.winner === seat ? "You won by " + fmt.int(margin) : them.name + " won by " + fmt.int(margin);
      const order = [seat, opp].sort((a, b) => r.scores[b] - r.scores[a]);
      const humanFirst = seat === 0;
      overlay = h("div", { class: "board-overlay" },
        h("div", { class: "result-card", role: "dialog", "aria-label": "Result" },
          h("button", { class: "result-close", type: "button", title: "Hide the result", "aria-label": "Hide the result",
            onclick: hideResult, html: U.ICONS.clear }),
          h("div", { class: "result-kicker" }, r.ended_by === "out" ? "Game over · out in " + U.plural(state.history.length, "turn")
            : "Game over · six scoreless turns"),
          h("div", { class: "result-headline" }, headline),
          h("div", { class: "result-scores" }, order.map((i) => [
            h("span", { class: "name " + (r.winner === i ? "is-winner" : "is-loser") },
              h("span", { class: "who " + (i === seat ? "who-you" : "who-b") }), name(i)),
            h("span", { class: "pts " + (r.winner === i ? "is-winner" : "is-loser") }, fmt.int(r.scores[i])),
          ])),
          h("p", { class: "result-note" }, adjustmentText(r, name, state.racks)),
          h("div", { class: "result-actions" },
            h("button", { class: "button button-primary", type: "button", onclick: () => again(false) }, "Play again"),
            h("button", { class: "button", type: "button", onclick: () => again(true),
              title: "The same tiles from the bag, with " + (humanFirst ? them.name : "you") + " moving first" }, "Rematch, swap who goes first"))));
      board.layer.appendChild(overlay);
    }

    function hideResult() {
      if (overlay) overlay.remove();
      overlay = h("button", { class: "button board-pill", type: "button", onclick: showResult }, "Show the result");
      board.layer.appendChild(overlay);
    }

    async function again(swap) {
      const humanFirst = seat === 0;
      const s = Object.assign({}, gameSetup);
      if (swap) {
        s.first = humanFirst ? "opponent" : "you";
        s.seed = state.seed;
      }
      try {
        await create(container, s, data);
      } catch (error) {
        App.toast(U.sentence(error.message), "critical");
      }
    }

    // -- start

    board.setPosition(state.board, state.last_placed, { animate: false });
    renderAll();
    if (state.over) finish();
    else if (state.to_move === seat) yourTurn();
    else theirTurn();

    return {
      board, rack, get state() { return state; }, openExchange, toggleHints, previewCandidate, send, say,
      set showOpp(v) { showOpp = v; renderScore(); },
    };
  }

  function adjustmentText(r, name, racks) {
    const adj = r.adjustment || [0, 0];
    if (r.ended_by === "out") {
      const out = adj[0] > 0 ? 0 : adj[1] > 0 ? 1 : null;
      if (out === null) return "The game ended when a player used their last tile.";
      const left = adj[out] / 2;
      const who = name(out);
      return who + (who === "You" ? " went out and get" : " went out and gets") + " twice the " + U.plural(left, "point") +
        " left on " + (name(1 - out) === "You" ? "your" : name(1 - out) + "'s") + " rack: +" + fmt.int(adj[out]) + ".";
    }
    return "Six scoreless turns in a row ended the game. Each player loses the value of their own rack: " +
      name(0) + " " + fmt.signed(adj[0], 0) + ", " + name(1) + " " + fmt.signed(adj[1], 0) + ".";
  }

  // ---- Demo states (mock mode only) -------------------------------------------------------

  async function demo(container, q, data) {
    const kind = q.demo;
    const opponent = U.defaultOpponent(data);
    const s = { opponent, first: "you", hints: true, advisor: opponent };
    const base = { seed: 7 };
    if (kind === "setup") return setup(container, q, data, null);
    if (kind === "resume") {
      const state = await App.post("/api/games", { players: ["human", opponent], seed: 7, mock_turns: 8 });
      remember({ id: state.id, setup: s });
      return setup(container, q, data, { state, setup: s });
    }
    if (kind === "gameover") {
      const state = await App.post("/api/games", Object.assign({ players: ["human", opponent], mock_turns: 23 }, base));
      return game(container, state, s, data);
    }
    const state = await App.post("/api/games", Object.assign({ players: ["human", opponent], mock_turns: 11 }, base));
    const ctl = game(container, state, s, data, kind);
    if (kind === "thinking") return;
    // Wait for the opponent's scripted move (13C DITTO) and the human's turn.
    for (let i = 0; i < 100 && ctl.state.to_move !== 0; i++) await U.sleep(50);
    if (q.show === "1") ctl.showOpp = true;
    if (kind === "placing") {
      ctl.board.setPlacement([[9, 13, "Y", false], [10, 13, "A", false], [11, 13, "R", false], [12, 13, "N", false]]);
    } else if (kind === "hint") {
      await ctl.toggleHints(true);
      const res = await App.get("/api/games/" + ctl.state.id + "/hint?top=15");
      ctl.previewCandidate(res.candidates[0], ctl.state.board);
    } else if (kind === "error") {
      ctl.board.setPlacement([[13, 1, "D", false], [13, 2, "A", false], [13, 3, "N", false], [13, 4, "E", false]]);
      await ctl.send({ tiles: ctl.board.placement() });
    } else if (kind === "exchange") {
      ctl.openExchange("EY");
    }
  }
})();
