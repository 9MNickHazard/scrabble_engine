/* Train from games: #/games[?section=sets|train|runs].
 *
 * The second way to train a network (docs/APP_API.md, "Training from saved games"), in two
 * steps. Step 1 makes a game set: games the static evaluator plays against itself on the CPU,
 * saved to disk (GET and POST /api/gamesets, polled while a set is being built). Step 2 trains a
 * network on one, one position from every game per pass: an ordinary POST /api/runs whose
 * overrides name the set in `offline.games`. Below them, every run that trains from games
 * (GET /api/runs, `mode` "games"), judged by its win rate against the static evaluator.
 *
 * The form is built from GET /api/config/schema: the fields whose `modes` include "games", each
 * shown with the default that really applies (`games_defaults` over the schema's), and
 * `games_presets` as the model picker. Only values changed from those defaults are sent; the
 * server applies `games_defaults` itself.
 *
 * Uses window.LabKit from views/training.js (icons, win-rate cells, health chips, sparklines,
 * the rename dialog and the settings-form pieces the Training form uses too).
 *
 * Mock-only demo states (with ?mock=1):
 *   #/games?demo=more | add | log | stop | delete | settings | errors | resume | rename
 */
(function () {
  "use strict";
  const { h, fmt } = App;
  const K = window.LabKit;
  const enc = encodeURIComponent;

  const KEY = { name: "run.name", hours: "run.hours", blocks: "model.blocks", channels: "model.channels", kind: "model.kind",
    set: "offline.games", passes: "offline.passes", batch: "offline.batch_size", held: "offline.validation_games",
    checkpointMinutes: "offline.checkpoint_minutes" };
  // Laid out on their own at the top of the form, not in "All settings".
  const SPECIAL = new Set([KEY.hours, KEY.blocks, KEY.channels, KEY.set, KEY.passes]);
  // The app sets these itself (the server refuses them in overrides).
  const HIDDEN = new Set([KEY.name, "run.out_dir"]);
  const NAME_RE = /^[A-Za-z0-9_-]+$/;
  const ACTIVE = new Set(["running", "queued", "stopping"]);
  const RUN_STATUS = {
    running: { label: "Running", tone: "info", order: 0 },
    stopping: { label: "Stopping", tone: "warning", order: 1 },
    queued: { label: "Queued", tone: "", order: 2 },
    finished: { label: "Finished", tone: "good", order: 3 },
    stopped: { label: "Stopped", tone: "", order: 4 },
    failed: { label: "Failed", tone: "critical", order: 5 },
  };
  const SET_STATUS = {
    building: { label: "Building", tone: "info" },
    stopping: { label: "Stopping", tone: "warning" },
    complete: { label: "Complete", tone: "good" },
    partial: { label: "Partial", tone: "" },
    empty: { label: "Empty", tone: "" },
    failed: { label: "Failed", tone: "critical" },
  };
  const BUSY = new Set(["building", "stopping"]);
  const QUICK = [[1e6, "1 million"], [5e6, "5 million"], [20e6, "20 million"], [50e6, "50 million"]];
  // What a new set gets when the request leaves the opening settings out (docs/APP_API.md,
  // POST /api/gamesets). The API does not report them, so they are restated here; the page sends
  // the two settings only when they are changed, and the server's own values apply otherwise.
  const OPENING = { plies: 4, temperature: 6.0 };

  // Kept across re-mounts (a theme switch, a visit to a run page), so a half-made form survives.
  let S = null;        // the schema, indexed for this form (LabKit.indexSchema in "games" mode)
  let draft = null;    // the run being set up
  let make = null;     // the new game set being set up
  let runsCache = null;
  let setsCache = null;

  App.registerView("games", {
    title: "Train from games", nav: true, order: 22, icon: K.svg("layers", 18), mount,
  });

  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const trimZero = (text) => text.replace(/\.0$/, "");
  /** A server message as a sentence. Only a plain first word is capitalized: one that starts
   *  with a name ("twenty-million is being built") or a setting's key keeps it as it is. */
  const sentence = (text) => {
    const t = String(text || "").trim();
    const head = /^[a-z]+(\s|$)/.test(t) ? t.charAt(0).toUpperCase() : t.charAt(0);
    return head + t.slice(1) + (/[.!?]$/.test(t) ? "" : ".");
  };
  const gamesRuns = () => (runsCache || []).filter(K.isGames);
  const sets = () => (setsCache && setsCache.sets) || [];
  /** A set a run can be trained from: it has games and no process is adding to it. */
  const trainable = (set) => set && (set.games || 0) > 0 && !BUSY.has(set.status);

  function datePart() {
    const d = new Date();
    return String(d.getMonth() + 1).padStart(2, "0") + String(d.getDate()).padStart(2, "0");
  }

  function finishWords(hours) {
    const end = new Date(Date.now() + hours * 3600 * 1000);
    const time = end.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const days = Math.round((new Date(end.toDateString()) - new Date(new Date().toDateString())) / 86400000);
    if (days === 0) return `${time} today`;
    if (days === 1) return `${time} tomorrow`;
    if (days < 7) return `${time} on ${end.toLocaleDateString("en-US", { weekday: "long" })}`;
    return fmt.datetime(end.getTime() / 1000);
  }

  function listNames(names) {
    if (names.length <= 1) return names.join("");
    if (names.length > 3) return `${names.slice(0, 2).join(", ")} and ${names.length - 2} others`;
    return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  }

  /** "pass 2.1 of 6" */
  function passText(run) {
    if (typeof run.pass !== "number") return null;
    const of = typeof run.passes === "number" ? ` of ${trimZero(fmt.fixed(run.passes, 1))}` : "";
    return `pass ${fmt.fixed(run.pass, 1)}${of}`;
  }

  // ---- The page -------------------------------------------------------------------------

  async function mount(container, route) {
    const demo = App.mock.enabled ? route.query.demo || "" : "";
    const refs = {};
    let controls = {};
    let setsError = null;
    let runsError = null;
    let lastSetsKey = "";
    let lastRunsKey = "";
    let lastActiveKey = "";
    let setsAt = 0;
    let resolveSets;
    let resolveRuns;
    const setsLoaded = new Promise((resolve) => { resolveSets = resolve; });
    const runsLoaded = new Promise((resolve) => { resolveRuns = resolve; });

    const setsHost = h("div");
    const trainHost = h("div", null, h("div", { class: "card tm-placeholder" }, h("span", { class: "muted" }, "Loading the training settings…")));
    const runsHost = h("div");
    container.appendChild(h("div", { class: "tm-page gm-page" },
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Train from games"),
          h("p", { class: "lede gm-lede" },
            "First make a game set: millions of games that the static evaluator (the bot that plays the highest score plus " +
            "leave value) plays against itself, saved to disk. Then train a network on it: each pass takes one position from " +
            "every game, and the network learns to predict whether that position ends in a win, a draw or a loss. Unlike the " +
            "Training tab, where a network learns from games it plays against itself, these networks learn from the saved " +
            "games alone and are judged by how often they beat the static evaluator: 50% is level.")),
        h("div", { class: "page-actions" },
          h("button", { type: "button", class: "button button-ghost", onclick: () => runsHost.scrollIntoView({ behavior: "smooth", block: "start" }) },
            K.icon("clock", 16), "Runs"),
          h("a", { class: "button button-ghost", href: "#/guide?from=games", title: "Every setting: what it does, and what turning it down or up does" },
            K.icon("book", 16), "Tuning guide"))),
      setsHost, trainHost, runsHost));

    buildSets();
    buildRuns();
    App.every(3000, () => loadSets(false));
    App.every(5000, () => loadRuns(false));

    try {
      S = K.indexSchema(await App.get("/api/config/schema"), "games");
    } catch (error) {
      trainHost.replaceChildren(h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
        h("div", null, h("strong", null, "Could not load the training settings. "), String(error.message || error))));
      return;
    }
    if (!draft) freshDraft();
    else reconcileDraft();
    renderForm();

    let leavesWere = leavesReady();
    const warningKinds = () => ((App.status && App.status.warnings) || []).map((w) => w.kind).join(",");
    let warningsWere = warningKinds();
    App.onCleanup(App.on("status", () => {
      const ok = leavesReady();
      if (ok !== leavesWere) { leavesWere = ok; renderForm(); }
      else if (warningKinds() !== warningsWere) renderBasket();
      warningsWere = warningKinds();
    }));

    const target = { sets: setsHost, train: trainHost, runs: runsHost }[route.query.section];
    if (target) {
      await Promise.all([setsLoaded, runsLoaded]);
      requestAnimationFrame(() => target.scrollIntoView({ block: "start" }));
    }
    if (demo) {
      await Promise.all([setsLoaded, runsLoaded]);
      applyDemo();
    }

    function leavesReady() {
      const status = App.status;
      return !status || !status.leaves || status.leaves.ok !== false;
    }

    // ---- Loading ------------------------------------------------------------------------

    async function loadSets(force) {
      const busy = sets().some((s) => BUSY.has(s.status));
      // While a set is being built, every 3 s; otherwise now and then, in case one was started elsewhere.
      if (!force && setsCache && !busy && Date.now() - setsAt < 15000) return;
      let data;
      try {
        data = await App.get("/api/gamesets");
      } catch (error) {
        setsError = String(error.message || error);
        if (!setsCache) renderSets();
        resolveSets();
        return;
      }
      setsAt = Date.now();
      setsError = null;
      setsCache = { sets: data.sets || [], estimate: data.estimate || {} };
      const key = JSON.stringify(data);
      if (key !== lastSetsKey || force) {
        lastSetsKey = key;
        renderSets();
        if (make && refs.makeEstimate) { checkMakeName(); refreshMake(); }
        if (draft && S && refs.basket) {
          chooseSet();
          renderSetPicker();
          refreshEditor();
          renderBasket();
        }
      }
      resolveSets();
    }

    async function loadRuns(force) {
      let data;
      try {
        data = await App.get("/api/runs");
      } catch (error) {
        runsError = String(error.message || error);
        if (!runsCache) renderRuns();
        resolveRuns();
        return;
      }
      runsError = null;
      runsCache = data.runs || [];
      const key = JSON.stringify(runsCache);
      if (key !== lastRunsKey || force) {
        lastRunsKey = key;
        renderRuns();
        if (draft && S && refs.basket) {
          if (draft.autoName && nameTaken(draft.name)) {
            draft.name = suggestRunName();
            if (refs.nameInput) refs.nameInput.value = draft.name;
          }
          refreshEditor();
          if (activeKey() !== lastActiveKey) renderBasket();
        }
      }
      resolveRuns();
    }

    function activeKey() {
      return (runsCache || []).filter((r) => ACTIVE.has(r.status)).map((r) => r.name + ":" + r.status).join(",");
    }

    // ---- Step 1: game sets ----------------------------------------------------------------

    function buildSets() {
      refs.setsTable = h("div", { class: "tm-table-wrap" });
      refs.makeHost = h("div", { class: "gm-make" });
      setsHost.replaceChildren(h("section", { class: "card card-flush gm-card", id: "gm-sets", "aria-labelledby": "gm-sets-title" },
        h("div", { class: "gm-card-head" },
          h("div", null,
            h("div", { class: "gm-step" }, "Step 1"),
            h("h2", { id: "gm-sets-title" }, "Make a game set"),
            h("p", { class: "card-sub" }, "Games the static evaluator plays against itself, saved to disk to learn from. " +
              "Building a set uses every CPU core and no GPU, and carries on with the app closed."))),
        refs.setsTable, refs.makeHost));
      renderSets();
      renderMake();
    }

    function renderSets() {
      if (!setsCache) {
        refs.setsTable.replaceChildren(setsError
          ? h("div", { class: "tm-pad" }, h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
            h("div", null, h("strong", null, "Could not load the game sets. "), setsError)))
          : h("div", { class: "empty" }, "Loading game sets…"));
        return;
      }
      const list = sets();
      if (!list.length) {
        refs.setsTable.replaceChildren(h("div", { class: "empty tm-empty gm-empty" },
          h("span", { class: "tm-empty-icon" }, K.icon("layers", 28)),
          h("strong", null, "No game sets yet"),
          h("span", null, "Make the first one below. It is listed here while it is being built.")));
        return;
      }
      refs.setsTable.replaceChildren(h("table", { class: "table tm-table gm-sets-table" },
        h("thead", null, h("tr", null,
          h("th", { scope: "col" }, "Game set"),
          h("th", { scope: "col" }, "Status"),
          h("th", { scope: "col", class: "num", title: "Games in the set, and the number it is being built to" }, "Games"),
          h("th", { scope: "col", class: "num", title: "Space the set takes on disk" }, "On disk"),
          h("th", { scope: "col", class: "num", title: "Games played a second while it was built, and the time spent" }, "Speed"),
          h("th", { scope: "col", class: "num", title: "A player's mean score; bingos (both players together) and turns per game" }, "Per game"),
          h("th", { class: "tm-actions-cell" }, h("span", { class: "tm-sr" }, "Actions")))),
        h("tbody", null, list.map(setRow))));
    }

    function setRow(set) {
      const st = SET_STATUS[set.status] || { label: set.status, tone: "" };
      const busy = BUSY.has(set.status);
      const games = set.games || 0;
      const target = set.target || 0;
      const opening = typeof set.opening_plies === "number"
        ? `Opening varied over up to ${set.opening_plies} plies, at ${trimZero(fmt.fixed(set.opening_temperature, 1))} points` : null;
      let errorLine = null;
      if (set.status === "failed" && set.error) {
        const lines = String(set.error).trim().split("\n").map((l) => l.trim()).filter(Boolean);
        const line = lines.find((l) => /(Error|Exception)\b/.test(l)) || lines[lines.length - 1];
        errorLine = h("span", { class: "tm-sub tm-error-line", title: set.error }, line);
      }
      const gps = num(set.games_per_second);
      const left = busy && gps && target > games ? (target - games) / gps / 3600 : null;
      return h("tr", { class: busy ? "tm-row-active" : null },
        h("td", { class: "gm-set-cell" },
          h("span", { class: "gm-set-name" }, set.name),
          opening ? h("span", { class: "tm-sub" }, opening) : null,
          set.created ? h("span", { class: "tm-sub", title: `Started ${fmt.datetime(set.created)}` }, `Started ${fmt.ago(set.created)}`) : null,
          errorLine),
        h("td", { class: "tm-status-cell" },
          h("div", { class: "tm-status-line" }, h("span", { class: "badge" + (st.tone ? " badge-" + st.tone : "") }, st.label),
            busy ? h("span", { class: "dot dot-live", "aria-hidden": "true" }) : null)),
        h("td", { class: "num", title: target ? `${fmt.int(games)} of ${fmt.int(target)} games` : `${fmt.int(games)} games` },
          target && target !== games ? [fmt.compact(games), h("span", { class: "muted" }, ` of ${fmt.compact(target)}`)] : fmt.compact(games),
          busy && typeof set.progress === "number"
            ? h("span", { class: "tm-progress-line gm-progress" },
              h("span", { class: "progress", role: "progressbar", "aria-valuenow": Math.round(set.progress * 100), "aria-valuemin": 0, "aria-valuemax": 100,
                "aria-label": `${set.name}: ${fmt.pct(set.progress, 0)} built` },
              h("span", { style: { width: `${Math.min(100, set.progress * 100)}%` } })),
              h("span", { class: "tm-sub" }, fmt.pct(set.progress, 0)))
            : null),
        h("td", { class: "num" }, set.bytes ? fmt.bytes(set.bytes) : "–"),
        h("td", { class: "num" }, gps ? `${fmt.int(gps)}/s` : "–",
          left !== null ? h("span", { class: "tm-sub" }, `about ${fmt.duration(left)} to go`)
            : set.seconds ? h("span", { class: "tm-sub" }, `${fmt.duration(set.seconds / 3600)} of play`) : null),
        h("td", { class: "num" }, num(set.mean_score) !== null ? `${fmt.fixed(set.mean_score, 0)} points` : "–",
          num(set.bingos_per_game) !== null
            ? h("span", { class: "tm-sub" }, `${fmt.fixed(set.bingos_per_game, 2)} bingos` +
              (num(set.mean_turns) !== null ? ` · ${fmt.fixed(set.mean_turns, 1)} turns` : ""))
            : null),
        h("td", { class: "tm-actions-cell" }, setActions(set)));
    }

    function setActions(set) {
      const busy = BUSY.has(set.status);
      const items = [];
      if (set.status === "building") {
        items.push(h("button", { type: "button", class: "button button-small", onclick: () => stopSet(set, false) }, K.icon("stop", 13), "Stop"));
      } else if (set.status === "stopping") {
        items.push(h("button", { type: "button", class: "button button-small", title: "Kill the process now instead of waiting for it to finish its block",
          onclick: () => stopSet(set, true) }, K.icon("stop", 13), "Force stop"));
      } else {
        items.push(h("button", { type: "button", class: "button button-small", title: `Carry on building ${set.name}, up to a larger number of games`,
          onclick: () => addGames(set) }, K.icon("plus", 14), "Add more games"));
      }
      items.push(h("button", { type: "button", class: "button button-small button-ghost", onclick: () => openLog(set),
        title: `The output of the processes that built ${set.name}` }, "Log"));
      items.push(h("button", { type: "button", class: "button button-small button-ghost tm-iconbtn", disabled: busy,
        "aria-label": `Delete ${set.name}`, title: busy ? "Stop building the set before deleting it" : `Delete ${set.name}`,
        onclick: () => deleteSet(set) }, K.icon("trash", 15)));
      return h("div", { class: "tm-actions" }, items);
    }

    /** How much disk `games` games take, and how long they take to play, from the estimate. */
    function estimateOf(games) {
      const e = (setsCache && setsCache.estimate) || {};
      const bytes = num(e.bytes_per_game) !== null ? e.bytes_per_game * games : null;
      const rate = num(e.games_per_second);
      return { bytes, rate, hours: rate ? games / rate / 3600 : null };
    }

    function estimateNode(games, adding) {
      if (!(games > 0)) return null;
      const e = estimateOf(games);
      const what = `${fmt.int(games)} ${adding ? "more " : ""}game${games === 1 ? "" : "s"}`;
      const disk = e.bytes !== null ? `about ${fmt.bytes(e.bytes)} on disk` : null;
      let main;
      let sub;
      if (e.hours !== null) {
        main = `${what} take ${disk ? disk + " and " : ""}about ${fmt.duration(e.hours)} to play, at ${fmt.int(e.rate)} games a second.`;
        sub = `That is the speed of the latest set built on this PC. Done around ${finishWords(e.hours)} if it starts now.`;
      } else {
        main = `${what} take ${disk || "an unknown amount of disk"}.`;
        sub = "The time is not known until a set has been timed on this PC.";
      }
      return h("div", { class: "tm-speed gm-estimate" + (e.hours === null ? " is-unknown" : "") }, K.icon("gauge", 18),
        h("div", null, h("p", { class: "tm-speed-main" }, main), h("p", { class: "tm-speed-sub" }, sub)));
    }

    function quickPicks(current, onPick, above = 0) {
      return h("div", { class: "tm-sizes gm-quick", role: "group", "aria-label": "Number of games" },
        h("span", { class: "tm-sizes-label" }, "Quick picks"),
        QUICK.filter(([n]) => n > above).map(([n, label]) => h("button", {
          type: "button", class: "tm-size" + (n === current ? " is-active" : ""), "aria-pressed": n === current ? "true" : "false",
          onclick: () => onPick(n),
        }, label)));
    }

    // -- the new set form --

    function freshMake() {
      return { name: suggestSetName(), auto: true, games: 20e6, gamesText: null, plies: OPENING.plies, temperature: OPENING.temperature,
        more: false, busy: false, error: null, attempted: false };
    }

    function setNameTaken(name) {
      const lower = String(name || "").trim().toLowerCase();
      return sets().some((s) => s.name.toLowerCase() === lower);
    }

    function suggestSetName() {
      const base = `set-${datePart()}`;
      for (let i = 0; i < 26; i += 1) {
        const name = `${base}-${String.fromCharCode(97 + i)}`;
        if (!setNameTaken(name)) return name;
      }
      for (let i = 2; ; i += 1) if (!setNameTaken(`${base}-${i}`)) return `${base}-${i}`;
    }

    function checkMakeName() {
      if (make && make.auto && setNameTaken(make.name)) {
        make.name = suggestSetName();
        if (refs.makeName) refs.makeName.value = make.name;
      }
    }

    function makeErrors() {
      const errors = {};
      const name = make.name.trim();
      if (!name) errors.name = "Give the set a name.";
      else if (!NAME_RE.test(name)) errors.name = "Use only letters, digits, - and _.";
      else if (name.length > 64) errors.name = "Use at most 64 characters.";
      else if (setNameTaken(name)) errors.name = "A game set with this name exists. To add games to it, use Add more games on its row.";
      if (!(Number.isInteger(make.games) && make.games >= 1)) errors.games = "Enter a whole number of games, at least 1.";
      if (!(Number.isInteger(make.plies) && make.plies >= 0)) errors.plies = "Enter a whole number, 0 or more.";
      if (!(typeof make.temperature === "number" && Number.isFinite(make.temperature) && make.temperature >= 0)) errors.temperature = "Enter a number of points, 0 or more.";
      return errors;
    }

    function numberBox(id, value, unit, onInput) {
      const input = h("input", { type: "text", id, inputmode: "decimal", autocomplete: "off", spellcheck: "false", value });
      const frame = h("div", { class: "tm-num" }, input, unit ? h("span", { class: "tm-num-unit" }, unit) : null);
      frame.addEventListener("mousedown", (event) => { if (event.target !== input) { event.preventDefault(); input.focus(); } });
      input.addEventListener("input", () => onInput(input.value));
      return { input, frame };
    }

    function renderMake() {
      if (!make) make = freshMake();
      const nameInput = h("input", { class: "input", id: "gm-set-name", type: "text", value: make.name, spellcheck: "false",
        autocomplete: "off", maxlength: 64 });
      nameInput.addEventListener("input", () => { make.name = nameInput.value; make.auto = false; make.error = null; refreshMake(); });
      refs.makeName = nameInput;
      refs.makeNameErr = h("div", { class: "tm-field-error", hidden: true });
      const games = numberBox("gm-set-games", make.gamesText !== null ? make.gamesText : fmt.int(make.games), "games", (text) => {
        const n = K.parseNumber(text);
        make.games = n === null ? NaN : n;
        make.gamesText = text;
        make.error = null;
        refreshMake();
      });
      games.input.addEventListener("blur", () => { if (Number.isInteger(make.games)) { make.gamesText = null; games.input.value = fmt.int(make.games); } });
      refs.makeGames = games;
      refs.makeGamesErr = h("div", { class: "tm-field-error", hidden: true });
      refs.makeQuick = h("div");
      refs.makeEstimate = h("div", { "aria-live": "polite" });
      refs.makeError = h("div", { hidden: true });
      refs.makeButton = h("button", { type: "button", class: "button button-primary", onclick: submitMake }, K.icon("play", 15), "Make game set");

      const more = h("div", { class: "gm-more" + (make.more ? " is-open" : "") },
        h("button", { type: "button", class: "tm-all-toggle", "aria-expanded": make.more ? "true" : "false", "aria-controls": "gm-more-body",
          onclick: () => { make.more = !make.more; renderMake(); } }, K.icon("chevron", 16, "tm-chev"), "More options"),
        h("span", { class: "tm-all-summary" }, "How the openings of the games are varied"));
      if (make.more) {
        const plies = numberBox("gm-plies", String(make.plies), "plies", (text) => {
          const n = K.parseNumber(text); make.plies = n === null ? NaN : n; refreshMake();
        });
        const temp = numberBox("gm-temp", String(make.temperature), "points", (text) => {
          const n = K.parseNumber(text); make.temperature = n === null ? NaN : n; refreshMake();
        });
        refs.makePliesErr = h("div", { class: "tm-field-error", hidden: true });
        refs.makeTempErr = h("div", { class: "tm-field-error", hidden: true });
        refs.makePlies = plies;
        refs.makeTemp = temp;
        more.appendChild(h("div", { class: "gm-more-body", id: "gm-more-body" },
          h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-plies" }, "Opening plies"), plies.frame, refs.makePliesErr,
            h("p", { class: "field-help" }, "Each game varies some of its first moves, anywhere from none to this many (half of it on average). " +
              "On those moves the static evaluator draws its move from its best 16 by score plus leave value instead of always taking the best.")),
          h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-temp" }, "Opening temperature"), temp.frame, refs.makeTempErr,
            h("p", { class: "field-help" }, "How freely those moves are drawn, in points: a move this many points worse than the best is chosen " +
              "about a third as often. After the opening, every move is the static evaluator's first choice.")),
          h("p", { class: "field-help gm-more-note" }, "A set keeps the opening settings it was made with; games added to it later are made the same way.")));
      } else {
        refs.makePliesErr = refs.makeTempErr = refs.makePlies = refs.makeTemp = null;
      }

      refs.makeHost.replaceChildren(
        h("div", { class: "gm-make-head" }, h("h3", { class: "tm-block-title" }, "New game set")),
        h("div", { class: "gm-make-grid" },
          h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-set-name" }, "Name"), nameInput, refs.makeNameErr),
          h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-set-games" }, "Number of games"), games.frame, refs.makeGamesErr)),
        refs.makeQuick,
        more,
        refs.makeEstimate,
        h("p", { class: "gm-note" }, K.icon("alert", 14),
          h("span", null, "Building uses every CPU core and no GPU, so the PC is slow for anything else while it runs. It carries on with the app " +
            "closed; stop it at any time and the games made so far are kept, to carry on with later.")),
        refs.makeError,
        h("div", { class: "gm-make-actions" }, refs.makeButton));
      refreshMake();
    }

    function refreshMake() {
      if (!refs.makeEstimate) return;
      const errors = makeErrors();
      const show = (node, frame, message) => {
        if (!node) return;
        const visible = !!message;
        node.hidden = !visible;
        if (visible) node.replaceChildren(K.icon("alert", 14), h("span", null, message));
        if (frame) frame.classList.toggle("is-error", !!visible);
      };
      show(refs.makeNameErr, refs.makeName, errors.name);
      show(refs.makeGamesErr, refs.makeGames && refs.makeGames.frame, errors.games);
      show(refs.makePliesErr, refs.makePlies && refs.makePlies.frame, errors.plies);
      show(refs.makeTempErr, refs.makeTemp && refs.makeTemp.frame, errors.temperature);
      refs.makeQuick.replaceChildren(h("div", { class: "gm-quick-wrap" },
        quickPicks(make.games, (n) => {
          make.games = n; make.gamesText = null; make.error = null;
          refs.makeGames.input.value = fmt.int(n);
          refreshMake();
        }),
        h("p", { class: "field-help gm-quick-note" }, "How many is enough: 5.4 million was too few in the one published study; about 20 million was enough.")));
      refs.makeEstimate.replaceChildren(...[estimateNode(errors.games ? 0 : make.games, false)].filter(Boolean));
      refs.makeError.hidden = !make.error;
      if (make.error) {
        refs.makeError.replaceChildren(h("div", { class: "notice notice-critical", role: "alert" }, K.icon("alert", 18, "tm-tone-critical"),
          h("div", null, h("strong", null, "The server did not start the set. "), make.error)));
      }
      refs.makeButton.disabled = make.busy;
      refs.makeButton.lastChild.textContent = make.busy ? "Starting…" : "Make game set";
    }

    async function submitMake() {
      make.attempted = true;
      const errors = makeErrors();
      if (Object.keys(errors).length) {
        if ((errors.plies || errors.temperature) && !make.more) { make.more = true; renderMake(); }
        refreshMake();
        const first = errors.name ? refs.makeName : errors.games ? refs.makeGames.input
          : errors.plies ? refs.makePlies.input : refs.makeTemp && refs.makeTemp.input;
        if (first) first.focus();
        return;
      }
      const body = { name: make.name.trim(), games: make.games };
      if (make.plies !== OPENING.plies) body.opening_plies = make.plies;
      if (make.temperature !== OPENING.temperature) body.opening_temperature = make.temperature;
      make.busy = true;
      make.error = null;
      refreshMake();
      try {
        await App.post("/api/gamesets", body);
        App.toast(`Building ${body.name}: ${fmt.int(body.games)} games.`, "good");
        make = freshMake();
        await loadSets(true);
        make.name = suggestSetName();
        renderMake();
      } catch (error) {
        make.busy = false;
        make.error = sentence(error.message || error);
        refreshMake();
      }
    }

    // -- row actions --

    function addGames(set) {
      const have = set.games || 0;
      const next = QUICK.map(([n]) => n).find((n) => n > have) || Math.max(have * 2, have + 1e6);
      let value = next;
      const box = numberBox("gm-add-games", fmt.int(next), "games", (text) => {
        const n = K.parseNumber(text); value = n === null ? NaN : n; refresh();
      });
      const quick = h("div");
      const estimate = h("div", { "aria-live": "polite" });
      const errors = h("div", { class: "tm-resume-errors", role: "alert", hidden: true });
      const go = h("button", { type: "button", class: "button button-primary" }, K.icon("play", 15), "Add games");
      const cancel = h("button", { type: "button", class: "button" }, "Cancel");
      const problem = () => (!Number.isInteger(value) ? "Enter a whole number of games."
        : value <= have ? `Ask for more than the ${fmt.int(have)} it has.` : null);
      const showError = (text) => {
        errors.hidden = !text;
        errors.replaceChildren(...(text ? [h("p", null, K.icon("alert", 14), h("span", null, text))] : []));
      };
      function refresh() {
        quick.replaceChildren(quickPicks(value, (n) => { value = n; box.input.value = fmt.int(n); refresh(); }, have));
        const bad = problem();
        box.frame.classList.toggle("is-error", !!bad && !errors.hidden);
        estimate.replaceChildren(...[bad ? null : estimateNode(value - have, true)].filter(Boolean));
        go.disabled = !!bad;
      }
      const content = h("div", { class: "tm-resume" },
        h("p", null, `${set.name} has ${fmt.int(have)} games. Building carries on from there up to the number below, ` +
          "with the opening settings the set was made with."),
        h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-add-games" }, "Games in the set when it is done"), box.frame),
        quick, estimate,
        h("p", { class: "field-help" }, "Building uses every CPU core and no GPU, and carries on with the app closed. " +
          "A run trained from this set cannot be resumed once the set has grown."),
        errors,
        h("div", { class: "modal-actions" }, cancel, go));
      const close = App.modal({ title: `Add games to ${set.name}`, content });
      cancel.addEventListener("click", close);
      go.addEventListener("click", async () => {
        const bad = problem();
        if (bad) { showError(bad); return; }
        go.disabled = true;
        try {
          await App.post("/api/gamesets", { name: set.name, games: value });
          close();
          App.toast(`Building ${set.name} up to ${fmt.int(value)} games.`, "good");
          await loadSets(true);
        } catch (error) {
          go.disabled = false;
          showError(sentence(error.message || error));
        }
      });
      refresh();
      setTimeout(() => { box.input.focus(); box.input.select(); }, 0);
    }

    function openLog(set) {
      const pre = h("pre", { class: "log gm-log", tabindex: "0", "aria-label": `Log of ${set.name}` }, "Loading the log…");
      const note = h("span", { class: "dash-live" });
      let closed = false;
      let first = true;
      const load = async () => {
        const current = sets().find((s) => s.name === set.name) || set;
        const live = BUSY.has(current.status);
        note.replaceChildren(...(live ? [h("span", { class: "dot dot-live" }), "Following the log"] : ["Not being built"]));
        if (!first && !live) return;
        try {
          const res = await App.get(`/api/gamesets/${enc(set.name)}/log?tail=200`);
          if (closed) return;
          const atBottom = first || pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 16;
          pre.textContent = (res.lines || []).length ? res.lines.join("\n") : "The log is empty.";
          if (atBottom) pre.scrollTop = pre.scrollHeight;
        } catch (error) {
          if (!closed && first) pre.textContent = String(error.message || error);
        }
        first = false;
      };
      const timer = setInterval(() => { if (!closed && !document.hidden) load(); }, 3000);
      App.modal({
        title: `Log of ${set.name}`, wide: true,
        content: h("div", { class: "stack-tight" },
          h("div", { class: "row row-between" }, h("span", { class: "muted" }, "The last 200 lines of the processes that built the set"), note), pre),
        onClose: () => { closed = true; clearInterval(timer); },
      });
      load();
    }

    async function stopSet(set, force) {
      const ok = await App.confirm(force
        ? { title: `Force stop ${set.name}?`, confirmLabel: "Force stop", danger: true,
          body: "This kills the process at once. The set keeps every block of games that was complete; the block in progress is lost. You can carry on later with Add more games." }
        : { title: `Stop building ${set.name}?`, confirmLabel: "Stop building",
          body: "It stops once it has finished its current block of games, so nothing is lost. You can carry on later with Add more games, and train from it as it is." });
      if (!ok) return;
      try {
        await App.post(`/api/gamesets/${enc(set.name)}/stop`, force ? { force: true } : {});
        App.toast(force ? `${set.name} was stopped.` : `Stopping ${set.name} after its current block.`, "info");
        await loadSets(true);
      } catch (error) {
        App.toast(`Could not stop ${set.name}: ${sentence(error.message || error)}`, "critical");
      }
    }

    async function deleteSet(set) {
      const users = gamesRuns().filter((r) => r.game_set === set.name).map((r) => r.name);
      const ok = await App.confirm({
        title: `Delete ${set.name}?`, confirmLabel: "Delete game set", danger: true,
        body: h("div", { class: "stack-tight" },
          h("p", null, `This deletes the game set and its ${fmt.int(set.games || 0)} games` + (set.bytes ? ` (${fmt.bytes(set.bytes)} on disk).` : ".")),
          users.length ? h("p", null, `${listNames(users)} ${users.length > 1 ? "were" : "was"} trained from it: ` +
            `${users.length > 1 ? "their networks stay, but they" : "its networks stay, but it"} can no longer be resumed.`) : null,
          h("p", null, h("strong", null, "This cannot be undone."))),
      });
      if (!ok) return;
      try {
        await App.del(`/api/gamesets/${enc(set.name)}`);
        App.toast(`Deleted ${set.name}.`, "info");
        await loadSets(true);
      } catch (error) {
        App.toast(`Could not delete ${set.name}: ${sentence(error.message || error)}`, "critical");
      }
    }

    // ---- Step 2: train a network ----------------------------------------------------------

    function nameTaken(name) {
      const lower = String(name || "").trim().toLowerCase();
      return (runsCache || []).some((r) => r.name.toLowerCase() === lower);
    }

    function suggestRunName() {
      const base = `games-${datePart()}`;
      for (let i = 0; i < 26; i += 1) {
        const name = `${base}-${String.fromCharCode(97 + i)}`;
        if (!nameTaken(name)) return name;
      }
      for (let i = 2; ; i += 1) if (!nameTaken(`${base}-${i}`)) return `${base}-${i}`;
    }

    function freshDraft() {
      const values = {};
      for (const field of S.fields) values[field.key] = field.default;
      draft = { values, name: suggestRunName(), autoName: true, set: null, mode: "queue", showAll: false, search: "",
        attempted: false, busy: false, serverError: null, serverField: null };
      chooseSet();
    }

    function reconcileDraft() {
      for (const field of S.fields) if (!(field.key in draft.values)) draft.values[field.key] = field.default;
      draft.busy = false;
      chooseSet();
    }

    /** Keep the chosen set if it can still be trained from; otherwise the first that can. */
    function chooseSet() {
      if (!draft || !setsCache) return;
      const current = sets().find((s) => s.name === draft.set);
      if (trainable(current)) return;
      const first = sets().find(trainable);
      draft.set = first ? first.name : null;
    }

    function chosenSet() { return sets().find((s) => s.name === draft.set) || null; }
    function isTransformer() { return draft.values[KEY.kind] === "transformer"; }
    function formFields() { return S.fields.filter((f) => !HIDDEN.has(f.key)); }
    function advancedFields() { return formFields().filter((f) => f.advanced && !SPECIAL.has(f.key)); }

    function activePreset() {
      return S.presets.find((p) => S.presetKeys.every((k) =>
        K.same(draft.values[k], p.overrides && k in p.overrides ? p.overrides[k] : S.byKey[k].default))) || null;
    }

    function applyPreset(preset) {
      for (const key of S.presetKeys) {
        const overrides = preset.overrides || {};
        draft.values[key] = key in overrides ? overrides[key] : S.byKey[key].default;
      }
    }

    function overridesOf() {
      const out = {};
      for (const field of formFields()) {
        if (field.key === KEY.set) continue;
        if (!K.same(draft.values[field.key], field.default)) out[field.key] = draft.values[field.key];
      }
      out[KEY.set] = draft.set;
      return out;
    }

    function errorsOf() {
      const errors = {};
      const name = draft.name.trim();
      if (!name) errors[KEY.name] = "Give the run a name.";
      else if (!NAME_RE.test(name)) errors[KEY.name] = "Use only letters, digits, - and _.";
      else if (nameTaken(name)) errors[KEY.name] = "A run with this name already exists.";
      for (const field of formFields()) {
        if (field.key === KEY.set) continue;
        const error = K.fieldError(field, draft.values[field.key]);
        if (error) errors[field.key] = error;
      }
      if (!draft.set && sets().some(trainable)) errors[KEY.set] = "Choose a game set to learn from.";
      const server = draft.serverField;
      if (server && !errors[server.key]) errors[server.key] = server.message;
      return errors;
    }

    /** What training will work through: positions and steps per pass, as offline.py works them out. */
    function planOf() {
      const set = chosenSet();
      const passes = num(draft.values[KEY.passes]);
      const batchSize = num(draft.values[KEY.batch]);
      const held = num(draft.values[KEY.held]);
      if (!set || !(set.games > 0) || passes === null || !(batchSize > 0) || held === null) return null;
      const heldOut = Math.min(held, Math.floor(set.games / 10));
      const perPassPositions = set.games - heldOut;
      if (perPassPositions < 1) return null;
      const batch = Math.min(batchSize, perPassPositions);
      const stepsPerPass = Math.max(1, Math.floor(perPassPositions / batch));
      const totalSteps = Math.max(1, Math.ceil(passes * stepsPerPass));
      return { set, games: set.games, heldOut, perPassPositions, batch, stepsPerPass, totalSteps, passes };
    }

    function shareContext() {
      const all = runsCache || [];
      const active = all.filter((r) => r.status === "running" || r.status === "stopping");
      const queued = all.filter((r) => r.status === "queued");
      const modeVisible = active.length + queued.length > 0;
      return { active, queued, modeVisible, parallel: modeVisible && draft.mode === "parallel" };
    }

    function renderForm() {
      const ok = leavesReady();
      refs.notices = h("div", { class: "tm-notices" });
      refs.editor = h("div", { class: "tm-editor" });
      refs.basket = h("aside", { class: "tm-basket", "aria-label": "Run to start" });
      trainHost.replaceChildren(h("section", { class: "card tm-new gm-train", id: "gm-train", "aria-labelledby": "gm-train-title" },
        h("div", { class: "tm-new-head" },
          h("div", { class: "gm-step" }, "Step 2"),
          h("h2", { id: "gm-train-title" }, "Train a network"),
          h("p", { class: "card-sub" }, "A network learns from a game set, one position from every game per pass. Anything you leave alone " +
            "keeps its default for training from games. Training uses the GPU, like a self-play run.")),
        h("fieldset", { class: "tm-fieldset", disabled: !ok },
          h("div", { class: "tm-new-grid" }, h("div", { class: "tm-left" }, refs.notices, refs.editor), refs.basket))));
      renderNotices();
      renderEditor();
      renderBasket();
    }

    function renderNotices() {
      const items = [];
      if (!leavesReady()) {
        items.push(h("div", { class: "notice notice-warning" }, K.icon("alert", 18, "tm-tone-warning"),
          h("div", null, h("strong", null, "Leave values have not been fitted yet. "),
            "Game sets and training both rank moves with them. In a terminal in the project folder run ",
            h("code", { class: "tm-code" }, "uv run scrabble-ai fit-leaves"), ". This form unlocks when they are ready.")));
      }
      if (draft.serverError) {
        items.push(h("div", { class: "notice notice-critical", role: "alert" }, K.icon("alert", 18, "tm-tone-critical"),
          h("div", null, h("strong", null, "The server did not start the run. "), draft.serverError)));
      }
      refs.notices.replaceChildren(...items);
      refs.notices.hidden = !items.length;
    }

    function renderEditor() {
      controls = {};
      const parts = [setBlock(), presetBlock(), lengthBlock()];
      const net = networkBlock();
      if (net) parts.push(net);
      const others = formFields().filter((f) => !f.advanced && !SPECIAL.has(f.key));
      if (others.length) {
        parts.push(h("div", { class: "tm-block" }, h("h3", { class: "tm-block-title" }, "Also"),
          h("div", { class: "tm-ess-grid" }, others.map((f) => fieldCell(f)))));
      }
      const all = allSettingsBlock();
      if (all) parts.push(all);
      refs.editor.replaceChildren(...parts);
      renderSetPicker();
      refreshEditor();
    }

    function setBlock() {
      refs.setPicker = h("div", { class: "gm-setcards", role: "radiogroup", "aria-label": "Game set to learn from" });
      refs.setErr = h("div", { class: "tm-field-error", hidden: true });
      refs.setNote = h("p", { class: "field-help", hidden: true });
      return h("div", { class: "tm-block" },
        h("div", { class: "tm-block-head" }, h("h3", { class: "tm-block-title" }, "Game set to learn from"),
          h("span", { class: "tm-block-note" }, K.icon("lock", 12), "Fixed for the life of the run")),
        refs.setPicker, refs.setErr, refs.setNote);
    }

    function renderSetPicker() {
      if (!refs.setPicker) return;
      if (!setsCache) {
        refs.setPicker.replaceChildren(h("p", { class: "muted" }, "Loading game sets…"));
        return;
      }
      const usable = sets().filter(trainable);
      const building = sets().filter((s) => BUSY.has(s.status));
      if (!usable.length) {
        refs.setPicker.replaceChildren(h("div", { class: "notice gm-noset" }, K.icon("layers", 18),
          h("div", null, h("strong", null, building.length ? "No game set is ready to learn from yet. " : "Make a game set first. "),
            building.length
              ? `${listNames(building.map((s) => s.name))} ${building.length > 1 ? "are" : "is"} being built; a set can be trained from once it has stopped or is complete.`
              : "Step 1 above makes one: millions of games of the static evaluator against itself, saved to disk.")));
        refs.setNote.hidden = true;
        return;
      }
      refs.setPicker.replaceChildren(...usable.map((set) => {
        const chosen = set.name === draft.set;
        const st = SET_STATUS[set.status] || { label: set.status };
        return h("button", {
          type: "button", class: "tm-preset gm-setcard" + (chosen ? " is-active" : ""), role: "radio", "aria-checked": chosen ? "true" : "false",
          onclick: () => { if (draft.set !== set.name) { draft.set = set.name; clearServer(KEY.set); renderSetPicker(); refreshEditor(); renderBasket(); } },
        },
        h("span", { class: "tm-preset-top" }, h("span", { class: "tm-radio", "aria-hidden": "true" }), h("span", { class: "tm-preset-name" }, set.name)),
        h("span", { class: "gm-setcard-games" }, `${fmt.int(set.games)} games`,
          set.status !== "complete" ? h("span", { class: "muted" }, ` · ${st.label.toLowerCase()}`) : null),
        num(set.mean_score) !== null ? h("span", { class: "tm-preset-desc" }, `${fmt.fixed(set.mean_score, 0)} points a player · ` +
          `${fmt.fixed(set.bingos_per_game, 2)} bingos a game`) : null);
      }));
      refs.setNote.hidden = !building.length;
      refs.setNote.textContent = building.length
        ? `${listNames(building.map((s) => s.name))} ${building.length > 1 ? "are" : "is"} being built and can be chosen once ${building.length > 1 ? "they have" : "it has"} stopped.` : "";
    }

    function presetBlock() {
      refs.presets = h("div", { class: "tm-presets gm-models", role: "radiogroup", "aria-label": "Model" });
      refs.presetNote = h("span", { class: "tm-block-note" });
      return h("div", { class: "tm-block" },
        h("div", { class: "tm-block-head" }, h("h3", { class: "tm-block-title" }, "Model"), refs.presetNote),
        refs.presets);
    }

    function renderPresets() {
      if (!refs.presets) return;
      const active = activePreset();
      refs.presets.replaceChildren(...S.presets.map((preset) => h("button", {
        type: "button", class: "tm-preset" + (preset === active ? " is-active" : ""), role: "radio",
        "aria-checked": preset === active ? "true" : "false",
        onclick: () => { applyPreset(preset); clearServer(null); renderEditor(); renderBasket(); },
      }, h("span", { class: "tm-preset-top" }, h("span", { class: "tm-radio", "aria-hidden": "true" }),
        h("span", { class: "tm-preset-name" }, preset.name)),
      preset.description ? h("span", { class: "tm-preset-desc" }, preset.description) : null)));
      refs.presetNote.textContent = active ? "" : "Custom: no model matches the settings below";
    }

    function lengthBlock() {
      const input = h("input", { class: "input", id: "gm-run-name", type: "text", value: draft.name, spellcheck: "false",
        autocomplete: "off", maxlength: 64, "aria-describedby": "gm-run-name-hint" });
      input.addEventListener("input", () => {
        draft.name = input.value;
        draft.autoName = false;
        clearServer(KEY.name);
        refreshEditor();
        renderBasket();
      });
      refs.nameInput = input;
      const err = h("div", { class: "tm-field-error", hidden: true });
      refs.nameHint = h("div", { class: "tm-name-hint", id: "gm-run-name-hint" });
      controls[KEY.name] = { field: { key: KEY.name, label: "Run name", type: "str" }, frame: input, err, kind: "name", input };
      const cells = [h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-run-name" }, "Run name"), input, err, refs.nameHint)];
      if (S.byKey[KEY.hours]) cells.push(fieldCell(S.byKey[KEY.hours]));
      if (S.byKey[KEY.passes]) cells.push(fieldCell(S.byKey[KEY.passes]));
      return h("div", { class: "tm-block" },
        h("h3", { class: "tm-block-title" }, "Name and length"),
        h("div", { class: "tm-ess-grid" }, cells),
        h("p", { class: "field-help" }, "A pass takes one position from every game in the set. With a time limit of 0 the run stops when its " +
          "passes are done; otherwise at whichever comes first. A run keeps going when the app is closed."));
    }

    function networkBlock() {
      const blocks = S.byKey[KEY.blocks];
      const channels = S.byKey[KEY.channels];
      if (!blocks || !channels) return null;
      const locked = blocks.locked_on_resume || channels.locked_on_resume;
      refs.netNote = h("p", { class: "field-help" });
      return h("div", { class: "tm-block" },
        h("div", { class: "tm-block-head" }, h("h3", { class: "tm-block-title" }, "Network size"),
          locked ? h("span", { class: "tm-block-note" }, K.icon("lock", 12), "Fixed for the life of the run") : null),
        h("div", { class: "tm-network" },
          h("div", { class: "tm-network-inputs" }, fieldCell(blocks), h("span", { class: "tm-times", "aria-hidden": "true" }, "×"), fieldCell(channels))),
        refs.netNote);
    }

    /** A labelled control in the grid at the top. */
    function fieldCell(field) {
      const id = "gm-e-" + field.key.replace(/[^a-z0-9]/gi, "-");
      const control = K.settingControl(field, draft.values, id, () => onValue(field.key));
      const err = h("div", { class: "tm-field-error", hidden: true });
      const label = h("label", { class: "field-label", for: id, title: field.help || null }, field.label);
      controls[field.key] = Object.assign(control, { field, err, label, kind: "cell" });
      return h("div", { class: "field" }, label, control.node, err);
    }

    function allSettingsBlock() {
      const fields = advancedFields();
      if (!fields.length) return null;
      const sectionCount = S.sections.filter((s) => s.fields.some((f) => fields.includes(f))).length;
      refs.allCount = h("span", { class: "tm-pill tm-pill-accent" });
      refs.allErrors = h("span", { class: "tm-pill tm-pill-critical" });
      const toggle = h("button", { type: "button", class: "tm-all-toggle", "aria-expanded": draft.showAll ? "true" : "false",
        "aria-controls": "gm-all-body", onclick: () => { draft.showAll = !draft.showAll; renderEditor(); } },
      K.icon("chevron", 16, "tm-chev"), "All settings");
      const head = h("div", { class: "tm-all-head" }, toggle, refs.allCount, refs.allErrors,
        h("span", { class: "tm-all-summary" }, `${fields.length} more in ${sectionCount} sections, as they apply to training from games · `,
          h("a", { href: "#/guide?from=games", title: "Every setting in one table: what it does, lower, higher, typical range" }, "tuning guide")));
      const wrap = h("div", { class: "tm-all" + (draft.showAll ? " is-open" : "") }, head);
      refs.sections = [];
      refs.noMatch = null;
      refs.resetAll = null;
      if (!draft.showAll) return wrap;

      const search = h("input", { class: "input", type: "search", placeholder: "Search names, help and effects", value: draft.search,
        "aria-label": "Search settings by name, key, help or effect" });
      search.addEventListener("input", () => { draft.search = search.value; applySearch(); });
      refs.resetAll = h("button", { type: "button", class: "button button-small button-ghost",
        onclick: () => { for (const f of fields) draft.values[f.key] = f.default; clearServer(null); renderEditor(); renderBasket(); } },
      K.icon("reset", 14), "Reset all");
      head.appendChild(h("div", { class: "tm-all-tools" }, h("div", { class: "tm-search" }, K.icon("search", 15), search), refs.resetAll));

      const body = h("div", { class: "tm-all-body", id: "gm-all-body" });
      for (const section of S.sections) {
        const list = section.fields.filter((f) => fields.includes(f));
        if (!list.length) continue;
        const node = h("section", { class: "tm-section" },
          h("div", { class: "tm-section-head" }, h("h4", { class: "tm-section-title" }, section.title || section.name),
            h("code", { class: "tm-key" }, `[${section.name}]`)),
          section.description ? h("p", { class: "tm-section-desc" }, section.description) : null,
          list.map((f) => {
            const ctl = K.settingRow(f, draft.values, () => onValue(f.key), "gm-f-");
            controls[f.key] = ctl;
            return ctl.row;
          }));
        refs.sections.push({ node, fields: list });
        body.appendChild(node);
      }
      refs.noMatch = h("div", { class: "tm-nomatch", hidden: true });
      body.appendChild(refs.noMatch);
      wrap.appendChild(body);
      applySearch();
      return wrap;
    }

    function applySearch() {
      if (!refs.noMatch) return;
      let shown = 0;
      for (const section of refs.sections) {
        let any = false;
        for (const field of section.fields) {
          const match = K.settingMatches(field, draft.search);
          controls[field.key].row.hidden = !match;
          if (match) { any = true; shown += 1; }
        }
        section.node.hidden = !any;
      }
      refs.noMatch.hidden = shown > 0;
      refs.noMatch.replaceChildren(...K.noMatchContent(draft.search));
    }

    function refreshEditor() {
      if (!refs.editor) return;
      const errors = errorsOf();
      for (const [key, ctl] of Object.entries(controls)) {
        let error = errors[key] || null;
        if (key === KEY.name && draft.autoName && !draft.attempted && !nameTaken(draft.name)) error = null;
        ctl.err.hidden = !error;
        if (error) ctl.err.replaceChildren(K.icon("alert", 14), h("span", null, error));
        if (ctl.frame) ctl.frame.classList.toggle("is-error", !!error);
        if (ctl.kind !== "row") continue;
        K.markSettingRow(ctl, draft.values[key], ctl.field.default, error,
          () => { draft.values[key] = ctl.field.default; ctl.set(ctl.field.default); onValue(key); });
      }
      if (refs.setErr) {
        const error = errors[KEY.set] && (draft.attempted || (draft.serverField && draft.serverField.key === KEY.set)) ? errors[KEY.set] : null;
        refs.setErr.hidden = !error;
        if (error) refs.setErr.replaceChildren(K.icon("alert", 14), h("span", null, error));
      }
      if (refs.nameHint) {
        const ok = !errors[KEY.name];
        refs.nameHint.hidden = !ok;
        if (ok) refs.nameHint.replaceChildren(K.icon("check", 13), `Saved to runs/${draft.name.trim()}/`);
      }
      // A transformer's "blocks" are its layers and its "channels" the width of each square's token.
      const t = isTransformer();
      const labels = { [KEY.blocks]: t ? "Layers" : null, [KEY.channels]: t ? "Token width" : null };
      for (const key of [KEY.blocks, KEY.channels]) {
        const ctl = controls[key];
        if (ctl && ctl.label) ctl.label.textContent = labels[key] || ctl.field.label;
      }
      if (refs.netNote) {
        refs.netNote.textContent = t
          ? "For the transformer: the number of layers, and the width of the token each square of the board becomes."
          : "Blocks of the convolutional network, and the channels in each.";
      }
      renderPresets();
      if (refs.allCount) {
        const fields = advancedFields();
        const changed = fields.filter((f) => !K.same(draft.values[f.key], f.default)).length;
        refs.allCount.textContent = `${changed} changed`;
        refs.allCount.hidden = !changed;
        const bad = fields.filter((f) => errors[f.key]).length;
        refs.allErrors.textContent = `${bad} to fix`;
        refs.allErrors.hidden = !bad;
        if (refs.resetAll) refs.resetAll.disabled = !changed;
      }
    }

    function clearServer(key) {
      if (draft.serverField && (key === null || draft.serverField.key === key)) {
        draft.serverField = null;
        draft.serverError = null;
        renderNotices();
      } else if (!draft.serverField && draft.serverError && key === null) {
        draft.serverError = null;
        renderNotices();
      }
    }

    function onValue(key) {
      clearServer(key);
      refreshEditor();
      renderBasket();
    }

    // -- the run to start --

    function renderBasket() {
      if (!refs.basket) return;
      lastActiveKey = activeKey();
      const ctx = shareContext();
      const plan = planOf();
      const preset = activePreset();
      const errors = errorsOf();
      const errorCount = Object.keys(errors).length;
      const items = [h("div", { class: "tm-basket-head" }, h("h3", null, "Run to start"))];
      const card = h("div", { class: "tm-runcard gm-runcard is-selected" },
        h("div", { class: "tm-runcard-top" }, h("span", { class: "tm-runcard-name" }, draft.name.trim() || "Unnamed run"),
          errorCount ? h("span", { class: "tm-runcard-err", title: `${errorCount} setting${errorCount > 1 ? "s" : ""} to fix` }, K.icon("alert", 14)) : null),
        h("div", { class: "tm-runcard-meta" },
          h("span", { class: "tm-runcard-net" }, K.networkLabel({ mode: "games", network: `${draft.values[KEY.blocks]}x${draft.values[KEY.channels]}`,
            changed: { [KEY.kind]: draft.values[KEY.kind] } })),
          preset ? null : h("span", null, "custom")),
        h("div", { class: "tm-runcard-rate" }, draft.set ? `From ${draft.set}` : "No game set chosen"));
      items.push(card);
      items.push(planBlock(plan));
      if (ctx.modeVisible) items.push(modeBlock(ctx));
      const warnings = [];
      for (const w of (App.status && App.status.warnings) || []) warnings.push({ tone: "warning", text: w.message });
      if (errorCount && (draft.attempted || draft.serverField)) {
        warnings.push({ tone: "critical", text: `${errorCount} setting${errorCount > 1 ? "s" : ""} to fix before starting.` });
      }
      if (warnings.length) {
        items.push(h("div", { class: "gm-warnings" }, warnings.map((w) => h("p", { class: "tm-warn tm-warn-" + w.tone }, K.icon("alert", 14), h("span", null, w.text)))));
      }
      const queueing = ctx.modeVisible && !ctx.parallel;
      const label = draft.busy ? "Starting…" : queueing ? "Add to the queue" : "Start training";
      items.push(h("button", { type: "button", class: "button button-primary tm-start", disabled: draft.busy || !leavesReady() || !draft.set,
        title: draft.set ? null : "Choose a game set first", onclick: submit }, K.icon("play", 15), label));
      refs.basket.replaceChildren(h("div", { class: "tm-basket-inner" }, items));
    }

    function planBlock(plan) {
      if (!plan) {
        return h("div", { class: "tm-summary" }, h("p", { class: "tm-summary-sub" }, !draft.set
          ? "Choose a game set to see how many positions and steps the run takes."
          : "Fix the settings marked in red to see how many positions and steps the run takes."));
      }
      const hours = num(draft.values[KEY.hours]) || 0;
      const passes = trimZero(fmt.fixed(plan.passes, plan.passes < 1 ? 2 : 1)).replace(/(\.\d)0$/, "$1");
      return h("div", { class: "tm-summary" },
        h("dl", { class: "kv gm-plan" },
          h("dt", null, "Positions per pass"), h("dd", null, fmt.int(plan.perPassPositions)),
          h("dt", null, "Steps per pass"), h("dd", null, fmt.int(plan.stepsPerPass), h("span", { class: "muted" }, ` of ${fmt.int(plan.batch)} positions`)),
          h("dt", null, "Total steps"), h("dd", null, fmt.int(plan.totalSteps), h("span", { class: "muted" }, ` · ${passes} pass${plan.passes === 1 ? "" : "es"}`))),
        h("p", { class: "tm-summary-sub" }, `One position from each of the set's ${fmt.int(plan.games)} games, less the ${fmt.int(plan.heldOut)} ` +
          "held out to check the network on games it never trains on."),
        h("p", { class: "tm-summary-sub" }, hours > 0
          ? `It stops after ${trimZero(fmt.fixed(hours, 1))} hour${hours === 1 ? "" : "s"} if the passes are not done by then.`
          : "No time limit: it stops when its passes are done."));
    }

    function modeBlock(ctx) {
      const running = ctx.active.map((r) => r.name);
      const waiting = ctx.queued.map((r) => r.name);
      const bits = [];
      if (running.length) bits.push(`${listNames(running)} ${running.length > 1 ? "are" : "is"} running`);
      if (waiting.length) bits.push(`${listNames(waiting)} ${waiting.length > 1 ? "are" : "is"} queued`);
      const existing = running.length + waiting.length;
      return h("div", { class: "tm-mode" },
        h("div", { class: "tm-mode-label" }, "When to start it"),
        h("div", { class: "segmented tm-mode-seg", role: "group", "aria-label": "When to start it" },
          [["queue", "One after another"], ["parallel", "At the same time"]].map(([mode, label]) => h("button", {
            type: "button", "aria-pressed": draft.mode === mode ? "true" : "false",
            onclick: () => { draft.mode = mode; renderBasket(); },
          }, label))),
        h("p", { class: "tm-mode-note" }, `${bits.join(" and ")}. One after another, the new run waits until ${existing > 1 ? "they finish" : "it finishes"}.`),
        h("p", { class: "tm-mode-note" }, "At the same time, the runs share the GPU and each goes slower, so it saves little time in all."));
    }

    async function submit() {
      draft.attempted = true;
      const errors = errorsOf();
      const keys = Object.keys(errors);
      if (keys.length) {
        const order = [KEY.set, KEY.name, ...S.fields.map((f) => f.key)];
        keys.sort((a, b) => order.indexOf(a) - order.indexOf(b));
        const target = S.byKey[keys[0]];
        if (target && target.advanced && !SPECIAL.has(target.key)) { draft.showAll = true; draft.search = ""; }
        renderEditor();
        renderBasket();
        const ctl = controls[keys[0]];
        const node = keys[0] === KEY.set ? refs.setPicker : ctl && ctl.input;
        if (node) {
          if (node.focus) node.focus({ preventScroll: true });
          node.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        return;
      }
      draft.busy = true;
      draft.serverError = null;
      draft.serverField = null;
      renderNotices();
      renderBasket();
      const ctx = shareContext();
      const name = draft.name.trim();
      const body = { mode: ctx.modeVisible ? draft.mode : "queue", runs: [{ name, overrides: overridesOf() }] };
      try {
        const result = await App.post("/api/runs", body);
        const started = result.started || [];
        const queued = result.queued || [];
        const parts = [];
        if (started.length) parts.push(`Started ${listNames(started)}.`);
        if (queued.length) parts.push(`${listNames(queued)} ${queued.length > 1 ? "are" : "is"} queued.`);
        App.toast(parts.join(" ") || "Started.", "good");
        const keep = draft.set;
        freshDraft();
        draft.set = keep;
        chooseSet();
        renderForm();
        await loadRuns(true);
        App.refreshStatus();
        runsHost.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        draft.busy = false;
        mapServerError(String(error.message || error));
        renderNotices();
        renderEditor();
        renderBasket();
        trainHost.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    /** Show the server's message at the top, and beside the setting it names when it names one. */
    function mapServerError(message) {
      draft.serverError = sentence(message);
      const name = draft.name.trim();
      let text = message;
      if (name && text.startsWith(name + ":")) text = text.slice(name.length + 1).trim();
      const field = formFields().filter((f) => f.key !== KEY.set && message.includes(f.key))
        .sort((a, b) => message.indexOf(a.key) - message.indexOf(b.key) || b.key.length - a.key.length)[0];
      let key = field ? field.key : null;
      if (!key && /game set|offline\.games/i.test(message)) key = KEY.set;
      if (!key && /\bname\b|already exists|already in use/i.test(message)) key = KEY.name;
      if (!key) return;
      draft.serverField = { key, message: sentence(text) };
      if (S.byKey[key] && S.byKey[key].advanced && !SPECIAL.has(key)) { draft.showAll = true; draft.search = ""; }
    }

    // ---- Runs trained from games -----------------------------------------------------------

    function buildRuns() {
      refs.runsSub = h("p", { class: "card-sub" }, "Loading…");
      refs.runsTable = h("div", { class: "tm-table-wrap" });
      refs.runsFoot = h("div", { class: "tm-footnote", hidden: true }, winLegend());
      runsHost.replaceChildren(h("section", { class: "card card-flush tm-history gm-runs", id: "gm-runs", "aria-labelledby": "gm-runs-title" },
        h("div", { class: "tm-history-head" },
          h("div", { class: "tm-history-title" }, h("h2", { id: "gm-runs-title" }, "Runs trained from games"), refs.runsSub)),
        refs.runsTable, refs.runsFoot));
    }

    function winLegend() {
      return [
        h("span", null, "Win rate: share of games won against the static evaluator over seat-swapped pairs, ± its 95% interval. 50% is level."),
        h("span", { class: "tm-legend-item" }, h("span", { class: "tm-sig tm-sig-up" }, h("span", { class: "tm-icon", html: K.svg("up", 10) })),
          "interval above 50%: it wins more than it loses"),
        h("span", { class: "tm-legend-item" }, h("span", { class: "tm-sig tm-sig-down" }, h("span", { class: "tm-icon", html: K.svg("down", 10) })),
          "interval below 50%"),
      ];
    }

    function renderRuns() {
      for (const chart of refs.sparks || []) chart.destroy();
      refs.sparks = [];
      if (!runsCache) {
        refs.runsSub.textContent = runsError ? "" : "Loading…";
        refs.runsTable.replaceChildren(runsError
          ? h("div", { class: "tm-pad" }, h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
            h("div", null, h("strong", null, "Could not load the runs. "), runsError)))
          : h("div", { class: "empty" }, "Loading runs…"));
        return;
      }
      const list = gamesRuns().slice().sort((a, b) => {
        const order = (r) => (r.status === "running" || r.status === "stopping" ? 0 : r.status === "queued" ? 1 : 2);
        return order(a) - order(b) || (b.created || 0) - (a.created || 0);
      });
      const positions = list.reduce((sum, r) => sum + (r.positions || 0), 0);
      const hours = list.reduce((sum, r) => sum + (r.hours || 0), 0);
      refs.runsSub.textContent = list.length
        ? `${list.length} run${list.length > 1 ? "s" : ""}: ${fmt.duration(hours)} of training on ${fmt.compact(positions)} positions in all. ` +
          "Each is judged by how often it beats the static evaluator."
        : "None yet.";
      refs.runsFoot.hidden = !list.length;
      if (!list.length) {
        refs.runsTable.replaceChildren(h("div", { class: "empty tm-empty" },
          h("span", { class: "tm-empty-icon" }, K.icon("training", 28)),
          h("strong", null, "No runs trained from games yet"),
          h("span", null, "Start one above. It appears here straight away, and keeps going when the app is closed.")));
        return;
      }
      refs.runsTable.replaceChildren(h("table", { class: "table tm-table tm-history-table gm-runs-table" },
        h("thead", null, h("tr", null,
          h("th", { scope: "col", title: "Click a row, or Open, for the run's page: every chart, the report, settings, log and checkpoints" }, "Run"),
          h("th", { scope: "col", title: "Where the run is, and below it the verdict on its win rate (hover a verdict for the reason)" }, "Status and verdict"),
          h("th", { scope: "col", class: "tm-trend-cell", title: "Every match against the static evaluator, oldest first, with its 95% interval shaded. The grey line is 50%: level." }, "Win rate"),
          h("th", { scope: "col", class: "num", title: "Share of games won against the static evaluator at the latest match, ± its 95% interval; below it, the spread in points, which is secondary for these runs" }, "vs static bot"),
          h("th", { scope: "col", class: "num", title: "Passes through the game set done, and positions trained a second" }, "Progress"),
          h("th", { scope: "col", class: "num", title: "Time trained, and when the run last reported" }, "Time"),
          h("th", { class: "tm-actions-cell" }, h("span", { class: "tm-sr" }, "Actions")))),
        h("tbody", null, list.map(runRow))));
    }

    function runRow(run) {
      const status = RUN_STATUS[run.status] || { label: run.status, tone: "" };
      const href = "#/run/" + enc(run.name);
      const stop = (event) => event.stopPropagation();
      const live = run.status === "running" || run.status === "stopping";
      const e = run.latest_eval && run.latest_eval["greedy-leave"];
      const spark = K.trendSpark(run);
      if (spark.chart) refs.sparks.push(spark.chart);
      let errorLine = null;
      if (run.status === "failed" && run.error) {
        const lines = String(run.error).trim().split("\n").map((l) => l.trim()).filter(Boolean);
        const line = lines.find((l) => /(Error|Exception)\b/.test(l)) || lines[lines.length - 1];
        errorLine = h("span", { class: "tm-sub tm-error-line", title: run.error }, line);
      }
      const healthChip = K.healthChip(run.health, "tm-health", true);
      const pass = passText(run);
      const tr = h("tr", { class: "clickable tm-run-row" + (live || run.status === "queued" ? " tm-row-active" : ""),
        title: `Open ${run.name}: charts, report, settings, log and checkpoints`,
        onclick: (event) => { if (!window.getSelection().toString()) App.go(href); else event.preventDefault(); } },
      h("td", { class: "tm-name-cell" },
        h("a", { class: "tm-run-link", href, onclick: stop }, run.name, K.icon("chevron", 13, "tm-run-link-go")),
        h("span", { class: "tm-sub", title: run.parameters ? `${fmt.int(run.parameters)} parameters` : null },
          h("span", { class: "tm-net" }, K.networkLabel(run)), run.game_set ? ` · from ${run.game_set}` : ""),
        errorLine),
      h("td", { class: "tm-status-cell" },
        h("div", { class: "tm-status-line" }, h("span", { class: "badge" + (status.tone ? " badge-" + status.tone : "") }, status.label)),
        healthChip || (run.status === "queued" ? null
          : h("span", { class: "tm-sub tm-health" }, live ? "First match to come" : "Never played a match"))),
      h("td", { class: "tm-trend-cell" }, spark.chart ? spark.node : h("span", { class: "tm-sub" }, run.status === "queued" ? "" : "no match yet")),
      h("td", { class: "num" }, K.winCell(e), e && typeof e.spread === "number"
        ? h("span", { class: "tm-sub tm-greedy-line", title: "Average points a game against the static evaluator, ± its 95% interval. Secondary for these runs: a network trained to win can be behind on points and ahead on wins." },
          "spread ", fmt.spread(e.spread, e.ci)) : null),
      h("td", { class: "num" }, pass || (run.status === "queued" ? "waiting" : "–"),
        live && typeof run.progress === "number"
          ? h("span", { class: "tm-progress-line gm-progress" },
            h("span", { class: "progress", role: "progressbar", "aria-valuenow": Math.round(run.progress * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
              h("span", { style: { width: `${Math.min(100, run.progress * 100)}%` } })),
            h("span", { class: "tm-sub" }, fmt.pct(run.progress, 0)))
          : null,
        num(run.positions_per_second) !== null ? h("span", { class: "tm-sub" }, `${fmt.int(run.positions_per_second)} positions/s`) : null),
      h("td", { class: "num" }, run.hours ? fmt.duration(run.hours) : "–",
        live
          ? (run.limit && run.limit.hours ? h("span", { class: "tm-sub" }, `limit ${fmt.duration(run.limit.hours).replace(/ 00 m$/, "")}`) : null)
          : h("span", { class: "tm-sub", title: `Last report ${fmt.datetime(run.updated)}` }, fmt.ago(run.updated))),
      h("td", { class: "tm-actions-cell", onclick: stop, title: "" }, runActions(run)));
      return tr;
    }

    function runActions(run) {
      const items = [h("a", { class: "button button-small button-primary tm-charts", href: "#/run/" + enc(run.name),
        title: `${run.name}'s page: win rate, how it learned, its report` }, K.icon("chart", 14), "Open")];
      if (run.status === "running") {
        items.push(h("button", { type: "button", class: "button button-small", onclick: () => stopRun(run) }, K.icon("stop", 13), "Stop"));
      } else if (run.status === "queued") {
        items.push(h("button", { type: "button", class: "button button-small", onclick: () => stopRun(run) }, K.icon("x", 13), "Cancel"));
      } else if (run.status === "stopping") {
        items.push(h("button", { type: "button", class: "button button-small", disabled: true }, "Stopping…"));
      } else {
        items.push(h("button", { type: "button", class: "button button-small", disabled: !run.checkpoints,
          title: run.checkpoints ? `Train ${run.name} some more from its newest checkpoint` : "No checkpoint to resume from",
          onclick: () => openResume(run) }, K.icon("resume", 13), "Resume"));
      }
      const blocked = K.renameBlocked(run);
      items.push(h("span", { class: "tm-tip", title: blocked || `Rename ${run.name}, and with it its models` },
        h("button", { type: "button", class: "button button-small", disabled: !!blocked, "aria-label": `Rename ${run.name}`,
          onclick: () => renameRun(run) }, K.icon("pencil", 13), "Rename")));
      const busy = run.status === "running" || run.status === "stopping";
      items.push(h("button", { type: "button", class: "button button-small button-ghost tm-iconbtn", disabled: busy,
        "aria-label": `Delete ${run.name}`, title: busy ? "Stop the run before deleting it" : `Delete ${run.name}`,
        onclick: () => deleteRun(run) }, K.icon("trash", 15)));
      return h("div", { class: "tm-actions" }, items);
    }

    function renameRun(run) {
      K.renameDialog({ name: run.name, run, names: (runsCache || []).map((r) => r.name), onRenamed: () => loadRuns(true) });
    }

    async function stopRun(run) {
      const queued = run.status === "queued";
      const ok = await App.confirm(queued
        ? { title: `Take ${run.name} out of the queue?`, confirmLabel: "Take out of queue",
          body: "It has not started, so there is nothing to save. It stays in the list; delete it if you no longer want it." }
        : { title: `Stop ${run.name}?`, confirmLabel: "Stop run",
          body: "The run finishes its current step, checks its held-out games, plays a match against the static evaluator and saves a checkpoint, then stops. You can resume it later from that checkpoint." });
      if (!ok) return;
      try {
        await App.post(`/api/runs/${enc(run.name)}/stop`);
        App.toast(queued ? `${run.name} is out of the queue.` : `Stopping ${run.name}. It plays a match and saves first.`, "info");
        await loadRuns(true);
        App.refreshStatus();
      } catch (error) {
        App.toast(`Could not stop ${run.name}: ${sentence(error.message || error)}`, "critical");
      }
    }

    async function deleteRun(run) {
      const parts = [];
      if (run.checkpoints) parts.push(`${run.checkpoints} checkpoint${run.checkpoints > 1 ? "s" : ""}`);
      if (run.has_best) parts.push("the best network");
      parts.push("the metrics", "the log");
      const list = parts.slice(0, -1).join(", ") + " and " + parts[parts.length - 1];
      const ok = await App.confirm({
        title: `Delete ${run.name}?`, confirmLabel: "Delete run", danger: true,
        body: h("div", { class: "stack-tight" },
          h("p", null, "This deletes the folder ", h("code", { class: "tm-code" }, `runs/${run.name}/`), ` and everything in it: ${list}.` +
            (run.has_best || run.checkpoints ? " Its models leave the library." : "") + " The game set it learned from stays."),
          h("p", null, h("strong", null, "This cannot be undone."))),
      });
      if (!ok) return;
      try {
        await App.del(`/api/runs/${enc(run.name)}`);
        App.toast(`Deleted ${run.name}.`, "info");
        await loadRuns(true);
      } catch (error) {
        App.toast(`Could not delete ${run.name}: ${sentence(error.message || error)}`, "critical");
      }
    }

    /** Resume takes a time limit only: a run that trains from games carries on with its own settings. */
    function openResume(run) {
      const input = h("input", { type: "number", id: "gm-resume-hours", min: 0, max: 168, step: 0.5,
        value: String(run.limit && run.limit.hours ? run.limit.hours : 0) });
      const errors = h("div", { class: "tm-resume-errors", role: "alert", hidden: true });
      const go = h("button", { type: "button", class: "button button-primary" }, K.icon("resume", 15), "Resume");
      const cancel = h("button", { type: "button", class: "button" }, "Cancel");
      const left = typeof run.pass === "number" && typeof run.passes === "number" && run.passes > run.pass
        ? ` It has done ${fmt.fixed(run.pass, 1)} of its ${trimZero(fmt.fixed(run.passes, 1))} passes.` : "";
      const content = h("div", { class: "tm-resume" },
        h("p", null, `${run.name} continues from its newest checkpoint, with the settings and the game set it was started with.${left}`),
        h("div", { class: "field" }, h("label", { class: "field-label", for: "gm-resume-hours" }, "Time limit for this session"),
          h("div", { class: "tm-num" }, input, h("span", { class: "tm-num-unit" }, "hours"))),
        h("p", { class: "field-help" }, "0 means no time limit: it stops when its passes are done."),
        S ? K.lockedNote(S.sections, "It keeps the settings it started with. Fixed for the life of the run") : null,
        errors,
        h("div", { class: "modal-actions" }, cancel, go));
      const close = App.modal({ title: `Resume ${run.name}`, content });
      cancel.addEventListener("click", close);
      const show = (text) => {
        errors.hidden = !text;
        errors.replaceChildren(...(text ? [h("p", null, K.icon("alert", 14), h("span", null, text))] : []));
      };
      go.addEventListener("click", async () => {
        const hours = Number(input.value);
        if (input.value === "" || !Number.isFinite(hours) || hours < 0) { show("The time limit must be a number of hours, 0 or more."); return; }
        go.disabled = true;
        try {
          await App.post(`/api/runs/${enc(run.name)}/resume`, { hours });
          close();
          App.toast(`Resumed ${run.name}.`, "good");
          await loadRuns(true);
          App.refreshStatus();
        } catch (error) {
          go.disabled = false;
          show(sentence(error.message || error));
        }
      });
      setTimeout(() => input.focus(), 0);
    }

    // ---- Mock-only demo states ------------------------------------------------------------

    function applyDemo() {
      const list = sets();
      const done = list.find((s) => s.status === "complete") || list[0];
      const building = list.find((s) => s.status === "building");
      const runs = gamesRuns();
      const finished = runs.find((r) => r.status === "finished");
      if (demo === "more") { make.more = true; renderMake(); refs.makeHost.scrollIntoView({ block: "center" }); }
      else if (demo === "add" && done) addGames(done);
      else if (demo === "log" && (building || done)) openLog(building || done);
      else if (demo === "stop" && building) stopSet(building, false);
      else if (demo === "delete" && done) deleteSet(done);
      else if (demo === "resume" && finished) openResume(finished);
      else if (demo === "rename" && finished) renameRun(finished);
      else if (demo === "settings" || demo === "errors") {
        draft.showAll = true;
        draft.values["offline.learning_rate"] = 0.0005;
        if (demo === "errors") {
          draft.attempted = true;
          draft.values["offline.batch_size"] = 0;
          mapServerError("offline.eval_pairs must be at least 2");
        }
        renderNotices();
        renderEditor();
        renderBasket();
        trainHost.scrollIntoView({ block: "start" });
      }
    }
  }
})();
