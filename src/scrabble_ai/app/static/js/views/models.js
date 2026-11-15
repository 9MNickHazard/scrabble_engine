/* Models: the library of everything that can play. The built-in bots in a strip, the trained
 * networks in a sortable table (the run's best by default, every checkpoint on request), stars
 * and notes, "Rename run" on every row (models are named after their run), and a small
 * leave-value explorer. GET /api/runs says which runs are busy (no renaming while one trains,
 * stops or waits in the queue); without it the server still refuses those.
 *
 * Uses window.LabKit from views/training.js (icons, spread cells, sortable headers, menus, the
 * rename dialog).
 *
 * Mock-only demo states (with ?mock=1):
 *   #/models?demo=empty | many | checkpoints | menu | note | rename
 */
(function () {
  "use strict";
  const { h, fmt } = App;
  const K = window.LabKit;

  // Kept across re-mounts.
  const lib = { sort: "gl", dir: -1, run: "", text: "", starred: false, all: false };
  const leave = { rack: "ERS", bag: 60 };

  const TILE_POINTS = { A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8, K: 5, L: 1, M: 3, N: 1, O: 1,
    P: 3, Q: 10, R: 1, S: 1, T: 1, U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10 };
  const EXAMPLES = ["S", "?", "ERS", "AEINST", "Q", "IIU", "VVW"];
  const INCOMPATIBLE = "Trained with a different word list or leave values than the ones installed, so it cannot play here.";

  App.registerView("models", { title: "Models", nav: true, order: 30, icon: K.svg("models", 18), mount });

  const evalOf = (m, opponent) => (m.eval && m.eval[opponent]) || null;
  const spreadOf = (m, opponent) => { const e = evalOf(m, opponent); return e ? e.spread : null; };
  const metaPath = (m) => "/api/models/" + m.id.split("/").map(encodeURIComponent).join("/") + "/meta";

  const COLUMNS = [
    { key: "starred", label: () => [K.icon("star", 13), h("span", { class: "tm-sr" }, "Starred")], dir: -1, get: (m) => (m.starred ? 1 : 0), cls: "tm-star-cell", title: "Sort by star" },
    { key: "name", label: "Model", dir: 1, get: (m) => `${m.run.toLowerCase()}\u0000${m.kind === "best" ? "0" : "1"}${String(1e12 - (m.step || 0)).padStart(13, "0")}` },
    { key: "network", label: "Network", dir: -1, get: (m) => m.parameters },
    { key: "games", label: "Games", num: true, dir: -1, get: (m) => m.games, title: "Self-play games the run had played when this network was saved" },
    { key: "gl", label: "vs greedy-leave", num: true, dir: -1, get: (m) => spreadOf(m, "greedy-leave"),
      title: "Average spread a game against the greedy + leave bot, from the run's evaluation of this network, ± its 95% interval. " +
        "For a network trained from saved games, its win rate instead: those are judged by wins." },
    { key: "g", label: "vs greedy", num: true, dir: -1, get: (m) => spreadOf(m, "greedy"),
      title: "Average spread a game against the greedy bot, ± its 95% interval" },
    { key: "size", label: "Size", num: true, dir: -1, get: (m) => m.size_bytes },
    { key: "modified", label: "Saved", num: true, dir: -1, get: (m) => m.modified },
  ];

  async function mount(container, route) {
    const demo = App.mock.enabled ? route.query.demo || "" : "";
    if (demo === "checkpoints") lib.all = true;
    let bots = [];
    let models = [];
    let runs = null;  // GET /api/runs, for each run's status; null if it could not be read
    let lastKey = "";
    const refs = {};
    const loadRuns = async () => {
      try {
        runs = (await App.get("/api/runs")).runs || [];
      } catch (error) {
        runs = null;
      }
    };

    const botsHost = h("section", { class: "tm-strip", "aria-labelledby": "tm-bots-title" });
    const libHost = h("div");
    const leavesHost = h("div");
    container.appendChild(h("div", { class: "tm-page" },
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Models"),
          h("p", { class: "lede" }, "Every network your training runs have saved, ready to play, watch or match, and the built-in bots they are measured against."))),
      botsHost, libHost, leavesHost));

    const path = "/api/models" + (demo === "empty" || demo === "many" ? `?demo=${demo}` : "");
    try {
      const [data] = await Promise.all([App.get(path), loadRuns()]);
      bots = data.bots || [];
      models = data.models || [];
      lastKey = JSON.stringify([data, runs && runs.map((r) => [r.name, r.status])]);
    } catch (error) {
      libHost.replaceChildren(h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
        h("div", null, h("strong", null, "Could not load the models. "), String(error.message || error))));
      renderLeaves();
      return;
    }
    renderBots();
    renderLibrary();
    renderLeaves();

    let first = true;
    App.every(30000, async () => {
      if (first) { first = false; return; }
      if (K.isMenuOpen() || (document.activeElement && document.activeElement.classList.contains("tm-note-input"))) return;
      await reload(false);
    });

    /** Fetch the models and the runs' states again; re-render when anything changed. */
    async function reload(force) {
      const [data] = await Promise.all([App.get(path), loadRuns()]);
      const key = JSON.stringify([data, runs && runs.map((r) => [r.name, r.status])]);
      if (key === lastKey && !force) return;
      lastKey = key;
      bots = data.bots || [];
      models = data.models || [];
      renderBots();
      renderLibrary();
    }

    if (demo === "rename") setTimeout(() => { const b = libHost.querySelector(".tm-rename-run:not(:disabled)"); if (b) b.click(); }, 600);

    // Mock-only demo states; the delay lets a headless browser finish its start-up resizes,
    // which would close a menu.
    if (demo === "menu" || demo === "note") {
      setTimeout(() => {
        if (demo === "menu") {
          const button = libHost.querySelector(".tm-match");
          if (button) button.click();
        } else {
          const add = libHost.querySelector(".tm-note-add:not([hidden])");
          if (add) add.click();
        }
      }, 1200);
    }

    // ---- Bots ---------------------------------------------------------------------------

    function tileMark(name) {
      return name.split(/\s+/).filter((w) => /^[A-Za-z]/.test(w)).map((w) => w[0].toUpperCase()).join("").slice(0, 2) || "?";
    }

    function renderBots() {
      if (!bots.length) { botsHost.replaceChildren(); return; }
      botsHost.replaceChildren(
        h("div", { class: "tm-strip-head" },
          h("h2", { id: "tm-bots-title" }, "Built-in bots"),
          h("p", { class: "card-sub" }, "Opponents and baselines, not trained models. Every evaluation measures a network against the first two.")),
        h("div", { class: "tm-bots" }, bots.map((bot) => h("div", { class: "tm-bot" },
          h("span", { class: "tm-bot-tile", "aria-hidden": "true" }, tileMark(bot.name)),
          h("div", { class: "tm-bot-text" },
            h("div", { class: "tm-bot-name" }, bot.name, h("code", { class: "tm-key" }, bot.spec)),
            h("p", { class: "tm-bot-desc" }, bot.description)),
          h("div", { class: "tm-bot-actions" },
            h("button", { type: "button", class: "button button-small button-ghost", title: `Play a game against ${bot.name}`,
              onclick: () => App.go(`#/play?opponent=${encodeURIComponent(bot.spec)}`) }, K.icon("play", 13), "Play"))))));
    }

    // ---- Library ------------------------------------------------------------------------

    function renderLibrary() {
      if (!models.length) {
        libHost.replaceChildren(h("section", { class: "card tm-lib-empty" },
          h("span", { class: "tm-empty-icon" }, K.icon("models", 30)),
          h("h2", null, "No trained models yet"),
          h("p", null, "A model appears here once a training run has evaluated its first network. With the default settings that is about 20 minutes after the warm start, and the run's best network stays here when it finishes."),
          h("a", { class: "button button-primary", href: "#/training" }, K.icon("training", 16), "Go to Training")));
        return;
      }
      if (lib.run && !models.some((m) => m.run === lib.run)) lib.run = "";
      const runs = [...new Set(models.map((m) => m.run))].sort((a, b) => a.localeCompare(b));
      const text = h("input", { class: "input", type: "search", placeholder: "Filter by name, network or note", value: lib.text, "aria-label": "Filter models" });
      text.addEventListener("input", () => { lib.text = text.value; renderTable(); });
      const runSelect = h("select", { class: "select tm-run-select", "aria-label": "Run" },
        h("option", { value: "" }, "All runs"), runs.map((r) => h("option", { value: r, selected: r === lib.run }, r)));
      runSelect.addEventListener("change", () => { lib.run = runSelect.value; renderTable(); });
      const toggle = (label, key) => {
        const input = h("input", { type: "checkbox", checked: lib[key] });
        input.addEventListener("change", () => { lib[key] = input.checked; renderTable(); });
        return h("label", { class: "toggle tm-toggle" }, input, h("span", null, label));
      };
      refs.count = h("p", { class: "card-sub" });
      refs.table = h("div", { class: "tm-table-wrap" });
      refs.foot = h("div", { class: "tm-footnote" }, K.spreadLegend());
      libHost.replaceChildren(h("section", { class: "card card-flush tm-lib", "aria-labelledby": "tm-lib-title" },
        h("div", { class: "tm-history-head" },
          h("div", { class: "tm-history-title" }, h("h2", { id: "tm-lib-title" }, "Trained models"), refs.count),
          h("div", { class: "tm-history-tools" },
            h("div", { class: "tm-search tm-search-wide" }, K.icon("search", 15), text), runSelect,
            toggle("Starred only", "starred"), toggle("Show all checkpoints", "all"))),
        refs.table, refs.foot));
      renderTable();
    }

    function visibleModels() {
      const q = lib.text.trim().toLowerCase();
      return models.filter((m) => (lib.all || m.kind === "best") &&
        (!lib.run || m.run === lib.run) &&
        (!lib.starred || m.starred) &&
        (!q || `${m.name} ${m.run} ${m.file} ${m.network} ${m.note || ""}`.toLowerCase().includes(q)));
    }

    function onSort(key) {
      const column = COLUMNS.find((c) => c.key === key);
      if (lib.sort === key) lib.dir = -lib.dir;
      else { lib.sort = key; lib.dir = column.dir; }
      renderTable();
    }

    // Runs that train from saved games are judged by wins: their models show a win rate against
    // greedy-leave where the others show a spread (the run modes come from GET /api/runs).
    function gamesRunOf(m) {
      const run = runs ? runs.find((r) => r.name === m.run) : null;
      return run && K.isGames(run) ? run : null;
    }
    function glValue(m) {
      const run = gamesRunOf(m);
      if (!run) return spreadOf(m, "greedy-leave");
      const w = K.modelWin(m, run);
      return w ? w.win_rate : null;
    }
    function glCell(m) {
      const run = gamesRunOf(m);
      if (!run) return K.spread(evalOf(m, "greedy-leave"), "greedy-leave");
      return h("span", { class: "tm-win-cell", title: "Trained from saved games, so judged by wins: the share of games it won against " +
        "greedy + leave in its run's match at this step, ± its 95% interval. 50% is level." }, K.winCell(K.modelWin(m, run), "greedy + leave"),
        h("span", { class: "tm-sub" }, "of games"));
    }

    function renderTable() {
      const list = visibleModels();
      const runCount = new Set(models.map((m) => m.run)).size;
      const best = models.filter((m) => m.kind === "best").length;
      const hidden = lib.all ? 0 : models.length - best;
      refs.count.textContent = lib.all
        ? `${models.length} saved network${models.length > 1 ? "s" : ""} from ${runCount} run${runCount > 1 ? "s" : ""}.`
        : `The best network of each run: ${best} from ${runCount} run${runCount > 1 ? "s" : ""}` +
          (hidden ? `, with ${hidden} more checkpoint${hidden > 1 ? "s" : ""} hidden.` : ".");
      if (!list.length) {
        refs.foot.hidden = true;
        refs.table.replaceChildren(h("div", { class: "empty tm-empty" },
          h("strong", null, lib.starred && !models.some((m) => m.starred) ? "No starred models yet." : "No models match these filters."),
          h("span", null, lib.starred ? "Star a model with the star at the start of its row." : ""),
          h("button", { type: "button", class: "button button-small", onclick: () => {
            Object.assign(lib, { run: "", text: "", starred: false }); renderLibrary();
          } }, "Show every model")));
        return;
      }
      refs.foot.hidden = false;
      const anyGames = models.some((m) => gamesRunOf(m));
      refs.foot.replaceChildren(...K.spreadLegend(), anyGames ? h("span", null,
        "Networks trained from games show their win rate instead: share of games won, ± its 95% interval, marked when the whole interval is above 50%.") : null);
      const column = COLUMNS.find((c) => c.key === lib.sort) || COLUMNS[4];
      const byName = COLUMNS[1].get;
      // A spread and a win rate do not compare: by that column, self-play networks come first, then those trained from games.
      const main = column.key === "gl"
        ? (a, b) => (gamesRunOf(a) ? 1 : 0) - (gamesRunOf(b) ? 1 : 0) || K.compareBy(glValue, lib.dir)(a, b)
        : K.compareBy(column.get, lib.dir);
      const sorted = list.slice().sort((a, b) => main(a, b) || byName(a).localeCompare(byName(b)));
      refs.table.replaceChildren(h("table", { class: "table tm-table tm-models" },
        h("thead", null, h("tr", null, COLUMNS.map((c) => K.sortHeader(c, lib, onSort)),
          h("th", { class: "tm-actions-cell" }, h("span", { class: "tm-sr" }, "Actions")))),
        h("tbody", null, sorted.map(modelRow))));
    }

    function modelRow(m) {
      const ok = m.compatible !== false;
      const note = noteParts(m);
      return h("tr", { class: ok ? null : "tm-dim" },
        h("td", { class: "tm-star-cell" }, starButton(m)),
        h("td", { class: "tm-model-cell" },
          h("div", { class: "tm-model-line" },
            h("div", { class: "tm-model-name", title: m.name },
              h("a", { class: "tm-run-link", href: "#/run/" + encodeURIComponent(m.run) }, m.run),
              m.kind === "best"
                ? h("span", { class: "tm-kind tm-kind-best", title: "The run's gated best network" }, "Best")
                : h("span", { class: "tm-kind", title: m.file }, m.step ? `Step ${fmt.int(m.step)}` : m.file)),
            note.add),
          ok ? null : h("div", { class: "tm-incompat" }, K.icon("alert", 13), h("span", null, m.incompatible_reason || INCOMPATIBLE)),
          note.line),
        h("td", { class: "tm-fade" }, h("span", { class: "tm-net" }, m.network || "–"),
          m.parameters ? h("span", { class: "tm-sub" }, `${fmt.compact(m.parameters)} parameters`) : null),
        gamesRunOf(m)
          ? h("td", { class: "num tm-fade", title: m.games !== null && m.games !== undefined ? `${fmt.int(m.games)} positions trained on, one from each saved game` : null },
            fmt.compact(m.games), h("span", { class: "tm-sub" }, "positions"))
          : h("td", { class: "num tm-fade", title: m.games !== null && m.games !== undefined ? `${fmt.int(m.games)} games` : null }, fmt.compact(m.games)),
        h("td", { class: "num tm-fade" }, glCell(m)),
        h("td", { class: "num tm-fade" }, K.spread(evalOf(m, "greedy"), "greedy")),
        h("td", { class: "num secondary tm-fade" }, fmt.bytes(m.size_bytes)),
        h("td", { class: "num secondary nowrap tm-fade", title: fmt.datetime(m.modified) }, fmt.ago(m.modified)),
        h("td", { class: "tm-actions-cell" }, modelActions(m)));
    }

    function starButton(m) {
      const button = h("button", { type: "button", class: "tm-star", "aria-label": `Star ${m.name}` }, K.icon("star", 17));
      const paint = () => {
        button.classList.toggle("is-on", !!m.starred);
        button.setAttribute("aria-pressed", m.starred ? "true" : "false");
        button.title = m.starred ? "Starred. Click to remove the star." : "Star this model";
      };
      paint();
      button.addEventListener("click", async () => {
        m.starred = !m.starred;
        paint();
        try {
          const updated = await App.post(metaPath(m), { starred: m.starred });
          if (updated && updated.id === m.id) Object.assign(m, updated);
        } catch (error) {
          m.starred = !m.starred;
          paint();
          App.toast(`Could not change the star: ${error.message}`, "critical");
        }
        if (lib.starred || lib.sort === "starred") renderTable();
      });
      return button;
    }

    /** The note line under the name (only when there is a note, or while editing) and an
     * "Add a note" control that sits on the name line and shows on hover when there is none. */
    function noteParts(m) {
      const wrap = h("div", { class: "tm-note" });
      const add = h("button", { type: "button", class: "tm-note-edit tm-note-add", "aria-label": `Add a note to ${m.name}` },
        K.icon("pencil", 12), "Add a note");
      const show = () => {
        add.hidden = !!m.note;
        wrap.hidden = !m.note;
        App.append(App.clear(wrap), [
          h("span", { class: "tm-note-text" }, m.note || ""),
          h("button", { type: "button", class: "tm-note-edit", "aria-label": `Edit the note on ${m.name}`, title: "Edit the note",
            onclick: edit }, K.icon("pencil", 12))]);
      };
      const edit = () => {
        add.hidden = true;
        wrap.hidden = false;
        const input = h("input", { class: "input tm-note-input", type: "text", value: m.note || "", maxlength: 200,
          placeholder: "A note for yourself", "aria-label": `Note on ${m.name}` });
        let done = false;
        const finish = async (save) => {
          if (done) return;
          done = true;
          const next = input.value.trim();
          if (!save || next === (m.note || "")) { show(); return; }
          const previous = m.note;
          m.note = next;
          show();
          try {
            const updated = await App.post(metaPath(m), { note: next });
            if (updated && updated.id === m.id) Object.assign(m, updated);
          } catch (error) {
            m.note = previous;
            show();
            App.toast(`Could not save the note: ${error.message}`, "critical");
          }
        };
        input.addEventListener("keydown", (event) => {
          if (event.key === "Enter") { event.preventDefault(); finish(true); }
          if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); finish(false); }
        });
        input.addEventListener("blur", () => finish(true));
        wrap.replaceChildren(input, h("span", { class: "tm-note-hint" }, "Enter saves, Esc cancels"));
        input.focus();
      };
      add.addEventListener("click", edit);
      show();
      return { line: wrap, add };
    }

    function modelActions(m) {
      const ok = m.compatible !== false;
      const why = ok ? null : INCOMPATIBLE;
      const spec = encodeURIComponent(m.spec);
      const match = h("button", { type: "button", class: "button button-small tm-match", disabled: !ok, "aria-haspopup": "menu",
        "aria-expanded": "false", title: why || `Play a fast match between ${m.name} and another player` },
      K.icon("versus", 14), "Match", h("span", { class: "tm-caret" }, K.icon("chevronDown", 12)));
      match.addEventListener("click", () => K.toggleMenu(match, () => matchItems(m)));
      return h("div", { class: "tm-actions" },
        h("button", { type: "button", class: "button button-small", disabled: !ok, title: why || `Play a game against ${m.name}`,
          onclick: () => App.go(`#/play?opponent=${spec}`) }, K.icon("play", 13), "Play"),
        h("button", { type: "button", class: "button button-small", disabled: !ok, title: why || `Watch ${m.name} play`,
          onclick: () => App.go(`#/watch?a=${spec}`) }, K.icon("eye", 14), "Watch"),
        match,
        renameButton(m),
        h("button", { type: "button", class: "button button-small button-ghost tm-iconbtn", "aria-label": `Open run ${m.run}`,
          title: `Open run ${m.run}`, onclick: () => App.go("#/run/" + encodeURIComponent(m.run)) }, K.icon("open", 15)));
    }

    /** "Rename run": models are named after their run, so renaming one is renaming the run. */
    function renameButton(m) {
      const run = runs ? runs.find((r) => r.name === m.run) || null : null;
      const blocked = K.renameBlocked(run);
      return h("span", { class: "tm-tip", title: blocked || `Rename the run ${m.run}, and with it all its models` },
        h("button", { type: "button", class: "button button-small button-ghost tm-rename-run", disabled: !!blocked,
          "aria-label": `Rename run ${m.run}`, onclick: () => K.renameDialog({
            name: m.run, run, names: runs ? runs.map((r) => r.name) : null, fromModels: true,
            modelCount: models.filter((x) => x.run === m.run).length,
            onRenamed: async (to) => {
              if (lib.run === m.run) lib.run = to;
              await reload(true);
            },
          }) }, K.icon("pencil", 13), "Rename run"));
    }

    function matchItems(m) {
      const a = encodeURIComponent(m.spec);
      const to = (spec) => () => App.go(`#/arena?a=${a}&b=${encodeURIComponent(spec)}`);
      const items = [{ heading: `Match ${m.name} against` }];
      const baselines = bots.filter((b) => b.spec !== "bot:random");
      for (const bot of baselines.length ? baselines : bots) items.push({ label: bot.name, onClick: to(bot.spec) });
      const others = models.filter((o) => o.id !== m.id && o.compatible !== false)
        .sort((x, y) => (x.kind === "best" ? 0 : 1) - (y.kind === "best" ? 0 : 1) ||
          (spreadOf(y, "greedy-leave") ?? -1e9) - (spreadOf(x, "greedy-leave") ?? -1e9));
      if (others.length) {
        items.push({ divider: true }, { heading: "Another model" });
        for (const o of others.slice(0, 12)) items.push({ label: o.name, hint: o.network, onClick: to(o.spec) });
        if (others.length > 12) items.push({ label: `Choose from all ${others.length} in the arena`, icon: "versus", onClick: () => App.go(`#/arena?a=${a}`) });
      }
      return items;
    }

    // ---- Leave values -------------------------------------------------------------------

    function renderLeaves() {
      const rackInput = h("input", { class: "input tm-rack-input", id: "tm-rack", type: "text", value: leave.rack, maxlength: 7,
        spellcheck: "false", autocomplete: "off", placeholder: "ERS", "aria-describedby": "tm-rack-help" });
      const bagInput = h("input", { type: "number", id: "tm-bag", min: 0, max: 86, step: 1, value: String(leave.bag) });
      const result = h("div", { class: "tm-leave-result", "aria-live": "polite" });
      const chips = h("div", { class: "tm-leave-examples" }, h("span", { class: "tm-sizes-label" }, "Try"),
        EXAMPLES.map((rack) => h("button", { type: "button", class: "tm-size", onclick: () => { rackInput.value = rack; onRack(); } }, rack)));
      let timer = null;
      let seq = 0;

      async function lookup() {
        const rack = leave.rack;
        const mine = ++seq;
        if (!rack) {
          result.replaceChildren(h("p", { class: "muted" }, "Type up to seven tiles; ? is a blank."));
          return;
        }
        try {
          const data = await App.get(`/api/leaves?rack=${encodeURIComponent(rack)}&bag=${leave.bag}`);
          if (mine !== seq) return;
          showResult(data);
        } catch (error) {
          if (mine !== seq) return;
          result.replaceChildren(h("p", { class: "tm-warn tm-warn-critical" }, K.icon("alert", 14), h("span", null, String(error.message || error))));
        }
      }

      function showResult(data) {
        const rel = data.relative;
        const nothing = data.value - data.relative;
        const width = Math.min(50, (Math.abs(rel) / 30) * 50);
        result.replaceChildren(
          h("div", { class: "tm-tiles", "aria-label": `Tiles: ${data.rack.split("").join(" ")}` },
            data.rack.split("").map((ch) => h("span", { class: "tm-tile" + (ch === "?" ? " tm-tile-blank" : ""), title: ch === "?" ? "Blank" : null },
              ch === "?" ? "" : ch, ch === "?" ? null : h("sub", null, String(TILE_POINTS[ch] || ""))))),
          h("div", { class: "tm-leave-figure" },
            h("span", { class: "tm-leave-value" }, fmt.signed(rel)),
            h("span", { class: "tm-leave-unit" }, rel >= 0 ? "points better than keeping nothing" : "points worse than keeping nothing")),
          h("div", { class: "tm-meter", "aria-hidden": "true" },
            h("span", { class: "tm-meter-zero" }),
            h("span", { class: "tm-meter-bar " + (rel >= 0 ? "is-pos" : "is-neg"),
              style: rel >= 0 ? { left: "50%", width: width + "%" } : { left: 50 - width + "%", width: width + "%" } })),
          h("p", { class: "tm-leave-detail" }, `Leave value ${fmt.signed(data.value)}; keeping nothing is ${fmt.signed(nothing)}. With ${fmt.int(data.bag)} tiles in the bag.`));
      }

      function onRack() {
        const clean = rackInput.value.toUpperCase().replace(/[^A-Z?]/g, "").slice(0, 7);
        if (clean !== rackInput.value) rackInput.value = clean;
        leave.rack = clean;
        clearTimeout(timer);
        timer = setTimeout(lookup, 160);
      }
      rackInput.addEventListener("input", onRack);
      bagInput.addEventListener("input", () => {
        const n = Number(bagInput.value);
        if (bagInput.value === "" || !Number.isInteger(n) || n < 0 || n > 86) return;
        leave.bag = n;
        clearTimeout(timer);
        timer = setTimeout(lookup, 160);
      });
      App.onCleanup(() => clearTimeout(timer));

      leavesHost.replaceChildren(h("section", { class: "card tm-leaves", "aria-labelledby": "tm-leaves-title" },
        h("div", { class: "tm-leaves-grid" },
          h("div", { class: "tm-leaves-form" },
            h("h2", { id: "tm-leaves-title" }, "Leave values"),
            h("p", { class: "tm-leaves-lede", id: "tm-rack-help" },
              "A leave is the tiles you keep after a move. Its value is what keeping them is worth to your next turns, in points; the bots and the networks add it to a move's score."),
            h("div", { class: "tm-leaves-inputs" },
              h("div", { class: "field" }, h("label", { class: "field-label", for: "tm-rack" }, "Tiles to keep"), rackInput),
              h("div", { class: "field tm-bag-field" }, h("label", { class: "field-label", for: "tm-bag" }, "Tiles in the bag"),
                h("div", { class: "tm-num" }, bagInput))),
            chips),
          result)));
      lookup();
    }
  }
})();
