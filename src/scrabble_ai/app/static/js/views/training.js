/* Training: start runs (one, or several to compare) and the history of every run.
 *
 * The form is built from GET /api/config/schema alone: sections, labels, help, defaults, limits,
 * which fields are advanced, presets and the speed table. Fields with `advanced: false` are the
 * essentials at the top (run.name, run.hours, run.max_games, model.blocks and model.channels get
 * a special layout; any other non-advanced field is shown with them in the generic style). The
 * rest live under "All settings". `step` is how far the arrow keys move; `multiple_of` is a rule.
 *
 * This file also defines window.LabKit, a few helpers shared with the other views (models, run,
 * overview, guide): icons, the spread cell with its significance marker, sortable headers, a
 * popover menu, the health chip and strength sparkline of a run, the rename dialog, and the
 * "what turning it down or up does" lines of a setting.
 *
 * Mock-only demo states (with ?mock=1):
 *   #/training?demo=multi | settings | search | errors | noleaves | resume | compare | delete | stop | rename
 */

window.LabKit = (function () {
  "use strict";
  const { h, fmt } = App;

  const PATHS = {
    plus: '<path d="M10 4.5v11M4.5 10h11"/>',
    x: '<path d="M5.5 5.5l9 9M14.5 5.5l-9 9"/>',
    trash: '<path d="M4 6h12M8 6V4.5h4V6M5.8 6l.8 10h6.8l.8-10M8.6 9v4.5M11.4 9v4.5"/>',
    stop: '<rect x="5.5" y="5.5" width="9" height="9" rx="1.5"/>',
    play: '<path d="M7 4.8v10.4l8.2-5.2z"/>',
    resume: '<path d="M15.6 10a5.6 5.6 0 1 1-1.7-4M14.8 3.4v3.2h-3.2"/>',
    search: '<circle cx="8.8" cy="8.8" r="5"/><path d="M12.6 12.6l3.6 3.6"/>',
    chevron: '<path d="M8 5l5 5-5 5"/>',
    chevronDown: '<path d="M5.5 8l4.5 4.5L14.5 8"/>',
    star: '<path d="M10 3.2l2.05 4.3 4.7.6-3.45 3.25.88 4.65L10 13.75 5.82 16l.88-4.65L3.25 8.1l4.7-.6z"/>',
    eye: '<path d="M2.2 10s2.9-5.2 7.8-5.2 7.8 5.2 7.8 5.2-2.9 5.2-7.8 5.2S2.2 10 2.2 10z"/><circle cx="10" cy="10" r="2.3"/>',
    versus: '<path d="M3.5 7.5h11l-3-3M16.5 12.5h-11l3 3"/>',
    open: '<path d="M8.5 4.5h-4v11h11v-4M11.5 4.5h4v4M15.5 4.5l-6.5 6.5"/>',
    reset: '<path d="M4.4 10a5.6 5.6 0 1 0 1.7-4M5.2 3.4v3.2h3.2"/>',
    gauge: '<path d="M3.2 14.5a7 7 0 1 1 13.6 0"/><path d="M10 14.2l3.4-4.6"/><circle cx="10" cy="14.5" r="0.6"/>',
    check: '<path d="M4.5 10.5l3.6 3.5 7.4-8"/>',
    alert: '<path d="M10 3.6l7.2 12.6H2.8z"/><path d="M10 8.4v3.6M10 14.1v.1"/>',
    pencil: '<path d="M12.6 4.4l3 3L7.2 15.8H4.2v-3z"/>',
    lock: '<rect x="5" y="9" width="10" height="7.5" rx="1.6"/><path d="M7.2 9V6.9a2.8 2.8 0 0 1 5.6 0V9"/>',
    clock: '<circle cx="10" cy="10" r="6.6"/><path d="M10 6.4V10l2.6 1.6"/>',
    compare: '<path d="M3 16.5h14"/><path d="M4 13l3.6-4.2 3 2.6L16 5.5"/><path d="M4 10.8l3.6-1.4 3 3.2 5.4-2.8" opacity="0.55"/>',
    sortUp: '<path d="M10 15V5M6 9l4-4 4 4"/>',
    sortDown: '<path d="M10 5v10M6 11l4 4 4-4"/>',
    sortNone: '<path d="M7 8l3-3 3 3M7 12l3 3 3-3"/>',
    up: '<path d="M10 4.5l5.5 9.5h-11z"/>',
    down: '<path d="M10 15.5l5.5-9.5h-11z"/>',
    training: '<path d="M3 15.5l4.2-5 3.2 2.8L16.5 5.5"/><path d="M12.5 5.5h4v4"/>',
    models: '<path d="M10 2.8l6.6 3.6v7.2L10 17.2l-6.6-3.6V6.4z"/><path d="M3.6 6.5L10 10l6.4-3.5M10 10v7"/>',
    layers: '<path d="M10 3.5l7 3.6-7 3.6-7-3.6z"/><path d="M3 10.4l7 3.6 7-3.6M3 13.6l7 3.6 7-3.6"/>',
    chart: '<path d="M3.5 16.5h13"/><path d="M4.5 13l3.6-4.1 3 2.6 4.4-6"/>',
    book: '<path d="M10 6.2C8.6 5 6.4 4.6 3.5 4.8v10c2.9-.2 5.1.2 6.5 1.4 1.4-1.2 3.6-1.6 6.5-1.4v-10c-2.9-.2-5.1.2-6.5 1.4z"/><path d="M10 6.2v10"/>',
    trendUp: '<path d="M3 14.5l4.8-4.8 3 3L17 6.5"/><path d="M12.5 6.5H17V11"/>',
    trendDown: '<path d="M3 5.5l4.8 4.8 3-3L17 13.5"/><path d="M12.5 13.5H17V9"/>',
    equals: '<path d="M5 8h10M5 12h10"/>',
    arrowDown: '<path d="M10 4v11.5M5.5 11l4.5 4.5 4.5-4.5"/>',
  };

  function svg(name, size = 16) {
    return `<svg viewBox="0 0 20 20" width="${size}" height="${size}" fill="none" stroke="currentColor" ` +
      `stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${PATHS[name] || ""}</svg>`;
  }

  function icon(name, size = 16, cls = "") {
    return h("span", { class: "tm-icon" + (cls ? " " + cls : ""), html: svg(name, size) });
  }

  /** "up" when the whole interval is above zero, "down" when below, else null. */
  function significance(entry) {
    if (!entry || entry.spread === null || entry.spread === undefined || entry.ci === null || entry.ci === undefined) return null;
    if (entry.spread - entry.ci > 0) return "up";
    if (entry.spread + entry.ci < 0) return "down";
    return null;
  }

  /** A spread with its interval and a marker when the interval excludes zero. */
  function spread(entry, against) {
    if (!entry || entry.spread === null || entry.spread === undefined) return h("span", { class: "muted" }, "–");
    const sig = significance(entry);
    const title = sig === "up" ? `Better than ${against}: the whole 95% interval is above zero.`
      : sig === "down" ? `Worse than ${against}: the whole 95% interval is below zero.`
        : entry.ci === null || entry.ci === undefined ? `Average spread a game against ${against}.`
          : `Not yet distinguishable from ${against}: the 95% interval includes zero.`;
    return h("span", { class: "tm-spread", title },
      h("span", { class: "tm-spread-value" }, fmt.signed(entry.spread)),
      entry.ci === null || entry.ci === undefined ? null : h("span", { class: "tm-spread-ci" }, "± " + fmt.fixed(entry.ci)),
      h("span", { class: "tm-sig" + (sig ? " tm-sig-" + sig : "") },
        sig ? h("span", { class: "tm-icon", html: svg(sig, 10) }) : null,
        sig ? h("span", { class: "tm-sr" }, sig === "up" ? "significantly better" : "significantly worse") : null));
  }

  /** The legend line that explains the spread columns. */
  function spreadLegend() {
    return [
      h("span", null, "Spread: average points a game against the bot over seat-swapped pairs, ± its 95% interval."),
      h("span", { class: "tm-legend-item" }, h("span", { class: "tm-sig tm-sig-up" }, h("span", { class: "tm-icon", html: svg("up", 10) })),
        "interval above zero: a real gain"),
      h("span", { class: "tm-legend-item" }, h("span", { class: "tm-sig tm-sig-down" }, h("span", { class: "tm-icon", html: svg("down", 10) })),
        "interval below zero"),
    ];
  }

  /** A rough count in words: "1.6 million", "790,000", "4,100". */
  function approx(n) {
    if (n === null || n === undefined || !Number.isFinite(n)) return "–";
    if (n >= 1e6) return fmt.fixed(n / 1e6, n >= 1e7 ? 0 : 1) + " million";
    if (n >= 1000) {
      const magnitude = Math.pow(10, Math.floor(Math.log10(n)) - 1);
      return fmt.int(Math.round(n / magnitude) * magnitude);
    }
    return fmt.int(n);
  }

  /** A sortable column header. `state` has {sort, dir}; `onSort(key)` re-sorts. */
  function sortHeader(column, state, onSort) {
    const active = state.sort === column.key;
    const dir = active ? (state.dir > 0 ? "sortUp" : "sortDown") : "sortNone";
    return h("th", {
      class: "sortable" + (column.num ? " num" : "") + (active ? " is-sorted" : "") + (column.cls ? " " + column.cls : ""),
      scope: "col", "aria-sort": active ? (state.dir > 0 ? "ascending" : "descending") : "none",
    }, h("button", { type: "button", class: "tm-sort", title: column.title || null, onclick: () => onSort(column.key) },
      typeof column.label === "function" ? column.label() : column.label, h("span", { class: "tm-sort-icon", html: svg(dir, 12) })));
  }

  /** Compare two rows by `get`, with null and undefined always last. */
  function compareBy(get, dir) {
    return (a, b) => {
      const x = get(a);
      const y = get(b);
      const xn = x === null || x === undefined;
      const yn = y === null || y === undefined;
      if (xn || yn) return xn && yn ? 0 : xn ? 1 : -1;
      if (typeof x === "string" && typeof y === "string") return x.localeCompare(y) * dir;
      return (x < y ? -1 : x > y ? 1 : 0) * dir;
    };
  }

  // ---- Popover menu ---------------------------------------------------------------------

  let current = null;

  function closeMenu() {
    if (current) current();
  }

  /** items: [{label, hint, icon, onClick, disabled, title} | {heading} | {divider: true}] */
  function menu(anchor, items) {
    closeMenu();
    const node = h("div", { class: "tm-menu", role: "menu" });
    for (const item of items) {
      if (item.divider) node.appendChild(h("div", { class: "tm-menu-sep", role: "separator" }));
      else if (item.heading) node.appendChild(h("div", { class: "tm-menu-heading" }, item.heading));
      else {
        node.appendChild(h("button", {
          type: "button", class: "tm-menu-item", role: "menuitem", disabled: !!item.disabled, title: item.title || null,
          onclick: () => { close(); item.onClick(); },
        }, item.icon ? icon(item.icon, 15) : null, h("span", { class: "tm-menu-label" }, item.label),
        item.hint ? h("span", { class: "tm-menu-hint" }, item.hint) : null));
      }
    }
    document.body.appendChild(node);
    const place = () => {
      const rect = anchor.getBoundingClientRect();
      const width = node.offsetWidth;
      const height = node.offsetHeight;
      const left = Math.min(window.innerWidth - width - 8, Math.max(8, rect.right - width));
      let top = rect.bottom + 4;
      if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 4);
      node.style.left = left + "px";
      node.style.top = top + "px";
    };
    place();
    anchor.setAttribute("aria-expanded", "true");

    const buttons = () => [...node.querySelectorAll(".tm-menu-item:not(:disabled)")];
    const onDown = (event) => { if (!node.contains(event.target) && !anchor.contains(event.target)) close(); };
    const onKey = (event) => {
      if (event.key === "Escape") { close(); anchor.focus(); return; }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        const list = buttons();
        const at = list.indexOf(document.activeElement);
        const next = event.key === "ArrowDown" ? (at + 1) % list.length : (at - 1 + list.length) % list.length;
        if (list[next]) list[next].focus();
      }
    };
    const onScroll = () => close();
    const view = document.getElementById("view");
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", place);
    if (view) view.addEventListener("scroll", onScroll, { passive: true });

    function close() {
      if (!node.isConnected) return;
      node.remove();
      anchor.setAttribute("aria-expanded", "false");
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("resize", place);
      if (view) view.removeEventListener("scroll", onScroll);
      if (current === close) current = null;
    }
    current = close;
    App.onCleanup(close);
    const first = buttons()[0];
    if (first) first.focus({ preventScroll: true });
    return close;
  }

  function toggleMenu(anchor, itemsFn) {
    if (anchor.getAttribute("aria-expanded") === "true") closeMenu();
    else menu(anchor, itemsFn());
  }

  function isMenuOpen() {
    return current !== null;
  }

  // ---- How a run went: the health chip and the strength sparkline --------------------------

  /** run.health.state -> label, tone (a reserved status colour, or neutral) and icon. */
  const HEALTH = {
    ahead: { label: "Beats the baseline", tone: "good", icon: "trendUp" },
    level: { label: "Level with the baseline", tone: "level", icon: "equals" },
    behind: { label: "Behind the baseline", tone: "warning", icon: "arrowDown" },
    worse: { label: "Got worse", tone: "critical", icon: "trendDown" },
  };
  /** The same states for a run that trains from games, where they are about its win rate
   *  against the static evaluator (ahead: the whole interval above 50%). */
  const HEALTH_GAMES = {
    ahead: { label: "Beats the static bot", tone: "good", icon: "trendUp" },
    level: { label: "Level with the static bot", tone: "level", icon: "equals" },
    behind: { label: "Behind the static bot", tone: "warning", icon: "arrowDown" },
    worse: { label: "Got worse", tone: "critical", icon: "trendDown" },
  };
  const HEALTH_ORDER = { ahead: 0, level: 1, behind: 2, worse: 3 };

  /** The verdict on a run as a chip, its sentence as the tooltip. null for no verdict. `games`:
   *  the run trains from games, so the states are about its win rate. */
  function healthChip(health, extraClass, games = false) {
    const table = games ? HEALTH_GAMES : HEALTH;
    if (!health || !table[health.state]) return null;
    const meta = table[health.state];
    return h("span", { class: "hl-chip hl-" + meta.tone + (extraClass ? " " + extraClass : ""),
      title: health.note ? App.fmt.prose(health.note) : meta.label },
    icon(meta.icon, 14), h("span", null, meta.label));
  }

  // ---- Runs that train from saved games ----------------------------------------------------
  // Such a run (`mode` "games") is judged by its win rate against the static evaluator
  // (greedy-leave): 50% is level, and only a whole 95% interval above one half is a win.

  const isGames = (run) => !!run && run.mode === "games";

  /** "up" when the whole interval of a win rate is above one half, "down" when below, else null. */
  function winSignificance(rate, ci) {
    if (typeof rate !== "number" || typeof ci !== "number") return null;
    if (rate - ci > 0.5) return "up";
    if (rate + ci < 0.5) return "down";
    return null;
  }

  /** "53.4% ± 1.3": a win rate and the half-width of its interval, in percentage points. */
  function winText(rate, ci) {
    if (typeof rate !== "number") return "–";
    return fmt.pct(rate) + (typeof ci === "number" ? " ± " + (100 * ci).toFixed(1) : "");
  }

  /** A few words on what a win rate's interval shows, for the hint under a figure. */
  function winVerdict(rate, ci) {
    if (typeof rate !== "number") return "";
    if (typeof ci !== "number") return "no interval recorded";
    const sig = winSignificance(rate, ci);
    if (sig === "up") return "clears 50%: wins more than it loses";
    if (sig === "down") return "below 50%: loses more than it wins";
    return "includes 50%: not shown to win more";
  }

  /** A win rate with its interval and a marker when the interval excludes one half; the
   *  counterpart of `spread` for runs trained from games. entry: {win_rate, win_ci}. */
  function winCell(entry, against = "the static evaluator") {
    const rate = entry && typeof entry.win_rate === "number" ? entry.win_rate : null;
    if (rate === null) return h("span", { class: "muted" }, "–");
    const ci = typeof entry.win_ci === "number" ? entry.win_ci : null;
    const sig = winSignificance(rate, ci);
    const title = sig === "up" ? `Wins more than half its games against ${against}: the whole 95% interval is above 50%.`
      : sig === "down" ? `Wins less than half its games against ${against}: the whole 95% interval is below 50%.`
        : ci === null ? `Share of games won against ${against}.`
          : `Not yet distinguishable from half against ${against}: the 95% interval includes 50%.`;
    return h("span", { class: "tm-spread", title },
      h("span", { class: "tm-spread-value" }, fmt.pct(rate)),
      ci === null ? null : h("span", { class: "tm-spread-ci" }, "± " + (100 * ci).toFixed(1)),
      h("span", { class: "tm-sig" + (sig ? " tm-sig-" + sig : "") },
        sig ? h("span", { class: "tm-icon", html: svg(sig, 10) }) : null,
        sig ? h("span", { class: "tm-sr" }, sig === "up" ? "significantly more than half" : "significantly less than half") : null));
  }

  /** The network of a run in words: "6x64", or for a run that chose another kind of network
   *  "6x64 CNN with board-wide pooling", "8x192 transformer". */
  function networkLabel(run) {
    if (!run) return "–";
    const net = run.network || "–";
    const kind = run.changed && run.changed["model.kind"];
    if (kind === "transformer") return `${net} transformer`;
    if (kind === "cnn-global") return `${net} CNN with board-wide pooling`;
    return isGames(run) ? `${net} CNN` : net;
  }

  /** A model's win rate against greedy-leave with its interval. GET /api/models gives the rate
   *  alone; the interval is in its run's `trend` (the match at the model's step). */
  function modelWin(model, run) {
    const e = model && model.eval && model.eval["greedy-leave"];
    const rate = e && typeof e.win_rate === "number" ? e.win_rate : null;
    let ci = null;
    const t = run && run.trend;
    if (t && Array.isArray(t.step) && Array.isArray(t.win_ci) && model && typeof model.step === "number") {
      const i = t.step.lastIndexOf(model.step);
      if (i >= 0 && typeof t.win_ci[i] === "number") ci = t.win_ci[i];
    }
    return rate === null ? null : { win_rate: rate, win_ci: ci };
  }

  /** run.trend as a sparkline against zero, with each evaluation's interval as a band; for a run
   *  that trains from games, its win rate against 50%, by training step.
   *  Returns {node, chart}; chart is null when there is nothing to draw. */
  function trendSpark(run, options = {}) {
    const t = run && run.trend;
    const node = h("span", { class: "hl-spark" });
    if (isGames(run)) return winSpark(run, node, options);
    if (!t || !Array.isArray(t.games) || !Array.isArray(t.spread) || !t.games.length) return { node, chart: null };
    const points = [];
    const band = [];
    t.games.forEach((games, i) => {
      const spread = t.spread[i];
      if (typeof games !== "number" || typeof spread !== "number") return;
      points.push([games, spread]);
      const ci = Array.isArray(t.ci) ? t.ci[i] : null;
      if (typeof ci === "number") band.push([games, spread - ci, spread + ci]);
    });
    if (!points.length) return { node, chart: null };
    const chart = Charts.sparkline(node, Object.assign({
      points, band, referenceY: 0, width: 112, height: 30, color: "var(--series-1)",
      format: (v) => fmt.signed(v) + " a game", xFormat: (v) => fmt.compact(v) + " games",
      label: `Spread against greedy + leave at each of ${points.length} evaluations; the latest is ${fmt.signed(points[points.length - 1][1])}`,
    }, options));
    return { node, chart };
  }

  function winSpark(run, node, options) {
    const t = run.trend;
    if (!t || !Array.isArray(t.win_rate) || !t.win_rate.length) return { node, chart: null };
    const xs = Array.isArray(t.step) && t.step.length === t.win_rate.length ? t.step : t.games;
    const byStep = xs === t.step;
    const points = [];
    const band = [];
    t.win_rate.forEach((rate, i) => {
      const x = xs ? xs[i] : i;
      if (typeof x !== "number" || typeof rate !== "number") return;
      points.push([x, rate]);
      const ci = Array.isArray(t.win_ci) ? t.win_ci[i] : null;
      if (typeof ci === "number") band.push([x, rate - ci, rate + ci]);
    });
    if (!points.length) return { node, chart: null };
    const chart = Charts.sparkline(node, Object.assign({
      points, band, referenceY: 0.5, width: 112, height: 30, color: "var(--series-1)",
      format: (v) => fmt.pct(v) + " of games",
      xFormat: (v) => (byStep ? "Step " + fmt.int(v) : fmt.compact(v) + " positions"),
      label: `Win rate against the static evaluator at each of ${points.length} matches; the latest is ${fmt.pct(points[points.length - 1][1])}`,
    }, options));
    return { node, chart };
  }

  // ---- Settings forms: the pieces the Training and Train from games forms share -------------

  const same = (a, b) => a === b ||
    (typeof a === "number" && typeof b === "number" && Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a), Math.abs(b)));

  /** The schema indexed for one form. `mode` is "selfplay" or "games": only the fields whose
   *  `modes` include it (an older server sends no `modes`: every field then), sections left
   *  empty dropped. For "games" each field's `default` is the one that will really apply
   *  (`games_defaults` over the schema's; the schema's own stays as `schemaDefault`) and the
   *  presets are `games_presets`. */
  function indexSchema(schema, mode = "selfplay") {
    const games = mode === "games";
    const over = (games && schema.games_defaults) || {};
    const shown = (f) => !Array.isArray(f.modes) || f.modes.includes(mode);
    const fields = [];
    const byKey = {};
    const sections = [];
    for (const section of schema.sections || []) {
      const list = [];
      for (const field of section.fields || []) {
        if (!shown(field)) continue;
        const copy = Object.assign({ section: section.name, schemaDefault: field.default }, field);
        if (Object.prototype.hasOwnProperty.call(over, field.key)) copy.default = over[field.key];
        fields.push(copy);
        byKey[field.key] = copy;
        list.push(copy);
      }
      if (list.length) sections.push(Object.assign({}, section, { fields: list }));
    }
    const presets = (games ? schema.games_presets : schema.presets) || [];
    const presetKeys = [...new Set(presets.flatMap((p) => Object.keys(p.overrides || {})))].filter((k) => byKey[k]);
    return { sections, fields, byKey, presets, presetKeys, speed: schema.speed || [] };
  }

  /** "3,000,000", "2M", "300k", "1e-4", "0.0003" -> a number; anything else -> null. */
  function parseNumber(text) {
    const t = String(text).trim().replace(/[,_\s]/g, "");
    const match = t.match(/^([-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?)([km]?)$/i);
    if (!match) return null;
    let n = Number(match[1]);
    if (match[2]) {
      n *= match[2].toLowerCase() === "k" ? 1e3 : 1e6;
      if (Math.abs(n - Math.round(n)) < 1e-6) n = Math.round(n);
    }
    return Number.isFinite(n) ? n : null;
  }

  function showNumber(field, value) {
    if (value === null || value === undefined) return "";
    if (typeof value !== "number") return String(value);
    if (field.type === "int" && Number.isInteger(value)) return fmt.int(value);
    return String(Number(value.toPrecision(10)));
  }

  function formatValue(field, value) {
    if (value === null || value === undefined) return "–";
    if (field.type === "bool") return value ? "on" : "off";
    if (typeof value !== "number") return value === "" ? "empty" : String(value);
    if (field.type === "int") return fmt.int(value);
    return String(Number(value.toPrecision(6)));
  }

  function fieldError(field, value) {
    if (field.type === "int" || field.type === "float") {
      if (typeof value !== "number" || !Number.isFinite(value)) return "Enter a number.";
      if (field.type === "int" && !Number.isInteger(value)) return "Enter a whole number.";
      const lo = field.min;
      const hi = field.max;
      const has = (v) => v !== null && v !== undefined;
      if (has(lo) && has(hi) && (value < lo || value > hi)) return `Must be between ${formatValue(field, lo)} and ${formatValue(field, hi)}.`;
      if (has(lo) && value < lo) return `Must be at least ${formatValue(field, lo)}.`;
      if (has(hi) && value > hi) return `Must be at most ${formatValue(field, hi)}.`;
      if (field.type === "int" && field.multiple_of > 1 && value % field.multiple_of !== 0) return `Must be a multiple of ${field.multiple_of}.`;
    }
    if (field.type === "choice" && field.options && !field.options.includes(value)) return "Choose one of the options.";
    return null;
  }

  /** The input for one setting: a switch, a select, a number box that takes "2M" and "300k" and
   *  shows separators, or a text box. It reads and writes values[field.key] and calls
   *  onChange() after every edit. Returns {node, input, set(value), frame}. */
  function settingControl(field, values, id, onChange) {
    const key = field.key;
    if (field.type === "bool") {
      const input = h("input", { type: "checkbox", id, role: "switch" });
      const text = h("span", { class: "tm-toggle-text" });
      const set = (v) => { input.checked = !!v; text.textContent = v ? "On" : "Off"; };
      input.addEventListener("change", () => { values[key] = input.checked; text.textContent = input.checked ? "On" : "Off"; onChange(); });
      set(values[key]);
      return { node: h("label", { class: "toggle tm-toggle" }, input, text), input, set, frame: null };
    }
    if (field.type === "choice") {
      const select = h("select", { class: "select", id }, (field.options || []).map((o) => h("option", { value: o }, o)));
      const set = (v) => { select.value = v; };
      select.addEventListener("change", () => { values[key] = select.value; onChange(); });
      set(values[key]);
      return { node: select, input: select, set, frame: select };
    }
    if (field.type === "int" || field.type === "float") {
      const input = h("input", { type: "text", id, inputmode: field.type === "int" ? "numeric" : "decimal",
        autocomplete: "off", spellcheck: "false", "aria-describedby": null });
      const frame = h("div", { class: "tm-num" }, input, field.unit ? h("span", { class: "tm-num-unit" }, field.unit) : null);
      frame.addEventListener("mousedown", (event) => { if (event.target !== input) { event.preventDefault(); input.focus(); } });
      const set = (v) => { input.value = showNumber(field, v); };
      input.addEventListener("input", () => {
        const n = parseNumber(input.value);
        values[key] = n === null ? input.value.trim() : n;
        onChange();
      });
      input.addEventListener("blur", () => { if (typeof values[key] === "number") set(values[key]); });
      input.addEventListener("keydown", (event) => {
        if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
        const current = typeof values[key] === "number" ? values[key] : field.default;
        if (typeof current !== "number") return;
        event.preventDefault();
        const step = field.step || (field.type === "int" ? 1 : 0.1);
        let next = current + (event.key === "ArrowUp" ? step : -step);
        if (field.min !== null && field.min !== undefined) next = Math.max(field.min, next);
        if (field.max !== null && field.max !== undefined) next = Math.min(field.max, next);
        next = field.type === "int" ? Math.round(next) : Number(next.toFixed(10));
        values[key] = next;
        set(next);
        onChange();
      });
      set(values[key]);
      return { node: frame, input, set, frame };
    }
    const input = h("input", { type: "text", class: "input", id, spellcheck: "false", autocomplete: "off",
      placeholder: field.default === "" ? "Default" : null });
    const set = (v) => { input.value = v === null || v === undefined ? "" : String(v); };
    input.addEventListener("input", () => { values[key] = input.value; onChange(); });
    set(values[key]);
    return { node: input, input, set, frame: input };
  }

  /** One row of "All settings": label, lock, key, help, what lower and higher do, and the control
   *  with room for an error and the "Default … Reset" line. Returns the control with
   *  {field, err, meta, row, kind: "row"} added; markSettingRow keeps it up to date. */
  function settingRow(field, values, onChange, idPrefix = "tm-f-") {
    const id = idPrefix + field.key.replace(/[^a-z0-9]/gi, "-");
    const control = settingControl(field, values, id, onChange);
    const err = h("div", { class: "tm-field-error", hidden: true });
    const meta = h("div", { class: "tm-setting-meta", hidden: true });
    const row = h("div", { class: "tm-setting", dataset: { key: field.key } },
      h("div", { class: "tm-setting-text" },
        h("div", { class: "tm-setting-name" },
          h("label", { class: "tm-setting-label", for: id }, field.label),
          field.locked_on_resume ? h("span", { class: "tm-lock", title: "Fixed for the life of the run: it cannot be changed when resuming" }, icon("lock", 12)) : null,
          h("code", { class: "tm-key" }, field.key)),
        field.help ? h("p", { class: "tm-setting-help" }, field.help) : null,
        effectLines(field)),
      h("div", { class: "tm-setting-control" }, control.node, err, meta));
    return Object.assign(control, { field, err, meta, row, kind: "row", metaValue: undefined });
  }

  /** Mark a settings row as changed (from `def`) or in error, with the default and a Reset
   *  button under its control while it differs. */
  function markSettingRow(ctl, value, def, error, onReset) {
    const changed = !same(value, def);
    ctl.row.classList.toggle("is-changed", changed);
    ctl.row.classList.toggle("is-error", !!error);
    if (ctl.frame) ctl.frame.classList.toggle("is-changed", changed && !error);
    ctl.meta.hidden = !changed;
    if (changed && ctl.metaValue !== String(value)) {
      ctl.metaValue = String(value);
      ctl.meta.replaceChildren(h("span", null, "Default ", h("span", { class: "tm-default" }, formatValue(ctl.field, def))),
        h("button", { type: "button", class: "tm-reset", "aria-label": `Reset ${ctl.field.label} to its default`, onclick: onReset },
          icon("reset", 12), "Reset"));
    }
    if (!changed) ctl.metaValue = undefined;
  }

  /** Whether a setting matches a search of "All settings": its label, key, help and effects. */
  function settingMatches(field, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return true;
    return `${field.label} ${field.key} ${field.help || ""} ${effectsText(field)}`.toLowerCase().includes(q);
  }

  /** The settings a run keeps for life (`locked_on_resume`) as a note with a lock, from the
   *  sections of an indexed schema: a section locked whole is named once ("every network setting
   *  (model.*)"), the rest one by one. null when none is locked. */
  function lockedNote(sections, lead = "Fixed for the life of the run, so not changeable here") {
    const parts = [];
    for (const section of sections || []) {
      const fields = (section.fields || []).filter((f) => f.locked_on_resume);
      if (!fields.length) continue;
      if (fields.length === section.fields.length) parts.push(`every ${String(section.title || section.name).toLowerCase()} setting (${section.name}.*)`);
      else for (const f of fields) parts.push(`${f.label.toLowerCase()} (${f.key})`);
    }
    if (!parts.length) return null;
    const list = parts.length === 1 ? parts[0] : `${parts.slice(0, -1).join(", ")} and ${parts[parts.length - 1]}`;
    return h("p", { class: "tm-locked-note" }, icon("lock", 13), h("span", null, `${lead}: ${list}.`));
  }

  /** What "All settings" says when a search matches nothing. */
  function noMatchContent(query) {
    return [h("strong", null, `No setting matches “${String(query || "").trim()}”.`),
      h("span", null, "Search looks at the label, the key, the help text and what turning a setting down or up does.")];
  }

  // ---- Renaming a run (and so its models) ---------------------------------------------------

  const NAME_RULE = /^[A-Za-z0-9_-]+$/;

  /** Why a run cannot be renamed now, or null when it can. */
  function renameBlocked(run) {
    if (!run) return null;
    if (run.status === "running") return "Stop the run before renaming it.";
    if (run.status === "stopping") return "The run is stopping; rename it once it has stopped.";
    if (run.status === "queued") return "Take the run out of the queue before renaming it.";
    return null;
  }

  function modelsSentence(count) {
    const tail = " Matches already played keep working.";
    if (count === null || count === undefined) return "Renames the run folder and all of its models." + tail;
    if (count === 0) return "Renames the run folder. It has no saved models yet.";
    if (count === 1) return "Renames the run folder and its one model." + tail;
    return `Renames the run folder and all ${count} of its models.` + tail;
  }

  /** The shared rename dialog.
   *  name: the run's name. run: its summary, if the caller has it. names: every run name, if
   *  the caller has them (else they are fetched). modelCount: how many models it has, if known.
   *  fromModels: opened from the Models page, so the dialog explains that models follow their run.
   *  onRenamed(newName, summary): after the server has renamed it. */
  function renameDialog({ name, run = null, names = null, modelCount = null, fromModels = false, onRenamed = null }) {
    let taken = names ? new Set(names.map((n) => n.toLowerCase())) : null;
    let summary = run;
    let serverMessage = null;
    let busy = false;
    const input = h("input", { class: "input rn-input", id: "rn-name", type: "text", value: name, spellcheck: "false",
      autocomplete: "off", maxlength: 64, "aria-describedby": "rn-msg rn-effect" });
    const msg = h("div", { class: "rn-msg", id: "rn-msg", role: "alert" });
    const effect = h("p", { class: "rn-effect", id: "rn-effect" });
    const example = h("strong", null, name);
    const go = h("button", { type: "button", class: "button button-primary" }, icon("pencil", 14), "Rename");
    const cancel = h("button", { type: "button", class: "button" }, "Cancel");
    const content = h("div", { class: "rn" },
      fromModels ? h("p", { class: "rn-lede" }, "Models are named after the run that trained them, so this renames the run and every model it saved: ",
        h("strong", null, `${name} (best)`), " becomes ", h("strong", null, example, " (best)"), ".") : null,
      h("div", { class: "rn-current" }, h("span", { class: "field-label" }, "Current name"), h("span", { class: "rn-current-name" }, name)),
      h("div", { class: "field" }, h("label", { class: "field-label", for: "rn-name" }, "New name"), input, msg),
      effect,
      h("div", { class: "modal-actions" }, cancel, go));

    const problem = (value) => {
      const v = value.trim();
      if (!v) return "Give the run a name.";
      if (!NAME_RULE.test(v)) return "Use only letters, digits, - and _.";
      if (v === name) return "That is its name already.";
      if (taken && v.toLowerCase() !== name.toLowerCase() && taken.has(v.toLowerCase())) return "A run with this name already exists.";
      return null;
    };
    const refresh = () => {
      const blocked = renameBlocked(summary);
      const v = input.value.trim();
      const bad = blocked || serverMessage || problem(v);
      // "That is its name already" is the starting state, not a mistake: no red for it.
      const quiet = !blocked && !serverMessage && v === name;
      msg.hidden = !bad || quiet;
      if (bad && !quiet) msg.replaceChildren(icon("alert", 14), h("span", null, bad));
      input.classList.toggle("is-error", !!bad && !quiet);
      go.disabled = busy || !!bad;
      example.textContent = v && NAME_RULE.test(v) && v !== name ? v : "…";
      const count = modelCount !== null && modelCount !== undefined ? modelCount
        : summary ? (summary.checkpoints || 0) + (summary.has_best ? 1 : 0) : null;
      effect.textContent = modelsSentence(count);
    };
    const submit = async () => {
      if (busy || go.disabled) return;
      const to = input.value.trim();
      busy = true;
      go.disabled = true;
      go.lastChild.textContent = "Renaming…";
      try {
        const result = await App.post(`/api/runs/${encodeURIComponent(name)}/rename`, { name: to });
        close();
        App.toast(`Renamed ${name} to ${to}.`, "good");
        if (onRenamed) onRenamed(to, result);
      } catch (error) {
        busy = false;
        go.lastChild.textContent = "Rename";
        const text = String(error.message || error).trim();
        serverMessage = text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".");
        refresh();
      }
    };
    input.addEventListener("input", () => { serverMessage = null; refresh(); });
    input.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); submit(); } });
    go.addEventListener("click", submit);
    const close = App.modal({ title: fromModels ? "Rename run" : `Rename ${name}`, content });
    cancel.addEventListener("click", close);
    refresh();
    setTimeout(() => { input.focus(); input.select(); }, 0);
    // The names of the other runs (for the live check) and this run's state, when not given.
    if (!taken || !summary) {
      App.get("/api/runs").then((data) => {
        const list = (data && data.runs) || [];
        if (!taken) taken = new Set(list.map((r) => r.name.toLowerCase()));
        if (!summary) summary = list.find((r) => r.name === name) || null;
        if (content.isConnected) refresh();
      }).catch(() => { /* the server checks the name anyway */ });
    }
    return close;
  }

  // ---- Settings: what turning one down or up does ------------------------------------------

  /** [[label, sentence], ...] for a field's effects: Lower/Higher, Off/On for a switch, and the
   *  two option names for a two-way choice. Empty sentences are left out. When the server's
   *  sentence already starts with its label ("Off: nothing stops…", "cuda: the GPU."), the label
   *  is taken from the sentence rather than said twice. */
  function effectRows(field) {
    return [effectSide(field, "lower"), effectSide(field, "higher")].filter(Boolean);
  }

  /** One side of a field's effects as [label, sentence], or null when the server gives none. */
  function effectSide(field, which) {
    const e = field && field.effects;
    const text = e && typeof e[which] === "string" ? e[which].trim() : "";
    if (!text) return null;
    let labels = ["Lower", "Higher"];
    if (field.type === "bool") labels = ["Off", "On"];
    else if (field.type === "choice" && Array.isArray(field.options) && field.options.length === 2) labels = field.options.map(String);
    const label = labels[which === "lower" ? 0 : 1];
    const lead = text.match(/^([^\s:]{1,24}):\s+/);
    if (lead && lead[1].toLowerCase() === label.toLowerCase()) return [label, text.slice(lead[0].length)];
    return [label, text];
  }

  /** Everything a search over settings should look at in a field's effects. */
  function effectsText(field) {
    const e = (field && field.effects) || {};
    return [e.lower, e.higher, e.typical].filter((t) => typeof t === "string").join(" ");
  }

  /** The quiet "Lower: … / Higher: … / Typical: …" lines under a setting's help. */
  function effectLines(field) {
    const rows = effectRows(field);
    const typical = field && field.effects && typeof field.effects.typical === "string" ? field.effects.typical.trim() : "";
    if (!rows.length && !typical) return null;
    return h("div", { class: "tm-effects" },
      rows.map(([label, text]) => h("p", { class: "tm-effect" }, h("span", { class: "tm-effect-k" }, label + ":"), " ", text)),
      typical ? h("p", { class: "tm-effect" }, h("span", { class: "tm-effect-k" }, "Typical:"), " ", typical) : null);
  }

  return {
    svg, icon, significance, spread, spreadLegend, approx, sortHeader, compareBy, menu, toggleMenu, closeMenu, isMenuOpen,
    HEALTH, HEALTH_GAMES, HEALTH_ORDER, healthChip, trendSpark, renameBlocked, renameDialog, effectRows, effectSide, effectsText, effectLines,
    isGames, winSignificance, winText, winVerdict, winCell, networkLabel, modelWin,
    same, indexSchema, parseNumber, showNumber, formatValue, fieldError, settingControl, settingRow, markSettingRow,
    settingMatches, noMatchContent, lockedNote,
  };
})();

(function () {
  "use strict";
  const { h, fmt } = App;
  const K = window.LabKit;

  const KEY = { name: "run.name", hours: "run.hours", games: "run.max_games", blocks: "model.blocks", channels: "model.channels" };
  const LIMIT_KEYS = new Set([KEY.hours, KEY.games]);
  const SPECIAL = new Set(Object.values(KEY));
  const SUGGESTED_HOURS = 8; // a fresh form suggests an overnight run rather than no limit at all
  const NAME_RE = /^[A-Za-z0-9_-]+$/;
  const ACTIVE = new Set(["running", "queued", "stopping"]);
  const STATUS = {
    running: { label: "Running", tone: "info", order: 0 },
    stopping: { label: "Stopping", tone: "warning", order: 1 },
    queued: { label: "Queued", tone: "", order: 2 },
    finished: { label: "Finished", tone: "good", order: 3 },
    stopped: { label: "Stopped", tone: "", order: 4 },
    failed: { label: "Failed", tone: "critical", order: 5 },
  };
  const PHASES = { warmstart: "Warm start", pretraining: "Pretraining", selfplay: "Self-play" };
  const evalOf = (run, opponent) => (run.latest_eval && run.latest_eval[opponent]) || null;
  const spreadOf = (run, opponent) => { const e = evalOf(run, opponent); return e ? e.spread : null; };
  // To fit a 1440 px window with the verdict and the strength sparkline added, some columns
  // share a cell: the network size sits under the run's name (a name need not say the size), the
  // verdict under the status, the result against greedy under the one against greedy + leave,
  // games per second under the games, and the time of the last update under the time trained.
  const COLUMNS = [
    { key: "name", label: "Run", dir: 1, get: (r) => r.name.toLowerCase(),
      title: "Click a row, or its Charts button, for the run's page: every chart, the report, settings, log and checkpoints" },
    { key: "status", label: "Status and verdict", dir: 1, get: (r) => (STATUS[r.status] ? STATUS[r.status].order : 9),
      title: "Where the run is, and below it the verdict on how it went against greedy + leave (hover a verdict for the reason)" },
    { key: "trend", label: "Strength", sortable: false, cls: "tm-trend-cell",
      title: "Every evaluation against greedy + leave, oldest first, with its 95% interval shaded. The grey line is zero: level with the bot." },
    { key: "gl", label: "vs greedy-leave", num: true, dir: -1, get: (r) => spreadOf(r, "greedy-leave"),
      title: "Average spread a game against the greedy + leave bot at the latest evaluation, ± its 95% interval; below it, the same against greedy" },
    { key: "games", label: "Games", num: true, dir: -1, get: (r) => r.games, title: "Self-play games, and games per second at the last measurement" },
    { key: "hours", label: "Time", num: true, dir: -1, get: (r) => r.hours, title: "Time trained, and when the run last reported" },
  ];
  const FILTERS = [["all", "All"], ["active", "Active"], ["finished", "Finished"], ["stopped", "Stopped"], ["failed", "Failed"]];

  // Kept across re-mounts (theme switch, a visit to a run page) so a half-made form survives.
  let S = null; // the indexed schema
  let draft = null; // the runs being set up
  let uid = 0;
  let runsCache = null; // latest GET /api/runs
  const hist = { filter: "all", text: "", sort: null, dir: -1, selected: new Set() };

  App.registerView("training", {
    title: "Training", nav: true, order: 20, covers: ["run", "guide"], icon: K.svg("training", 18), mount,
  });

  // ---- Schema and values ----------------------------------------------------------------

  // This form starts self-play runs: only the settings whose `modes` include "selfplay" (the
  // `offline` section belongs to Train from games).
  const indexSchema = (schema) => K.indexSchema(schema, "selfplay");
  const { same, parseNumber, showNumber, formatValue, fieldError } = K;
  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const enc = encodeURIComponent;
  /** History lists self-play runs; runs that train from games have their own page. */
  const selfPlayRuns = () => (runsCache || []).filter((r) => !K.isGames(r));

  function initialValue(key) {
    const field = S.byKey[key];
    if (!field) return undefined;
    if (key === KEY.hours && !(field.default > 0)) return SUGGESTED_HOURS;
    return field.default;
  }

  /** The label, prefixed with its section when two settings share it ("Self-play candidate moves"). */
  function labelOf(field) {
    const twins = S.fields.filter((f) => f.label === field.label).length;
    if (twins < 2) return field.label;
    const section = S.sections.find((s) => s.name === field.section);
    return `${section ? section.title || section.name : field.section} ${field.label.toLowerCase()}`;
  }

  function networkOf(run) {
    return `${run.values[KEY.blocks]}x${run.values[KEY.channels]}`;
  }

  function speedRow(network) {
    return S.speed.find((row) => row.network === network) || null;
  }

  function activePreset(run) {
    const matchKeys = S.presetKeys.filter((k) => !LIMIT_KEYS.has(k));
    return S.presets.find((p) => matchKeys.every((k) =>
      same(run.values[k], p.overrides && k in p.overrides ? p.overrides[k] : initialValue(k)))) || null;
  }

  /** Set every key any preset touches: from this preset, else back to its starting value. The
   * time and game limits are kept unless this preset sets them or the previous one did. */
  function applyPreset(run, preset) {
    const previous = activePreset(run);
    for (const key of S.presetKeys) {
      const overrides = preset.overrides || {};
      if (key in overrides) run.values[key] = overrides[key];
      else if (LIMIT_KEYS.has(key)) {
        const before = previous && previous.overrides && key in previous.overrides;
        if (before && same(run.values[key], previous.overrides[key])) run.values[key] = initialValue(key);
      } else run.values[key] = initialValue(key);
    }
  }

  function overridesOf(run) {
    const out = {};
    for (const field of S.fields) {
      if (field.key === KEY.name) continue;
      if (!same(run.values[field.key], field.default)) out[field.key] = run.values[field.key];
    }
    return out;
  }

  const advancedFields = () => S.fields.filter((f) => f.advanced && f.key !== KEY.name);
  const changedAdvanced = (run) => advancedFields().filter((f) => !same(run.values[f.key], f.default));
  const cardDiffs = (run) => S.fields.filter((f) => !SPECIAL.has(f.key) && !same(run.values[f.key], f.default));

  // ---- Names ----------------------------------------------------------------------------

  function existingNames() {
    return new Set((runsCache || []).map((r) => r.name.toLowerCase()));
  }

  function nameTaken(name, except) {
    const lower = name.trim().toLowerCase();
    return existingNames().has(lower) || draft.runs.some((r) => r !== except && r.name.trim().toLowerCase() === lower);
  }

  function nameBase() {
    const d = new Date();
    return "run-" + String(d.getMonth() + 1).padStart(2, "0") + String(d.getDate()).padStart(2, "0");
  }

  function suggestName(base, except, firstLetter = 0) {
    for (let i = firstLetter; i < 26; i += 1) {
      const name = `${base}-${String.fromCharCode(97 + i)}`;
      if (!nameTaken(name, except)) return name;
    }
    for (let i = 2; ; i += 1) if (!nameTaken(`${base}-${i}`, except)) return `${base}-${i}`;
  }

  /** "run-1112-a" -> "run-1112-b"; "night3" -> "night3-b". */
  function copyName(name) {
    const match = name.match(/^(.*)-([a-y])$/);
    return match ? suggestName(match[1], null, match[2].charCodeAt(0) - 96) : suggestName(name, null, 1);
  }

  // ---- Validation -----------------------------------------------------------------------

  function nameError(run) {
    const name = run.name.trim();
    if (!name) return "Give the run a name.";
    if (!NAME_RE.test(name)) return "Use only letters, digits, - and _.";
    if (existingNames().has(name.toLowerCase())) return "A run with this name already exists.";
    if (draft.runs.some((r) => r !== run && r.name.trim().toLowerCase() === name.toLowerCase())) return "Another run in this list has the same name.";
    return null;
  }

  function errorsOf(run) {
    const errors = {};
    const bad = nameError(run);
    if (bad) errors[KEY.name] = bad;
    for (const field of S.fields) {
      if (field.key === KEY.name) continue;
      const error = fieldError(field, run.values[field.key]);
      if (error) errors[field.key] = error;
    }
    const server = draft.serverField;
    if (server && server.runId === run.id && !errors[server.key]) errors[server.key] = server.message;
    return errors;
  }

  // ---- Time and speed -------------------------------------------------------------------

  function planOf(run, share) {
    const network = networkOf(run);
    const row = speedRow(network);
    const hours = num(run.values[KEY.hours]) || 0;
    const maxGames = num(run.values[KEY.games]) || 0;
    const rate = row ? row.games_per_second / share : null;
    const plan = { network, row, rate, hours, maxGames, share, duration: null, games: null, by: "none" };
    if (hours > 0 && maxGames > 0 && rate) {
      const t = maxGames / rate / 3600;
      if (t < hours) Object.assign(plan, { duration: t, games: maxGames, by: "games-first" });
      else Object.assign(plan, { duration: hours, games: rate * hours * 3600, by: "time-first" });
    } else if (hours > 0) {
      Object.assign(plan, { duration: hours, games: rate ? rate * hours * 3600 : null, by: "time" });
    } else if (maxGames > 0) {
      Object.assign(plan, { duration: rate ? maxGames / rate / 3600 : null, games: maxGames, by: "games" });
    }
    return plan;
  }

  const trimZero = (text) => text.replace(/\.0$/, "");
  const rateWords = (r) => (r >= 10 ? fmt.int(r) : trimZero(fmt.fixed(r, 1)));
  /** "the 8-hour limit", "the 30-minute limit" */
  const limitWords = (hours) => (hours < 1 ? `${fmt.int(hours * 60)}-minute` : `${trimZero(fmt.fixed(hours, 1))}-hour`) + " limit";

  function hoursWords(hours) {
    if (hours < 1) return `${fmt.int(hours * 60)} minutes`;
    if (hours === 1) return "1 hour";
    return `${trimZero(fmt.fixed(hours, 1))} hours`;
  }

  /** Rounded for a total: "about 16 hours", "about 45 minutes". */
  function aboutHours(hours) {
    if (hours < 1) return `${fmt.int(Math.max(5, Math.round(hours * 12) * 5))} minutes`;
    if (hours < 10) return trimZero(fmt.fixed(hours, 1)) === "1" ? "1 hour" : `${trimZero(fmt.fixed(hours, 1))} hours`;
    return `${fmt.int(hours)} hours`;
  }

  function limitShort(plan) {
    const parts = [];
    if (plan.hours > 0) parts.push(plan.hours < 1 ? `${fmt.int(plan.hours * 60)} min` : `${trimZero(fmt.fixed(plan.hours, 1))} h`);
    if (plan.maxGames > 0) parts.push(`${fmt.compact(plan.maxGames)} games`);
    return parts.length ? parts.join(" or ") : "no limit";
  }

  function shareWords(n) {
    return n === 2 ? "half" : n === 3 ? "a third" : n === 4 ? "a quarter" : `1/${n}`;
  }

  function shareOfSpeed(n) {
    return n === 2 ? "half the speed" : `${shareWords(n)} of the speed`;
  }

  function listNames(names) {
    if (names.length <= 1) return names.join("");
    if (names.length > 3) return `${names.slice(0, 2).join(", ")} and ${names.length - 2} others`;
    return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  }

  function finishWords(hours) {
    const end = new Date(Date.now() + hours * 3600 * 1000);
    const time = end.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" });
    const today = new Date();
    const days = Math.round((new Date(end.toDateString()) - new Date(today.toDateString())) / 86400000);
    if (days === 0) return `${time} today`;
    if (days === 1) return `${time} tomorrow`;
    if (days < 7) return `${time} on ${end.toLocaleDateString("en-US", { weekday: "long" })}`;
    return fmt.datetime(end.getTime() / 1000);
  }

  // ---- The page -------------------------------------------------------------------------

  async function mount(container, route) {
    const demo = App.mock.enabled ? route.query.demo || "" : "";
    let controls = {};
    let refs = {};
    let lastRunsKey = "";
    let lastActiveKey = "";
    let runsError = null;
    let resolveFirst;
    const firstLoad = new Promise((resolve) => { resolveFirst = resolve; });

    const formHost = h("div", { class: "tm-form-host" },
      h("div", { class: "card tm-placeholder" }, h("span", { class: "muted" }, "Loading the training settings…")));
    const historyHost = h("div", { class: "tm-history-host" });
    const page = h("div", { class: "tm-page" },
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Training"),
          h("p", { class: "lede" }, "Start training runs, one at a time or several to compare, and look back at every run so far.")),
        h("div", { class: "page-actions" },
          h("button", { type: "button", class: "button button-ghost", onclick: () => historyHost.scrollIntoView({ behavior: "smooth", block: "start" }) },
            K.icon("clock", 16), "History"),
          h("a", { class: "button button-ghost", href: "#/guide?from=training", title: "Every setting: what it does, and what turning it down or up does" },
            K.icon("book", 16), "Tuning guide"))),
      formHost, historyHost);
    container.appendChild(page);

    buildHistory();
    App.every(5000, () => loadRuns(false));

    try {
      S = indexSchema(await App.get("/api/config/schema"));
    } catch (error) {
      formHost.replaceChildren(h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
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
      if (ok !== leavesWere) {
        leavesWere = ok;
        renderForm();
      } else if (warningKinds() !== warningsWere) {
        renderBasket();  // a warning appeared or cleared: it is shown beside the Start button
      }
      warningsWere = warningKinds();
    }));

    if (demo) {
      await firstLoad;
      applyDemo();
    }

    // ---- Draft ----------------------------------------------------------------------------

    function freshDraft() {
      draft = { runs: [], selected: null, mode: "queue", showAll: false, search: "", attempted: false,
        busy: false, serverError: null, serverField: null };
      const run = newRun();
      draft.runs.push(run);
      draft.selected = run.id;
    }

    function newRun() {
      const values = {};
      for (const field of S.fields) values[field.key] = initialValue(field.key);
      const run = { id: ++uid, name: "", autoName: true, values };
      run.name = suggestName(nameBase(), run);
      return run;
    }

    /** After a schema change, give every run a value for every field. */
    function reconcileDraft() {
      for (const run of draft.runs) {
        for (const field of S.fields) if (!(field.key in run.values)) run.values[field.key] = initialValue(field.key);
      }
      draft.busy = false;
    }

    function selectedRun() {
      return draft.runs.find((r) => r.id === draft.selected) || draft.runs[0];
    }

    function leavesReady() {
      if (demo === "noleaves") return false;
      const status = App.status;
      return !status || !status.leaves || status.leaves.ok !== false;
    }

    function shareContext() {
      const n = draft.runs.length;
      const all = runsCache || [];
      const active = all.filter((r) => r.status === "running" || r.status === "stopping");
      const queued = all.filter((r) => r.status === "queued");
      const modeVisible = n >= 2 || active.length + queued.length > 0;
      const parallel = modeVisible && draft.mode === "parallel";
      return { n, active, queued, modeVisible, parallel, share: parallel ? n + active.length : 1,
        ordered: modeVisible && !parallel && n > 1 };
    }

    // ---- Form -----------------------------------------------------------------------------

    function renderForm() {
      const ok = leavesReady();
      refs.notices = h("div", { class: "tm-notices" });
      refs.editor = h("div", { class: "tm-editor" });
      refs.basket = h("aside", { class: "tm-basket", "aria-label": "Runs to start" });
      const card = h("section", { class: "card tm-new", "aria-labelledby": "tm-new-title" },
        h("div", { class: "tm-new-head" },
          h("div", null, h("h2", { id: "tm-new-title" }, "Start training"),
            h("p", { class: "card-sub" }, "Set up a run, or several to compare. Anything you leave alone keeps its default from configs/train.toml."))),
        h("fieldset", { class: "tm-fieldset", disabled: !ok },
          h("div", { class: "tm-new-grid" }, h("div", { class: "tm-left" }, refs.notices, refs.editor), refs.basket)));
      formHost.replaceChildren(card);
      renderNotices();
      renderEditor();
      renderBasket();
    }

    function renderNotices() {
      const items = [];
      if (!leavesReady()) {
        items.push(h("div", { class: "notice notice-warning" }, K.icon("alert", 18, "tm-tone-warning"),
          h("div", null, h("strong", null, "Leave values have not been fitted yet. "),
            "Training ranks candidate moves with them and the network learns a correction to them, so they come first. In a terminal in the project folder run ",
            h("code", { class: "tm-code" }, "uv run scrabble-ai fit-leaves"),
            " (about 15 minutes). This form unlocks when they are ready.")));
      }
      if (draft.serverError) {
        items.push(h("div", { class: "notice notice-critical", role: "alert" }, K.icon("alert", 18, "tm-tone-critical"),
          h("div", null, h("strong", null, "The server did not start the runs. "), draft.serverError)));
      }
      refs.notices.replaceChildren(...items);
      refs.notices.hidden = !items.length;
    }

    function renderEditor() {
      const run = selectedRun();
      controls = {};
      const parts = [];
      if (draft.runs.length > 1) {
        const index = draft.runs.indexOf(run);
        refs.editingName = h("strong", null, run.name.trim() || "unnamed run");
        parts.push(h("div", { class: "tm-editing" }, K.icon("pencil", 14), h("span", null, "Editing ", refs.editingName),
          h("span", { class: "muted" }, `run ${index + 1} of ${draft.runs.length}`)));
      } else refs.editingName = null;
      if (S.presets.length) parts.push(presetBlock(run));
      parts.push(limitsBlock(run));
      const net = networkBlock(run);
      if (net) parts.push(net);
      const others = S.fields.filter((f) => !f.advanced && !SPECIAL.has(f.key));
      if (others.length) {
        parts.push(h("div", { class: "tm-block" }, h("h3", { class: "tm-block-title" }, "Also"),
          h("div", { class: "tm-ess-grid" }, others.map((f) => fieldCell(f, run)))));
      }
      const all = allSettingsBlock(run);
      if (all) parts.push(all);
      refs.editor.replaceChildren(...parts);
      refreshEditor();
    }

    function presetBlock(run) {
      refs.presets = h("div", { class: "tm-presets", role: "radiogroup", "aria-label": "Starting point" });
      refs.presetNote = h("span", { class: "tm-block-note" });
      return h("div", { class: "tm-block" },
        h("div", { class: "tm-block-head" }, h("h3", { class: "tm-block-title" }, "Starting point"), refs.presetNote),
        refs.presets);
    }

    function renderPresets(run) {
      const active = activePreset(run);
      refs.presets.replaceChildren(...S.presets.map((preset) => h("button", {
        type: "button", class: "tm-preset" + (preset === active ? " is-active" : ""), role: "radio",
        "aria-checked": preset === active ? "true" : "false",
        onclick: () => { applyPreset(run, preset); renderEditor(); renderBasket(); },
      }, h("span", { class: "tm-preset-top" }, h("span", { class: "tm-radio", "aria-hidden": "true" }),
        h("span", { class: "tm-preset-name" }, preset.name)),
      preset.description ? h("span", { class: "tm-preset-desc" }, preset.description) : null)));
      refs.presetNote.textContent = active ? "" : "Custom: no preset matches the settings below";
    }

    function limitsBlock(run) {
      const nameField = S.byKey[KEY.name];
      const cells = [];
      const input = h("input", { class: "input", id: "tm-name", type: "text", value: run.name, spellcheck: "false",
        autocomplete: "off", maxlength: 64, "aria-describedby": "tm-name-hint" });
      input.addEventListener("input", () => {
        run.name = input.value;
        run.autoName = false;
        if (refs.editingName) refs.editingName.textContent = run.name.trim() || "unnamed run";
        onValue(run, KEY.name);
      });
      const err = h("div", { class: "tm-field-error", hidden: true });
      refs.nameHint = h("div", { class: "tm-name-hint", id: "tm-name-hint" });
      controls[KEY.name] = { field: nameField || { key: KEY.name, label: "Run name", type: "str" }, frame: input, err, kind: "name", input };
      cells.push(h("div", { class: "field" },
        h("label", { class: "field-label", for: "tm-name" }, (nameField && nameField.label) || "Run name"), input, err, refs.nameHint));
      if (S.byKey[KEY.hours]) cells.push(fieldCell(S.byKey[KEY.hours], run));
      if (S.byKey[KEY.games]) cells.push(fieldCell(S.byKey[KEY.games], run));
      return h("div", { class: "tm-block" },
        h("h3", { class: "tm-block-title" }, "Name and limits"),
        h("div", { class: "tm-ess-grid" }, cells),
        h("p", { class: "field-help" }, "Training stops at whichever limit it reaches first, and 0 means no limit. A run keeps going when the app is closed."));
    }

    function networkBlock(run) {
      const blocks = S.byKey[KEY.blocks];
      const channels = S.byKey[KEY.channels];
      if (!blocks || !channels) return null;
      refs.sizes = h("div", { class: "tm-sizes", role: "group", "aria-label": "Benchmarked sizes" });
      refs.speed = h("div", { class: "tm-speed", "aria-live": "polite" });
      const locked = blocks.locked_on_resume || channels.locked_on_resume;
      return h("div", { class: "tm-block" },
        h("div", { class: "tm-block-head" }, h("h3", { class: "tm-block-title" }, "Network size"),
          locked ? h("span", { class: "tm-block-note" }, K.icon("lock", 12), "Fixed for the life of the run") : null),
        h("div", { class: "tm-network" },
          h("div", { class: "tm-network-inputs" }, fieldCell(blocks, run), h("span", { class: "tm-times", "aria-hidden": "true" }, "×"), fieldCell(channels, run)),
          refs.sizes),
        refs.speed);
    }

    function renderSizes(run) {
      if (!refs.sizes) return;
      const current = networkOf(run);
      App.append(App.clear(refs.sizes), [
        S.speed.length ? h("span", { class: "tm-sizes-label" }, "Benchmarked") : null,
        S.speed.map((row) => h("button", {
          type: "button", class: "tm-size" + (row.network === current ? " is-active" : ""),
          "aria-pressed": row.network === current ? "true" : "false",
          title: `${row.network}: about ${rateWords(row.games_per_second)} games a second (${row.measured ? "measured" : "estimated"})`,
          onclick: () => {
            const [b, c] = row.network.split("x").map(Number);
            run.values[KEY.blocks] = b;
            run.values[KEY.channels] = c;
            controls[KEY.blocks].set(b);
            controls[KEY.channels].set(c);
            onValue(run, KEY.blocks);
          },
        }, row.network, h("span", { class: "tm-size-rate" }, `${rateWords(row.games_per_second)}/s`)))]);
    }

    function renderSpeed(run) {
      if (!refs.speed) return;
      const ctx = shareContext();
      const plan = planOf(run, ctx.share);
      let main;
      let sub;
      if (!plan.row) {
        main = `${plan.network} has not been measured on this PC, so there is no speed figure for it.`;
        sub = S.speed.length ? "The benchmarked sizes have one; for any other size the time a run takes is not known." : "No sizes have been benchmarked yet.";
      } else {
        const how = plan.row.measured ? "measured" : "estimated";
        const others = ctx.share - 1;
        const lead = ctx.share > 1
          ? `Sharing the GPU with ${others === 1 ? "one other run" : `${others} other runs`}, about ${rateWords(plan.rate)} games a second (${shareWords(ctx.share)} of ${rateWords(plan.row.games_per_second)}, ${how})`
          : `About ${rateWords(plan.rate)} games a second on this PC (${how})`;
        let tail;
        if (plan.by === "time") tail = `roughly ${K.approx(plan.games)} games in ${hoursWords(plan.hours)}`;
        else if (plan.by === "time-first") tail = `roughly ${K.approx(plan.games)} games in ${hoursWords(plan.hours)}, so the time limit comes before the game limit`;
        else if (plan.by === "games-first") tail = `${fmt.int(plan.maxGames)} games take about ${fmt.duration(plan.duration)}, within the ${limitWords(plan.hours)}`;
        else if (plan.by === "games") tail = `${fmt.int(plan.maxGames)} games would take about ${fmt.duration(plan.duration)}`;
        else tail = `roughly ${K.approx(plan.rate * 3600)} games an hour, with no limit set`;
        main = `${lead}: ${tail}.`;
        sub = [plan.row.parameters ? `${fmt.compact(plan.row.parameters)} parameters` : null,
          plan.row.vram_gb ? `about ${fmt.fixed(plan.row.vram_gb, 1)} GB of GPU memory` : null].filter(Boolean).join(" · ");
      }
      refs.speed.className = "tm-speed" + (plan.row ? "" : " is-unknown");
      refs.speed.replaceChildren(K.icon("gauge", 18),
        h("div", null, h("p", { class: "tm-speed-main" }, main), sub ? h("p", { class: "tm-speed-sub" }, sub) : null));
    }

    /** A labelled control in the essentials grid. */
    function fieldCell(field, run) {
      const id = "tm-e-" + field.key.replace(/[^a-z0-9]/gi, "-");
      const control = makeControl(field, run, id);
      const err = h("div", { class: "tm-field-error", hidden: true });
      controls[field.key] = Object.assign(control, { field, err, kind: "cell" });
      return h("div", { class: "field" }, h("label", { class: "field-label", for: id, title: field.help || null }, field.label), control.node, err);
    }

    /** The input for one setting of `run` (LabKit.settingControl, shared with Train from games). */
    function makeControl(field, run, id) {
      return K.settingControl(field, run.values, id, () => onValue(run, field.key));
    }

    function allSettingsBlock(run) {
      const fields = advancedFields();
      if (!fields.length) return null;
      const sectionCount = S.sections.filter((s) => (s.fields || []).some((f) => S.byKey[f.key].advanced && f.key !== KEY.name)).length;
      refs.allCount = h("span", { class: "tm-pill tm-pill-accent" });
      refs.allErrors = h("span", { class: "tm-pill tm-pill-critical" });
      const toggle = h("button", { type: "button", class: "tm-all-toggle", "aria-expanded": draft.showAll ? "true" : "false",
        "aria-controls": "tm-all-body", onclick: () => { draft.showAll = !draft.showAll; renderEditor(); } },
      K.icon("chevron", 16, "tm-chev"), "All settings");
      const head = h("div", { class: "tm-all-head" }, toggle, refs.allCount, refs.allErrors,
        h("span", { class: "tm-all-summary" }, `${fields.length} more in ${sectionCount} sections · `,
          h("a", { href: "#/guide?from=training", title: "Every setting in one table: what it does, lower, higher, typical range" }, "tuning guide")));
      const wrap = h("div", { class: "tm-all" + (draft.showAll ? " is-open" : "") }, head);
      refs.sections = [];
      refs.noMatch = null;
      refs.resetAll = null;
      if (!draft.showAll) return wrap;

      const search = h("input", { class: "input", type: "search", placeholder: "Search names, help and effects", value: draft.search,
        "aria-label": "Search settings by name, key, help or effect" });
      search.addEventListener("input", () => { draft.search = search.value; applySearch(); });
      refs.resetAll = h("button", { type: "button", class: "button button-small button-ghost",
        onclick: () => { for (const f of fields) run.values[f.key] = f.default; onValue(run, null); renderEditor(); } },
      K.icon("reset", 14), "Reset all");
      head.appendChild(h("div", { class: "tm-all-tools" }, h("div", { class: "tm-search" }, K.icon("search", 15), search), refs.resetAll));

      const body = h("div", { class: "tm-all-body", id: "tm-all-body" });
      for (const section of S.sections) {
        const list = (section.fields || []).map((f) => S.byKey[f.key]).filter((f) => f.advanced && f.key !== KEY.name);
        if (!list.length) continue;
        const node = h("section", { class: "tm-section" },
          h("div", { class: "tm-section-head" }, h("h4", { class: "tm-section-title" }, section.title || section.name),
            h("code", { class: "tm-key" }, `[${section.name}]`)),
          section.description ? h("p", { class: "tm-section-desc" }, section.description) : null,
          list.map((f) => settingRow(f, run)));
        refs.sections.push({ node, fields: list });
        body.appendChild(node);
      }
      refs.noMatch = h("div", { class: "tm-nomatch", hidden: true });
      body.appendChild(refs.noMatch);
      wrap.appendChild(body);
      applySearch();
      return wrap;
    }

    function settingRow(field, run) {
      const ctl = K.settingRow(field, run.values, () => onValue(run, field.key));
      controls[field.key] = ctl;
      return ctl.row;
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

    /** Update error and changed markers, and everything derived from the values. */
    function refreshEditor() {
      const run = selectedRun();
      const errors = errorsOf(run);
      const showName = draft.attempted || !run.autoName;
      for (const [key, ctl] of Object.entries(controls)) {
        let error = errors[key] || null;
        if (key === KEY.name && !showName && error && !nameTaken(run.name, run)) error = null;
        ctl.err.hidden = !error;
        if (error) ctl.err.replaceChildren(K.icon("alert", 14), h("span", null, error));
        if (ctl.frame) ctl.frame.classList.toggle("is-error", !!error);
        if (ctl.kind !== "row") continue;
        K.markSettingRow(ctl, run.values[key], ctl.field.default, error,
          () => { run.values[key] = ctl.field.default; ctl.set(ctl.field.default); onValue(run, key); });
      }
      if (refs.nameHint) {
        const out = (run.values["run.out_dir"] || "runs").replace(/[\\/]+$/, "");
        const ok = !errors[KEY.name];
        refs.nameHint.hidden = !ok;
        if (ok) refs.nameHint.replaceChildren(K.icon("check", 13), `Saved to ${out}/${run.name.trim()}/`);
      }
      if (refs.presets) renderPresets(run);
      renderSizes(run);
      renderSpeed(run);
      if (refs.allCount) {
        const changed = changedAdvanced(run).length;
        refs.allCount.textContent = `${changed} changed`;
        refs.allCount.hidden = !changed;
        const bad = advancedFields().filter((f) => errors[f.key]).length;
        refs.allErrors.textContent = `${bad} to fix`;
        refs.allErrors.hidden = !bad;
        if (refs.resetAll) refs.resetAll.disabled = !changed;
      }
    }

    function onValue(run, key) {
      if (draft.serverField && (key === null || draft.serverField.key === key)) {
        draft.serverField = null;
        draft.serverError = null;
        renderNotices();
      }
      refreshEditor();
      renderBasket();
    }

    // ---- Basket: the runs to start ----------------------------------------------------------

    function renderBasket() {
      const ctx = shareContext();
      const n = ctx.n;
      lastActiveKey = activeKey();
      const items = [
        h("div", { class: "tm-basket-head" }, h("h3", null, n === 1 ? "Run to start" : "Runs to start"),
          h("span", { class: "tm-pill" }, String(n))),
        h("div", { class: "tm-runlist" }, draft.runs.map((run, i) => runCard(run, i, ctx))),
        h("button", { type: "button", class: "tm-add", onclick: addRun,
          title: "Adds a copy of the run you are editing, with a new name, so you can change one thing" },
        K.icon("plus", 16), "Add another run"),
      ];
      if (n === 1) items.push(h("p", { class: "tm-hint" }, "To compare settings, add another run: it starts as a copy of this one."));
      if (ctx.modeVisible) items.push(modeBlock(ctx));
      items.push(summaryBlock(ctx));
      const label = draft.busy ? "Starting…" : n === 1 ? "Start training" : `Start ${n} runs`;
      items.push(h("button", { type: "button", class: "button button-primary tm-start", disabled: draft.busy || !leavesReady(), onclick: submit },
        K.icon("play", 15), label));
      refs.basket.replaceChildren(h("div", { class: "tm-basket-inner" }, items));
    }

    function runCard(run, index, ctx) {
      const plan = planOf(run, ctx.share);
      const errorCount = Object.keys(errorsOf(run)).length;
      const diffs = cardDiffs(run);
      const selected = run.id === draft.selected;
      const networkChanged = !same(run.values[KEY.blocks], S.byKey[KEY.blocks] && S.byKey[KEY.blocks].default) ||
        !same(run.values[KEY.channels], S.byKey[KEY.channels] && S.byKey[KEY.channels].default);
      const rateLine = plan.rate
        ? `${rateWords(plan.rate)} games/s${ctx.share > 1 ? " each" : ""}${plan.games ? ` · about ${fmt.compact(plan.games)} games` : ""}`
        : "Speed not measured for this size";
      const choose = () => { if (draft.selected !== run.id) { draft.selected = run.id; renderEditor(); renderBasket(); } };
      return h("div", {
        class: "tm-runcard" + (selected ? " is-selected" : "") + (errorCount ? " has-error" : ""),
        role: "button", tabindex: "0", "aria-pressed": selected ? "true" : "false",
        "aria-label": `${run.name || "Unnamed run"}${selected ? ", being edited" : ", select to edit"}`,
        onclick: choose,
        onkeydown: (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); } },
      },
      h("div", { class: "tm-runcard-top" },
        ctx.ordered ? h("span", { class: "tm-runcard-index", title: `Runs ${index === 0 ? "first" : `after run ${index}`}` }, String(index + 1)) : null,
        h("span", { class: "tm-runcard-name" }, run.name.trim() || "Unnamed run"),
        errorCount ? h("span", { class: "tm-runcard-err", title: `${errorCount} setting${errorCount > 1 ? "s" : ""} to fix` }, K.icon("alert", 14)) : null,
        draft.runs.length > 1 ? h("button", { type: "button", class: "tm-runcard-remove", "aria-label": `Remove ${run.name}`, title: "Remove this run",
          onclick: (event) => { event.stopPropagation(); removeRun(run); } }, K.icon("x", 14)) : null),
      h("div", { class: "tm-runcard-meta" }, h("span", { class: "tm-runcard-net" }, plan.network), h("span", null, limitShort(plan))),
      h("div", { class: "tm-runcard-rate" }, rateLine),
      diffs.length
        ? h("div", { class: "tm-runcard-diff" }, diffs.slice(0, 3).map((f) => h("span", { class: "tm-diff", title: f.key },
          labelOf(f), " ", h("b", null, formatValue(f, run.values[f.key])))),
        diffs.length > 3 ? h("span", { class: "tm-diff tm-diff-more", title: diffs.slice(3).map((f) => f.key).join(", ") }, `+${diffs.length - 3} more`) : null)
        : h("div", { class: "tm-runcard-diff tm-runcard-plain" }, networkChanged ? "No other changes" : "Default settings"));
    }

    function modeBlock(ctx) {
      const existing = [...ctx.active, ...ctx.queued].map((r) => r.name);
      const would = ctx.n + ctx.active.length;
      const notes = [];
      if (existing.length) {
        const running = ctx.active.map((r) => r.name);
        const waiting = ctx.queued.map((r) => r.name);
        const bits = [];
        if (running.length) bits.push(`${listNames(running)} ${running.length > 1 ? "are" : "is"} running`);
        if (waiting.length) bits.push(`${listNames(waiting)} ${waiting.length > 1 ? "are" : "is"} queued`);
        notes.push(`${bits.join(" and ")}. One after another, the new ${ctx.n > 1 ? "runs wait" : "run waits"} until ${existing.length > 1 ? "they finish" : "it finishes"}.`);
      }
      notes.push(`Runs at the same time share the GPU, so each goes at about ${shareOfSpeed(Math.max(2, would))} and the total time is about the same as running them one after another (three at once measured 4 to 8% more total work).`);
      return h("div", { class: "tm-mode" },
        h("div", { class: "tm-mode-label" }, ctx.n > 1 ? "How to run them" : "When to start it"),
        h("div", { class: "segmented tm-mode-seg", role: "group", "aria-label": "How to run them" },
          [["queue", "One after another"], ["parallel", "At the same time"]].map(([mode, label]) => h("button", {
            type: "button", "aria-pressed": draft.mode === mode ? "true" : "false",
            onclick: () => { draft.mode = mode; renderBasket(); refreshEditor(); },
          }, label))),
        notes.map((text) => h("p", { class: "tm-mode-note" }, text)));
    }

    function summaryBlock(ctx) {
      const plans = draft.runs.map((r) => planOf(r, ctx.share));
      const n = ctx.n;
      const head = n === 1 ? "1 run" : ctx.parallel ? `${n} runs at the same time` : `${n} runs, one after another`;
      const unlimited = draft.runs.filter((r, i) => plans[i].by === "none");
      const unknown = plans.filter((p) => p.by !== "none" && p.duration === null);
      const lines = [];
      const warnings = [];
      let main;
      let total = null;
      if (unlimited.length) {
        main = n === 1 ? "1 run with no limit: it trains until you stop it."
          : `${head}, with no limit on ${unlimited.length === n ? "any of them" : listNames(unlimited.map((r) => r.name))}.`;
        if (!ctx.parallel && n > 1) {
          const blocking = draft.runs.find((r, i) => plans[i].by === "none" && i < n - 1);
          if (blocking) warnings.push({ tone: "warning", text: `${blocking.name.trim() || "A run"} has no limit, so the runs after it wait until you stop it.` });
        }
      } else if (unknown.length) {
        const known = plans.map((p) => p.duration).filter((d) => d !== null);
        const partial = known.length ? (ctx.parallel ? Math.max(...known) : known.reduce((a, b) => a + b, 0)) : null;
        const nets = [...new Set(unknown.map((p) => p.network))];
        main = `${head}. The time for ${listNames(nets)} is not known${partial !== null ? `; the rest take about ${aboutHours(partial)}` : ""}.`;
      } else {
        const durations = plans.map((p) => p.duration);
        total = ctx.parallel ? Math.max(...durations) : durations.reduce((a, b) => a + b, 0);
        main = n === 1 ? `1 run, about ${aboutHours(total)}.` : `${head}, about ${aboutHours(total)} in total.`;
      }
      const waiting = !ctx.parallel ? [...ctx.active, ...ctx.queued].map((r) => r.name) : [];
      if (waiting.length) lines.push(`Starts when ${listNames(waiting)} ${waiting.length > 1 ? "finish" : "finishes"}.`);
      else if (total !== null) lines.push(`Done around ${finishWords(total)}.`);

      const gpu = App.status && App.status.gpu;
      if (gpu && gpu.memory_total_mb) {
        const rows = plans.map((p) => p.row);
        if (rows.every(Boolean)) {
          const need = ctx.parallel ? rows.reduce((a, r) => a + (r.vram_gb || 0), 0) : Math.max(...rows.map((r) => r.vram_gb || 0));
          const have = gpu.memory_total_mb / 1024;
          if (need > have * 0.9) {
            warnings.push({ tone: "warning", text: `${ctx.parallel && n > 1 ? "Together these runs need" : "This needs"} about ${fmt.fixed(need, 1)} GB of GPU memory; the GPU has ${fmt.fixed(have, 0)} GB.` });
          }
        }
      }
      // What the server warns about (other programs holding GPU memory, for one) belongs here,
      // beside the button that would start a run into it.
      for (const w of (App.status && App.status.warnings) || []) warnings.push({ tone: "warning", text: w.message });
      const errorCount = draft.runs.reduce((sum, r) => sum + Object.keys(errorsOf(r)).length, 0);
      if (errorCount) warnings.push({ tone: "critical", text: `${errorCount} setting${errorCount > 1 ? "s" : ""} to fix before starting.` });
      return h("div", { class: "tm-summary" },
        h("p", { class: "tm-summary-main" }, main),
        lines.map((text) => h("p", { class: "tm-summary-sub" }, text)),
        warnings.map((w) => h("p", { class: "tm-warn tm-warn-" + w.tone }, K.icon("alert", 14), h("span", null, w.text))));
    }

    function addRun() {
      const source = selectedRun();
      const copy = { id: ++uid, name: "", autoName: false, values: Object.assign({}, source.values) };
      copy.name = copyName(source.name.trim() || nameBase());
      draft.runs.push(copy);
      draft.selected = copy.id;
      renderEditor();
      renderBasket();
    }

    function removeRun(run) {
      const index = draft.runs.indexOf(run);
      draft.runs.splice(index, 1);
      if (draft.selected === run.id) draft.selected = draft.runs[Math.max(0, index - 1)].id;
      if (draft.serverField && draft.serverField.runId === run.id) { draft.serverField = null; draft.serverError = null; renderNotices(); }
      renderEditor();
      renderBasket();
    }

    async function submit() {
      draft.attempted = true;
      const bad = draft.runs.find((r) => Object.keys(errorsOf(r)).length);
      if (bad) {
        draft.selected = bad.id;
        const keys = Object.keys(errorsOf(bad));
        const order = [KEY.name, ...S.fields.map((f) => f.key)];
        keys.sort((a, b) => order.indexOf(a) - order.indexOf(b));
        const target = S.byKey[keys[0]];
        if (target && target.advanced && keys[0] !== KEY.name) { draft.showAll = true; draft.search = ""; }
        renderEditor();
        renderBasket();
        const ctl = controls[keys[0]];
        if (ctl && ctl.input) {
          ctl.input.focus({ preventScroll: true });
          ctl.input.scrollIntoView({ behavior: "smooth", block: "center" });
        }
        return;
      }
      draft.busy = true;
      draft.serverError = null;
      draft.serverField = null;
      renderNotices();
      renderBasket();
      const ctx = shareContext();
      const body = {
        mode: ctx.modeVisible ? draft.mode : "queue",
        runs: draft.runs.map((r) => ({ name: r.name.trim(), overrides: overridesOf(r) })),
      };
      try {
        const result = await App.post("/api/runs", body);
        const started = result.started || [];
        const queued = result.queued || [];
        const parts = [];
        if (started.length) parts.push(`Started ${listNames(started)}.`);
        if (queued.length) parts.push(`${listNames(queued)} ${queued.length > 1 ? "are" : "is"} queued.`);
        App.toast(parts.join(" ") || "Started.", "good");
        freshDraft();
        renderForm();
        await loadRuns(true);
        App.refreshStatus();
        historyHost.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        draft.busy = false;
        mapServerError(String(error.message || error));
        renderNotices();
        renderEditor();
        renderBasket();
        formHost.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }

    /** Show the server's message at the top, and beside the field it names when it names one. */
    function mapServerError(message) {
      draft.serverError = message;
      const run = draft.runs.find((r) => r.name.trim() && message.includes(r.name.trim())) || selectedRun();
      let text = message;
      if (run.name.trim() && text.startsWith(run.name.trim() + ":")) text = text.slice(run.name.trim().length + 1).trim();
      // The setting at fault is the one the message names first ("train.min_replay must be at least train.batch_size").
      const field = S.fields.filter((f) => message.includes(f.key))
        .sort((a, b) => message.indexOf(a.key) - message.indexOf(b.key) || b.key.length - a.key.length)[0];
      let key = field ? field.key : null;
      if (!key && /\bname\b|already exists|already in use/i.test(message)) key = KEY.name;
      if (!key) return;
      draft.serverField = { runId: run.id, key, message: text.charAt(0).toUpperCase() + text.slice(1) + (/[.!?]$/.test(text) ? "" : ".") };
      draft.selected = run.id;
      if (S.byKey[key] && S.byKey[key].advanced) { draft.showAll = true; draft.search = ""; }
    }

    // ---- History --------------------------------------------------------------------------

    function buildHistory() {
      refs.filters = h("div", { class: "tm-filter-host" });
      refs.compare = h("div", { class: "tm-compare-host" });
      refs.countLine = h("p", { class: "card-sub" }, "Loading…");
      refs.table = h("div", { class: "tm-table-wrap" });
      refs.foot = h("div", { class: "tm-footnote", hidden: true }, K.spreadLegend());
      const text = h("input", { class: "input", type: "search", placeholder: "Filter by name or setting", value: hist.text, "aria-label": "Filter runs" });
      text.addEventListener("input", () => { hist.text = text.value; renderTable(); });
      refs.tools = h("div", { class: "tm-history-tools" }, refs.filters, h("div", { class: "tm-search tm-search-wide" }, K.icon("search", 15), text), refs.compare);
      historyHost.replaceChildren(h("section", { class: "card card-flush tm-history", id: "tm-history", "aria-labelledby": "tm-history-title" },
        h("div", { class: "tm-history-head" },
          h("div", { class: "tm-history-title" }, h("h2", { id: "tm-history-title" }, "History"), refs.countLine),
          refs.tools),
        refs.table, refs.foot));
    }

    async function loadRuns(force) {
      let data;
      try {
        data = await App.get("/api/runs");
      } catch (error) {
        runsError = String(error.message || error);
        if (!runsCache) renderHistory();
        resolveFirst();
        return;
      }
      runsError = null;
      const runs = data.runs || [];
      const key = JSON.stringify(runs);
      const changed = key !== lastRunsKey;
      lastRunsKey = key;
      runsCache = runs;
      if (changed || force) {
        renderHistory();
        if (draft && S && refs.basket) {
          let renamed = false;
          for (const run of draft.runs) {
            if (run.autoName && nameTaken(run.name, run)) { run.name = suggestName(nameBase(), run); renamed = true; }
          }
          if (renamed) renderEditor();
          else refreshEditor();
          if (renamed || activeKey() !== lastActiveKey) renderBasket();
        }
      }
      resolveFirst();
    }

    function activeKey() {
      return (runsCache || []).filter((r) => ACTIVE.has(r.status)).map((r) => r.name + ":" + r.status).join(",");
    }

    function renderHistory() {
      if (!runsCache) {
        refs.countLine.textContent = runsError ? "" : "Loading…";
        refs.table.replaceChildren(runsError
          ? h("div", { class: "tm-pad" }, h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
            h("div", null, h("strong", null, "Could not load the runs. "), runsError)))
          : h("div", { class: "empty" }, "Loading runs…"));
        return;
      }
      const runs = selfPlayRuns();
      const fromGames = runsCache.length - runs.length;
      refs.tools.hidden = !runs.length;
      const counts = {
        all: runs.length,
        active: runs.filter((r) => ACTIVE.has(r.status)).length,
        finished: runs.filter((r) => r.status === "finished").length,
        stopped: runs.filter((r) => r.status === "stopped").length,
        failed: runs.filter((r) => r.status === "failed").length,
      };
      refs.filters.replaceChildren(h("div", { class: "segmented", role: "group", "aria-label": "Show runs" },
        FILTERS.map(([key, label]) => h("button", { type: "button", "aria-pressed": hist.filter === key ? "true" : "false",
          onclick: () => { hist.filter = key; renderHistory(); } }, label, h("span", { class: "tm-seg-count" }, String(counts[key]))))));
      const hours = runs.reduce((sum, r) => sum + (r.hours || 0), 0);
      const games = runs.reduce((sum, r) => sum + (r.games || 0), 0);
      App.append(App.clear(refs.countLine), [runs.length
        ? `${runs.length} self-play run${runs.length > 1 ? "s" : ""}: ${fmt.duration(hours)} of training and ${fmt.compact(games)} games in all.`
        : "No self-play runs yet.",
      fromGames ? [" ", h("a", { href: "#/games?section=runs" }, `${fromGames} run${fromGames > 1 ? "s" : ""} trained from saved games`),
        fromGames > 1 ? " are on the Train from games page." : " is on the Train from games page."] : null]);
      renderCompareButton();
      renderTable();
    }

    function renderCompareButton() {
      const names = new Set((runsCache || []).map((r) => r.name));
      for (const name of [...hist.selected]) if (!names.has(name)) hist.selected.delete(name);
      const selected = [...hist.selected];
      App.append(App.clear(refs.compare), [
        selected.length ? h("button", { type: "button", class: "button button-small button-ghost",
          onclick: () => { hist.selected.clear(); renderHistory(); } }, "Clear") : null,
        h("button", { type: "button", class: "button button-small" + (selected.length >= 2 ? " button-primary" : ""),
          disabled: selected.length < 2, title: selected.length < 2 ? "Tick two or more runs to compare them" : null,
          onclick: () => openCompare(selected) }, K.icon("compare", 15), selected.length >= 2 ? `Compare ${selected.length} runs` : "Compare")]);
    }

    function matchesFilter(run) {
      if (hist.filter === "all") return true;
      if (hist.filter === "active") return ACTIVE.has(run.status);
      return run.status === hist.filter;
    }

    function matchesText(run) {
      const q = hist.text.trim().toLowerCase();
      if (!q) return true;
      const changed = Object.entries(run.changed || {}).map(([k, v]) => `${k}=${v}`).join(" ");
      return `${run.name} ${run.network || ""} ${run.status} ${changed}`.toLowerCase().includes(q);
    }

    function sortRuns(list) {
      const column = COLUMNS.find((c) => c.key === hist.sort);
      const newest = (a, b) => (b.created || 0) - (a.created || 0);
      const byStatus = (a, b) => (STATUS[a.status] ? STATUS[a.status].order : 9) - (STATUS[b.status] ? STATUS[b.status].order : 9);
      const main = column ? K.compareBy(column.get, hist.dir) : newest;
      const cmp = (a, b) => main(a, b) || newest(a, b);
      const pinned = list.filter((r) => ACTIVE.has(r.status)).sort(column ? cmp : (a, b) => byStatus(a, b) || newest(a, b));
      return pinned.concat(list.filter((r) => !ACTIVE.has(r.status)).sort(cmp));
    }

    function onSort(key) {
      const column = COLUMNS.find((c) => c.key === key);
      if (hist.sort === key) hist.dir = -hist.dir;
      else { hist.sort = key; hist.dir = column.dir; }
      renderTable();
    }

    function renderTable() {
      const runs = selfPlayRuns();
      if (!runs.length) {
        refs.foot.hidden = true;
        refs.table.replaceChildren(h("div", { class: "empty tm-empty" },
          h("span", { class: "tm-empty-icon" }, K.icon("training", 28)),
          h("strong", null, runsCache && runsCache.length ? "No self-play runs yet" : "No training runs yet"),
          h("span", null, "Start one above. It appears here straight away, and keeps going when the app is closed.")));
        return;
      }
      const list = sortRuns(runs.filter(matchesFilter).filter(matchesText));
      refs.foot.hidden = !list.length;
      if (!list.length) {
        refs.table.replaceChildren(h("div", { class: "empty tm-empty" }, h("strong", null, "No runs match these filters."),
          h("button", { type: "button", class: "button button-small", onclick: () => { hist.filter = "all"; hist.text = ""; buildHistory(); renderHistory(); } }, "Show every run")));
        return;
      }
      for (const chart of refs.sparks || []) chart.destroy();
      refs.sparks = [];
      refs.table.replaceChildren(h("table", { class: "table tm-table tm-history-table" },
        h("thead", null, h("tr", null, h("th", { class: "tm-check" }, h("span", { class: "tm-sr" }, "Select to compare")),
          COLUMNS.map((c) => (c.sortable === false
            ? h("th", { class: c.cls || null, scope: "col", title: c.title || null }, c.label)
            : K.sortHeader(c, hist, onSort))),
          h("th", { class: "tm-actions-cell" }, h("span", { class: "tm-sr" }, "Actions")))),
        h("tbody", null, list.map(historyRow))));
    }

    function historyRow(run) {
      const status = STATUS[run.status] || { label: run.status, tone: "" };
      const href = "#/run/" + enc(run.name);
      const stop = (event) => event.stopPropagation();
      const checked = hist.selected.has(run.name);
      const tr = h("tr", { class: "clickable tm-run-row" + (ACTIVE.has(run.status) ? " tm-row-active" : "") + (checked ? " is-checked" : ""),
        title: `Open ${run.name}: charts, report, settings, log and checkpoints`,
        onclick: (event) => { if (!window.getSelection().toString()) App.go(href); else event.preventDefault(); } });
      const box = h("input", { type: "checkbox", checked, "aria-label": `Select ${run.name} to compare` });
      box.addEventListener("change", () => {
        if (box.checked) hist.selected.add(run.name); else hist.selected.delete(run.name);
        tr.classList.toggle("is-checked", box.checked);
        renderCompareButton();
      });
      const spark = K.trendSpark(run);
      if (spark.chart) refs.sparks.push(spark.chart);
      const rate = run.games_per_second;
      tr.append(
        h("td", { class: "tm-check", onclick: stop, title: "" }, box),
        h("td", { class: "tm-name-cell" },
          h("a", { class: "tm-run-link", href, onclick: stop }, run.name, K.icon("chevron", 13, "tm-run-link-go")),
          h("span", { class: "tm-sub" }, h("span", { class: "tm-net" }, run.network || "–"),
            run.parameters ? ` · ${fmt.compact(run.parameters)} parameters` : ""),
          nameSubline(run)),
        h("td", { class: "tm-status-cell" },
          h("div", { class: "tm-status-line" }, h("span", { class: "badge" + (status.tone ? " badge-" + status.tone : "") }, status.label),
            run.status === "running" && run.phase ? h("span", { class: "tm-sub" }, PHASES[run.phase] || run.phase) : null),
          healthCell(run)),
        h("td", { class: "tm-trend-cell" }, spark.chart ? spark.node
          : h("span", { class: "tm-sub" }, run.trend && Array.isArray(run.trend.games) ? (run.status === "queued" ? "" : "no evaluation yet") : "–")),
        h("td", { class: "num" }, K.spread(evalOf(run, "greedy-leave"), "greedy-leave"), greedyLine(run)),
        h("td", { class: "num", title: run.games !== null && run.games !== undefined ? `${fmt.int(run.games)} games` : null }, fmt.compact(run.games),
          rate ? h("span", { class: "tm-sub" }, `${fmt.fixed(rate, 1)}/s`) : null),
        h("td", { class: "num" }, run.hours ? fmt.duration(run.hours) : "–",
          ACTIVE.has(run.status)
            ? (run.limit && run.limit.hours ? h("span", { class: "tm-sub" }, `of ${fmt.duration(run.limit.hours).replace(/ 00 m$/, "")}`) : null)
            : h("span", { class: "tm-sub", title: `Last report ${fmt.datetime(run.updated)}` }, fmt.ago(run.updated))),
        h("td", { class: "tm-actions-cell", onclick: stop, title: "" }, rowActions(run)));
      return tr;
    }

    /** The verdict chip, or why there is none. An older server sends no verdict: nothing then. */
    function healthCell(run) {
      const chip = K.healthChip(run.health, "tm-health");
      if (chip) return chip;
      if (run.health === undefined || run.status === "queued") return null;
      return h("span", { class: "tm-sub tm-health" }, run.status === "running" || run.status === "stopping" ? "First evaluation to come" : "Never evaluated");
    }

    /** The latest result against greedy, small, under the one against greedy + leave. */
    function greedyLine(run) {
      const e = evalOf(run, "greedy");
      if (!e || e.spread === null || e.spread === undefined) return null;
      const sig = K.significance(e);
      return h("span", { class: "tm-sub tm-greedy-line", title: `Against greedy at the latest evaluation, ± its 95% interval` +
        (sig === "up" ? ": significantly better" : sig === "down" ? ": significantly worse" : "") },
      "greedy ", fmt.spread(e.spread, e.ci));
    }

    function nameSubline(run) {
      if (run.status === "running" || run.status === "stopping") {
        if (run.progress === null || run.progress === undefined) return h("span", { class: "tm-sub" }, "No limit");
        return h("span", { class: "tm-progress-line" },
          h("span", { class: "progress", role: "progressbar", "aria-valuenow": Math.round(run.progress * 100), "aria-valuemin": 0, "aria-valuemax": 100 },
            h("span", { style: { width: `${Math.min(100, run.progress * 100)}%` } })),
          h("span", { class: "tm-sub" }, fmt.pct(run.progress, 0)));
      }
      if (run.status === "queued") return h("span", { class: "tm-sub" }, "Waiting for the GPU");
      if (run.status === "failed" && run.error) {
        const lines = String(run.error).trim().split("\n").map((l) => l.trim()).filter(Boolean);
        const line = lines.find((l) => /(Error|Exception)\b/.test(l)) || lines[lines.length - 1];
        return h("span", { class: "tm-sub tm-error-line", title: run.error }, line);
      }
      return null;
    }

    function rowActions(run) {
      const items = [h("a", { class: "button button-small button-primary tm-charts", href: "#/run/" + enc(run.name),
        title: `${run.name}'s charts: strength, learning, self-play, and its report` }, K.icon("chart", 14), "Charts")];
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
      K.renameDialog({
        name: run.name, run, names: (runsCache || []).map((r) => r.name),
        onRenamed: async (to) => {
          if (hist.selected.delete(run.name)) hist.selected.add(to);
          await loadRuns(true);
        },
      });
    }

    async function stopRun(run) {
      const queued = run.status === "queued";
      const ok = await App.confirm(queued
        ? { title: `Take ${run.name} out of the queue?`, confirmLabel: "Take out of queue",
          body: "It has not started, so there is nothing to save. It stays in the history; delete it if you no longer want it." }
        : { title: `Stop ${run.name}?`, confirmLabel: "Stop run",
          body: "The run finishes its current training step, evaluates against the baselines and saves a checkpoint, which takes a minute or two. You can resume it later from that checkpoint." });
      if (!ok) return;
      try {
        await App.post(`/api/runs/${enc(run.name)}/stop`);
        App.toast(queued ? `${run.name} is out of the queue.` : `Stopping ${run.name}. It saves a checkpoint first.`, "info");
        await loadRuns(true);
        App.refreshStatus();
      } catch (error) {
        App.toast(`Could not stop ${run.name}: ${error.message}`, "critical");
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
            (run.has_best || run.checkpoints ? " Its models leave the library." : "")),
          h("p", null, h("strong", null, "This cannot be undone."))),
      });
      if (!ok) return;
      try {
        await App.del(`/api/runs/${enc(run.name)}`);
        hist.selected.delete(run.name);
        App.toast(`Deleted ${run.name}.`, "info");
        await loadRuns(true);
      } catch (error) {
        App.toast(`Could not delete ${run.name}: ${error.message}`, "critical");
      }
    }

    // ---- Resume dialog ----------------------------------------------------------------------

    function parseOverrides(text) {
      const overrides = {};
      const errors = [];
      text.split(/\r?\n/).forEach((raw, i) => {
        const line = raw.trim();
        if (!line || line.startsWith("#")) return;
        const where = `Line ${i + 1}`;
        const match = line.match(/^([A-Za-z_]+\.[A-Za-z_]+)\s*=\s*(.*)$/);
        if (!match) { errors.push(`${where}: write it as section.key=value.`); return; }
        const field = S.byKey[match[1]];
        if (!field) { errors.push(`${where}: there is no setting called ${match[1]}.`); return; }
        if (field.key === KEY.name) { errors.push(`${where}: a run cannot be renamed.`); return; }
        if (field.locked_on_resume) { errors.push(`${where}: ${field.label} (${field.key}) cannot be changed when resuming.`); return; }
        const text = match[2].trim().replace(/^["']|["']$/g, "");
        let value = text;
        if (field.type === "int" || field.type === "float") { const n = parseNumber(text); value = n === null ? NaN : n; }
        else if (field.type === "bool") {
          const lower = text.toLowerCase();
          value = ["true", "1", "yes", "on"].includes(lower) ? true : ["false", "0", "no", "off"].includes(lower) ? false : null;
          if (value === null) { errors.push(`${where}: ${field.key} is on or off (true or false).`); return; }
        }
        const problem = fieldError(field, value);
        if (problem) { errors.push(`${where}: ${field.key}: ${problem.charAt(0).toLowerCase() + problem.slice(1)}`); return; }
        overrides[field.key] = value;
      });
      return { overrides, errors };
    }

    function lockedSentence() {
      const parts = [];
      for (const section of S.sections) {
        const fields = (section.fields || []).filter((f) => f.locked_on_resume);
        if (!fields.length) continue;
        if (fields.length === section.fields.length) parts.push(`every ${String(section.title || section.name).toLowerCase()} setting (${section.name}.*)`);
        else for (const f of fields) parts.push(`${f.label.toLowerCase()} (${f.key})`);
      }
      return parts.length ? `Fixed for the life of the run, so not changeable here: ${listNames(parts)}.` : null;
    }

    function smallNumber(id, value, unit, min, max, step) {
      const input = h("input", { type: "number", id, min, max, step, value: String(value) });
      return { input, node: h("div", { class: "tm-num" }, input, h("span", { class: "tm-num-unit" }, unit)) };
    }

    async function openResume(run) {
      if (!S) return;
      const hours = smallNumber("tm-resume-hours", run.limit && run.limit.hours ? run.limit.hours : SUGGESTED_HOURS, "hours", 0, 168, 0.5);
      const games = smallNumber("tm-resume-games", run.limit && run.limit.max_games ? run.limit.max_games : 0, "games", 0, null, 1);
      const area = h("textarea", { class: "input tm-overrides", id: "tm-resume-overrides", rows: 4, spellcheck: "false",
        placeholder: "train.learning_rate=0.0003\nselfplay.temperature_end=0.3" });
      const errors = h("div", { class: "tm-resume-errors", role: "alert", hidden: true });
      const locked = lockedSentence();
      const go = h("button", { type: "button", class: "button button-primary" }, K.icon("resume", 15), "Resume");
      const cancel = h("button", { type: "button", class: "button" }, "Cancel");
      const content = h("div", { class: "tm-resume" },
        h("p", null, `${run.name} continues from its newest checkpoint with the settings saved in it. The limits apply to this session.`),
        h("div", { class: "tm-resume-grid" },
          h("div", { class: "field" }, h("label", { class: "field-label", for: "tm-resume-hours" }, "Time limit"), hours.node),
          h("div", { class: "field" }, h("label", { class: "field-label", for: "tm-resume-games" }, "Game limit"), games.node)),
        h("p", { class: "field-help" }, "0 means no limit."),
        h("div", { class: "field" }, h("label", { class: "field-label", for: "tm-resume-overrides" }, "Change settings (optional)"), area,
          h("p", { class: "field-help" }, "One section.key=value per line. The keys are shown under All settings in the form.")),
        locked ? h("p", { class: "tm-locked-note" }, K.icon("lock", 13), h("span", null, locked)) : null,
        errors,
        h("div", { class: "modal-actions" }, cancel, go));
      const close = App.modal({ title: `Resume ${run.name}`, content });
      cancel.addEventListener("click", close);
      const showErrors = (list) => {
        errors.hidden = !list.length;
        errors.replaceChildren(...list.map((text) => h("p", null, K.icon("alert", 14), h("span", null, text))));
      };
      area.addEventListener("input", () => { if (!errors.hidden) showErrors(parseOverrides(area.value).errors); });
      go.addEventListener("click", async () => {
        const list = [];
        const h1 = Number(hours.input.value);
        const g1 = Number(games.input.value);
        if (hours.input.value === "" || !Number.isFinite(h1) || h1 < 0) list.push("The time limit must be a number of hours, 0 or more.");
        if (games.input.value === "" || !Number.isInteger(g1) || g1 < 0) list.push("The game limit must be a whole number, 0 or more.");
        const parsed = parseOverrides(area.value);
        list.push(...parsed.errors);
        showErrors(list);
        if (list.length) return;
        go.disabled = true;
        try {
          await App.post(`/api/runs/${enc(run.name)}/resume`, { hours: h1, max_games: g1, overrides: parsed.overrides });
          close();
          App.toast(`Resumed ${run.name}.`, "good");
          await loadRuns(true);
          App.refreshStatus();
        } catch (error) {
          go.disabled = false;
          showErrors([String(error.message || error)]);
        }
      });
      if (demo === "resume") area.value = "train.learning_rate=0.0003\nmodel.blocks=10";
      if (demo === "resume") go.click();
      setTimeout(() => hours.input.focus(), 0);
    }

    // ---- Compare dialog ---------------------------------------------------------------------

    async function openCompare(names) {
      const runs = names.map((n) => (runsCache || []).find((r) => r.name === n)).filter(Boolean);
      const content = h("div", { class: "tm-compare" }, h("p", { class: "muted" }, "Loading the evaluations…"));
      App.modal({ title: `Compare ${runs.length} runs`, content, wide: true });
      const metrics = await Promise.all(runs.map((r) => App.get(`/api/runs/${enc(r.name)}/metrics?since=0`).catch(() => ({ records: [] }))));
      const evals = metrics.map((m) => (m.records || [])
        .filter((r) => r.kind === "eval" && r.opponent === "greedy-leave" && r.spread !== null && r.spread !== undefined)
        .sort((a, b) => a.games - b.games));
      if (!content.isConnected) return;
      const parts = [];
      let charted = false;
      const chartHost = h("div", { class: "tm-compare-chart" });
      if (window.Charts && Charts.line && evals.some((e) => e.length)) {
        content.replaceChildren(chartHost);
        try {
          Charts.line(chartHost, {
            title: "Spread against greedy + leave",
            series: runs.map((r, i) => ({
              name: r.name,
              points: evals[i].map((e) => [e.games, e.spread]),
              band: evals[i].filter((e) => e.spread_ci !== null && e.spread_ci !== undefined)
                .map((e) => [e.games, e.spread - e.spread_ci, e.spread + e.spread_ci]),
            })),
            x: { label: "Games", format: (v) => fmt.compact(v) },
            y: { label: "Points a game", format: (v) => fmt.signed(v), zero: true },
            referenceY: [{ y: 0, label: "Level with greedy + leave" }],
            height: 300,
          });
          charted = true;
        } catch (error) {
          console.warn(error);
        }
      }
      if (charted) parts.push(chartHost);
      parts.push(h("h3", { class: "tm-compare-title" }, "Evaluations against greedy + leave"));
      parts.push(resultsTable(runs, evals));
      parts.push(h("h3", { class: "tm-compare-title" }, "Settings that differ"));
      parts.push(settingsDiffTable(runs));
      content.replaceChildren(...parts);
    }

    function resultsTable(runs, evals) {
      return h("div", { class: "tm-table-wrap" }, h("table", { class: "table tm-table tm-compact" },
        h("thead", null, h("tr", null, ["Run", "Network", "Games", "Evaluations", "Latest", "Best", "First above zero"].map((label, i) =>
          h("th", { class: i >= 2 ? "num" : null }, label)))),
        h("tbody", null, runs.map((run, i) => {
          const list = evals[i];
          const last = list[list.length - 1];
          const best = list.reduce((a, e) => (!a || e.spread > a.spread ? e : a), null);
          const first = list.find((e) => e.spread_ci !== null && e.spread - e.spread_ci > 0);
          const entry = (e) => (e ? { spread: e.spread, ci: e.spread_ci } : null);
          return h("tr", null,
            h("td", null, h("strong", null, run.name)),
            h("td", null, run.network || "–"),
            h("td", { class: "num" }, fmt.compact(run.games)),
            h("td", { class: "num" }, fmt.int(list.length)),
            h("td", { class: "num" }, K.spread(entry(last), "greedy-leave")),
            h("td", { class: "num" }, best ? h("span", null, K.spread(entry(best), "greedy-leave"), h("span", { class: "tm-sub" }, `at ${fmt.compact(best.games)} games`)) : "–"),
            h("td", { class: "num" }, first ? `${fmt.compact(first.games)} games` : h("span", { class: "muted" }, "not yet")));
        }))));
    }

    function settingsDiffTable(runs) {
      const keys = new Set();
      for (const run of runs) for (const key of Object.keys(run.changed || {})) keys.add(key);
      const valueOf = (run, key) => (run.changed && key in run.changed ? run.changed[key] : S && S.byKey[key] ? S.byKey[key].default : undefined);
      const order = S ? S.fields.map((f) => f.key) : [];
      const rows = [...keys].filter((key) => {
        const first = valueOf(runs[0], key);
        return runs.some((r) => !same(valueOf(r, key), first));
      }).sort((a, b) => order.indexOf(a) - order.indexOf(b));
      const nets = runs.map((r) => r.network);
      const showNet = nets.some((n) => n !== nets[0]);
      if (showNet) for (const key of [KEY.blocks, KEY.channels]) if (rows.includes(key)) rows.splice(rows.indexOf(key), 1);
      if (!rows.length && !showNet) return h("p", { class: "muted" }, "Apart from their names, these runs use the same settings.");
      const fieldOf = (key) => (S && S.byKey[key]) || { key, label: key, type: "str" };
      return h("div", { class: "tm-table-wrap" }, h("table", { class: "table tm-table tm-compact" },
        h("thead", null, h("tr", null, h("th", null, "Setting"), runs.map((r) => h("th", null, r.name)))),
        h("tbody", null,
          showNet ? h("tr", null, h("td", null, "Network"), runs.map((r) => h("td", null, h("strong", null, r.network || "–")))) : null,
          rows.map((key) => {
            const field = fieldOf(key);
            return h("tr", null,
              h("td", null, S && S.byKey[key] ? labelOf(S.byKey[key]) : key, h("span", { class: "tm-sub tm-mono" }, key)),
              runs.map((r) => {
                const changed = r.changed && key in r.changed && S && S.byKey[key] && !same(r.changed[key], S.byKey[key].default);
                const value = valueOf(r, key);
                const unit = field.unit && typeof value === "number" ? ` ${field.unit}` : "";
                return h("td", { class: changed ? "tm-cell-changed" : "secondary" }, formatValue(field, value) + unit,
                  changed ? null : h("span", { class: "tm-sub" }, "default"));
              }));
          }))));
    }

    // ---- Mock-only demo states ------------------------------------------------------------

    function applyDemo() {
      const run = selectedRun();
      const finished = selfPlayRuns().filter((r) => r.status === "finished");
      if (demo === "multi" || demo === "parallel") {
        const second = { id: ++uid, name: copyName(run.name), autoName: false, values: Object.assign({}, run.values) };
        second.values[KEY.blocks] = 10;
        second.values[KEY.channels] = 128;
        draft.runs.push(second);
        const third = { id: ++uid, name: "", autoName: false, values: Object.assign({}, run.values) };
        draft.runs.push(third);
        third.name = copyName(second.name);
        third.values["train.learning_rate"] = 0.0005;
        third.values["train.td_lambda"] = 0.9;
        draft.selected = second.id;
        draft.mode = demo === "parallel" ? "parallel" : "queue";
      } else if (demo === "settings" || demo === "search") {
        draft.showAll = true;
        run.values["train.learning_rate"] = 0.0005;
        run.values["train.td_lambda"] = 0.9;
        run.values["selfplay.temperature_end"] = 0.8;
        if (demo === "search") draft.search = "temperature";
      } else if (demo === "errors") {
        draft.showAll = true;
        draft.attempted = true;
        run.name = (runsCache && runsCache[0] && runsCache[0].name) || "night1";
        run.autoName = false;
        run.values[KEY.channels] = 100;
        run.values["train.learning_rate"] = 2;
        run.values["selfplay.max_candidates"] = 4;
        mapServerError(`${run.name}: selfplay.max_exchanges must be below max_candidates`);
      } else if (demo === "resume") {
        const target = selfPlayRuns().find((r) => r.status === "stopped" && r.checkpoints) || finished[0];
        if (target) openResume(target);
        return;
      } else if (demo === "compare") {
        const pick = selfPlayRuns().filter((r) => r.latest_eval).slice(0, 3);
        pick.forEach((r) => hist.selected.add(r.name));
        renderHistory();
        openCompare(pick.map((r) => r.name));
        return;
      } else if (demo === "delete") {
        if (finished[0]) deleteRun(finished[0]);
        return;
      } else if (demo === "rename") {
        if (finished[0]) renameRun(finished[0]);
        return;
      } else if (demo === "stop") {
        const running = selfPlayRuns().find((r) => r.status === "running");
        if (running) stopRun(running);
        return;
      }
      renderForm();
    }
  }
})();
