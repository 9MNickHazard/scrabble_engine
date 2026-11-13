/* Tuning guide: #/guide[?from=training|run/<name>|models|overview].
 *
 * Every training setting in one table, grouped by section as in configs/train.toml: what it
 * does, what turning it down does, what turning it up does, the range worth trying, and its
 * default. All of it comes from GET /api/config/schema (each field's `help` and `effects`), so the
 * server owns the words. A filter box narrows the rows. Not in the sidebar: the Training page
 * header, "All settings" and a run's verdict banner link here, and "Back" returns to where the
 * reader came from (`from`). run.name and run.out_dir are left out: the app sets them.
 *
 * Uses window.LabKit from views/training.js (icons, the effects labels).
 */
(function () {
  "use strict";

  const { h, fmt } = App;
  const SKIP = new Set(["run.name", "run.out_dir"]);
  let filterText = "";  // kept across visits

  App.registerView("guide", { title: "Tuning guide", nav: false, order: 26, mount });

  /** Where "Back" goes, and what it says. Anything unexpected goes back to Training. */
  function backTarget(from) {
    const value = String(from || "");
    const run = value.match(/^run\/(.+)$/);
    if (run) return { href: "#/run/" + encodeURIComponent(run[1]), label: `Back to ${run[1]}` };
    if (value === "models") return { href: "#/models", label: "Back to models" };
    if (value === "overview") return { href: "#/overview", label: "Back to the overview" };
    return { href: "#/training", label: "Back to training" };
  }

  function defaultText(field) {
    const v = field.default;
    if (v === null || v === undefined) return h("span", { class: "muted" }, "–");
    let text;
    if (field.type === "bool") text = v ? "On" : "Off";
    else if (typeof v === "number") text = field.type === "int" ? fmt.int(v) : String(Number(v.toPrecision(6)));
    else if (v === "") return h("span", { class: "muted" }, "empty");
    else text = String(v);
    return [h("span", { class: "gd-default" }, text), field.unit && typeof v === "number" ? h("span", { class: "gd-unit" }, " " + field.unit) : null];
  }

  async function mount(container, route) {
    const K = window.LabKit;
    const back = backTarget(route.query.from);
    const search = h("input", { class: "input", type: "search", value: filterText, placeholder: "Filter: a name, a key or a word",
      "aria-label": "Filter the settings" });
    const count = h("span", { class: "gd-count" });
    const notices = h("div", { class: "gd-notices" });
    const tableHost = h("div", { class: "card gd-card" }, h("p", { class: "muted gd-loading" }, "Loading the settings…"));
    container.appendChild(h("div", { class: "gd-page" },
      h("div", { class: "page-head gd-head" },
        h("div", null,
          h("a", { class: "dash-back", href: back.href }, K.icon("chevron", 13, "gd-back-icon"), back.label),
          h("h1", null, "Tuning guide"),
          h("p", { class: "lede" }, "Every training setting: what it does, what turning it down or up does, and the range worth trying. " +
            "A lock marks the settings fixed for the life of a run. Change them in the new-run form's All settings, or when resuming a run.")),
        h("div", { class: "gd-tools" }, h("div", { class: "tm-search gd-search" }, K.icon("search", 15), search), count)),
      notices, tableHost));

    let schema;
    try {
      schema = await App.get("/api/config/schema");
    } catch (error) {
      tableHost.replaceChildren(h("div", { class: "notice notice-critical" }, K.icon("alert", 18, "tm-tone-critical"),
        h("div", null, h("strong", null, "Could not load the settings. "), String(error.message || error))));
      return;
    }

    const rows = [];  // {node, hay, section}
    const sections = [];  // {node, rows}
    const fields = [];
    for (const section of schema.sections || []) for (const field of section.fields || []) if (!SKIP.has(field.key)) fields.push(field);
    const described = fields.filter((f) => f.effects && (f.effects.lower || f.effects.higher || f.effects.typical)).length;
    if (!described) {
      notices.appendChild(h("div", { class: "notice" }, K.icon("alert", 18, "tm-tone-warning"),
        h("div", null, h("strong", null, "This version of the server does not describe what turning each setting down or up does. "),
          "The table shows what each setting does and its default; the other columns fill in once the server sends them.")));
    }

    // The column says Lower or Higher; an on/off or two-way setting names its side in the cell.
    const effectCell = (field, which) => {
      const side = K.effectSide(field, which);
      if (!side) return h("td", { class: "gd-effect" }, h("span", { class: "muted" }, "–"));
      const named = field.type === "bool" || (field.type === "choice" && Array.isArray(field.options) && field.options.length === 2);
      return h("td", { class: "gd-effect" }, named ? h("span", { class: "gd-effect-k" }, side[0] + ": ") : null, side[1]);
    };

    const tbody = h("tbody");
    for (const section of schema.sections || []) {
      const list = (section.fields || []).filter((f) => !SKIP.has(f.key));
      if (!list.length) continue;
      const head = h("tr", { class: "gd-section" }, h("th", { colspan: "6", scope: "rowgroup" },
        h("span", { class: "gd-section-title" }, section.title || section.name), " ",
        h("code", { class: "tm-key" }, `[${section.name}]`),
        section.description ? h("span", { class: "gd-section-desc" }, section.description) : null));
      tbody.appendChild(head);
      const entry = { node: head, rows: [] };
      for (const field of list) {
        const e = field.effects || {};
        const typical = typeof e.typical === "string" && e.typical.trim() ? e.typical.trim() : null;
        const tr = h("tr", { class: "gd-row" },
          h("th", { scope: "row", class: "gd-setting" },
            h("div", { class: "gd-label" }, field.label,
              field.locked_on_resume ? h("span", { class: "tm-lock", title: "Fixed for the life of the run: it cannot be changed when resuming" }, K.icon("lock", 12)) : null),
            h("code", { class: "tm-key" }, field.key)),
          h("td", { class: "gd-help" }, field.help || h("span", { class: "muted" }, "–")),
          effectCell(field, "lower"),
          effectCell(field, "higher"),
          h("td", { class: "gd-typical" }, typical || h("span", { class: "muted" }, "–")),
          h("td", { class: "gd-default-cell" }, defaultText(field)));
        tbody.appendChild(tr);
        const hay = [section.title, section.name, field.label, field.key, field.help, e.lower, e.higher, e.typical]
          .filter((t) => typeof t === "string").join(" ").toLowerCase();
        const row = { node: tr, hay };
        rows.push(row);
        entry.rows.push(row);
      }
      sections.push(entry);
    }
    const none = h("tr", { class: "gd-none" }, h("td", { colspan: "6" }, h("strong", null, "No setting matches."),
      " The filter looks at names, keys, what each setting does and its effects."));
    tbody.appendChild(none);
    tableHost.replaceChildren(h("table", { class: "table gd-table" },
      h("colgroup", null, ["gd-c-setting", "gd-c-help", "gd-c-effect", "gd-c-effect", "gd-c-typical", "gd-c-default"].map((c) => h("col", { class: c }))),
      h("thead", null, h("tr", null,
        h("th", { scope: "col" }, "Setting"), h("th", { scope: "col" }, "What it does"),
        h("th", { scope: "col" }, "Lower"), h("th", { scope: "col" }, "Higher"),
        h("th", { scope: "col" }, "Typical range"), h("th", { scope: "col" }, "Default"))),
      tbody));

    const apply = () => {
      const words = filterText.trim().toLowerCase().split(/\s+/).filter(Boolean);
      let shown = 0;
      for (const section of sections) {
        let any = false;
        for (const row of section.rows) {
          const match = words.every((w) => row.hay.includes(w));
          row.node.hidden = !match;
          if (match) { any = true; shown += 1; }
        }
        section.node.hidden = !any;
      }
      none.hidden = shown > 0;
      count.textContent = words.length ? `${shown} of ${rows.length} settings` : `${rows.length} settings`;
    };
    search.addEventListener("input", () => { filterText = search.value; apply(); });
    apply();
    if (!filterText) search.focus({ preventScroll: true });
  }
})();
