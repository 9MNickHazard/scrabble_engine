/* Settings: #/settings. How the app looks (theme, the glass and what is behind it) and what it
 * is running on. Choices are kept in the browser's local storage by app.js. */
(function () {
  "use strict";

  const { h } = App;

  const ICON = '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" ' +
    'stroke-linecap="round" stroke-linejoin="round"><circle cx="10" cy="10" r="2.6"/><path d="M10 2.2v2.1M10 15.7v2.1M2.2 10h2.1' +
    'M15.7 10h2.1M4.5 4.5l1.5 1.5M14 14l1.5 1.5M4.5 15.5L6 14M14 6l1.5-1.5"/></svg>';

  const BACKDROPS = [
    { mode: "frost", title: "Frosted desktop", swatch: "set-swatch-frost",
      help: "Your desktop and other windows show through the app, blurred by Windows." },
    { mode: "clear", title: "Clear desktop", swatch: "set-swatch-clear",
      help: "They show through sharp. Harder to read over a busy window." },
    { mode: "builtin", title: "Built-in", swatch: "set-swatch-builtin",
      help: "The app's own soft backdrop. Nothing shows through." },
  ];

  App.registerView("settings", { title: "Settings", nav: true, foot: true, order: 99, icon: ICON, mount });

  function mount(container) {
    const body = h("div", { class: "set-grid" });
    container.append(
      h("div", { class: "page-head" },
        h("div", null, h("h1", null, "Settings"), h("p", { class: "lede" }, "How the app looks, and what it is running on."))),
      body);
    const draw = () => body.replaceChildren(h("div", { class: "stack" }, appearanceCard(), aboutCard()), glassCard());
    draw();
    App.onCleanup(App.on("appearance", draw));
    App.onCleanup(App.on("status", () => {
      const about = body.querySelector(".set-about");
      if (about) about.replaceWith(aboutCard());
    }));
  }

  function row(title, help, control) {
    return h("div", { class: "set-row" },
      h("div", { class: "set-row-text" }, h("div", { class: "set-row-title" }, title),
        help ? h("div", { class: "set-row-help" }, help) : null),
      control);
  }

  function appearanceCard() {
    const a = App.appearance;
    const choice = (theme, label) => h("button", { type: "button", "aria-pressed": String(a.theme === theme),
      onclick: () => App.setTheme(theme) }, label);
    return h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", { class: "card-title" }, "Appearance")),
      row("Theme", "Light is white glass with black outlines. Dark is smoked glass.",
        h("div", { class: "segmented", role: "group", "aria-label": "Theme" }, choice("light", "Light"), choice("dark", "Dark"))));
  }

  function glassCard() {
    const a = App.appearance;
    const canSeeDesktop = a.native && a.seeThrough !== false;
    const options = BACKDROPS.map((option) => {
      const unavailable = option.mode !== "builtin" && (!canSeeDesktop || (option.mode === "frost" && a.frost === false));
      return h("button", { class: "set-option", type: "button", disabled: unavailable,
        "aria-pressed": String(a.backdropInUse === option.mode), onclick: () => App.setBackdrop(option.mode) },
        h("span", { class: "set-swatch " + option.swatch, "aria-hidden": "true" }),
        h("span", { class: "set-option-title" }, option.title),
        h("span", { class: "set-option-help" }, option.help));
    });
    let note = null;
    if (!a.native) {
      note = "Seeing the desktop through the app needs the app's own window (the Scrabble Lab shortcut). " +
        "This is an ordinary browser window, which cannot be see-through.";
    } else if (a.seeThrough === false) {
      note = "This window could not be made see-through" + (a.problem ? ` (${a.problem})` : "") + ", so the built-in backdrop is used.";
    } else if (a.frost === false) {
      note = "Windows did not accept the frosted effect" + (a.problem ? ` (${a.problem})` : "") + ". Clear and built-in still work.";
    }
    const value = h("span", { class: "num" }, a.tint + "%");
    const range = h("input", { type: "range", min: "0", max: "100", step: "5", value: String(a.tint), id: "set-tint",
      "aria-label": "How milky the glass is",
      oninput: () => { App.setTint(Number(range.value)); value.textContent = App.appearance.tint + "%"; } });
    return h("section", { class: "card" },
      h("div", { class: "card-head" }, h("h2", { class: "card-title" }, "Glass")),
      h("div", { class: "set-row-title" }, "What is behind the glass"),
      h("div", { class: "set-options", role: "group", "aria-label": "What is behind the glass" }, options),
      note ? h("p", { class: "set-note" }, note) : null,
      h("div", { style: { height: "var(--space-4)" } }),
      row("Tint", "Clearer glass shows more of what is behind it. Milkier glass is easier to read.",
        h("label", { class: "set-range", for: "set-tint" }, h("span", null, "Clear"), range, h("span", null, "Milky"), value)));
  }

  function aboutCard() {
    const status = App.status;
    const a = App.appearance;
    const items = [];
    const add = (name, text) => { items.push(h("dt", null, name), h("dd", null, text)); };
    add("Window", a.native ? "The app's own window" : "A browser window");
    if (status) {
      add("Version", status.version);
      add("Word list", `${status.lexicon.id.split("-")[0]} · ${App.fmt.int(status.lexicon.words)} words`);
      add("Leave values", status.leaves.ok ? "Fitted" : "Not fitted yet");
      add("GPU", status.gpu ? `${status.gpu.name} · ${(status.gpu.memory_total_mb / 1024).toFixed(0)} GB` : "Not found");
    }
    return h("section", { class: "card set-about" },
      h("div", { class: "card-head" }, h("h2", { class: "card-title" }, "About")),
      h("dl", { class: "kv" }, items),
      h("p", { class: "set-note" }, "Training runs are kept in the runs folder and everything else under data/app, next to the code. " +
        "Nothing is sent anywhere."));
  }
})();
