/* Scrabble Lab: application core.
 *
 * One global, `App`, shared by every view file:
 *
 *   App.registerView(name, {title, nav, order, icon, mount})   declare a page
 *   App.get / post / del                                       call the API (docs/APP_API.md)
 *   App.go("#/runs/night1")                                    navigate
 *   App.h(tag, attrs, ...children)                             build DOM
 *   App.fmt.*                                                  number, time and size formatting
 *   App.toast(message, tone) / App.confirm({...})              feedback
 *   App.every(ms, fn)                                          polling that stops when the page changes
 *   App.onCleanup(fn)                                          run when the page changes
 *   App.status                                                 latest GET /api/status
 *
 * Routes are hash based: "#/<view>/<arg>/<arg>?key=value". A view's mount(container, route)
 * receives route = {name, args: [...], query: {...}} and renders into the container.
 *
 * Mock mode: open the page with ?mock=1 and API calls are answered by handlers registered with
 * App.mock.on(method, /regex/, (match, body, query) => response) instead of the server. The
 * files in js/mock/ do that; they are only loaded in mock mode.
 */
window.App = (function () {
  "use strict";

  const views = {};
  const cleanups = [];
  const listeners = {};
  const state = { status: null, route: null };

  // ---- DOM ------------------------------------------------------------------------------

  /** h("div", {class: "card", onclick: fn}, "text", childNode, [more, children]) */
  function h(tag, attrs, ...children) {
    const node = tag === "svg" || (attrs && attrs.xmlns) || SVG_TAGS.has(tag)
      ? document.createElementNS("http://www.w3.org/2000/svg", tag)
      : document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.setAttribute("class", value);
      else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
      else if (key === "html") node.innerHTML = value;
      else if (value === true) node.setAttribute(key, "");
      else node.setAttribute(key, value);
    }
    append(node, children);
    return node;
  }
  const SVG_TAGS = new Set(["g", "path", "line", "rect", "circle", "text", "polyline", "polygon", "defs",
    "clipPath", "linearGradient", "stop", "title", "tspan", "pattern", "ellipse"]);

  function append(node, children) {
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  }

  // ---- Formatting -----------------------------------------------------------------------

  const fmt = {
    /** 1234567 -> "1,234,567" */
    int(value) {
      return value === null || value === undefined ? "–" : Math.round(value).toLocaleString("en-US");
    },
    /** 1234 -> "1,234", 12900 -> "12.9K", 4200000 -> "4.2M" */
    compact(value) {
      if (value === null || value === undefined) return "–";
      const abs = Math.abs(value);
      if (abs < 10000) return Math.round(value).toLocaleString("en-US");
      if (abs < 1e6) return (value / 1e3).toFixed(abs < 1e5 ? 1 : 0) + "K";
      if (abs < 1e9) return (value / 1e6).toFixed(abs < 1e7 ? 2 : 1) + "M";
      return (value / 1e9).toFixed(1) + "B";
    },
    fixed(value, digits = 1) {
      return value === null || value === undefined ? "–" : Number(value).toFixed(digits);
    },
    /** +4.2 / -3.1 / 0.0, with a real minus sign */
    signed(value, digits = 1) {
      if (value === null || value === undefined) return "–";
      const text = Math.abs(value).toFixed(digits);
      return (value > 0 ? "+" : value < 0 ? "−" : "") + text;
    },
    /** 0.531 -> "53.1%" */
    pct(value, digits = 1) {
      return value === null || value === undefined ? "–" : (100 * value).toFixed(digits) + "%";
    },
    /** A spread with its interval: "+4.2 ± 7.8" */
    spread(value, ci) {
      if (value === null || value === undefined) return "–";
      return fmt.signed(value) + (ci === null || ci === undefined ? "" : " ± " + Number(ci).toFixed(1));
    },
    /** Hours -> "8 h 05 m", "12 m", "45 s" */
    duration(hours) {
      if (hours === null || hours === undefined) return "–";
      const seconds = Math.round(hours * 3600);
      if (seconds < 60) return seconds + " s";
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return minutes + " m";
      return Math.floor(minutes / 60) + " h " + String(minutes % 60).padStart(2, "0") + " m";
    },
    /** Unix seconds -> "just now", "5 min ago", "3 h ago", "Nov 12" */
    ago(time) {
      if (!time) return "–";
      const seconds = Date.now() / 1000 - time;
      if (seconds < 45) return "just now";
      if (seconds < 3600) return Math.round(seconds / 60) + " min ago";
      if (seconds < 86400) return Math.round(seconds / 3600) + " h ago";
      return new Date(time * 1000).toLocaleDateString("en-US", { month: "short", day: "numeric" });
    },
    datetime(time) {
      if (!time) return "–";
      return new Date(time * 1000).toLocaleString("en-US",
        { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    },
    bytes(value) {
      if (value === null || value === undefined) return "–";
      if (value < 1e6) return (value / 1e3).toFixed(0) + " KB";
      if (value < 1e9) return (value / 1e6).toFixed(1) + " MB";
      return (value / 1e9).toFixed(2) + " GB";
    },
    /** Sentences written by the server, in the UI's typography: "+4.2 +/- 7.8 against greedy-leave"
     *  -> "+4.2 ± 7.8 against greedy + leave", and a real minus sign before a number. */
    prose(text) {
      if (text === null || text === undefined) return "";
      return String(text).replace(/ ?\+\/- ?/g, " ± ").replace(/greedy-leave/g, "greedy + leave")
        .replace(/(^|[\s(])-(?=\d)/g, "$1−");
    },
  };

  // ---- API ------------------------------------------------------------------------------

  const mockHandlers = [];
  const mock = {
    enabled: new URLSearchParams(location.search).has("mock"),
    /** Register a fake endpoint for mock mode. `handler(match, body, query)` returns the JSON. */
    on(method, pattern, handler) {
      mockHandlers.push({ method, pattern, handler });
    },
    /** True if some handler already answers this call (so a later file can leave it alone). */
    has(method, path) {
      return mockHandlers.some((entry) => entry.method === method && entry.pattern.test(path.split("?")[0]));
    },
  };

  async function api(method, path, body) {
    if (mock.enabled) return mockCall(method, path, body);
    let response;
    try {
      response = await fetch(path, {
        method,
        headers: body === undefined ? {} : { "Content-Type": "application/json" },
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch (failure) {
      const error = new Error("Scrabble Lab's server is not answering. Close this window and open the app again.");
      error.status = 0;
      throw error;
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok) {
      const error = new Error((data && data.error) || `${method} ${path} failed (${response.status})`);
      error.status = response.status;  // 400: the request was refused; 404: no such thing; 500: a bug
      throw error;
    }
    return data;
  }

  async function mockCall(method, path, body) {
    const [pathname, search] = path.split("?");
    const query = Object.fromEntries(new URLSearchParams(search || ""));
    for (const { method: m, pattern, handler } of mockHandlers) {
      const match = m === method && pathname.match(pattern);
      if (match) {
        await new Promise((resolve) => setTimeout(resolve, 60));
        const result = handler(match, body, query);
        if (result && result.error) throw new Error(result.error);
        return JSON.parse(JSON.stringify(result));
      }
    }
    throw new Error(`mock: no handler for ${method} ${pathname}`);
  }

  const get = (path) => api("GET", path);
  const post = (path, body) => api("POST", path, body === undefined ? {} : body);
  const del = (path) => api("DELETE", path);

  // ---- Events ---------------------------------------------------------------------------

  function on(event, handler) {
    (listeners[event] = listeners[event] || []).push(handler);
    return () => { listeners[event] = listeners[event].filter((x) => x !== handler); };
  }

  function emit(event, payload) {
    for (const handler of listeners[event] || []) handler(payload);
  }

  // ---- Page lifecycle -------------------------------------------------------------------

  /** Run `fn` when the current page is left (navigation or re-render). */
  function onCleanup(fn) {
    cleanups.push(fn);
  }

  /** Call `fn` now and every `ms` until the page is left. Overlapping calls are skipped. */
  function every(ms, fn) {
    let busy = false;
    let stopped = false;
    const tick = async () => {
      if (busy || stopped || document.hidden) return;
      busy = true;
      try {
        await fn();
      } catch (error) {
        console.warn(error);
      } finally {
        busy = false;
      }
    };
    tick();
    const timer = setInterval(tick, ms);
    const stop = () => { stopped = true; clearInterval(timer); };
    onCleanup(stop);
    return stop;
  }

  function registerView(name, definition) {
    views[name] = Object.assign({ name, title: name, nav: false, order: 100, icon: "" }, definition);
  }

  function parseRoute() {
    const hash = location.hash.replace(/^#\/?/, "");
    const [pathPart, queryPart] = hash.split("?");
    const parts = pathPart.split("/").filter(Boolean).map(decodeURIComponent);
    const name = parts[0] && views[parts[0]] ? parts[0] : "overview";
    return { name, args: parts.slice(1), query: Object.fromEntries(new URLSearchParams(queryPart || "")) };
  }

  function go(hash) {
    if (location.hash === hash) render();
    else location.hash = hash;
  }

  function render() {
    if (dismissDialog) dismissDialog();  // a dialog belongs to the page it was opened on
    while (cleanups.length) {
      try { cleanups.pop()(); } catch (error) { console.warn(error); }
    }
    const route = parseRoute();
    state.route = route;
    const view = views[route.name];
    const container = clear(document.getElementById("view"));
    container.className = "view view-" + route.name;
    document.title = (view ? view.title : "Scrabble Lab") + " · Scrabble Lab";
    renderSidebar(route.name);
    if (!view) {
      container.appendChild(h("p", { class: "empty" }, "This page does not exist."));
      return;
    }
    try {
      const result = view.mount(container, route);
      if (result && typeof result.catch === "function") result.catch((error) => showError(container, error));
    } catch (error) {
      showError(container, error);
    }
    container.focus({ preventScroll: true });
  }

  function showError(container, error) {
    console.error(error);
    container.appendChild(h("div", { class: "notice notice-critical" },
      h("strong", null, "Something went wrong. "), String(error.message || error)));
  }

  // ---- Chrome ---------------------------------------------------------------------------

  function renderSidebar(active) {
    const nav = clear(document.getElementById("sidebar"));
    nav.appendChild(h("a", { class: "brand", href: "#/overview" },
      h("span", { class: "brand-tile" }, "S"), h("span", { class: "brand-name" }, "Scrabble Lab")));
    const items = Object.values(views).filter((v) => v.nav).sort((a, b) => a.order - b.order);
    // Views marked `foot` (Settings) sit at the bottom of the sidebar.
    let spaced = false;
    for (const view of [...items.filter((v) => !v.foot), ...items.filter((v) => v.foot)]) {
      if (view.foot && !spaced) { nav.appendChild(h("div", { class: "nav-spacer" })); spaced = true; }
      // A view that owns sub-pages (run detail under Training) lists them in `covers`.
      const isActive = view.name === active || (view.covers || []).includes(active);
      nav.appendChild(h("a", { class: "nav-item" + (isActive ? " active" : ""), href: "#/" + view.name,
        "aria-current": isActive ? "page" : null },
        h("span", { class: "nav-icon", html: view.icon || "" }), h("span", null, view.title)));
    }
  }

  // ---- Appearance -----------------------------------------------------------------------
  // The theme (light by default), how milky the glass is, and what is behind it. In the app's
  // own window (scrabble_ai/app/window.py; the address carries ?glass=1) the desktop can show
  // through, frosted by Windows or clear; anywhere else the page draws its own backdrop.

  const appearance = {
    native: false,      // in the app's own window
    seeThrough: null,   // what that window reported: true, false, or null before it has answered
    frost: null,        // whether Windows accepted the frosted effect
    problem: "",
    theme: "light",
    backdrop: "frost",  // the choice: "frost", "clear" or "builtin"
    tint: 50,           // 0 (clearest) to 100 (milkiest)
  };

  function stored(key) {
    try { return localStorage.getItem(key); } catch (error) { return null; }
  }
  function store(key, value) {
    try { localStorage.setItem(key, String(value)); } catch (error) { /* private mode */ }
  }

  /** The backdrop actually in use: the choice, unless this window cannot do it. */
  function backdropInUse() {
    if (!appearance.native || appearance.seeThrough === false) return "builtin";
    if (appearance.backdrop === "frost" && appearance.frost === false) return "builtin";
    return appearance.backdrop;
  }

  function applyAppearance() {
    const root = document.documentElement;
    root.dataset.theme = appearance.theme;
    root.dataset.backdrop = backdropInUse();
    // Sharp things behind the glass compete with what is on it, so clear glass is tinted a step more.
    const tint = backdropInUse() === "clear" ? Math.min(100, appearance.tint + 25) : appearance.tint;
    root.style.setProperty("--tint", String(tint / 100));
  }

  function loadAppearance() {
    appearance.native = new URLSearchParams(location.search).get("glass") === "1";
    const theme = stored("theme");
    if (theme === "light" || theme === "dark") appearance.theme = theme;
    const backdrop = stored("backdrop");
    if (backdrop === "frost" || backdrop === "clear" || backdrop === "builtin") appearance.backdrop = backdrop;
    const tint = Number(stored("tint") ?? NaN);
    if (Number.isFinite(tint)) appearance.tint = Math.max(0, Math.min(100, Math.round(tint)));
    applyAppearance();
  }

  function setTheme(theme) {
    appearance.theme = theme === "dark" ? "dark" : "light";
    store("theme", appearance.theme);
    applyAppearance();
    tellWindow("set_theme", appearance.theme);
    emit("theme", appearance.theme);
    render();  // charts read their colours when they are drawn
  }

  function setBackdrop(mode) {
    appearance.backdrop = mode === "clear" || mode === "builtin" ? mode : "frost";
    store("backdrop", appearance.backdrop);
    syncWindow();
  }

  function setTint(tint) {
    appearance.tint = Math.max(0, Math.min(100, Math.round(tint)));
    store("tint", appearance.tint);
    applyAppearance();
  }

  /** Call a method of the native window, if there is one (see Bridge in window.py). */
  function tellWindow(method, ...args) {
    const api = window.pywebview && window.pywebview.api;
    if (!api || typeof api[method] !== "function") return Promise.resolve(null);
    try { return Promise.resolve(api[method](...args)).catch(() => null); } catch (error) { return Promise.resolve(null); }
  }

  /** Tell the window which backdrop is wanted and take note of what it could do. */
  async function syncWindow() {
    const answer = await tellWindow("set_backdrop", appearance.backdrop, appearance.theme);
    if (answer) {
      appearance.seeThrough = answer.ok !== false;
      appearance.frost = answer.frost_available !== false;
      appearance.problem = answer.problem || "";
    }
    applyAppearance();
    emit("appearance", appearance);
  }

  function connectWindow() {
    tellWindow("set_theme", appearance.theme);
    syncWindow();
  }

  function renderTopbar() {
    const bar = clear(document.getElementById("topbar"));
    const status = state.status;
    if (!status) {
      bar.appendChild(h("span", { class: "topbar-muted" }, "Connecting…"));
      return;
    }
    const chips = [];
    const running = status.active_runs.length;
    chips.push(h("a", { class: "chip" + (running ? " chip-live" : ""), href: "#/training" },
      h("span", { class: "dot" + (running ? " dot-live" : "") }),
      running ? `${running} training run${running > 1 ? "s" : ""} active` : "No training running"));
    if (status.queued_runs.length) chips.push(h("span", { class: "chip" }, `${status.queued_runs.length} queued`));
    if (status.active_matches.length) {
      chips.push(h("a", { class: "chip chip-live", href: "#/arena" }, h("span", { class: "dot dot-live" }),
        `${status.active_matches.length} match${status.active_matches.length > 1 ? "es" : ""} running`));
    }
    const right = [];
    if (status.gpu) {
      const g = status.gpu;
      right.push(h("span", { class: "chip", title: g.name },
        `GPU ${g.utilization}% · ${(g.memory_used_mb / 1024).toFixed(1)} / ${(g.memory_total_mb / 1024).toFixed(0)} GB` +
        (g.temperature_c ? ` · ${g.temperature_c}°C` : "")));
    }
    for (const warning of status.warnings || []) {
      right.push(h("span", { class: "chip chip-warning", title: warning.message },
        warning.kind === "gpu_memory" ? "GPU memory in use by other programs" : "Warning"));
    }
    if (!status.leaves.ok) {
      right.push(h("span", { class: "chip chip-warning", title: "Run `uv run scrabble-ai fit-leaves`" },
        "Leave values not fitted"));
    }
    right.push(h("span", { class: "chip", title: "Word list" }, status.lexicon.id.split("-")[0] +
      " · " + fmt.int(status.lexicon.words) + " words"));
    bar.appendChild(h("div", { class: "topbar-left" }, chips));
    bar.appendChild(h("div", { class: "topbar-right" }, right));
  }

  async function refreshStatus() {
    try {
      state.status = await get("/api/status");
    } catch (error) {
      state.status = null;
    }
    renderTopbar();
    emit("status", state.status);
  }

  // ---- Feedback -------------------------------------------------------------------------

  /** tone: "info" | "good" | "warning" | "critical" */
  function toast(message, tone = "info") {
    const node = h("div", { class: "toast toast-" + tone, role: "status" }, message);
    document.getElementById("toasts").appendChild(node);
    setTimeout(() => node.classList.add("toast-out"), 4200);
    setTimeout(() => node.remove(), 4700);
  }

  // Closes whichever dialog is open (answering "no" to a question); null when none is.
  let dismissDialog = null;

  /** Ask before something that cannot be undone. Resolves true or false. */
  function confirm({ title, body, confirmLabel = "Confirm", danger = false }) {
    if (dismissDialog) dismissDialog();
    return new Promise((resolve) => {
      const close = (answer) => {
        root.replaceChildren();
        document.removeEventListener("keydown", onKey);
        dismissDialog = null;
        resolve(answer);
      };
      const onKey = (event) => { if (event.key === "Escape") close(false); };
      const root = document.getElementById("modal-root");
      dismissDialog = () => close(false);
      const confirmButton = h("button", { class: "button " + (danger ? "button-danger" : "button-primary"),
        type: "button", onclick: () => close(true) }, confirmLabel);
      root.replaceChildren(h("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === e.currentTarget) close(false); } },
        h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title },
          h("h2", null, title),
          h("div", { class: "modal-body" }, body),
          h("div", { class: "modal-actions" },
            h("button", { class: "button", type: "button", onclick: () => close(false) }, "Cancel"),
            confirmButton))));
      document.addEventListener("keydown", onKey);
      confirmButton.focus();
    });
  }

  /** Show arbitrary content in a dialog. Returns a function that closes it. `onClose` is
   *  called once, however the dialog goes away (its button, Escape, a click outside, a change
   *  of page). */
  function modal({ title, content, wide = false, onClose = null }) {
    if (dismissDialog) dismissDialog();
    const root = document.getElementById("modal-root");
    let open = true;
    const close = () => {
      if (!open) return;
      open = false;
      root.replaceChildren();
      document.removeEventListener("keydown", onKey);
      dismissDialog = null;
      if (onClose) onClose();
    };
    const onKey = (event) => { if (event.key === "Escape") close(); };
    dismissDialog = close;
    root.replaceChildren(h("div", { class: "modal-backdrop", onclick: (e) => { if (e.target === e.currentTarget) close(); } },
      h("div", { class: "modal" + (wide ? " modal-wide" : ""), role: "dialog", "aria-modal": "true", "aria-label": title },
        h("div", { class: "modal-head" }, h("h2", null, title),
          h("button", { class: "button button-ghost", type: "button", onclick: close, "aria-label": "Close" }, "✕")),
        h("div", { class: "modal-body" }, content))));
    document.addEventListener("keydown", onKey);
    return close;
  }

  // ---- Start ----------------------------------------------------------------------------

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const node = document.createElement("script");
      node.src = src;
      node.onload = resolve;
      node.onerror = () => reject(new Error("could not load " + src));
      document.head.appendChild(node);
    });
  }

  async function start() {
    loadAppearance();
    // The native window announces itself once its bridge is ready.
    if (window.pywebview && window.pywebview.api) connectWindow();
    else window.addEventListener("pywebviewready", connectWindow, { once: true });
    if (mock.enabled) {
      for (const file of ["js/mock/training.js", "js/mock/library.js", "js/mock/games.js"]) {
        try { await loadScript(file); } catch (error) { console.warn(error.message); }
      }
    }
    window.addEventListener("hashchange", render);
    render();
    refreshStatus();
    setInterval(() => { if (!document.hidden) refreshStatus(); }, 4000);
  }

  return {
    registerView, start, go, render, h, clear, append, fmt, get, post, del, api, mock,
    on, emit, onCleanup, every, toast, confirm, modal, refreshStatus,
    get status() { return state.status; },
    get route() { return state.route; },
    get theme() { return document.documentElement.dataset.theme; },
    /** Theme, tint and backdrop: the choices, and `backdropInUse` for what this window can do. */
    get appearance() { return Object.assign({ backdropInUse: backdropInUse() }, appearance); },
    setTheme, setBackdrop, setTint,
  };
})();
