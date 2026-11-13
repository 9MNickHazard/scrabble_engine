/* Scrabble Lab: the board, racks and tiles.
 *
 *   const rack = Board.rack(container, {tiles: "AEINRST", hidden: false, ledge: true, shuffle: true,
 *                                       onShuffle() {}, size: 44})
 *     rack.set("AEINRS?")          new tiles; tiles kept from before stay where the user put them
 *     rack.setHidden(true, 7)      tile backs with a count
 *     rack.shuffle()
 *
 *   const board = Board.create(container, {
 *     interactive: false,          enter a play by keyboard (click a square, type) or by dragging
 *     rack,                        the rack the play is taken from (interactive mode)
 *     alignY: "center",            "start" | "center" | "end": where the board sits in a tall container
 *     onChange(tiles),             the pending placement changed: [{row, col, letter, blank}]
 *     onSubmit(tiles),             Enter
 *     onMessage(text, tone),       feedback for a refused key ("There is no Q on your rack")
 *     onResize(sizePx, cellPx),
 *   })
 *     board.setPosition(rows, lastPlaced)   rows: 15 strings as in the API; lastPlaced: [[row, col], ...]
 *     board.highlight(squares)              ring squares ([[row, col], ...]); null clears
 *     board.preview(tiles)                  ghost tiles for a candidate ([[row, col, letter, blank]]); null clears
 *     board.setInteractive(on)
 *     board.setPlacement(tiles) / clearPlacement() / placement()
 *     board.setCursor(row, col, down) / focus() / destroy()
 *
 * The board fits the largest square into its container (ResizeObserver) with whole-pixel cells, so
 * lines and letters stay crisp. Keyboard entry listens on the document while the board is
 * interactive and ignores keys meant for inputs, dialogs and modifier shortcuts.
 */
window.Board = (function () {
  "use strict";

  const N = 15;
  const COLS = "ABCDEFGHIJKLMNO";
  const VALUES = { A: 1, B: 3, C: 3, D: 2, E: 1, F: 4, G: 2, H: 4, I: 1, J: 8, K: 5, L: 1, M: 3, N: 1, O: 1,
    P: 3, Q: 10, R: 1, S: 1, T: 1, U: 1, V: 4, W: 4, X: 8, Y: 4, Z: 10, "?": 0 };
  const COUNTS = { A: 9, B: 2, C: 2, D: 4, E: 12, F: 2, G: 3, H: 2, I: 9, J: 1, K: 1, L: 4, M: 2, N: 6, O: 8,
    P: 2, Q: 1, R: 6, S: 4, T: 6, U: 4, V: 2, W: 2, X: 1, Y: 2, Z: 1, "?": 2 };
  const VOWELS = "AEIOU";

  // docs/RULES.md: = triple word, - double word, " triple letter, ' double letter.
  const HALF = ["=..'...=...'..=", ".-...\"...\"...-.", "..-...'.'...-..", "'..-...'...-..'",
    "....-.....-....", ".\"...\"...\"...\".", "..'...'.'...'..", "=..'...-...'..="];
  const LAYOUT = HALF.concat(HALF.slice(0, 7).reverse());
  const PREMIUM = { "=": "tw", "-": "dw", '"': "tl", "'": "dl" };
  const LABELS = { tw: "TW", dw: "DW", tl: "TL", dl: "DL" };

  const SVG_STAR = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M12 2.6l2.75 5.9 6.45.75-4.78 4.43 1.27 6.37L12 16.86l-5.69 3.19 1.27-6.37L2.8 9.25l6.45-.75z"/></svg>';
  const SVG_RIGHT = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M7 4.5l11 7.5-11 7.5z"/></svg>';
  const SVG_DOWN = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M4.5 7l7.5 11 7.5-11z"/></svg>';
  const SVG_SHUFFLE = '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2.5 5.5h3c3.5 0 5.5 9 9 9h3M2.5 14.5h3c1.4 0 2.5-1.4 3.4-3.2M11.1 8.7c.9-1.8 2-3.2 3.4-3.2h3M15.5 3.5l2 2-2 2M15.5 12.5l2 2-2 2"/></svg>';

  const premium = (row, col) => PREMIUM[LAYOUT[row][col]] || null;
  const squareName = (row, col) => COLS[col] + (row + 1);
  const emptyRows = () => Array.from({ length: N }, () => ".".repeat(N));
  const isLetter = (ch) => /^[A-Za-z]$/.test(ch);

  /** [row, col, letter, blank] from either the API's array form or the {row, col, letter, blank} form. */
  function normTile(t) {
    return Array.isArray(t) ? [t[0], t[1], String(t[2]).toUpperCase(), !!t[3]]
      : [t.row, t.col, String(t.letter).toUpperCase(), !!t.blank];
  }

  /** The board rows after placing tiles. */
  function applyTiles(rows, tiles) {
    const grid = (rows || emptyRows()).map((r) => r.split(""));
    for (const t of tiles || []) {
      const [r, c, letter, blank] = normTile(t);
      grid[r][c] = blank ? letter.toLowerCase() : letter;
    }
    return grid.map((r) => r.join(""));
  }

  /** Tiles nobody can see from one side: the full set less the board and that side's rack. */
  function unseen(rows, rack) {
    const left = Object.assign({}, COUNTS);
    for (const row of rows || []) {
      for (const ch of row) {
        if (ch === ".") continue;
        const key = ch === ch.toUpperCase() ? ch : "?";
        left[key] = Math.max(0, left[key] - 1);
      }
    }
    for (const ch of rack || "") if (left[ch] > 0) left[ch]--;
    return left;
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  /** A tile. `letter` "?" is an unplayed blank (an empty face). */
  function tile(letter, opts) {
    const o = opts || {};
    const L = String(letter || "").toUpperCase();
    const blank = L === "?" || !!o.blank;
    const node = el("div", "tile" + (blank ? " tile-blank" : "") + (o.cls ? " " + o.cls : ""));
    node.appendChild(el("span", "tile-letter", L === "?" ? "" : L));
    if (!blank && VALUES[L] !== undefined) node.appendChild(el("span", "tile-value", String(VALUES[L])));
    if (o.size) node.style.setProperty("--tile-size", o.size + "px");
    return node;
  }

  function tileBack(size) {
    const node = el("div", "tile tile-back");
    if (size) node.style.setProperty("--tile-size", size + "px");
    return node;
  }

  // ---- Rack ------------------------------------------------------------------------------

  let slotSeq = 0;

  function rack(container, options) {
    const opts = options || {};
    const root = el("div", "rack" + (opts.ledge ? " rack-ledge" : "") + (opts.className ? " " + opts.className : ""));
    if (opts.size) root.style.setProperty("--tile-size", opts.size + "px");
    const tilesEl = el("div", "rack-tiles");
    const countEl = el("span", "rack-count");
    let shuffleButton = null;
    if (opts.shuffle || opts.onShuffle) {
      shuffleButton = el("button", "rack-button");
      shuffleButton.type = "button";
      shuffleButton.title = "Shuffle the rack";
      shuffleButton.setAttribute("aria-label", "Shuffle the rack");
      shuffleButton.innerHTML = SVG_SHUFFLE;
      shuffleButton.addEventListener("mousedown", (e) => e.preventDefault());
      shuffleButton.addEventListener("click", () => {
        api.shuffle();
        if (opts.onShuffle) opts.onShuffle();
      });
      root.appendChild(shuffleButton);
    }
    root.appendChild(tilesEl);
    root.appendChild(countEl);
    container.appendChild(root);

    let slots = [];
    let hidden = !!opts.hidden;
    let hiddenCount = null;

    function render() {
      tilesEl.replaceChildren();
      root.classList.toggle("rack-is-hidden", hidden);
      if (shuffleButton) shuffleButton.disabled = hidden || slots.filter((s) => !s.used).length < 2;
      if (hidden) {
        const n = hiddenCount === null ? slots.length : hiddenCount;
        for (let i = 0; i < n; i++) tilesEl.appendChild(tileBack());
        countEl.textContent = n === 1 ? "1 tile" : n + " tiles";
        return;
      }
      countEl.textContent = "";
      for (const s of slots) {
        if (s.used) continue;
        const t = tile(s.letter, { cls: s.fresh ? "tile-fresh" : "" });
        t.dataset.slot = s.id;
        tilesEl.appendChild(t);
        s.fresh = false;
      }
      if (!slots.length && opts.emptyText) tilesEl.appendChild(el("span", "rack-empty", opts.emptyText));
    }

    const api = {
      el: root,
      tilesEl,
      /** New rack contents. Tiles that were already there keep their places; new ones go on the right. */
      set(tiles, setOpts) {
        const letters = String(tiles || "").toUpperCase().split("").filter((ch) => isLetter(ch) || ch === "?");
        const remaining = letters.slice();
        const kept = [];
        for (const s of slots) {
          const i = remaining.indexOf(s.letter);
          if (i >= 0) {
            remaining.splice(i, 1);
            kept.push({ id: s.id, letter: s.letter, used: false, fresh: false });
          }
        }
        const animate = !(setOpts && setOpts.animate === false) && slots.length > 0;
        for (const letter of remaining) kept.push({ id: ++slotSeq, letter, used: false, fresh: animate });
        slots = kept;
        if (setOpts && typeof setOpts.hidden === "boolean") hidden = setOpts.hidden;
        render();
      },
      setHidden(value, count) {
        hidden = !!value;
        hiddenCount = count === undefined ? null : count;
        render();
      },
      get hidden() { return hidden; },
      shuffle() {
        for (let i = slots.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [slots[i], slots[j]] = [slots[j], slots[i]];
        }
        render();
      },
      /** Take a tile for `letter`: the letter itself, else a blank (always a blank if forceBlank). */
      take(letter, forceBlank) {
        const L = letter.toUpperCase();
        if (!forceBlank) {
          const s = slots.find((x) => !x.used && x.letter === L);
          if (s) { s.used = true; render(); return { slot: s.id, blank: false }; }
        }
        const b = slots.find((x) => !x.used && x.letter === "?");
        if (b) { b.used = true; render(); return { slot: b.id, blank: true }; }
        return null;
      },
      takeSlot(id) {
        const s = slots.find((x) => x.id === id && !x.used);
        if (!s) return null;
        s.used = true;
        render();
        return s;
      },
      release(id) {
        const s = slots.find((x) => x.id === id);
        if (s) { s.used = false; render(); }
      },
      releaseAll() {
        for (const s of slots) s.used = false;
        render();
      },
      slot(id) { return slots.find((x) => x.id === id) || null; },
      /** Move a slot to sit before the index-th visible tile. */
      move(id, index) {
        const from = slots.findIndex((x) => x.id === id);
        if (from < 0) return;
        const [s] = slots.splice(from, 1);
        const visible = slots.filter((x) => !x.used);
        const before = visible[index];
        const at = before ? slots.indexOf(before) : slots.length;
        slots.splice(at, 0, s);
        render();
      },
      /** The visible index a pointer at clientX would drop at. */
      indexAt(clientX) {
        const nodes = Array.from(tilesEl.querySelectorAll(".tile:not(.tile-drag-source)"));
        let i = 0;
        for (const node of nodes) {
          const box = node.getBoundingClientRect();
          if (clientX > box.left + box.width / 2) i++;
        }
        return i;
      },
      letters() { return slots.map((s) => s.letter).join(""); },
      render,
    };
    api.set(opts.tiles || "", { animate: false });
    if (opts.hidden) api.setHidden(true, opts.count);
    return api;
  }

  // ---- Board -----------------------------------------------------------------------------

  function create(container, options) {
    const opts = Object.assign({ interactive: false, alignY: "center" }, options || {});
    const host = container;
    host.classList.add("board-host");

    const root = el("div", "board");
    root.setAttribute("role", "grid");
    root.setAttribute("aria-label", "Board");
    const colsEl = el("div", "board-cols");
    const rowsEl = el("div", "board-rows");
    const grid = el("div", "board-grid");
    for (let c = 0; c < N; c++) colsEl.appendChild(el("span", null, COLS[c]));
    for (let r = 0; r < N; r++) rowsEl.appendChild(el("span", null, String(r + 1)));

    const cells = [];
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const p = premium(r, c);
        const star = r === 7 && c === 7;
        const cell = el("div", "sq" + (p ? " sq-" + p : "") + (star ? " sq-star" : ""));
        cell.dataset.r = r;
        cell.dataset.c = c;
        cell.setAttribute("role", "gridcell");
        cell.setAttribute("aria-label", squareName(r, c));
        if (star) {
          const mark = el("span", "sq-mark");
          mark.innerHTML = SVG_STAR;
          cell.appendChild(mark);
        } else if (p) {
          cell.appendChild(el("span", "sq-label", LABELS[p]));
        }
        cells.push(cell);
        grid.appendChild(cell);
      }
    }
    const layer = el("div", "board-layer");
    root.append(colsEl, rowsEl, grid, layer);
    host.appendChild(root);

    const cursorEl = el("span", "sq-cursor");

    let position = emptyRows();
    let lastSet = new Set();
    let highlightSet = new Set();
    let previewMap = new Map();
    let pending = []; // {r, c, letter, blank, slot}
    let cursor = null; // {r, c, down, off}
    let interactive = !!opts.interactive;
    let picker = null;
    let cell = 0;
    let size = 0;
    let destroyed = false;
    let animateNext = true;
    const cellKeys = new Array(N * N).fill("");

    const idx = (r, c) => r * N + c;
    const inside = (r, c) => r >= 0 && r < N && c >= 0 && c < N;
    const onBoard = (r, c) => position[r][c] !== ".";
    const pendingAt = (r, c) => pending.find((p) => p.r === r && p.c === c) || null;
    const filled = (r, c) => onBoard(r, c) || !!pendingAt(r, c);
    const boardEmpty = () => position.every((row) => row === ".".repeat(N));

    // -- layout

    function layout() {
      if (destroyed) return;
      const w = host.clientWidth;
      const h = host.clientHeight;
      let S = h > 0 ? Math.min(w, h) : w;
      if (opts.maxSize) S = Math.min(S, opts.maxSize);
      if (S < 120) S = 120;
      const gap = S >= 520 ? 2 : 1;
      const c = Math.max(8, Math.floor((S - 16 * gap) / (N + 1.15)));
      const pad = Math.max(8, Math.floor((S - N * c - 16 * gap) / 2));
      const total = N * c + 16 * gap + 2 * pad;
      if (c === cell && total === size && root.dataset.w === String(w) && root.dataset.h === String(h)) return;
      cell = c;
      size = total;
      root.dataset.w = w;
      root.dataset.h = h;
      root.style.setProperty("--cell", c + "px");
      root.style.setProperty("--gap", gap + "px");
      root.style.setProperty("--pad", pad + "px");
      root.style.width = total + "px";
      root.style.height = total + "px";
      root.style.left = Math.max(0, Math.floor((w - total) / 2)) + "px";
      const free = Math.max(0, h - total);
      root.style.top = (h > 0 ? (opts.alignY === "start" ? 0 : opts.alignY === "end" ? free : Math.floor(free / 2)) : 0) + "px";
      if (opts.rack) opts.rack.el.style.setProperty("--tile-size", Math.round(c * 1.12) + "px");
      if (opts.onResize) opts.onResize(total, c);
    }
    // Lay out on the next frame, not inside the observer's callback: resizing the board there can
    // change the container again, which the browser reports as a "ResizeObserver loop" error.
    let layoutFrame = 0;
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(layoutFrame);
      layoutFrame = requestAnimationFrame(layout);
    });
    observer.observe(host);

    // -- rendering

    function render() {
      const pendMap = new Map(pending.map((p) => [idx(p.r, p.c), p]));
      for (let i = 0; i < N * N; i++) {
        const r = Math.floor(i / N);
        const c = i % N;
        const ch = position[r][c];
        let t = null;
        if (ch !== ".") t = { letter: ch.toUpperCase(), blank: ch !== ch.toUpperCase(), kind: "board" };
        else if (pendMap.has(i)) t = { letter: pendMap.get(i).letter, blank: pendMap.get(i).blank, kind: "pending" };
        else if (previewMap.has(i)) t = { letter: previewMap.get(i).letter, blank: previewMap.get(i).blank, kind: "ghost" };
        let key = "";
        let cls = "";
        if (t) {
          if (t.kind === "pending") cls = "tile-pending";
          else if (t.kind === "ghost") cls = "tile-ghost";
          else {
            if (lastSet.has(i)) cls += " tile-new";
            if (highlightSet.has(i)) cls += " tile-mark";
          }
          key = t.kind + ":" + t.letter + (t.blank ? "?" : "") + cls;
        }
        const node = cells[i];
        if (cellKeys[i] !== key) {
          const old = node.querySelector(".tile");
          if (old) old.remove();
          node.classList.toggle("sq-filled", !!t);
          if (t) {
            const wasEmpty = !cellKeys[i] || cellKeys[i].startsWith("ghost");
            const fresh = t.kind === "board" && lastSet.has(i) && wasEmpty && animateNext && opts.animate !== false;
            const node2 = tile(t.letter, { blank: t.blank, cls: cls.trim() + (fresh ? " tile-enter" : "") });
            if (t.kind === "pending") node2.dataset.pending = "1";
            node.appendChild(node2);
          }
          cellKeys[i] = key;
        }
      }
      // cursor
      const showCursor = interactive && cursor && !cursor.off && inside(cursor.r, cursor.c);
      for (const node of grid.querySelectorAll(".sq.is-cursor")) node.classList.remove("is-cursor");
      if (showCursor) {
        const node = cells[idx(cursor.r, cursor.c)];
        node.classList.add("is-cursor");
        cursorEl.innerHTML = cursor.down ? SVG_DOWN : SVG_RIGHT;
        cursorEl.classList.toggle("sq-cursor-down", !!cursor.down);
        node.appendChild(cursorEl);
      } else if (cursorEl.parentNode) {
        cursorEl.remove();
      }
      root.classList.toggle("board-interactive", interactive);
    }

    // -- placement

    function changed() {
      render();
      if (opts.onChange) opts.onChange(api.placement());
    }

    function message(text, tone) {
      if (opts.onMessage) opts.onMessage(text, tone || "warning");
    }

    function shake() {
      const target = opts.rack ? opts.rack.el : root;
      target.classList.remove("shake");
      void target.offsetWidth;
      target.classList.add("shake");
      setTimeout(() => target.classList.remove("shake"), 400);
    }

    function step(pos, dir) {
      return pos.down ? { r: pos.r + dir, c: pos.c, down: true } : { r: pos.r, c: pos.c + dir, down: false };
    }

    /** From pos, the first square along the direction that is not filled (pos itself included). */
    function nextFree(pos) {
      let p = { r: pos.r, c: pos.c, down: pos.down };
      while (inside(p.r, p.c) && filled(p.r, p.c)) p = step(p, 1);
      return p;
    }

    function ensureCursor() {
      if (cursor) return true;
      if (boardEmpty() && !pending.length) {
        cursor = { r: 7, c: 7, down: false };
        return true;
      }
      message("Click a square to start a word.", "info");
      shake();
      return false;
    }

    function typeLetter(letter, forceBlank) {
      if (!ensureCursor()) return;
      const at = nextFree(cursor);
      if (!inside(at.r, at.c)) {
        message("There is no room for another tile in this direction.");
        shake();
        return;
      }
      if (!opts.rack) return;
      const got = opts.rack.take(letter, forceBlank);
      if (!got) {
        message(forceBlank ? "There is no blank on your rack." : "There is no " + letter + " on your rack.");
        shake();
        return;
      }
      pending.push({ r: at.r, c: at.c, letter, blank: got.blank, slot: got.slot });
      const after = nextFree(step(at, 1));
      cursor = inside(after.r, after.c) ? after : { r: after.r, c: after.c, down: at.down, off: true };
      changed();
    }

    function backspace() {
      if (!pending.length) return;
      const p = pending.pop();
      if (opts.rack) opts.rack.release(p.slot);
      cursor = { r: p.r, c: p.c, down: cursor ? cursor.down : false };
      changed();
    }

    function clearPlacement(keepCursor) {
      const first = pending.length ? pending[0] : null;
      for (const p of pending) if (opts.rack) opts.rack.release(p.slot);
      pending = [];
      if (keepCursor && first) cursor = { r: first.r, c: first.c, down: cursor ? cursor.down : false };
      else if (!keepCursor) cursor = null;
      changed();
    }

    function moveCursor(dr, dc) {
      if (!cursor) cursor = { r: 7, c: 7, down: dr !== 0 };
      else {
        const r = Math.min(N - 1, Math.max(0, cursor.r + dr));
        const c = Math.min(N - 1, Math.max(0, cursor.c + dc));
        cursor = { r, c, down: cursor.down };
      }
      render();
    }

    function removePending(p) {
      pending = pending.filter((x) => x !== p);
      if (opts.rack) opts.rack.release(p.slot);
    }

    async function placeSlot(slotId, r, c) {
      if (!opts.rack || filled(r, c)) return false;
      const s = opts.rack.slot(slotId);
      if (!s || s.used) return false;
      let letter = s.letter;
      let blank = false;
      if (letter === "?") {
        letter = await pickLetter(r, c);
        if (!letter) return false;
        blank = true;
        if (filled(r, c)) return false;
      }
      if (!opts.rack.takeSlot(slotId)) return false;
      pending.push({ r, c, letter, blank, slot: slotId });
      return true;
    }

    // -- blank letter picker

    function pickLetter(r, c) {
      closePicker(null);
      return new Promise((resolve) => {
        const pop = el("div", "blank-picker");
        pop.appendChild(el("div", "blank-picker-title", "Play the blank as"));
        const keys = el("div", "blank-picker-keys");
        for (const L of "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
          const b = el("button", null, L);
          b.type = "button";
          b.addEventListener("click", (e) => { e.stopPropagation(); closePicker(L); });
          keys.appendChild(b);
        }
        pop.appendChild(keys);
        layer.appendChild(pop);
        const pitch = cell + parseInt(getComputedStyle(root).getPropertyValue("--gap"), 10);
        const pad = parseInt(getComputedStyle(root).getPropertyValue("--pad"), 10);
        const wPop = pop.offsetWidth;
        const hPop = pop.offsetHeight;
        let left = pad + c * pitch + cell / 2 - wPop / 2;
        left = Math.max(4, Math.min(size - wPop - 4, left));
        let top = pad + (r + 1) * pitch + 4;
        if (top + hPop > size - 4) top = pad + r * pitch - hPop - 4;
        pop.style.left = Math.round(left) + "px";
        pop.style.top = Math.round(Math.max(4, top)) + "px";
        picker = { el: pop, resolve };
        setTimeout(() => document.addEventListener("pointerdown", outsidePicker, true), 0);
      });
    }

    function outsidePicker(e) {
      if (picker && !picker.el.contains(e.target)) closePicker(null);
    }

    function closePicker(letter) {
      if (!picker) return;
      const p = picker;
      picker = null;
      document.removeEventListener("pointerdown", outsidePicker, true);
      p.el.remove();
      p.resolve(letter);
    }

    // -- keyboard

    function onKey(e) {
      if (destroyed || !interactive || e.defaultPrevented) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target;
      if (target && (target.closest && target.closest("input, textarea, select, [contenteditable]"))) return;
      const modal = document.getElementById("modal-root");
      if (modal && modal.childElementCount) return;
      if (!root.isConnected) return;
      const key = e.key;
      if (picker) {
        if (isLetter(key)) { e.preventDefault(); closePicker(key.toUpperCase()); }
        else if (key === "Escape") { e.preventDefault(); closePicker(null); }
        return;
      }
      const onButton = target && target.closest && target.closest("button, a, [role=button], summary");
      if (isLetter(key)) {
        e.preventDefault();
        typeLetter(key.toUpperCase(), e.shiftKey);
      } else if (key === "Backspace" || key === "Delete") {
        e.preventDefault();
        backspace();
      } else if (key === "Escape") {
        if (pending.length) clearPlacement(true);
        else if (cursor) { cursor = null; render(); }
        else return;
        e.preventDefault();
      } else if (key === "Enter") {
        if (onButton) return;
        e.preventDefault();
        if (opts.onSubmit) opts.onSubmit(api.placement());
      } else if (key === " " || key === "Spacebar") {
        if (onButton || !cursor) return;
        e.preventDefault();
        cursor = { r: cursor.r, c: cursor.c, down: !cursor.down, off: cursor.off };
        render();
      } else if (key === "ArrowLeft") { e.preventDefault(); moveCursor(0, -1); }
      else if (key === "ArrowRight") { e.preventDefault(); moveCursor(0, 1); }
      else if (key === "ArrowUp") { e.preventDefault(); moveCursor(-1, 0); }
      else if (key === "ArrowDown") { e.preventDefault(); moveCursor(1, 0); }
    }
    document.addEventListener("keydown", onKey);

    // -- clicks

    let swallowClick = false;

    grid.addEventListener("click", (e) => {
      if (swallowClick) { swallowClick = false; return; }
      const node = e.target.closest(".sq");
      if (!node) return;
      const r = +node.dataset.r;
      const c = +node.dataset.c;
      if (opts.onSquareClick) opts.onSquareClick(r, c);
      if (!interactive) return;
      const p = pendingAt(r, c);
      if (p) {
        removePending(p);
        cursor = { r, c, down: cursor ? cursor.down : false };
        changed();
        return;
      }
      if (onBoard(r, c)) return;
      if (cursor && !cursor.off && cursor.r === r && cursor.c === c) cursor = { r, c, down: !cursor.down };
      else cursor = { r, c, down: cursor ? cursor.down : false };
      render();
    });

    function onRackClick(e) {
      if (swallowClick) { swallowClick = false; return; }
      if (!interactive) return;
      const t = e.target.closest(".tile");
      if (!t || !t.dataset.slot || !cursor) return;
      const at = nextFree(cursor);
      if (!inside(at.r, at.c)) { message("There is no room for another tile in this direction."); shake(); return; }
      const dir = cursor.down;
      placeSlot(+t.dataset.slot, at.r, at.c).then((ok) => {
        if (!ok) return;
        const after = nextFree(step({ r: at.r, c: at.c, down: dir }, 1));
        cursor = inside(after.r, after.c) ? after : { r: after.r, c: after.c, down: dir, off: true };
        changed();
      });
    }

    // -- dragging tiles between the rack and the board

    let drag = null;

    function onPointerDown(e) {
      if (e.button !== 0 || picker) return;
      const t = e.target.closest && e.target.closest(".tile");
      if (!t) return;
      let source = null;
      if (opts.rack && opts.rack.el.contains(t) && t.dataset.slot) {
        source = { kind: "rack", slot: +t.dataset.slot, el: t };
      } else if (interactive && grid.contains(t) && t.dataset.pending) {
        const node = t.closest(".sq");
        const p = pendingAt(+node.dataset.r, +node.dataset.c);
        if (p) source = { kind: "board", pending: p, el: t };
      }
      if (!source) return;
      e.preventDefault();
      drag = { source, x0: e.clientX, y0: e.clientY, active: false, ghost: null, over: null };
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      window.addEventListener("pointercancel", onPointerCancel);
    }

    function dragLetter(source) {
      if (source.kind === "board") return { letter: source.pending.blank ? "?" : source.pending.letter };
      const s = opts.rack.slot(source.slot);
      return { letter: s ? s.letter : "?" };
    }

    function onPointerMove(e) {
      if (!drag) return;
      if (!drag.active) {
        if (Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) < 5) return;
        drag.active = true;
        const box = drag.source.el.getBoundingClientRect();
        const ghost = tile(dragLetter(drag.source).letter, { cls: "tile-dragging" });
        ghost.style.setProperty("--tile-size", Math.round(Math.max(box.width, cell * 1.1)) + "px");
        document.body.appendChild(ghost);
        drag.ghost = ghost;
        drag.source.el.classList.add("tile-drag-source");
        document.body.classList.add("is-dragging-tile");
      }
      const g = drag.ghost;
      g.style.left = e.clientX - g.offsetWidth / 2 + "px";
      g.style.top = e.clientY - g.offsetHeight / 2 + "px";
      const target = dropTarget(e.clientX, e.clientY, drag);
      if (drag.over && drag.over !== (target && target.node)) drag.over.classList.remove("is-drop-target");
      drag.over = target ? target.node : null;
      if (drag.over) drag.over.classList.add("is-drop-target");
    }

    function dropTarget(x, y, d) {
      const under = document.elementFromPoint(x, y);
      if (!under) return null;
      const node = under.closest(".sq");
      if (node && grid.contains(node) && interactive) {
        const r = +node.dataset.r;
        const c = +node.dataset.c;
        const p = pendingAt(r, c);
        const own = d.source.kind === "board" && p === d.source.pending;
        if (!onBoard(r, c) && (!p || own)) return { kind: "cell", r, c, node };
        return null;
      }
      if (opts.rack && opts.rack.el.contains(under)) return { kind: "rack", node: opts.rack.el };
      return null;
    }

    async function onPointerUp(e) {
      const d = drag;
      stopDrag();
      if (!d || !d.active) return;
      swallowClick = true;
      setTimeout(() => { swallowClick = false; }, 0);
      const target = dropTarget(e.clientX, e.clientY, d);
      // Where in the rack it lands, measured while the dragged tile is still left out of the count.
      const index = target && target.kind === "rack" ? opts.rack.indexAt(e.clientX) : 0;
      if (d.over) d.over.classList.remove("is-drop-target");
      if (d.ghost) d.ghost.remove();
      d.source.el.classList.remove("tile-drag-source");
      if (target && target.kind === "cell") {
        if (d.source.kind === "rack") {
          if (await placeSlot(d.source.slot, target.r, target.c)) {
            cursor = null;
            changed();
          }
        } else {
          const p = d.source.pending;
          if (p.r !== target.r || p.c !== target.c) {
            p.r = target.r;
            p.c = target.c;
            pending = pending.filter((x) => x !== p).concat([p]);
            cursor = null;
            changed();
          }
        }
      } else if (target && target.kind === "rack") {
        if (d.source.kind === "board") {
          removePending(d.source.pending);
          opts.rack.move(d.source.pending.slot, index);
          changed();
        } else {
          opts.rack.move(d.source.slot, index);
        }
      } else if (d.source.kind === "board") {
        removePending(d.source.pending);
        changed();
      }
    }

    function onPointerCancel() {
      const d = drag;
      stopDrag();
      if (!d) return;
      if (d.ghost) d.ghost.remove();
      if (d.over) d.over.classList.remove("is-drop-target");
      d.source.el.classList.remove("tile-drag-source");
    }

    function stopDrag() {
      drag = null;
      document.body.classList.remove("is-dragging-tile");
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerCancel);
    }

    root.addEventListener("pointerdown", onPointerDown);
    if (opts.rack) {
      opts.rack.el.addEventListener("pointerdown", onPointerDown);
      opts.rack.el.addEventListener("click", onRackClick);
    }

    // -- public

    const api = {
      el: root,
      /** An element over the board for overlays (result panel, banners); it does not block clicks itself. */
      layer,
      get cellSize() { return cell; },
      get size() { return size; },
      get position() { return position.slice(); },
      /** opts.animate false: no drop-in animation for the last move (used when restoring a view). */
      setPosition(rows, lastPlaced, setOpts) {
        position = rows && rows.length === N ? rows.slice() : emptyRows();
        lastSet = new Set((lastPlaced || []).map(([r, c]) => idx(r, c)));
        animateNext = !(setOpts && setOpts.animate === false);
        // A pending tile on a square that is now taken cannot stay.
        const clash = pending.filter((p) => onBoard(p.r, p.c));
        if (clash.length) {
          for (const p of clash) removePending(p);
          changed();
        } else {
          render();
        }
      },
      highlight(squares) {
        highlightSet = new Set((squares || []).map((s) => idx(s[0], s[1])));
        render();
      },
      preview(tiles) {
        previewMap = new Map();
        for (const t of tiles || []) {
          const [r, c, letter, blank] = normTile(t);
          previewMap.set(idx(r, c), { letter, blank });
        }
        render();
      },
      setInteractive(on) {
        interactive = !!on;
        if (!interactive) closePicker(null);
        render();
      },
      get interactive() { return interactive; },
      placement() {
        return pending.slice().sort((a, b) => a.r - b.r || a.c - b.c)
          .map((p) => ({ row: p.r, col: p.c, letter: p.letter, blank: p.blank }));
      },
      /** Put a candidate's tiles down as the pending placement (taking them from the rack). */
      setPlacement(tiles) {
        for (const p of pending) if (opts.rack) opts.rack.release(p.slot);
        pending = [];
        const list = (tiles || []).map(normTile);
        for (const [r, c, letter, blank] of list) {
          if (!inside(r, c) || filled(r, c) || !opts.rack) continue;
          const got = opts.rack.take(letter, blank);
          if (!got) continue;
          pending.push({ r, c, letter, blank: got.blank, slot: got.slot });
        }
        if (pending.length) {
          const sorted = pending.slice().sort((a, b) => a.r - b.r || a.c - b.c);
          const down = sorted.length > 1 && sorted[0].c === sorted[sorted.length - 1].c;
          const last = sorted[sorted.length - 1];
          const after = nextFree(step({ r: last.r, c: last.c, down }, 1));
          cursor = inside(after.r, after.c) ? after : { r: after.r, c: after.c, down, off: true };
        }
        changed();
      },
      clearPlacement() { clearPlacement(false); },
      setCursor(r, c, down) {
        cursor = r === null || r === undefined ? null : { r, c, down: !!down };
        render();
      },
      shake,
      focus() { root.focus && root.focus({ preventScroll: true }); },
      layout,
      destroy() {
        destroyed = true;
        closePicker(null);
        stopDrag();
        observer.disconnect();
        cancelAnimationFrame(layoutFrame);
        document.removeEventListener("keydown", onKey);
      },
    };
    layout();
    render();
    return api;
  }

  // ---- Shared pieces of the game pages (play.js and watch.js) ----------------------------

  const ICONS = {
    play: '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M6 4.2v11.6c0 .6.7 1 1.2.6l8.6-5.8a.75.75 0 000-1.2L7.2 3.6c-.5-.4-1.2 0-1.2.6z"/></svg>',
    pause: '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><rect fill="currentColor" x="5" y="4" width="3.4" height="12" rx="1"/><rect fill="currentColor" x="11.6" y="4" width="3.4" height="12" rx="1"/></svg>',
    step: '<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M4.5 4.6v10.8c0 .6.6.9 1.1.6l7.4-5.4a.7.7 0 000-1.2L5.6 4c-.5-.3-1.1 0-1.1.6z"/><rect fill="currentColor" x="13.6" y="4" width="2.6" height="12" rx="1"/></svg>',
    restart: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.8 10a6.2 6.2 0 101.9-4.5"/><path d="M3.5 3.5v3.6h3.6"/></svg>',
    check: '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 8.5l3.2 3L13 4.5"/></svg>',
    hint: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M7.5 14.5h5M8.3 17h3.4M10 2.8a5 5 0 00-3 9c.6.5 1 1.1 1 1.8v.4h4v-.4c0-.7.4-1.3 1-1.8a5 5 0 00-3-9z"/></svg>',
    clear: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" aria-hidden="true"><path d="M5 5l10 10M15 5L5 15"/></svg>',
    back: '<svg viewBox="0 0 20 20" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 5l-5 5 5 5M3.5 10H17"/></svg>',
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /** "8D QU(I)z" -> position, then the word with played-through letters muted and blanks marked. */
  function moveText(text) {
    const h = App.h;
    const t = String(text || "");
    if (!t || t === "pass" || t.startsWith("exchange")) {
      return h("span", { class: "mv" }, h("span", { class: "mv-other" }, t ? t[0].toUpperCase() + t.slice(1) : "–"));
    }
    const space = t.indexOf(" ");
    const pos = space > 0 ? t.slice(0, space) : "";
    const word = space > 0 ? t.slice(space + 1) : t;
    const parts = [];
    let through = false;
    let buffer = "";
    let bufferClass = "";
    const flush = () => {
      if (buffer) parts.push(h("span", { class: bufferClass || null }, buffer));
      buffer = "";
    };
    for (const ch of word) {
      if (ch === "(" || ch === ")") { flush(); through = ch === "("; continue; }
      const cls = through ? "mv-through" : ch !== ch.toUpperCase() ? "mv-blank" : "";
      if (cls !== bufferClass) { flush(); bufferClass = cls; }
      buffer += ch;
    }
    flush();
    return h("span", { class: "mv" }, pos ? h("span", { class: "mv-pos" }, pos) : null,
      h("span", { class: "mv-word" }, parts));
  }

  /** A ranked list of candidates: move, score, leave (with its value), value, total, win. */
  function candidateTable(cands, options) {
    const h = App.h;
    const fmt = App.fmt;
    const o = options || {};
    const list = cands.slice(0, o.limit || cands.length);
    const showWin = list.some((c) => c.win !== null && c.win !== undefined);
    // Bars show how close the alternatives were: scaled from a little below the worst listed total.
    const totals = list.map((c) => c.total);
    const hi = Math.max(...totals);
    const range = Math.max(hi - Math.min(...totals), 8) * 1.25;
    const rows = list.map((c, i) => {
      const width = Math.max(4, Math.min(100, Math.round(100 * (c.total - (hi - range)) / range)));
      const tr = h("tr", { class: c.chosen ? "is-chosen" : null },
        h("td", null,
          o.rank === false ? null : h("span", { class: "cand-rank" }, String(i + 1)),
          moveText(c.text),
          c.chosen ? h("span", { class: "cand-chosen", title: "Played", html: ICONS.check }) : null),
        h("td", null, fmt.int(c.score)),
        h("td", null,
          h("span", { class: "cand-leave" }, c.leave || "–"),
          o.leaveValue === false || !c.leave ? null : h("span", { class: "cand-lv" }, fmt.signed(c.leave_value, 1))),
        h("td", { class: c.value < 0 ? "cand-neg" : null }, fmt.signed(c.value, 1)),
        h("td", { class: "cand-total" },
          h("i", { class: "cand-total-bar", style: { width: width + "%" } }),
          h("span", null, fmt.fixed(c.total, 1))),
        showWin ? h("td", null, c.win === null || c.win === undefined ? "–" : fmt.pct(c.win, 0)) : null);
      if (o.onHover) tr.addEventListener("mouseenter", () => o.onHover(c));
      if (o.onClick) tr.addEventListener("click", () => o.onClick(c));
      return tr;
    });
    const table = h("table", { class: "cand-table" + (o.onClick ? " is-clickable" : "") },
      h("thead", null, h("tr", null,
        h("th", null, "Move"), h("th", { title: "Points scored" }, "Score"),
        h("th", { title: "Tiles kept, and their leave value" }, "Leave"),
        h("th", { title: "Estimated value of everything still to come, in points" }, "Value"),
        h("th", { title: "Score plus value: what the ranking is by" }, "Total"),
        showWin ? h("th", { title: "Estimated chance of winning after this move" }, "Win") : null)),
      h("tbody", null, rows));
    if (o.onHover) table.addEventListener("mouseleave", () => o.onHover(null));
    return h("div", { class: "cand-wrap" }, table);
  }

  /** Player lists from GET /api/models. */
  function playerChoices(data) {
    const bots = (data && data.bots) || [];
    const models = (data && data.models) || [];
    return { bots, models };
  }

  function specInfo(spec, data) {
    if (spec === "human") return { name: "You", kind: "human" };
    const { bots, models } = playerChoices(data);
    const bot = bots.find((b) => b.spec === spec);
    if (bot) return { name: bot.name, kind: "bot", description: bot.description };
    const model = models.find((m) => m.spec === spec);
    if (model) {
      const ev = model.eval && model.eval["greedy-leave"];
      const parts = [model.network + " network", "step " + App.fmt.int(model.step)];
      if (ev) parts.push(App.fmt.spread(ev.spread, ev.ci) + " a game against Greedy + leave");
      return { name: model.name, kind: "model", network: model.network, description: parts.join(" · "), model };
    }
    return { name: spec.replace(/^(model|bot):/, ""), kind: spec.startsWith("bot:") ? "bot" : "model" };
  }

  function kindLabel(spec, data) {
    const info = specInfo(spec, data);
    if (info.kind === "model") return info.network ? "model · " + info.network : "model";
    return info.kind === "bot" ? "bot" : "";
  }

  /** A <select> of bots and models (incompatible models are listed but cannot be chosen). */
  function playerSelect(data, value, options) {
    const h = App.h;
    const o = options || {};
    const { bots, models } = playerChoices(data);
    const select = h("select", { class: "select", id: o.id || null, "aria-label": o.label || null });
    const botGroup = h("optgroup", { label: "Bots" },
      bots.map((b) => h("option", { value: b.spec }, b.name)));
    select.appendChild(botGroup);
    if (models.length) {
      const sortedModels = models.slice().sort((a, b) =>
        (b.starred - a.starred) || (a.kind === "best" ? -1 : 0) - (b.kind === "best" ? -1 : 0) || (b.modified - a.modified));
      select.appendChild(h("optgroup", { label: "Models" }, sortedModels.map((m) =>
        h("option", { value: m.spec, disabled: !m.compatible },
          m.name + " · " + m.network + (m.starred ? " · starred" : "") + (m.compatible ? "" : " · other word list")))));
    }
    const all = bots.map((b) => b.spec).concat(models.filter((m) => m.compatible).map((m) => m.spec));
    select.value = all.includes(value) ? value : all[0] || "";
    return select;
  }

  /** The strongest-looking default: a starred or best compatible model, else Greedy + leave. */
  function defaultOpponent(data) {
    const models = ((data && data.models) || []).filter((m) => m.compatible);
    const starred = models.find((m) => m.starred);
    const best = models.filter((m) => m.kind === "best").sort((a, b) => b.modified - a.modified)[0];
    return (starred || best || models[0] || { spec: "bot:greedy-leave" }).spec;
  }

  /** Unseen tiles written the way a tile-tracking sheet does: AAAA EEE ... */
  function pool(counts, extra) {
    const h = App.h;
    const c = counts || {};
    const group = (L) => {
      const n = c[L] || 0;
      if (!n) return h("span", { class: "pool-group pool-gone", title: "All " + (L === "?" ? "blanks" : L + "s") + " are out" }, L === "?" ? "?" : L);
      return h("span", { class: "pool-group" + (L === "?" ? " is-blank" : ""), title: n + " × " + (L === "?" ? "blank" : L) },
        (L === "?" ? "?" : L).repeat(n));
    };
    const vowels = "AEIOU".split("");
    const consonants = "BCDFGHJKLMNPQRSTVWXYZ".split("");
    const total = Object.values(c).reduce((s, n) => s + n, 0);
    const nv = vowels.reduce((s, L) => s + (c[L] || 0), 0);
    const nb = c["?"] || 0;
    const nc = total - nv - nb;
    return h("div", { class: "pool" },
      h("div", { class: "pool-line" }, vowels.map(group)),
      h("div", { class: "pool-line" }, consonants.map(group), group("?")),
      h("div", { class: "pool-totals" },
        App.fmt.int(total) + " unseen: " + nv + " vowel" + (nv === 1 ? "" : "s") + ", " + nc + " consonant" +
        (nc === 1 ? "" : "s") + ", " + nb + " blank" + (nb === 1 ? "" : "s") + (extra ? ". " + extra : ".")));
  }

  /** "Ab" -> "Ab." ; server reasons arrive as lower-case fragments ("CATOX is not a word"). */
  function sentence(text) {
    const t = String(text || "").trim();
    if (!t) return "";
    const first = t[0].toUpperCase() + t.slice(1);
    return /[.!?]$/.test(first) ? first : first + ".";
  }

  function plural(n, word, many) {
    return App.fmt.int(n) + " " + (n === 1 ? word : many || word + "s");
  }

  /** Squares of a history entry or candidate. */
  const squaresOf = (tiles) => (tiles || []).map((t) => [normTile(t)[0], normTile(t)[1]]);

  const ui = {
    ICONS, sleep, moveText, candidateTable, playerSelect, specInfo, kindLabel, defaultOpponent, pool,
    sentence, plural, squaresOf,
  };

  return {
    create, rack, tile, tileBack, applyTiles, unseen, premium, squareName, emptyRows, normTile, ui,
    VALUES, COUNTS, VOWELS, COLS, N,
  };
})();
