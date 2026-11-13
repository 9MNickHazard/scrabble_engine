/* Scrabble Lab: charts.
 *
 * window.Charts draws every chart in the app as inline SVG and follows the same rules
 * everywhere: one y-axis per chart; categorical colours in their fixed order (var(--series-1..8),
 * never cycled: a ninth series is drawn in the muted ink); a legend for two or more series plus
 * direct end labels for up to four; 2px lines; bars at most 24px thick with 4px rounded data ends;
 * hairline solid gridlines; text in text tokens only; a hover layer on every chart (crosshair and
 * one tooltip listing every series on lines, a tooltip per mark on bars); a "Table" toggle that
 * shows the same numbers as a table.
 *
 *   Charts.line(container, options)        -> {update(options), destroy(), highlight(x, label)}
 *   Charts.bars(container, options)        -> {update, destroy}
 *   Charts.histogram(container, options)   -> {update, destroy}
 *   Charts.intervals(container, options)   -> {update, destroy}    estimates with 95% intervals
 *   Charts.sparkline(container, options)   -> {update, destroy}
 *   Charts.statTile(options)               -> element (.stat markup from app.css)
 *
 * Every chart replaces the container's content, re-renders when the container is resized or the
 * theme changes, and is destroyed when the page changes. update(options) merges the new options
 * into the old ones and redraws. Options beyond the core API are marked "optional extra" below.
 *
 * line options:
 *   title, subtitle
 *   series: [{name, points: [[x, y], ...], band?: [[x, low, high], ...], color?, dashed?}]
 *           a point with y null breaks the line (a gap); x going backwards also starts a new piece
 *   x: {label, format?, min?, max?}
 *   y: {label, format?, zero?, min?, max?}
 *   markers: [{x, label, tone?: "good"|"info"|"warning"}]    vertical event lines; the legend names
 *           each kind, the tooltip gives each marker's label
 *   referenceY: [{y, label}]
 *   height (default 260, includes the axis band)
 *   Optional extras:
 *   x/y.tickFormat: a function, or "auto" for round compact ticks while `format` serves the
 *           tooltip and table (by default `format` is used for ticks too)
 *   x.tooltipFormat: the tooltip heading, e.g. (v) => fmt.int(v) + " games"
 *   y.ciFormat: format of the "± half-width" in tooltips
 *   markers[].detail (tooltip text), .group (legend name), .emphasis (print the label on the chart)
 *   empty: text shown inside the frame when there is no data
 *   endLabels: false turns off the direct labels at the right end
 *   legend: false hides the legend
 *   margin: {left, right} minimum margins, to line up the plots of charts stacked together
 *
 * bars options: title, subtitle, categories, series: [{name, values, color?}], y: {label, format?},
 *   horizontal?, height?
 * histogram options: title, subtitle, series: [{name, values, color?}] (one or two), bins?: 30,
 *   x: {label, format?}, center?, height?, y?: {label} (optional extra),
 *   centerLabels?: ["left of centre", "right of centre"] (optional extra)
 * intervals options: title?, subtitle?, rows: [{label, sub?, value, ci, href?, detail?}],
 *   format?, center?: 0, centerLabel?, axisLabel?, color?
 * sparkline options: points, color?, width?: 120, height?: 32, band?, referenceY?: number,
 *   format?, xFormat?, label? (aria label)
 * statTile options: label, value (string or node), hint?, delta?, deltaTone?: "good"|"bad"|"neutral"
 */
window.Charts = (function () {
  "use strict";

  const h = App.h;
  const fmt = App.fmt;
  const MINUS = "−";
  let uid = 0;

  // ---- Styles ---------------------------------------------------------------------------

  const CSS = `
.chart { margin: 0; display: flex; flex-direction: column; gap: 10px; min-width: 0; --chart-surface: var(--surface-1); }
.chart-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
.chart-titles { min-width: 0; }
.chart-title { font-size: var(--text-m); font-weight: 650; color: var(--text-primary); line-height: 1.35; }
.chart-sub { font-size: var(--text-s); color: var(--text-secondary); margin-top: 2px; line-height: 1.45; }
.chart-tool { flex: none; display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 8px; margin: -3px -6px 0 0;
  border: 1px solid transparent; border-radius: 6px; background: none; color: var(--text-muted); font: inherit; font-size: var(--text-xs); cursor: pointer; }
.chart-tool:hover { color: var(--text-primary); background: var(--surface-2); }
.chart-tool[aria-pressed="true"] { color: var(--text-primary); background: var(--surface-2); border-color: var(--border); }
.chart-tool svg { display: block; }
.chart-legend { display: flex; flex-wrap: wrap; gap: 2px 16px; margin-top: -2px; }
.chart-legend-item { display: inline-flex; align-items: center; gap: 7px; padding: 1px 0; border: 0; background: none;
  color: var(--text-secondary); font: inherit; font-size: var(--text-s); line-height: 1.5; }
button.chart-legend-item { cursor: pointer; }
button.chart-legend-item:hover { color: var(--text-primary); }
.chart-legend-item[aria-pressed="false"] { color: var(--text-muted); text-decoration: line-through; }
.chart-legend-item[aria-pressed="false"] .chart-key { opacity: 0.3; }
.chart-key { flex: none; display: inline-block; position: relative; }
.chart-key-line { width: 16px; height: 2px; border-radius: 1px; }
.chart-key-dashed { width: 16px; height: 2px; }
.chart-key-rect { width: 10px; height: 10px; border-radius: 2px; }
.chart-key-band { width: 16px; height: 10px; border-radius: 2px; }
.chart-key-band > i { position: absolute; inset: 0; border-radius: 2px; opacity: 0.22; }
.chart-key-band > b { position: absolute; left: 0; right: 0; top: 4px; height: 2px; border-radius: 1px; }
.chart-key-marker { width: 10px; height: 8px; clip-path: polygon(0 0, 100% 0, 50% 100%); }
.chart-body { position: relative; }
.chart-body[hidden], .chart-table[hidden], .chart-legend[hidden], .chart-head[hidden] { display: none; }
.chart-svg { display: block; overflow: visible; font-family: var(--font); }
.chart-svg:focus { outline: none; }
.chart-svg:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 4px; }
.chart-svg text { font-family: var(--font); }
.chart-tick { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart-axis-label { fill: var(--text-muted); font-size: 11px; }
.chart-end-label { fill: var(--text-secondary); font-size: 12px; font-variant-numeric: tabular-nums; }
.chart-value { fill: var(--text-secondary); font-size: 11px; font-variant-numeric: tabular-nums; }
.chart-cat { fill: var(--text-secondary); font-size: 12px; }
.chart-grid { stroke: var(--grid); stroke-width: 1; shape-rendering: crispEdges; }
.chart-axis { stroke: var(--axis); stroke-width: 1; shape-rendering: crispEdges; }
.chart-ref { stroke: var(--text-muted); stroke-width: 1; opacity: 0.75; shape-rendering: crispEdges; }
.chart-ref-label { fill: var(--text-muted); font-size: 11px; }
.chart-halo { paint-order: stroke; stroke: var(--chart-surface); stroke-width: 3px; stroke-linejoin: round; }
.chart-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.chart-band { fill-opacity: 0.13; stroke: none; }
.chart-wash { fill-opacity: 0.12; stroke: none; }
.chart-dot { stroke: var(--chart-surface); stroke-width: 2; }
.chart-leader { fill: none; stroke: var(--text-muted); stroke-width: 1; opacity: 0.7; }
.chart-marker { stroke-width: 1; opacity: 0.5; shape-rendering: crispEdges; }
.chart-marker-label { fill: var(--text-secondary); font-size: 11px; }
.chart-highlight { stroke: var(--text-primary); stroke-width: 1.5; opacity: 0.8; }
.chart-pill { fill: var(--surface-3); stroke: var(--border-strong); stroke-width: 1; }
.chart-pill-text { fill: var(--text-primary); font-size: 11px; font-weight: 600; }
.chart-cross { stroke: var(--text-muted); stroke-width: 1; shape-rendering: crispEdges; pointer-events: none; }
.chart-hit { fill: transparent; }
.chart-hover-col { fill: var(--text-primary); opacity: 0.06; pointer-events: none; }
.chart-bar { transition: filter 0.1s, opacity 0.1s; }
.chart-bar.is-hover { filter: brightness(1.18); }
.chart-hit-bar { fill: transparent; cursor: default; outline: none; }
.chart-row-hit { fill: transparent; }
.chart-row:hover .chart-row-hit, .chart-row:focus .chart-row-hit { fill: var(--surface-2); }
.chart-row { outline: none; }
.chart-row.is-link { cursor: pointer; }
.chart-row-label { fill: var(--text-primary); font-size: 13px; font-weight: 550; }
.chart-row-sub { fill: var(--text-muted); font-size: 11px; }
.chart-row-value { fill: var(--text-primary); font-size: 13px; font-weight: 600; font-variant-numeric: tabular-nums; }
.chart-empty-text { fill: var(--text-muted); font-size: 13px; }
.chart-tooltip { position: absolute; z-index: 20; top: 0; left: 0; pointer-events: none; min-width: 120px; max-width: 300px;
  padding: 8px 10px; background: var(--surface-3); border: 1px solid var(--border-strong); border-radius: 8px;
  box-shadow: var(--shadow); font-size: var(--text-s); line-height: 1.45; color: var(--text-secondary); }
.chart-tooltip[hidden] { display: none; }
:root[data-theme="light"] .chart-tooltip { background: var(--surface-1); }
.tt-head { color: var(--text-secondary); margin-bottom: 4px; white-space: nowrap; }
.tt-row { display: flex; align-items: center; gap: 8px; white-space: nowrap; padding: 1px 0; }
.tt-row strong { color: var(--text-primary); font-weight: 650; font-variant-numeric: tabular-nums; }
.tt-key { flex: none; width: 12px; height: 2px; border-radius: 1px; }
.tt-key-rect { width: 9px; height: 9px; border-radius: 2px; }
.tt-key-marker { width: 9px; height: 7px; border-radius: 0; clip-path: polygon(0 0, 100% 0, 50% 100%); }
.tt-name { color: var(--text-secondary); }
.tt-wrap { white-space: normal; align-items: flex-start; }
.tt-wrap .tt-key { margin-top: 6px; }
.tt-note { color: var(--text-muted); margin-top: 3px; white-space: normal; }
.chart-table { max-height: 360px; overflow: auto; border: 1px solid var(--border); border-radius: 8px; }
.chart-table .table th { position: sticky; top: 0; background: var(--chart-surface); z-index: 1; }
.chart-table .table td, .chart-table .table th { padding: 6px 12px; font-size: var(--text-s); }
.chart-table-note { padding: 8px 12px; font-size: var(--text-s); color: var(--text-muted); border-bottom: 1px solid var(--border); }
.chart-table-empty { padding: 24px; text-align: center; color: var(--text-muted); }
.spark { position: relative; display: inline-block; line-height: 0; vertical-align: middle; }
.spark svg { display: block; overflow: visible; }
.spark .chart-tooltip { min-width: 0; padding: 4px 8px; white-space: nowrap; line-height: 1.4; }
.stat-delta { display: inline-flex; align-items: center; gap: 4px; font-size: var(--text-s); font-weight: 600; }
.stat-delta svg { flex: none; }
.stat-delta-neutral { color: var(--text-muted); }
.stat-value .stat-ci { font-size: 0.62em; font-weight: 550; color: var(--text-secondary); letter-spacing: 0; }
`;

  function injectStyles() {
    if (document.getElementById("charts-style")) return;
    const node = document.createElement("style");
    node.id = "charts-style";
    node.textContent = CSS;
    document.head.appendChild(node);
  }

  const ICON_TABLE = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4">' +
    '<rect x="2" y="2.5" width="12" height="11" rx="1.5"/><path d="M2 6.3h12M2 10h12M6.5 6.3v7.2"/></svg>';
  const ICON_CHART = '<svg viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.4" ' +
    'stroke-linecap="round" stroke-linejoin="round"><path d="M2 13.5h12"/><path d="M3 10.5l3.2-3.6 2.6 2.2L13 4"/></svg>';
  const ICON_UP = '<svg viewBox="0 0 10 10" width="10" height="10"><path d="M5 1.5L9 7.5H1z" fill="currentColor"/></svg>';
  const ICON_DOWN = '<svg viewBox="0 0 10 10" width="10" height="10"><path d="M5 8.5L1 2.5h8z" fill="currentColor"/></svg>';

  // ---- Numbers and measurement ----------------------------------------------------------

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const px = (v) => Math.round(v * 10) / 10;

  function colorOf(series, index) {
    if (series && series.color) return series.color;
    return index < 8 ? `var(--series-${index + 1})` : "var(--text-muted)";
  }

  function toneColor(tone) {
    if (tone === "good") return "var(--good)";
    if (tone === "warning") return "var(--warning)";
    if (tone === "critical") return "var(--critical)";
    return "var(--text-muted)";
  }

  let measureContext = null;
  let fontFamily = null;
  function textWidth(text, size = 11, weight = 400) {
    if (!measureContext) measureContext = document.createElement("canvas").getContext("2d");
    if (!fontFamily) {
      fontFamily = getComputedStyle(document.documentElement).getPropertyValue("--font").trim() ||
        "system-ui, sans-serif";
    }
    measureContext.font = `${weight} ${size}px ${fontFamily}`;
    return measureContext.measureText(String(text)).width;
  }

  /** A readable default for any value: 57.9, 0.142, 1,234, 12.9K. */
  function autoFormat(value) {
    if (!isNum(value)) return "–";
    const abs = Math.abs(value);
    let text;
    if (abs >= 10000) return fmt.compact(value).replace(/^-/, MINUS);
    if (abs >= 100) text = Math.round(value).toLocaleString("en-US");
    else if (abs >= 10) text = value.toFixed(1);
    else if (abs >= 1) text = value.toFixed(2);
    else if (abs === 0) text = "0";
    else text = Number(value.toPrecision(3)).toString();
    return text.replace(/^-/, MINUS);
  }

  /** One number of decimals for a set of values shown side by side (0.31, 0.40, not 0.4). */
  function consistentFormat(values) {
    let digits = 0;
    for (const v of values) {
      if (Math.abs(v) >= 10000) continue;
      for (let d = digits; d <= 2; d++) {
        if (Math.abs(Math.round(v * 10 ** d) - v * 10 ** d) < 1e-6) break;
        digits = Math.min(2, d + 1);
      }
    }
    return (v) => {
      if (!isNum(v)) return "–";
      if (Math.abs(v) >= 10000) return autoFormat(v);
      return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits }).replace(/^-/, MINUS);
    };
  }

  function niceStep(raw) {
    if (!(raw > 0) || !isFinite(raw)) return 1;
    const base = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / base;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * base;
  }

  /** The round step (1, 2, 2.5, 5 x 10^n) nearest to `raw`, for axis ticks. */
  function tickStep(raw) {
    if (!(raw > 0) || !isFinite(raw)) return 1;
    const base = Math.pow(10, Math.floor(Math.log10(raw)));
    const f = raw / base;
    return (f < 1.5 ? 1 : f < 2.25 ? 2 : f < 3.5 ? 2.5 : f < 7.5 ? 5 : 10) * base;
  }

  /** Ticks at round values. `extend` widens [min, max] out to whole steps. */
  function niceTicks(min, max, intervals, extend) {
    const step = tickStep((max - min) / Math.max(1, intervals));
    const eps = step * 1e-6;
    const lo = extend ? Math.floor(min / step + 1e-9) * step : Math.ceil(min / step - 1e-9) * step;
    const hi = extend ? Math.ceil(max / step - 1e-9) * step : Math.floor(max / step + 1e-9) * step;
    const ticks = [];
    for (let i = 0; i < 200; i++) {
      const v = lo + i * step;
      if (v > hi + eps) break;
      ticks.push(Math.abs(v) < eps ? 0 : Number(v.toPrecision(12)));
    }
    return { step, ticks, min: extend ? lo : min, max: extend ? hi : max };
  }

  /** One style per axis: in full under 10,000, otherwise compact (250K, 1.5M). */
  function tickFormatter(ticks, step) {
    const maxAbs = ticks.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    let unit = 1;
    let suffix = "";
    if (maxAbs >= 1e9) { unit = 1e9; suffix = "B"; } else if (maxAbs >= 1e6) { unit = 1e6; suffix = "M"; } else if (maxAbs >= 1e4) { unit = 1e3; suffix = "K"; }
    const scaled = step / unit;
    let digits = 0;
    while (digits < 8 && Math.abs(Math.round(scaled * 10 ** digits) - scaled * 10 ** digits) > 1e-6) digits++;
    return (value) => {
      if (Math.abs(value) < step * 1e-6) return "0";
      let text;
      if (suffix) {
        text = (value / unit).toFixed(digits);
        if (text.includes(".")) text = text.replace(/\.?0+$/, "");
        text += suffix;
      } else {
        text = value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
      }
      return text.replace(/^-/, MINUS);
    };
  }

  /** An axis's tick format: its tickFormat, else its format, else round compact ticks.
   * tickFormat "auto" keeps the round ticks while `format` is used for tooltips and tables. */
  function pickTickFormat(axis, ticks, step) {
    if (typeof axis.tickFormat === "function") return axis.tickFormat;
    if (axis.tickFormat !== "auto" && typeof axis.format === "function") return axis.format;
    return tickFormatter(ticks, step);
  }

  function linear(d0, d1, r0, r1) {
    const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
    const f = (v) => r0 + (v - d0) * k;
    f.invert = (p) => d0 + (p - r0) / (k || 1);
    return f;
  }

  function wrapText(text, maxWidth, size) {
    const words = String(text).split(/\s+/);
    const lines = [];
    let current = "";
    for (const word of words) {
      const next = current ? current + " " + word : word;
      if (current && textWidth(next, size) > maxWidth) {
        lines.push(current);
        current = word;
      } else current = next;
    }
    if (current) lines.push(current);
    return lines;
  }

  function ellipsize(text, maxWidth, size, weight) {
    text = String(text);
    if (textWidth(text, size, weight) <= maxWidth) return text;
    let lo = 0;
    let hi = text.length;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (textWidth(text.slice(0, mid) + "…", size, weight) <= maxWidth) lo = mid;
      else hi = mid - 1;
    }
    return text.slice(0, lo) + "…";
  }

  /** First index whose x (or value) is >= x, in an array sorted by x. */
  function lowerBound(array, x, key) {
    let lo = 0;
    let hi = array.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((key ? array[mid][0] : array[mid]) < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function nearestIndex(array, x, key) {
    if (!array.length) return -1;
    const i = lowerBound(array, x, key);
    if (i <= 0) return 0;
    if (i >= array.length) return array.length - 1;
    const a = key ? array[i - 1][0] : array[i - 1];
    const b = key ? array[i][0] : array[i];
    return x - a <= b - x ? i - 1 : i;
  }

  const isSorted = (points) => points.every((p, i) => i === 0 || points[i - 1][0] <= p[0]);
  const capitalize = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : "");

  // ---- Bars with a rounded data end and a square baseline -------------------------------

  function barPathV(x, w, base, tip, radius) {
    const len = Math.abs(base - tip);
    if (len < 0.5 || w <= 0) return "";
    const r = Math.max(0, Math.min(radius, w / 2, len));
    if (tip < base) {
      return `M${px(x)},${px(base)}V${px(tip + r)}A${r},${r} 0 0 1 ${px(x + r)},${px(tip)}` +
        `H${px(x + w - r)}A${r},${r} 0 0 1 ${px(x + w)},${px(tip + r)}V${px(base)}Z`;
    }
    return `M${px(x)},${px(base)}V${px(tip - r)}A${r},${r} 0 0 0 ${px(x + r)},${px(tip)}` +
      `H${px(x + w - r)}A${r},${r} 0 0 0 ${px(x + w)},${px(tip - r)}V${px(base)}Z`;
  }

  function barPathH(base, tip, y, w, radius) {
    const len = Math.abs(tip - base);
    if (len < 0.5 || w <= 0) return "";
    const r = Math.max(0, Math.min(radius, w / 2, len));
    if (tip > base) {
      return `M${px(base)},${px(y)}H${px(tip - r)}A${r},${r} 0 0 1 ${px(tip)},${px(y + r)}` +
        `V${px(y + w - r)}A${r},${r} 0 0 1 ${px(tip - r)},${px(y + w)}H${px(base)}Z`;
    }
    return `M${px(base)},${px(y)}H${px(tip + r)}A${r},${r} 0 0 0 ${px(tip)},${px(y + r)}` +
      `V${px(y + w - r)}A${r},${r} 0 0 0 ${px(tip + r)},${px(y + w)}H${px(base)}Z`;
  }

  // ---- The frame every chart shares: title, legend, body, tooltip, table ----------------

  function keyNode(kind, color) {
    if (kind === "band") {
      return h("span", { class: "chart-key chart-key-band" },
        h("i", { style: { background: color } }), h("b", { style: { background: color } }));
    }
    if (kind === "dashed") {
      return h("span", { class: "chart-key chart-key-dashed",
        style: { background: `repeating-linear-gradient(90deg, ${color} 0 5px, transparent 5px 8px)` } });
    }
    return h("span", { class: "chart-key chart-key-" + kind, style: { background: color } });
  }

  function tipRow(color, value, name, key = "line") {
    return h("div", { class: "tt-row" },
      h("span", { class: "tt-key" + (key === "line" ? "" : " tt-key-" + key),
        style: key === "dashed"
          ? { background: `repeating-linear-gradient(90deg, ${color} 0 4px, transparent 4px 6px)` }
          : { background: color } }),
      h("strong", null, value),
      name ? h("span", { class: "tt-name" }, name) : null);
  }

  function frame(container, options, kind, renderer) {
    injectStyles();
    const self = {
      kind, opts: Object.assign({}, options), hidden: new Set(), table: false, highlight: null,
      width: 0, destroyed: false, id: "chart" + (++uid),
    };
    const root = h("figure", { class: "chart chart-" + kind });
    const head = h("div", { class: "chart-head" });
    const legend = h("div", { class: "chart-legend" });
    const body = h("div", { class: "chart-body" });
    const tip = h("div", { class: "chart-tooltip", "aria-hidden": "true" });
    const tableBox = h("div", { class: "chart-table" });
    tip.hidden = true;
    tableBox.hidden = true;
    root.append(head, legend, body, tableBox);
    App.clear(container).appendChild(root);
    self.root = root;

    self.showTip = (nodes, x, y) => {
      tip.replaceChildren(...nodes.filter(Boolean));
      tip.hidden = false;
      const tw = tip.offsetWidth;
      const th = tip.offsetHeight;
      const bh = body.clientHeight || 200;
      let left = x + 14;
      if (left + tw > self.width) left = x - 14 - tw;
      if (left < 0) left = Math.max(0, Math.min(self.width - tw, x - tw / 2));
      const top = clamp(y - th / 2, -8, Math.max(-8, bh - th + 8));
      tip.style.left = Math.round(left) + "px";
      tip.style.top = Math.round(top) + "px";
    };
    self.hideTip = () => { tip.hidden = true; };

    function toggleSeries(name) {
      const names = renderer.names ? renderer.names(self) : [];
      if (!self.hidden.has(name) && names.filter((n) => !self.hidden.has(n)).length <= 1) return;
      if (self.hidden.has(name)) self.hidden.delete(name);
      else self.hidden.add(name);
      render();
    }

    function render() {
      if (self.destroyed) return;
      const o = self.opts;
      App.clear(head);
      head.appendChild(h("div", { class: "chart-titles" },
        o.title ? h("div", { class: "chart-title" }, o.title) : null,
        o.subtitle ? h("div", { class: "chart-sub" }, o.subtitle) : null));
      const tableable = renderer.table && o.table !== false;
      if (tableable) {
        head.appendChild(h("button", {
          class: "chart-tool", type: "button", "aria-pressed": String(self.table),
          title: self.table ? "Show the chart" : "Show these numbers as a table",
          onclick: () => { self.table = !self.table; render(); },
        }, h("span", { html: self.table ? ICON_CHART : ICON_TABLE }), self.table ? "Chart" : "Table"));
      }
      head.hidden = !o.title && !o.subtitle && !tableable;

      App.clear(legend);
      const items = renderer.legend && o.legend !== false ? renderer.legend(self) : [];
      for (const item of items) {
        const content = [keyNode(item.key, item.color), h("span", null, item.name)];
        legend.appendChild(item.toggle
          ? h("button", { class: "chart-legend-item", type: "button", "aria-pressed": String(item.active),
            title: item.active ? "Hide this series" : "Show this series", onclick: () => toggleSeries(item.name) }, content)
          : h("span", { class: "chart-legend-item" }, content));
      }
      legend.hidden = !items.length;

      self.hideTip();
      if (self.table && tableable) {
        body.hidden = true;
        tableBox.hidden = false;
        tableBox.replaceChildren(renderer.table(self));
        return;
      }
      body.hidden = false;
      tableBox.hidden = true;
      const width = Math.max(160, Math.floor(root.clientWidth || container.clientWidth || 640));
      self.width = width;
      body.replaceChildren(renderer.draw(self, width), tip);
    }

    let frameRequest = 0;
    const schedule = () => {
      if (self.destroyed || frameRequest) return;
      frameRequest = requestAnimationFrame(() => { frameRequest = 0; render(); });
    };
    const observer = new ResizeObserver(() => {
      if (self.destroyed || self.table) return;
      const w = Math.floor(root.clientWidth);
      if (w > 0 && Math.abs(w - self.width) >= 1) schedule();
    });
    observer.observe(root);
    const offTheme = App.on("theme", () => { if (!root.isConnected) destroy(); else schedule(); });
    function destroy() {
      if (self.destroyed) return;
      self.destroyed = true;
      observer.disconnect();
      offTheme();
      if (frameRequest) cancelAnimationFrame(frameRequest);
    }
    App.onCleanup(destroy);

    // A live update while the pointer is on the chart waits until it leaves, so the tooltip
    // the reader is looking at does not vanish under them.
    let hovering = false;
    let pending = false;
    root.addEventListener("pointerenter", () => { hovering = true; });
    root.addEventListener("pointerleave", () => {
      hovering = false;
      if (pending) { pending = false; render(); }
    });
    render();

    return {
      update(next) {
        Object.assign(self.opts, next || {});
        if (hovering && !self.table) pending = true;
        else render();
      },
      destroy,
      /** Emphasise one x position (for example an event in a timeline). null clears it. */
      highlight(x, label) {
        self.highlight = x === null || x === undefined ? null : { x, label };
        render();
      },
      get element() { return root; },
    };
  }

  function axisLabelText(svg, text, x, y, anchor) {
    if (text) svg.appendChild(h("text", { class: "chart-axis-label", x, y, "text-anchor": anchor || "start" }, text));
  }

  function emptyFrame(svg, L, R, T, B, message) {
    for (let i = 0; i <= 3; i++) {
      const y = Math.round(T + ((B - T) * i) / 3) + 0.5;
      svg.appendChild(h("line", { class: i === 3 ? "chart-axis" : "chart-grid", x1: L, x2: R, y1: y, y2: y }));
    }
    const lines = wrapText(message || "No data yet", Math.max(120, (R - L) - 48), 13).slice(0, 4);
    const cy = (T + B) / 2 - ((lines.length - 1) * 18) / 2;
    lines.forEach((text, i) => svg.appendChild(h("text", { class: "chart-empty-text chart-halo", x: (L + R) / 2,
      y: cy + i * 18, dy: "0.32em", "text-anchor": "middle" }, text)));
  }

  // ---- Line chart -----------------------------------------------------------------------

  function lineEntries(self) {
    return (self.opts.series || []).map((s, i) => ({ s, i, name: s.name || `Series ${i + 1}`, color: colorOf(s, i) }));
  }

  function prepareLine(self) {
    const visible = lineEntries(self).filter((e) => !self.hidden.has(e.name));
    for (const e of visible) {
      e.raw = (e.s.points || []).filter((p) => p && isNum(p[0]));
      e.valid = e.raw.filter((p) => isNum(p[1]));
      e.sorted = isSorted(e.valid) ? e.valid : e.valid.slice().sort((a, b) => a[0] - b[0]);
      e.bandRaw = (e.s.band || []).filter((b) => b && isNum(b[0]));
      const bandValid = e.bandRaw.filter((b) => isNum(b[1]) && isNum(b[2]));
      e.bandSorted = isSorted(bandValid) ? bandValid : bandValid.slice().sort((a, b) => a[0] - b[0]);
      e.last = e.valid.length ? e.valid[e.valid.length - 1] : null;
    }
    return visible;
  }

  function unionX(visible) {
    const set = new Set();
    for (const e of visible) for (const p of e.valid) set.add(p[0]);
    return Array.from(set).sort((a, b) => a - b);
  }

  /** Split into drawable pieces at null values and wherever x goes backwards. */
  function pieces(points, valueOk) {
    const out = [];
    let current = [];
    let prev = -Infinity;
    for (const p of points) {
      if (!valueOk(p)) {
        if (current.length) out.push(current);
        current = [];
        prev = -Infinity;
        continue;
      }
      if (p[0] < prev && current.length) {
        out.push(current);
        current = [];
      }
      current.push(p);
      prev = p[0];
    }
    if (current.length) out.push(current);
    return out;
  }

  /** Keep at most four points per pixel column (first, min, max, last) so long series stay light. */
  function decimate(screen, plotWidth) {
    if (screen.length <= plotWidth * 2) return screen;
    const out = [];
    let bucket = null;
    let items = [];
    const flush = () => {
      if (!items.length) return;
      let min = items[0];
      let max = items[0];
      for (const it of items) { if (it[1] < min[1]) min = it; if (it[1] > max[1]) max = it; }
      const keep = new Set([items[0], min, max, items[items.length - 1]]);
      for (const it of items) if (keep.has(it)) out.push(it);
    };
    for (const p of screen) {
      const b = Math.floor(p[0]);
      if (b !== bucket) { flush(); bucket = b; items = []; }
      items.push(p);
    }
    flush();
    return out;
  }

  function pathFrom(screen) {
    let d = "";
    for (let i = 0; i < screen.length; i++) d += (i ? "L" : "M") + px(screen[i][0]) + "," + px(screen[i][1]);
    return d;
  }

  function lineLegend(self) {
    const o = self.opts;
    const entries = lineEntries(self);
    const items = [];
    if (entries.length >= 2) {
      for (const e of entries) {
        items.push({ name: e.name, color: e.color, key: e.s.dashed ? "dashed" : e.s.band ? "band" : "line",
          toggle: true, active: !self.hidden.has(e.name) });
      }
    }
    const groups = new Map();
    for (const m of o.markers || []) {
      const name = m.group || m.label;
      if (!name) continue;
      const key = (m.tone || "info") + "|" + name;
      if (!groups.has(key)) groups.set(key, { name, color: toneColor(m.tone), key: "marker" });
    }
    if (groups.size && groups.size <= 3) items.push(...groups.values());
    return items;
  }

  function drawLine(self, width) {
    const o = self.opts;
    const xo = o.x || {};
    const yo = o.y || {};
    const height = o.height || 260;
    const visible = prepareLine(self);

    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, count = 0;
    for (const e of visible) {
      for (const p of e.valid) {
        if (p[0] < x0) x0 = p[0];
        if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1];
        if (p[1] > y1) y1 = p[1];
        count++;
      }
      for (const b of e.bandSorted) {
        if (b[1] < y0) y0 = b[1];
        if (b[2] > y1) y1 = b[2];
      }
    }
    const empty = count === 0;
    if (empty) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; } else {
      if (yo.zero) { y0 = Math.min(y0, 0); y1 = Math.max(y1, 0); }
      for (const r of o.referenceY || []) if (isNum(r.y)) { y0 = Math.min(y0, r.y); y1 = Math.max(y1, r.y); }
    }
    if (isNum(xo.min)) x0 = xo.min;
    if (isNum(xo.max)) x1 = xo.max;
    if (isNum(yo.min)) y0 = yo.min;
    if (isNum(yo.max)) y1 = yo.max;
    if (!(x1 > x0)) { const pad = Math.abs(x0) * 0.1 || 1; x0 -= pad; x1 += pad; }
    if (!(y1 > y0)) { const pad = Math.abs(y0) * 0.1 || 1; y0 -= pad; y1 += pad; }

    const markers = empty ? [] : (o.markers || []).filter((m) => isNum(m.x) && m.x >= x0 && m.x <= x1);
    const hl = !empty && self.highlight && isNum(self.highlight.x) && self.highlight.x >= x0 && self.highlight.x <= x1
      ? self.highlight : null;
    const top = (yo.label ? 18 : 6) + (markers.length ? 14 : 6);
    const bottom = 22 + (xo.label ? 14 : 0);
    const plotH = Math.max(40, height - top - bottom);

    const yt = niceTicks(y0, y1, Math.max(2, Math.round(plotH / 52)), true);
    if (!isNum(yo.min)) y0 = yt.min;
    if (!isNum(yo.max)) y1 = yt.max;
    const tol = Math.abs(yt.step) * 1e-6;
    const yTicks = empty ? [] : yt.ticks.filter((v) => v >= y0 - tol && v <= y1 + tol);
    const yTick = pickTickFormat(yo, yTicks, yt.step);
    const yValue = yo.format || autoFormat;
    const ciFormat = yo.ciFormat || ((v) => yValue(Math.abs(v)).replace(/^\+/, ""));
    const margin = o.margin || {};
    const left = Math.max(margin.left || 0, empty ? 12 : Math.ceil(Math.max(8, ...yTicks.map((v) => textWidth(yTick(v))))) + 10);

    // Direct labels at the right: the value for one series, names for two to four.
    let endItems = [];
    if (!empty && o.endLabels !== false) {
      const span = x1 - x0;
      const last = visible.length === 1 ? visible[0].last : null;
      if (last && last[0] >= x1 - 0.15 * span && last[0] <= x1) endItems = [{ e: visible[0], text: yValue(last[1]) }];
      else if (visible.length >= 2 && visible.length <= 4) {
        endItems = visible.filter((e) => e.last && e.last[0] >= x1 - 0.15 * span && e.last[0] <= x1)
          .map((e) => ({ e, text: e.name }));
      }
    }
    let right = 12;
    if (endItems.length) {
      const w = Math.max(...endItems.map((it) => textWidth(it.text, 12)));
      if (w + 18 <= Math.min(170, width * 0.3)) right = Math.ceil(w) + 18;
      else endItems = [];
    }
    right = Math.max(right, margin.right || 0);

    const L = left;
    const R = Math.max(L + 40, width - right);
    const T = top;
    const B = top + plotH;
    const sx = linear(x0, x1, L, R);
    const sy = linear(y0, y1, B, T);

    const svg = h("svg", { class: "chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": (o.title || "Line chart") + ". Use the Table button for the values.", tabindex: empty ? null : "0" });
    const clipId = self.id + "-clip";
    svg.appendChild(h("defs", null, h("clipPath", { id: clipId },
      h("rect", { x: L - 1, y: T - 3, width: R - L + 2, height: plotH + 6 }))));
    axisLabelText(svg, yo.label, 0, 11);

    if (empty) {
      emptyFrame(svg, L, R, T, B, o.empty);
      axisLabelText(svg, xo.label, R, height - 3, "end");
      return svg;
    }

    for (const v of yTicks) {
      const y = Math.round(sy(v)) + 0.5;
      if (Math.abs(v - y0) > tol) svg.appendChild(h("line", { class: "chart-grid", x1: L, x2: R, y1: y, y2: y }));
      svg.appendChild(h("text", { class: "chart-tick", x: L - 8, y, dy: "0.32em", "text-anchor": "end" }, yTick(v)));
    }
    svg.appendChild(h("line", { class: "chart-axis", x1: L, x2: R, y1: Math.round(B) + 0.5, y2: Math.round(B) + 0.5 }));

    // X ticks: as many round values as fit without their labels touching.
    let xTicks = [];
    let xTick = autoFormat;
    for (let n = Math.max(3, Math.floor((R - L) / 120)); n >= 1; n--) {
      const t = niceTicks(x0, x1, n, false);
      const f = pickTickFormat(xo, t.ticks, t.step);
      const widths = t.ticks.map((v) => textWidth(f(v)));
      let fits = true;
      for (let i = 1; i < t.ticks.length; i++) {
        if (sx(t.ticks[i]) - sx(t.ticks[i - 1]) < (widths[i] + widths[i - 1]) / 2 + 14) { fits = false; break; }
      }
      xTicks = t.ticks;
      xTick = f;
      if (fits) break;
    }
    for (const v of xTicks) {
      const x = sx(v);
      const w = textWidth(xTick(v));
      let anchor = "middle";
      let tx = x;
      if (x - w / 2 < 0) { anchor = "start"; tx = 0; } else if (x + w / 2 > width) { anchor = "end"; tx = width; }
      svg.appendChild(h("text", { class: "chart-tick", x: px(tx), y: B + 15, "text-anchor": anchor }, xTick(v)));
    }
    axisLabelText(svg, xo.label, R, B + 30, "end");

    // Event markers: a hairline and a small flag above the plot. The legend names each kind and
    // the tooltip gives every marker's label, so only emphasised markers print theirs.
    const markerLayer = h("g");
    let lastLabelEnd = -Infinity;
    for (const m of markers) {
      const x = Math.round(sx(m.x)) + 0.5;
      const c = toneColor(m.tone);
      markerLayer.appendChild(h("line", { class: "chart-marker", x1: x, x2: x, y1: T, y2: B, style: { stroke: c } }));
      markerLayer.appendChild(h("path", { d: `M${x - 4.5},${T - 10}h9l-4.5,7z`, style: { fill: c } }));
      if (m.emphasis && m.label) {
        const w = textWidth(m.label);
        let lx = x + 7;
        let anchor = "start";
        if (lx + w > width) { lx = x - 7; anchor = "end"; }
        const start = anchor === "start" ? lx : lx - w;
        if (start > lastLabelEnd + 10) {
          markerLayer.appendChild(h("text", { class: "chart-marker-label", x: lx, y: T - 4, "text-anchor": anchor }, m.label));
          lastLabelEnd = start + w;
        }
      }
    }

    const plot = h("g", { "clip-path": `url(#${clipId})` });
    for (const e of visible) {
      for (const piece of pieces(e.bandRaw, (b) => isNum(b[1]) && isNum(b[2]))) {
        if (piece.length === 1) {
          // A lone interval (the first evaluation, say): a narrow slice of the band.
          const b = piece[0];
          plot.appendChild(h("rect", { class: "chart-band", x: px(sx(b[0]) - 5), y: px(sy(b[2])), width: 10,
            height: px(Math.max(1, sy(b[1]) - sy(b[2]))), rx: 2, style: { fill: e.color } }));
          continue;
        }
        let d = "";
        piece.forEach((b, i) => { d += (i ? "L" : "M") + px(sx(b[0])) + "," + px(sy(b[2])); });
        for (let i = piece.length - 1; i >= 0; i--) d += "L" + px(sx(piece[i][0])) + "," + px(sy(piece[i][1]));
        plot.appendChild(h("path", { class: "chart-band", d: d + "Z", style: { fill: e.color } }));
      }
    }
    svg.appendChild(plot);
    svg.appendChild(markerLayer);
    for (const r of o.referenceY || []) {
      if (!isNum(r.y) || r.y < y0 - tol || r.y > y1 + tol) continue;
      const y = Math.round(sy(r.y)) + 0.5;
      svg.appendChild(h("line", { class: "chart-ref", x1: L, x2: R, y1: y, y2: y }));
    }
    const lines = h("g", { "clip-path": `url(#${clipId})` });
    for (const e of visible) {
      for (const piece of pieces(e.raw, (p) => isNum(p[1]))) {
        const screen = decimate(piece.map((p) => [sx(p[0]), sy(p[1])]), R - L);
        if (screen.length === 1) {
          lines.appendChild(h("circle", { cx: px(screen[0][0]), cy: px(screen[0][1]), r: 2.5, style: { fill: e.color } }));
          continue;
        }
        lines.appendChild(h("path", { class: "chart-line", d: pathFrom(screen),
          "stroke-dasharray": e.s.dashed ? "5 4" : null, style: { stroke: e.color } }));
      }
    }
    svg.appendChild(lines);
    for (const r of o.referenceY || []) {
      if (!r.label || !isNum(r.y) || r.y < y0 - tol || r.y > y1 + tol) continue;
      svg.appendChild(h("text", { class: "chart-ref-label chart-halo", x: R - 4, y: Math.round(sy(r.y)) - 5, "text-anchor": "end" }, r.label));
    }
    for (const e of visible) {
      if (!e.last || e.last[0] < x0 || e.last[0] > x1 || e.last[1] < y0 - tol || e.last[1] > y1 + tol) continue;
      svg.appendChild(h("circle", { class: "chart-dot", cx: px(sx(e.last[0])), cy: px(sy(e.last[1])), r: 4, style: { fill: e.color } }));
    }

    if (endItems.length) {
      const items = endItems.map((it) => ({ ...it, x: sx(it.e.last[0]), y: clamp(sy(it.e.last[1]), T, B) }))
        .sort((a, b) => a.y - b.y);
      items.forEach((it) => { it.ly = it.y; });
      const gap = 14;
      for (let i = 1; i < items.length; i++) if (items[i].ly - items[i - 1].ly < gap) items[i].ly = items[i - 1].ly + gap;
      const over = items[items.length - 1].ly - (B + 4);
      if (over > 0) {
        items[items.length - 1].ly -= over;
        for (let i = items.length - 2; i >= 0; i--) if (items[i + 1].ly - items[i].ly < gap) items[i].ly = items[i + 1].ly - gap;
      }
      for (const it of items) {
        if (Math.abs(it.ly - it.y) > 2 || it.x < R - 2) {
          svg.appendChild(h("path", { class: "chart-leader", d: `M${px(it.x + 6)},${px(it.y)}L${px(R + 5)},${px(it.ly)}` }));
        }
        svg.appendChild(h("text", { class: "chart-end-label", x: R + 9, y: px(it.ly), dy: "0.32em" }, it.text));
      }
    }

    if (hl) {
      const x = Math.round(sx(hl.x)) + 0.5;
      svg.appendChild(h("line", { class: "chart-highlight", x1: x, x2: x, y1: T - 2, y2: B }));
      if (hl.label) {
        const text = ellipsize(hl.label, Math.max(80, (R - L) * 0.6), 11, 600);
        const w = textWidth(text, 11, 600) + 14;
        let rx = x + 6;
        if (rx + w > R) rx = x - 6 - w;
        rx = Math.max(L, rx);
        svg.appendChild(h("rect", { class: "chart-pill", x: px(rx), y: T + 4, width: px(w), height: 20, rx: 5 }));
        svg.appendChild(h("text", { class: "chart-pill-text", x: px(rx + 7), y: T + 14, dy: "0.32em" }, text));
      }
    }

    attachLineHover(self, svg, { visible, sx, sy, L, R, T, B, xo, yValue, ciFormat, markers, x0, x1 });
    return svg;
  }

  function attachLineHover(self, svg, g) {
    const { visible, sx, sy, L, R, T, B, xo, yValue, ciFormat, markers } = g;
    const ux = unionX(visible);
    if (!ux.length) return;
    const cross = h("line", { class: "chart-cross", y1: T, y2: B, visibility: "hidden" });
    const dots = h("g", { "pointer-events": "none" });
    const hit = h("rect", { class: "chart-hit", x: L, y: T, width: Math.max(0, R - L), height: Math.max(0, B - T) });
    svg.append(cross, dots, hit);
    const xTip = xo.tooltipFormat || ((v) => (xo.format ? xo.format(v) : autoFormat(v)) + (xo.label ? " " + xo.label : ""));
    const tolerance = Math.max(8, 0.04 * (R - L));
    let current = -1;

    function show(k, pointerY) {
      current = k;
      const xv = ux[k];
      const x = sx(xv);
      cross.setAttribute("x1", px(x));
      cross.setAttribute("x2", px(x));
      cross.setAttribute("visibility", "visible");
      App.clear(dots);
      const rows = [];
      for (const e of visible) {
        const i = nearestIndex(e.sorted, xv, true);
        if (i < 0) continue;
        const p = e.sorted[i];
        if (Math.abs(sx(p[0]) - x) > tolerance) continue;
        dots.appendChild(h("circle", { class: "chart-dot", cx: px(sx(p[0])), cy: px(sy(p[1])), r: 4, style: { fill: e.color } }));
        let text = yValue(p[1]);
        if (e.bandSorted.length) {
          const b = e.bandSorted[nearestIndex(e.bandSorted, p[0], true)];
          if (b && b[0] === p[0]) {
            const half = (b[2] - b[1]) / 2;
            if (Math.abs((b[1] + b[2]) / 2 - p[1]) <= Math.max(1e-9, Math.abs(half) * 0.02)) text += " ± " + ciFormat(half);
            else text += ` (${yValue(b[1])} to ${yValue(b[2])})`;
          }
        }
        rows.push(tipRow(e.color, text, visible.length > 1 ? e.name : e.name || null, e.s.dashed ? "dashed" : "line"));
      }
      for (const m of markers) {
        if (Math.abs(sx(m.x) - x) > Math.max(6, tolerance / 2)) continue;
        rows.push(h("div", { class: "tt-row tt-wrap" },
          h("span", { class: "tt-key tt-key-marker", style: { background: toneColor(m.tone) } }),
          h("span", null, h("strong", null, m.label || ""), m.detail ? h("span", { class: "tt-name" }, " " + m.detail) : null)));
      }
      self.showTip([h("div", { class: "tt-head" }, xTip(xv)), ...rows], x, pointerY === undefined ? T + 30 : pointerY);
    }
    function hide() {
      cross.setAttribute("visibility", "hidden");
      App.clear(dots);
      self.hideTip();
    }
    svg.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      const x = event.clientX - box.left;
      const y = event.clientY - box.top;
      if (x < L - 10 || x > R + 10 || y < T - 16 || y > B + 10) { hide(); return; }
      show(nearestIndex(ux, sx.invert(clamp(x, L, R))), y);
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("focus", () => show(current >= 0 ? current : ux.length - 1));
    svg.addEventListener("blur", hide);
    svg.addEventListener("keydown", (event) => {
      let k = current < 0 ? ux.length - 1 : current;
      if (event.key === "ArrowLeft") k = Math.max(0, k - 1);
      else if (event.key === "ArrowRight") k = Math.min(ux.length - 1, k + 1);
      else if (event.key === "Home") k = 0;
      else if (event.key === "End") k = ux.length - 1;
      else if (event.key === "Escape") { hide(); return; } else return;
      event.preventDefault();
      show(k);
    });
  }

  function lineTable(self) {
    const o = self.opts;
    const xo = o.x || {};
    const yo = o.y || {};
    const visible = prepareLine(self);
    const ux = unionX(visible);
    if (!ux.length) return h("p", { class: "chart-table-empty" }, o.empty || "No data yet.");
    const yValue = yo.format || autoFormat;
    const ciFormat = yo.ciFormat || ((v) => yValue(Math.abs(v)).replace(/^\+/, ""));
    const xValue = xo.format || autoFormat;
    let indices = ux.map((_, i) => i);
    let note = null;
    const limit = 1000;
    if (ux.length > limit) {
      const every = Math.ceil(ux.length / limit);
      indices = indices.filter((i) => i % every === 0 || i === ux.length - 1);
      note = `Showing ${fmt.int(indices.length)} of ${fmt.int(ux.length)} rows, evenly spaced.`;
    }
    const values = visible.map((e) => new Map(e.valid.map((p) => [p[0], p[1]])));
    const bands = visible.map((e) => new Map(e.bandSorted.map((b) => [b[0], b])));
    const rows = indices.map((i) => {
      const x = ux[i];
      return h("tr", null, h("td", { class: "num" }, xValue(x)), visible.map((e, j) => {
        const v = values[j].get(x);
        if (!isNum(v)) return h("td", { class: "num muted" }, "–");
        const b = bands[j].get(x);
        return h("td", { class: "num" }, yValue(v) + (b ? " ± " + ciFormat((b[2] - b[1]) / 2) : ""));
      }));
    });
    return h("div", null, note ? h("div", { class: "chart-table-note" }, note) : null,
      h("table", { class: "table" },
        h("thead", null, h("tr", null, h("th", { class: "num" }, capitalize(xo.label || "x")),
          visible.map((e) => h("th", { class: "num" }, e.name + (yo.label && visible.length === 1 ? ` (${yo.label})` : ""))))),
        h("tbody", null, rows)));
  }

  function line(container, options) {
    return frame(container, options, "line", {
      draw: drawLine, legend: lineLegend, table: lineTable,
      names: (self) => lineEntries(self).map((e) => e.name),
    });
  }

  // ---- Bars -----------------------------------------------------------------------------

  function barEntries(self) {
    return (self.opts.series || []).map((s, i) => ({ s, i, name: s.name || `Series ${i + 1}`, color: colorOf(s, i) }));
  }

  function barsLegend(self) {
    const entries = barEntries(self);
    if (entries.length < 2) return [];
    return entries.map((e) => ({ name: e.name, color: e.color, key: "rect", toggle: true, active: !self.hidden.has(e.name) }));
  }

  function drawBars(self, width) {
    const o = self.opts;
    const yo = o.y || {};
    const categories = o.categories || [];
    const visible = barEntries(self).filter((e) => !self.hidden.has(e.name));
    const nv = Math.max(1, visible.length);
    let v0 = 0, v1 = 0, count = 0;
    for (const e of visible) {
      for (let c = 0; c < categories.length; c++) {
        const v = (e.s.values || [])[c];
        if (isNum(v)) { v0 = Math.min(v0, v); v1 = Math.max(v1, v); count++; }
      }
    }
    const empty = count === 0;
    if (!(v1 > v0)) v1 = v0 + 1;
    const value = yo.format || consistentFormat(visible.flatMap((e) => (e.s.values || []).filter(isNum)));
    const ctx = { self, o, yo, categories, visible, nv, v0, v1, empty, value };
    return o.horizontal ? drawBarsH(ctx, width) : drawBarsV(ctx, width);
  }

  function barHover(self, svg, groups, hits, categories, visible, value, place) {
    hits.forEach((hit, c) => {
      const enter = (event) => {
        groups.forEach((g, i) => g.forEach((bar) => bar.classList.toggle("is-hover", i === c)));
        const rows = visible.map((e) => {
          const v = (e.s.values || [])[c];
          return tipRow(e.color, isNum(v) ? value(v) : "–", visible.length > 1 || e.s.name ? e.name : null, "rect");
        });
        const box = svg.getBoundingClientRect();
        const pos = place(c, event && event.clientY !== undefined ? event.clientY - box.top : null);
        self.showTip([h("div", { class: "tt-head" }, categories[c]), ...rows], pos[0], pos[1]);
      };
      const leave = () => {
        groups.forEach((g) => g.forEach((bar) => bar.classList.remove("is-hover")));
        self.hideTip();
      };
      hit.addEventListener("pointerenter", enter);
      hit.addEventListener("pointermove", enter);
      hit.addEventListener("pointerleave", leave);
      hit.addEventListener("focus", () => enter(null));
      hit.addEventListener("blur", leave);
    });
  }

  function drawBarsV(ctx, width) {
    const { self, o, yo, categories, visible, nv, v0, v1, empty, value } = ctx;
    const height = o.height || 240;
    const showValues = !empty && categories.length * nv <= 24;
    const top = (yo.label ? 18 : 6) + (showValues ? 14 : 4);
    const provisionalBand = (width - 50) / Math.max(1, categories.length);
    const wraps = categories.some((c) => textWidth(c, 12) > provisionalBand - 8);
    const bottom = wraps ? 38 : 24;
    const plotH = Math.max(40, height - top - bottom);
    const yt = niceTicks(v0, v1, Math.max(2, Math.round(plotH / 50)), true);
    const tickFmt = pickTickFormat(yo, yt.ticks, yt.step);
    const left = Math.ceil(Math.max(8, ...yt.ticks.map((v) => textWidth(tickFmt(v))))) + 10;
    const L = left, R = width - 8, T = top, B = top + plotH;
    const sy = linear(yt.min, yt.max, B, T);
    const svg = h("svg", { class: "chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": (o.title || "Bar chart") + ". Use the Table button for the values." });
    axisLabelText(svg, yo.label, 0, 11);
    if (empty) { emptyFrame(svg, L, R, T, B, o.empty); return svg; }
    for (const v of yt.ticks) {
      const y = Math.round(sy(v)) + 0.5;
      if (v !== 0) svg.appendChild(h("line", { class: "chart-grid", x1: L, x2: R, y1: y, y2: y }));
      svg.appendChild(h("text", { class: "chart-tick", x: L - 8, y, dy: "0.32em", "text-anchor": "end" }, tickFmt(v)));
    }
    const base = Math.round(sy(0)) + 0.5;
    const band = (R - L) / Math.max(1, categories.length);
    const barW = Math.max(2, Math.min(24, (band * 0.7 - (nv - 1) * 2) / nv));
    const groupW = nv * barW + (nv - 1) * 2;
    let labelsFit = showValues;
    if (showValues) {
      for (const e of visible) {
        for (const v of e.s.values || []) {
          if (isNum(v) && textWidth(value(v), 11) > (nv === 1 ? band - 6 : barW + 8)) labelsFit = false;
        }
      }
    }
    const groups = [];
    const hits = [];
    const barLayer = h("g");
    const labelLayer = h("g");
    categories.forEach((cat, c) => {
      const gx = L + c * band + (band - groupW) / 2;
      const group = [];
      visible.forEach((e, j) => {
        const v = (e.s.values || [])[c];
        if (!isNum(v)) return;
        const x = gx + j * (barW + 2);
        const tipY = sy(v);
        const bar = h("path", { class: "chart-bar", d: barPathV(x, barW, base, tipY, 4), style: { fill: e.color } });
        barLayer.appendChild(bar);
        group.push(bar);
        if (labelsFit) {
          labelLayer.appendChild(h("text", { class: "chart-value", x: px(x + barW / 2), y: px(v >= 0 ? tipY - 6 : tipY + 15),
            "text-anchor": "middle" }, value(v)));
        }
      });
      groups.push(group);
      const cx = L + c * band + band / 2;
      const lines = textWidth(cat, 12) > band - 8 ? wrapText(cat, band - 8, 12).slice(0, 2) : [String(cat)];
      lines.forEach((text, i) => svg.appendChild(h("text", { class: "chart-cat", x: px(cx), y: B + 16 + i * 14,
        "text-anchor": "middle" }, ellipsize(text, band - 4, 12))));
      const hit = h("rect", { class: "chart-hit-bar", x: px(L + c * band), y: T, width: px(band), height: plotH,
        tabindex: "0", "aria-label": cat + ": " + visible.map((e) => `${e.name} ${value((e.s.values || [])[c])}`).join(", ") });
      hits.push(hit);
    });
    svg.append(barLayer, h("line", { class: "chart-axis", x1: L, x2: R, y1: base, y2: base }), labelLayer, ...hits);
    barHover(self, svg, groups, hits, categories, visible, value,
      (c, y) => [L + c * band + band / 2 + groupW / 2, y === null ? T + plotH / 2 : y]);
    return svg;
  }

  function drawBarsH(ctx, width) {
    const { self, o, yo, categories, visible, nv, v0, v1, empty, value } = ctx;
    const barW = nv === 1 ? 18 : 14;
    const rowH = Math.max(30, nv * barW + (nv - 1) * 2 + 12);
    const top = 4;
    const bottom = 22 + (yo.label ? 14 : 0);
    const height = o.height || top + Math.max(1, categories.length) * rowH + bottom;
    const plotH = height - top - bottom;
    const left = Math.ceil(Math.min(width * 0.4, Math.max(20, ...categories.map((c) => textWidth(c, 12))) + 14));
    const showValues = !empty && categories.length * nv <= 30;
    const valueRoom = showValues ? Math.ceil(Math.max(0, ...visible.flatMap((e) => (e.s.values || []).filter(isNum)
      .map((v) => textWidth(value(v), 11))))) + 12 : 10;
    const L = left, R = Math.max(left + 40, width - valueRoom), T = top, B = top + plotH;
    const svg = h("svg", { class: "chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": (o.title || "Bar chart") + ". Use the Table button for the values." });
    if (empty) { emptyFrame(svg, L, R, T, B, o.empty); return svg; }
    const xt = niceTicks(v0, v1, Math.max(2, Math.floor((R - L) / 80)), true);
    const tickFmt = pickTickFormat(yo, xt.ticks, xt.step);
    const sx = linear(xt.min, xt.max, L, R);
    for (const v of xt.ticks) {
      const x = Math.round(sx(v)) + 0.5;
      if (v !== 0) svg.appendChild(h("line", { class: "chart-grid", x1: x, x2: x, y1: T, y2: B }));
      svg.appendChild(h("text", { class: "chart-tick", x, y: B + 15, "text-anchor": "middle" }, tickFmt(v)));
    }
    axisLabelText(svg, yo.label, R, B + 30, "end");
    const base = Math.round(sx(0)) + 0.5;
    const groupH = nv * barW + (nv - 1) * 2;
    const groups = [];
    const hits = [];
    const barLayer = h("g");
    categories.forEach((cat, c) => {
      const rowTop = T + c * rowH;
      const gy = rowTop + (rowH - groupH) / 2;
      const group = [];
      visible.forEach((e, j) => {
        const v = (e.s.values || [])[c];
        if (!isNum(v)) return;
        const y = gy + j * (barW + 2);
        const bar = h("path", { class: "chart-bar", d: barPathH(base, sx(v), y, barW, 4), style: { fill: e.color } });
        barLayer.appendChild(bar);
        group.push(bar);
        if (showValues) {
          barLayer.appendChild(h("text", { class: "chart-value", x: px(v >= 0 ? sx(v) + 6 : sx(v) - 6), y: px(y + barW / 2),
            dy: "0.32em", "text-anchor": v >= 0 ? "start" : "end" }, value(v)));
        }
      });
      groups.push(group);
      svg.appendChild(h("text", { class: "chart-cat", x: L - 10, y: px(rowTop + rowH / 2), dy: "0.32em", "text-anchor": "end" },
        ellipsize(cat, L - 14, 12)));
      hits.push(h("rect", { class: "chart-hit-bar", x: 0, y: px(rowTop), width, height: rowH, tabindex: "0",
        "aria-label": cat + ": " + visible.map((e) => `${e.name} ${value((e.s.values || [])[c])}`).join(", ") }));
    });
    svg.append(barLayer, h("line", { class: "chart-axis", x1: base, x2: base, y1: T, y2: B }), ...hits);
    barHover(self, svg, groups, hits, categories, visible, value, (c) => [Math.min(width - 40, sx(Math.max(0, v1)) + 10), T + c * rowH + rowH / 2]);
    return svg;
  }

  function barsTable(self) {
    const o = self.opts;
    const value = (o.y || {}).format || autoFormat;
    const visible = barEntries(self).filter((e) => !self.hidden.has(e.name));
    return h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", null, ""), visible.map((e) => h("th", { class: "num" }, e.name)))),
      h("tbody", null, (o.categories || []).map((cat, c) => h("tr", null, h("td", null, cat),
        visible.map((e) => h("td", { class: "num" }, value((e.s.values || [])[c])))))));
  }

  function bars(container, options) {
    return frame(container, options, "bars", {
      draw: drawBars, legend: barsLegend, table: barsTable, names: (self) => barEntries(self).map((e) => e.name),
    });
  }

  // ---- Histogram ------------------------------------------------------------------------

  function binHistogram(self) {
    const o = self.opts;
    const visible = barEntries(self).filter((e) => !self.hidden.has(e.name));
    let lo = Infinity, hi = -Infinity;
    for (const e of visible) for (const v of e.s.values || []) if (isNum(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (!isFinite(lo)) return { empty: true, visible };
    if (isNum(o.center)) { lo = Math.min(lo, o.center); hi = Math.max(hi, o.center); }
    if (!(hi > lo)) { hi = lo + 1; lo -= 1; }
    const w = niceStep((hi - lo) / (o.bins || 30));
    const anchor = isNum(o.center) ? o.center : 0;
    const start = anchor + Math.floor((lo - anchor) / w + 1e-9) * w;
    const n = Math.max(1, Math.floor((hi - start) / w + 1e-9) + 1);
    for (const e of visible) {
      e.counts = new Array(n).fill(0);
      e.total = 0;
      for (const v of e.s.values || []) {
        if (!isNum(v)) continue;
        e.counts[clamp(Math.floor((v - start) / w + 1e-9), 0, n - 1)]++;
        e.total++;
      }
    }
    return { empty: false, visible, start, w, n };
  }

  function drawHistogram(self, width) {
    const o = self.opts;
    const xo = o.x || {};
    const label = (o.y || {}).label || "count";
    const height = o.height || 240;
    const bins = binHistogram(self);
    const centerLabels = Array.isArray(o.centerLabels) && isNum(o.center);
    const top = 18 + 6;
    const bottom = 22 + (xo.label ? 14 : 0);
    const plotH = Math.max(40, height - top - bottom);
    const svg = h("svg", { class: "chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": (o.title || "Histogram") + ". Use the Table button for the values." });
    axisLabelText(svg, label, 0, 11);
    if (bins.empty) { emptyFrame(svg, 12, width - 8, top, top + plotH, o.empty); return svg; }
    const { visible, start, w, n } = bins;
    const maxCount = Math.max(1, ...visible.map((e) => Math.max(...e.counts)));
    const yt = niceTicks(0, maxCount, Math.max(2, Math.round(plotH / 50)), true);
    const yFmt = tickFormatter(yt.ticks, Math.max(1, yt.step));
    const ticksY = yt.ticks.filter((v) => Number.isInteger(v) || yt.step >= 1);
    const left = Math.ceil(Math.max(8, ...ticksY.map((v) => textWidth(yFmt(v))))) + 10;
    const L = left, R = width - 10, T = top, B = top + plotH;
    const end = start + n * w;
    const sx = linear(start, end, L, R);
    const sy = linear(0, yt.max, B, T);
    for (const v of ticksY) {
      const y = Math.round(sy(v)) + 0.5;
      if (v !== 0) svg.appendChild(h("line", { class: "chart-grid", x1: L, x2: R, y1: y, y2: y }));
      svg.appendChild(h("text", { class: "chart-tick", x: L - 8, y, dy: "0.32em", "text-anchor": "end" }, yFmt(v)));
    }
    let xTicks = [];
    let xFmt = autoFormat;
    for (let k = Math.max(2, Math.floor((R - L) / 70)); k >= 1; k--) {
      const t = niceTicks(start, end, k, false);
      const f = pickTickFormat(xo, t.ticks, t.step);
      const widths = t.ticks.map((v) => textWidth(f(v)));
      let fits = true;
      for (let i = 1; i < t.ticks.length; i++) if (sx(t.ticks[i]) - sx(t.ticks[i - 1]) < (widths[i] + widths[i - 1]) / 2 + 12) fits = false;
      xTicks = t.ticks;
      xFmt = f;
      if (fits) break;
    }
    for (const v of xTicks) svg.appendChild(h("text", { class: "chart-tick", x: px(sx(v)), y: B + 15, "text-anchor": "middle" }, xFmt(v)));
    axisLabelText(svg, xo.label, R, B + 30, "end");

    const binW = (R - L) / n;
    if (visible.length === 1) {
      const e = visible[0];
      e.counts.forEach((c, i) => {
        if (!c) return;
        const gap = binW > 4 ? 1 : 0;
        const x = sx(start + i * w) + gap;
        svg.appendChild(h("path", { class: "chart-bar", d: barPathV(x, binW - gap * 2, sy(0), sy(c), Math.min(4, binW / 3)),
          style: { fill: e.color } }));
      });
    } else {
      for (const e of visible) {
        let d = `M${px(L)},${px(sy(0))}`;
        e.counts.forEach((c, i) => { d += `V${px(sy(c))}H${px(sx(start + (i + 1) * w))}`; });
        svg.appendChild(h("path", { class: "chart-wash", d: d + `V${px(sy(0))}Z`, style: { fill: e.color } }));
      }
      for (const e of visible) {
        let d = `M${px(L)},${px(sy(0))}`;
        e.counts.forEach((c, i) => { d += `V${px(sy(c))}H${px(sx(start + (i + 1) * w))}`; });
        d += `V${px(sy(0))}`;
        svg.appendChild(h("path", { class: "chart-line", d, style: { stroke: e.color } }));
      }
    }
    svg.appendChild(h("line", { class: "chart-axis", x1: L, x2: R, y1: Math.round(B) + 0.5, y2: Math.round(B) + 0.5 }));
    if (isNum(o.center) && o.center >= start && o.center <= end) {
      const x = Math.round(sx(o.center)) + 0.5;
      svg.appendChild(h("line", { class: "chart-ref", x1: x, x2: x, y1: centerLabels ? 3 : T - 6, y2: B }));
      if (centerLabels) {
        // In the label row above the plot, so the tallest bars never run into them.
        const leftText = "← " + o.centerLabels[0];
        const rightText = o.centerLabels[1] + " →";
        if (o.centerLabels[0] && x - 7 - textWidth(leftText) > textWidth(label) + 12) {
          svg.appendChild(h("text", { class: "chart-ref-label", x: x - 7, y: 11, "text-anchor": "end" }, leftText));
        }
        if (o.centerLabels[1] && x + 7 + textWidth(rightText) < width) {
          svg.appendChild(h("text", { class: "chart-ref-label", x: x + 7, y: 11 }, rightText));
        }
      }
    }

    const hover = h("rect", { class: "chart-hover-col", x: 0, y: T, width: binW, height: plotH, visibility: "hidden" });
    svg.appendChild(hover);
    const rangeFmt = xo.format || tickFormatter([start, end], w);
    const hit = h("rect", { class: "chart-hit", x: L, y: T, width: R - L, height: plotH, tabindex: "0",
      "aria-label": "Histogram bins; use the arrow keys to read each bin." });
    svg.appendChild(hit);
    let current = -1;
    const show = (i, y) => {
      current = i;
      hover.setAttribute("x", px(L + i * binW));
      hover.setAttribute("visibility", "visible");
      const a = start + i * w;
      const rows = visible.map((e) => tipRow(e.color, fmt.int(e.counts[i]) +
        (e.total ? ` (${(100 * e.counts[i] / e.total).toFixed(1)}%)` : ""), visible.length > 1 ? e.name : label, visible.length > 1 ? "line" : "rect"));
      self.showTip([h("div", { class: "tt-head" }, `${rangeFmt(a)} to ${rangeFmt(a + w)}`), ...rows], L + (i + 1) * binW, y);
    };
    const hide = () => { hover.setAttribute("visibility", "hidden"); self.hideTip(); };
    hit.addEventListener("pointermove", (event) => {
      const box = svg.getBoundingClientRect();
      show(clamp(Math.floor((event.clientX - box.left - L) / binW), 0, n - 1), event.clientY - box.top);
    });
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("focus", () => show(current < 0 ? Math.floor(n / 2) : current, T + plotH / 2));
    hit.addEventListener("blur", hide);
    hit.addEventListener("keydown", (event) => {
      let i = current < 0 ? 0 : current;
      if (event.key === "ArrowLeft") i = Math.max(0, i - 1);
      else if (event.key === "ArrowRight") i = Math.min(n - 1, i + 1);
      else return;
      event.preventDefault();
      show(i, T + plotH / 2);
    });
    return svg;
  }

  function histogramLegend(self) {
    const entries = barEntries(self);
    if (entries.length < 2) return [];
    return entries.map((e) => ({ name: e.name, color: e.color, key: "band", toggle: true, active: !self.hidden.has(e.name) }));
  }

  function histogramTable(self) {
    const o = self.opts;
    const bins = binHistogram(self);
    if (bins.empty) return h("p", { class: "chart-table-empty" }, o.empty || "No data yet.");
    const rangeFmt = (o.x || {}).format || tickFormatter([bins.start, bins.start + bins.n * bins.w], bins.w);
    const rows = [];
    for (let i = 0; i < bins.n; i++) {
      const a = bins.start + i * bins.w;
      rows.push(h("tr", null, h("td", { class: "num" }, `${rangeFmt(a)} to ${rangeFmt(a + bins.w)}`),
        bins.visible.map((e) => h("td", { class: "num" }, fmt.int(e.counts[i])))));
    }
    return h("table", { class: "table" },
      h("thead", null, h("tr", null, h("th", { class: "num" }, capitalize((o.x || {}).label || "bin")),
        bins.visible.map((e) => h("th", { class: "num" }, e.name || "Count")))),
      h("tbody", null, rows));
  }

  function histogram(container, options) {
    return frame(container, options, "histogram", {
      draw: drawHistogram, legend: histogramLegend, table: histogramTable,
      names: (self) => barEntries(self).map((e) => e.name),
    });
  }

  // ---- Intervals: one estimate per row with its 95% interval, on a shared scale ---------

  function drawIntervals(self, width) {
    const o = self.opts;
    const rows = (o.rows || []).filter((r) => isNum(r.value));
    const value = o.format || ((v) => fmt.signed(v));
    const ci = (v) => Math.abs(v).toFixed(1);
    const center = isNum(o.center) ? o.center : 0;
    const color = o.color || "var(--series-1)";
    const rowH = 40;
    const axisH = 34;
    const height = Math.max(1, rows.length) * rowH + axisH;
    const svg = h("svg", { class: "chart-svg", width, height, viewBox: `0 0 ${width} ${height}`, role: "img",
      "aria-label": (o.title || "Estimates with 95% intervals") });
    if (!rows.length) {
      svg.setAttribute("height", 90);
      svg.setAttribute("viewBox", `0 0 ${width} 90`);
      svg.appendChild(h("text", { class: "chart-empty-text", x: width / 2, y: 45, dy: "0.32em", "text-anchor": "middle" },
        o.empty || "Nothing to show yet"));
      return svg;
    }
    const labelW = Math.min(width * 0.42, Math.max(...rows.map((r) => Math.max(textWidth(r.label, 13, 550),
      r.sub ? textWidth(r.sub, 11) : 0))) + 16);
    const valueTexts = rows.map((r) => value(r.value) + (isNum(r.ci) ? " ± " + ci(r.ci) : ""));
    const valueW = Math.max(...valueTexts.map((t) => textWidth(t, 13, 600))) + 16;
    const L = labelW + 8;
    const R = width - valueW - 8;
    let lo = center, hi = center;
    for (const r of rows) {
      const half = isNum(r.ci) ? r.ci : 0;
      lo = Math.min(lo, r.value - half);
      hi = Math.max(hi, r.value + half);
    }
    const pad = (hi - lo) * 0.06 || 1;
    lo -= pad;
    hi += pad;
    const sx = linear(lo, hi, L, R);
    const cx = Math.round(sx(center)) + 0.5;
    const plotB = rows.length * rowH;
    svg.appendChild(h("line", { class: "chart-ref", x1: cx, x2: cx, y1: 2, y2: plotB + 4 }));
    rows.forEach((r, i) => {
      const y = i * rowH;
      const mid = y + rowH / 2;
      const g = h("g", { class: "chart-row" + (r.href ? " is-link" : ""), tabindex: "0",
        role: r.href ? "link" : null, "aria-label": `${r.label}: ${valueTexts[i]}` });
      g.appendChild(h("rect", { class: "chart-row-hit", x: 0, y: y + 2, width, height: rowH - 4, rx: 6 }));
      g.appendChild(h("text", { class: "chart-row-label", x: 8, y: r.sub ? mid - 6 : mid, dy: "0.32em" },
        ellipsize(r.label, labelW - 12, 13, 550)));
      if (r.sub) g.appendChild(h("text", { class: "chart-row-sub", x: 8, y: mid + 9, dy: "0.32em" }, ellipsize(r.sub, labelW - 12, 11)));
      if (i < rows.length - 1) g.appendChild(h("line", { class: "chart-grid", x1: 8, x2: width - 8, y1: y + rowH + 0.5, y2: y + rowH + 0.5 }));
      if (isNum(r.ci)) {
        g.appendChild(h("line", { class: "chart-line", x1: px(sx(r.value - r.ci)), x2: px(sx(r.value + r.ci)), y1: mid, y2: mid,
          style: { stroke: color } }));
      }
      g.appendChild(h("circle", { class: "chart-dot", cx: px(sx(r.value)), cy: mid, r: 4.5, style: { fill: color } }));
      g.appendChild(h("text", { class: "chart-row-value", x: width - 8, y: mid, dy: "0.32em", "text-anchor": "end" }, valueTexts[i]));
      const show = () => {
        const nodes = [h("div", { class: "tt-head" }, r.label),
          tipRow(color, valueTexts[i], o.valueLabel || null)];
        if (isNum(r.ci)) nodes.push(h("div", { class: "tt-note" }, `95% interval: ${value(r.value - r.ci)} to ${value(r.value + r.ci)}`));
        if (r.detail) nodes.push(h("div", { class: "tt-note" }, r.detail));
        self.showTip(nodes, sx(r.value), mid);
      };
      g.addEventListener("pointerenter", show);
      g.addEventListener("focus", show);
      g.addEventListener("pointerleave", () => self.hideTip());
      g.addEventListener("blur", () => self.hideTip());
      if (r.href) {
        g.addEventListener("click", () => App.go(r.href));
        g.addEventListener("keydown", (event) => { if (event.key === "Enter") App.go(r.href); });
      }
      svg.appendChild(g);
    });
    const t = niceTicks(lo, hi, Math.max(2, Math.floor((R - L) / 70)), false);
    const tf = o.tickFormat || ((v) => (v === 0 ? "0" : fmt.signed(v, t.step < 1 ? 1 : 0)));
    for (const v of t.ticks) {
      const x = sx(v);
      if (Math.abs(x - cx) > 2 && Math.abs(x - cx) < 18) continue;
      svg.appendChild(h("text", { class: "chart-tick", x: px(x), y: plotB + 16, "text-anchor": "middle" }, tf(v)));
    }
    if (o.axisLabel) svg.appendChild(h("text", { class: "chart-axis-label", x: R, y: plotB + 30, "text-anchor": "end" }, o.axisLabel));
    if (o.centerLabel) svg.appendChild(h("text", { class: "chart-axis-label", x: L, y: plotB + 30 }, o.centerLabel));
    return svg;
  }

  function intervals(container, options) {
    return frame(container, Object.assign({ table: false }, options), "intervals", { draw: drawIntervals });
  }

  // ---- Sparkline ------------------------------------------------------------------------

  function sparkline(container, options) {
    injectStyles();
    const self = { opts: Object.assign({}, options), destroyed: false };
    const wrap = h("span", { class: "spark" });
    const tip = h("span", { class: "chart-tooltip", "aria-hidden": "true" });
    tip.hidden = true;
    App.clear(container).appendChild(wrap);

    function render() {
      if (self.destroyed) return;
      const o = self.opts;
      const w = o.width || 120;
      const ht = o.height || 32;
      const color = o.color || "var(--series-1)";
      const points = (o.points || []).filter((p) => p && isNum(p[0]) && isNum(p[1]));
      const band = (o.band || []).filter((b) => b && isNum(b[0]) && isNum(b[1]) && isNum(b[2]));
      const refs = isNum(o.referenceY) ? [o.referenceY] : [];
      const svg = h("svg", { width: w, height: ht, viewBox: `0 0 ${w} ${ht}`, role: "img",
        "aria-label": o.label || (points.length ? `Trend, latest ${(o.format || autoFormat)(points[points.length - 1][1])}` : "No data yet") });
      if (!points.length) {
        svg.appendChild(h("line", { class: "chart-grid", x1: 0, x2: w, y1: Math.round(ht / 2) + 0.5, y2: Math.round(ht / 2) + 0.5 }));
        wrap.replaceChildren(svg);
        return;
      }
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const p of points) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
      for (const b of band) { y0 = Math.min(y0, b[1]); y1 = Math.max(y1, b[2]); }
      for (const r of refs) { y0 = Math.min(y0, r); y1 = Math.max(y1, r); }
      if (!(x1 > x0)) { x0 -= 1; x1 += 1; }
      if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
      const pad = 4;
      const sx = linear(x0, x1, pad, w - pad);
      const sy = linear(y0, y1, ht - pad, pad);
      for (const r of refs) {
        const y = Math.round(sy(r)) + 0.5;
        svg.appendChild(h("line", { class: "chart-ref", x1: 0, x2: w, y1: y, y2: y }));
      }
      if (band.length > 1) {
        let d = "";
        band.forEach((b, i) => { d += (i ? "L" : "M") + px(sx(b[0])) + "," + px(sy(b[2])); });
        for (let i = band.length - 1; i >= 0; i--) d += "L" + px(sx(band[i][0])) + "," + px(sy(band[i][1]));
        svg.appendChild(h("path", { class: "chart-band", d: d + "Z", style: { fill: color } }));
      }
      if (points.length > 1) {
        svg.appendChild(h("path", { class: "chart-line", d: pathFrom(points.map((p) => [sx(p[0]), sy(p[1])])),
          style: { stroke: color } }));
      }
      const last = points[points.length - 1];
      svg.appendChild(h("circle", { class: "chart-dot", cx: px(sx(last[0])), cy: px(sy(last[1])), r: 3.5, style: { fill: color } }));
      const dot = h("circle", { class: "chart-dot", r: 3.5, visibility: "hidden", style: { fill: color } });
      svg.appendChild(dot);
      svg.addEventListener("pointermove", (event) => {
        const box = svg.getBoundingClientRect();
        const p = points.slice().sort((a, b) => Math.abs(sx(a[0]) - (event.clientX - box.left)) - Math.abs(sx(b[0]) - (event.clientX - box.left)))[0];
        dot.setAttribute("cx", px(sx(p[0])));
        dot.setAttribute("cy", px(sy(p[1])));
        dot.setAttribute("visibility", "visible");
        tip.textContent = (o.xFormat ? o.xFormat(p[0]) + ": " : "") + (o.format || autoFormat)(p[1]);
        tip.hidden = false;
        const tw = tip.offsetWidth;
        tip.style.left = Math.round(clamp(sx(p[0]) - tw / 2, -20, w - tw + 20)) + "px";
        tip.style.top = Math.round(-tip.offsetHeight - 6) + "px";
      });
      svg.addEventListener("pointerleave", () => { dot.setAttribute("visibility", "hidden"); tip.hidden = true; });
      wrap.replaceChildren(svg, tip);
    }

    render();
    const offTheme = App.on("theme", () => { if (!wrap.isConnected) destroy(); else render(); });
    function destroy() {
      if (self.destroyed) return;
      self.destroyed = true;
      offTheme();
    }
    App.onCleanup(destroy);
    return {
      update(next) { Object.assign(self.opts, next || {}); render(); },
      destroy,
    };
  }

  // ---- Stat tile ------------------------------------------------------------------------

  function statTile({ label, value, hint, delta, deltaTone = "neutral" } = {}) {
    let deltaNode = null;
    if (delta !== undefined && delta !== null && delta !== "") {
      const text = typeof delta === "number"
        ? (delta > 0 ? "+" : delta < 0 ? MINUS : "") + autoFormat(Math.abs(delta))
        : String(delta);
      const up = typeof delta === "number" ? delta > 0 : /^\+/.test(text);
      const down = typeof delta === "number" ? delta < 0 : /^[-−]/.test(text);
      deltaNode = h("div", { class: "stat-delta stat-delta-" + deltaTone },
        up ? h("span", { html: ICON_UP }) : down ? h("span", { html: ICON_DOWN }) : null, text);
    }
    injectStyles();
    return h("div", { class: "stat" },
      h("div", { class: "stat-label" }, label),
      h("div", { class: "stat-value", title: typeof value === "string" ? value : null },
        value === null || value === undefined ? "–" : value),
      deltaNode,
      hint ? h("div", { class: "stat-hint" }, hint) : null);
  }

  return {
    line, bars, histogram, intervals, sparkline, statTile,
    // Helpers other pages may find useful.
    niceTicks, textWidth, autoFormat,
  };
})();
