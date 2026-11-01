// Deterministic harness for static/ui/film.js: node only, no network, no browser.
// Fakes the DOM pieces the player touches (canvas, fetch with a streamed body, Blob, Image.decode, matchMedia,
// navigator.connection, rAF, scroll) and runs the real script in a fresh vm context per scenario.
// Usage: node film_harness.js <public film config JSON file> <bundle plan JSON file> <served anchors JSON file>
// (tests/test_film_js.py writes all three; the anchors are the shipped render's, as tools/film_frames.py serves them)
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const SRC = fs.readFileSync(path.join(__dirname, "..", "..", "static", "ui", "film.js"), "utf8");
const CFG = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const PLAN = JSON.parse(fs.readFileSync(process.argv[3], "utf8")); // [[bundle name, [frame numbers]], ...]
const ANCHORS = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const N = CFG.frames;
require("v8").setFlagsFromString("--expose-gc");
const gcFn = vm.runInNewContext("gc");

// A bundle as tools/film_frames.py writes it; each fake webp is a RIFF/WEBP envelope whose body starts with its
// frame number (uint16 LE, at byte 12). lenDelta lies about the stated record length (a damaged bundle).
function webp(n) {
  const body = Buffer.from([n & 255, n >> 8, 0xaa, 0xbb]);
  const head = Buffer.alloc(12);
  head.write("RIFF", 0);
  head.writeUInt32LE(4 + body.length, 4);
  head.write("WEBP", 8);
  return Buffer.concat([head, body]);
}
function record(n, lenDelta = 0) {
  const body = webp(n);
  const head = Buffer.alloc(6);
  head.writeUInt16LE(n, 0);
  head.writeUInt32LE(body.length + lenDelta, 2);
  return Buffer.concat([head, body]);
}
const FRAME = (b) => b[12] + b[13] * 256;
const BYTES = Object.fromEntries(PLAN.map(([name, nums]) => [name, new Uint8Array(Buffer.concat(nums.map((n) => record(n))))]));
const tick = () => new Promise((r) => setImmediate(r));
const settle = async () => { for (let i = 0; i < 6; i++) await tick(); };

function el(id) {
  const classes = new Set();
  const attrs = {};
  return {
    id, textContent: "", children: [], style: { setProperty(k, v) { this[k] = v; } },
    appendChild(c) { this.children.push(c); return c; },
    classList: {
      add: (c) => classes.add(c), remove: (c) => classes.delete(c), contains: (c) => classes.has(c),
      toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
    },
    setAttribute: (k, v) => { attrs[k] = String(v); }, getAttribute: (k) => (k in attrs ? attrs[k] : null),
  };
}

// The stage geometry static/site.css gives the art: wide = right edge, min(56rem, 62vw) wide, centred; phone =
// phone_art_width_vw wide, centred, in flow under the opening/chapter copy (stage padding 24 px + the copy block).
const PHONE_COPY_PX = 300;  // shortcut: fixed copy-block height stands in for real flow, real-browser shots cover overlap
function artBox(layout, W, H) {
  if (layout === "phone") {
    const w = (CFG.phone_art_width_vw * W) / 100, h = (w * 9) / 16;
    return { left: (W - w) / 2, top: 24 + PHONE_COPY_PX, width: w, height: h };
  }
  const w = Math.min(56 * 16, 0.62 * W), h = (w * 9) / 16;
  return { left: W - w, top: H / 2 - h / 2, width: w, height: h };
}
// A label's rendered height: 7 px per character at 13 px type, wrapped at its max-width, 16 px a line.
const CHAR_PX = 7, LINE_PX = 16;
const textWidth = (t) => t.textContent.length * CHAR_PX;

function run({ width = 390, height = 844, dpr = 3, reduced = false, saveData = false, badFrames = [], bitmaps = false, slowDecode = false, bytes = {}, anchors = "ok", anchorFails = 0, plain = false } = {}) {
  const BUNDLES = { ...BYTES, ...bytes };
  const h = { fetches: [], draws: [], clears: 0, raf: [], listeners: {}, scroll: 0, inFlight: 0, maxInFlight: 0, styleWrites: 0, urls: 0, live: new Set(), blobRefs: [], allocs: 0 };
  const ids = ["film", "film-config", "film-stage", "film-canvas", "film-poster", "film-open", "film-art", "film-labels"];
  const els = Object.fromEntries(ids.map((i) => [i, el(i)]));
  h.anchorFetches = [];
  h.aborts = [];
  h.timers = [];
  // Runs the waits the page asked for, in order, letting each try settle; returns the delays it waited.
  h.waits = async () => {
    const out = [];
    for (let i = 0; i < 50 && h.timers.length; i++) { const t = h.timers.shift(); out.push(t.ms); t.fn(); await settle(); }
    return out;
  };
  const made = (tag) => {
    const e = el(tag);
    e.tag = tag;
    Object.defineProperty(e, "offsetHeight", {
      get() { const mw = parseFloat(e.style.maxWidth) || Infinity; return Math.max(1, Math.ceil(textWidth(e) / mw)) * LINE_PX; },
    });
    return e;
  };
  els["film-config"].textContent = JSON.stringify(CFG);
  const chaps = CFG.chapters.map((_, i) => el("chap" + i));
  els["film-stage"].querySelectorAll = () => chaps;
  const ctx = { drawImage: (im) => h.draws.push(im.frame), clearRect: () => h.clears++, fillRect: () => h.clears++ };
  els["film-canvas"].width = 300;
  els["film-canvas"].height = 150;
  els["film-canvas"].getContext = () => ctx;
  for (const e of [els["film-open"], ...chaps]) {
    e.style = new Proxy({}, { set: (o, k, v) => { h.styleWrites++; o[k] = v; return true; } });
  }
  const blobs = {};
  const g = {
    innerWidth: width, innerHeight: height, devicePixelRatio: dpr,
    navigator: { connection: saveData ? { saveData: true } : undefined },
    matchMedia: (q) => ({ matches: reduced && /prefers-reduced-motion:\s*reduce/.test(q) }),
    document: { getElementById: (i) => els[i] || null, createElement: made, createElementNS: (_, tag) => made(tag) },
    requestAnimationFrame: (cb) => { h.raf.push(cb); return h.raf.length; },
    cancelAnimationFrame: () => { h.raf.length = 0; },
    removeEventListener: (type, cb) => { h.listeners[type] = (h.listeners[type] || []).filter((x) => x !== cb); },
    setTimeout: (fn, ms) => { const t = { fn, ms }; h.timers.push(t); return t; },
    clearTimeout: (t) => { const i = h.timers.indexOf(t); if (i >= 0) h.timers.splice(i, 1); },
    AbortController: class { constructor() { this.signal = { aborted: false }; h.aborts.push(this.signal); } abort() { this.signal.aborted = true; } },
    addEventListener: (type, cb) => { (h.listeners[type] = h.listeners[type] || []).push(cb); },
    Uint8Array: class extends Uint8Array { constructor(...a) { super(...a); h.allocs++; } }, Promise, Error,
    Blob: class { constructor(parts) { this.bytes = parts[0].slice(); h.blobRefs.push(new WeakRef(this)); } },
    URL: { createObjectURL: (b) => { const u = "blob:" + (h.urls++); blobs[u] = b; h.live.add(u); return u; }, revokeObjectURL: (u) => { h.live.delete(u); delete blobs[u]; } },
    Image: class {
      set src(u) { const b = blobs[u].bytes; this.frame = FRAME(b); this._src = u; }
      get src() { return this._src; }
      decode() {
        if (badFrames.includes(this.frame)) return Promise.reject(new Error("EncodingError"));
        this.naturalWidth = 900; this.naturalHeight = 506;
        return Promise.resolve();
      }
    },
    // A fetch whose body the scenario feeds by hand: f.send(bytes) streams the next bytes, f.fail(status) refuses it.
    fetch: (url, opts) => new Promise((resolve) => {
      const name = /\/([^/?]+)\?v=/.exec(url)[1];
      if (name === CFG.anchors) {
        h.anchorFetches.push(url);
        if (anchors === "missing" || h.anchorFetches.length <= anchorFails) return resolve({ ok: false, status: 404 });
        const body = anchors === "ok" ? ANCHORS : anchors; // a scenario may pass a broken file
        return resolve({ ok: true, status: 200, json: () => Promise.resolve(JSON.parse(JSON.stringify(body))) });
      }
      const f = { url, name, signal: opts && opts.signal, sent: 0, waiting: null, queued: [], ended: false };
      h.fetches.push(f);
      h.inFlight++;
      h.maxInFlight = Math.max(h.maxInFlight, h.inFlight);
      const read = () => new Promise((r) => {
        if (f.queued.length) r(f.queued.shift()); else f.waiting = r;
      });
      const push = (x) => { if (f.waiting) { const w = f.waiting; f.waiting = null; w(x); } else f.queued.push(x); };
      f.send = (k = Infinity) => {
        f.respond();
        const all = BUNDLES[name];
        const part = all.subarray(f.sent, Math.min(all.length, f.sent + k));
        f.sent += part.length;
        if (part.length) push({ done: false, value: part });
        if (f.sent >= all.length && !f.ended) { f.ended = true; h.inFlight--; push({ done: true }); }
      };
      f.fail = (status = 503) => { f.ended = true; h.inFlight--; resolve({ ok: false, status }); };
      // The response (headers) arrives with the first bytes; until then fail() can still refuse it.
      f.respond = () => resolve({ ok: true, status: 200, body: { getReader: () => ({ read }) } });
      // plain: a response with no body.getReader(): the headers arrive at once, the whole body only at f.finish().
      if (plain) {
        let done, fail; const body = new Promise((r, j) => { done = r; fail = j; });
        resolve({ ok: true, status: 200, arrayBuffer: () => body });
        f.finish = () => { f.ended = true; h.inFlight--; done(BUNDLES[name].slice().buffer); };
        f.fail = () => { f.ended = true; h.inFlight--; fail(new Error("503")); };
        f.send = () => f.finish();
      }
    }),
  };
  if (bitmaps) {
    h.bitmaps = new Set();
    h.slow = [];
    h.calls = 0;
    h.peakPending = 0;
    h.peakLive = 0;
    g.createImageBitmap = (blob) => {
      h.calls++;
      const b = { frame: FRAME(blob.bytes), width: 900, height: 506, close() { h.bitmaps.delete(b); b.closed = true; } };
      const born = () => { h.bitmaps.add(b); h.peakLive = Math.max(h.peakLive, h.bitmaps.size); return b; };
      if (!slowDecode) return Promise.resolve(born());
      return new Promise((resolve) => {
        h.slow.push(() => resolve(born()));
        h.peakPending = Math.max(h.peakPending, h.slow.length);
      });
    };
    // Finishes every decode started so far (and any started by their completion, until none is left).
    h.finishDecodes = async () => { for (let i = 0; i < 50 && h.slow.length; i++) { h.slow.splice(0).forEach((f) => f()); await settle(); } };
  }
  h.gc = () => { for (let i = 0; i < 3; i++) gcFn(); };
  g.window = g;
  const filmHeight = () => (CFG.height_vh * g.innerHeight) / 100;
  els.film.getBoundingClientRect = () => ({ top: -h.scroll, height: filmHeight() });
  els["film-labels"].getBoundingClientRect = () => ({ left: 0, top: 0, width: g.innerWidth, height: g.innerHeight });
  els["film-art"].getBoundingClientRect = () =>
    artBox(els.film.getAttribute("data-layout"), g.innerWidth, g.innerHeight);
  // The labels as drawn: name, side, on, top (px), left edge and right edge of the text, and the leader line.
  h.labels = () => {
    const svg = els["film-labels"].children[0];
    return els["film-labels"].children.slice(1).map((t, i) => {
      const ln = svg.children[i], mw = parseFloat(t.style.maxWidth), w = Math.min(textWidth(t), mw);
      const left = t.classList.contains("is-left") ? g.innerWidth - parseFloat(t.style.right) - w : parseFloat(t.style.left);
      const num = (k) => Number(ln.getAttribute(k));
      return {
        name: t.textContent, left, right: left + w, maxWidth: mw, top: parseFloat(t.style.top), h: t.offsetHeight,
        side: t.classList.contains("is-left") ? "left" : "right", on: t.classList.contains("on"),
        lineOn: ln.getAttribute("class") === "on", line: [num("x1"), num("y1"), num("x2"), num("y2")],
      };
    });
  };

  h.els = els;
  h.global = g;
  h.open = () => h.fetches.filter((f) => !f.ended);
  // Streams every open bundle to the end, round after round, until nothing is left to fetch.
  h.loadAll = async () => {
    for (let i = 0; i < 100; i++) {
      await tick();
      const open = h.open();
      if (!open.length) break;
      open.forEach((f) => f.send());
    }
    await tick();
  };
  h.flush = () => { for (let i = 0; h.raf.length && i < 10000; i++) h.raf.shift()(); assert.strictEqual(h.raf.length, 0); };
  h.scrollToFrame = (f) => {
    h.scroll = ((f - 1) / (N - 1)) * (filmHeight() - g.innerHeight);
    (h.listeners.scroll || []).forEach((cb) => cb());
    h.flush();
  };
  vm.runInNewContext(SRC, g, { filename: "film.js" });
  return h;
}

const scenarios = {
  async "reduced motion fetches nothing"() {
    const h = run({ reduced: true });
    await tick();
    assert.strictEqual(h.fetches.length, 0);
    assert.ok(!h.els.film.classList.contains("is-live"));
  },
  async "Save-Data fetches nothing"() {
    const h = run({ saveData: true, width: 1440, dpr: 1 });
    await tick();
    assert.strictEqual(h.fetches.length, 0);
    assert.ok(!h.els.film.classList.contains("is-live"));
  },
  async "the set is chosen once; the layout follows the css-width breakpoints"() {
    const phone = CFG.sets[0], wide = CFG.sets[CFG.sets.length - 1];
    const h = run({ width: 390, dpr: 3 });
    assert.strictEqual(h.els.film.getAttribute("data-layout"), phone.layout);
    h.global.innerWidth = 1440;
    h.global.devicePixelRatio = 1;
    (h.listeners.resize || []).forEach((cb) => cb());
    await h.loadAll();
    assert.strictEqual(h.fetches.length, PLAN.length);
    assert.ok(h.fetches.every((f) => f.url.includes("/" + phone.dir + "/")), "every fetch stays on the first set");
    assert.strictEqual(h.els.film.getAttribute("data-layout"), phone.layout);
    const d = run({ width: 1440, dpr: 1 });
    assert.ok(d.fetches.length && d.fetches.every((f) => f.url.includes("/" + wide.dir + "/")));
    assert.strictEqual(d.els.film.getAttribute("data-layout"), wide.layout);
    // css width over the limit, device pixels under it: the first set's frames, the wide composition.
    const mid = run({ width: phone.max_css_width + 10, dpr: 1 });
    assert.ok(phone.max_css_width + 10 <= phone.max_device_width);
    assert.ok(mid.fetches[0].url.includes("/" + phone.dir + "/"));
    assert.strictEqual(mid.els.film.getAttribute("data-layout"), wide.layout);
    // at the breakpoint itself: phone composition and phone frames together.
    const edge = run({ width: phone.max_css_width, dpr: 1 });
    assert.ok(edge.fetches[0].url.includes("/" + phone.dir + "/"));
    assert.strictEqual(edge.els.film.getAttribute("data-layout"), phone.layout);
  },
  async "bundles load in plan order, a few at a time, and bring every frame once"() {
    assert.deepStrictEqual(CFG.passes, [8, 4, 1]);
    assert.deepStrictEqual(CFG.bundles, PLAN.map(([name]) => name));
    const flat = PLAN.flatMap(([, nums]) => nums);
    assert.strictEqual(flat.length, N);
    assert.strictEqual(new Set(flat).size, N);
    const passOf = (n) => (n === N || n % 8 === 1 ? 0 : n % 4 === 1 ? 1 : 2);
    assert.deepStrictEqual(flat.map(passOf), flat.map(passOf).sort((a, b) => a - b), "no later pass before an earlier one");
    assert.ok(PLAN.length * 20 < N, "a handful of requests, not one per frame");
    const h = run({ width: 1440, dpr: 1 });
    assert.strictEqual(h.fetches.length, CFG.parallel_loads);
    await h.loadAll();
    assert.deepStrictEqual(h.fetches.map((f) => f.name), CFG.bundles);
    assert.ok(h.maxInFlight <= CFG.parallel_loads && h.maxInFlight > 1);
    assert.ok(h.fetches[0].url.endsWith("/" + CFG.bundles[0] + "?v=" + CFG.version), h.fetches[0].url);
    h.scrollToFrame(N);
    await settle();
    assert.strictEqual(h.draws[h.draws.length - 1], N);
  },
  async "a frame draws as soon as its record is in, before its bundle ends"() {
    const h = run({ width: 390, dpr: 3 });
    await tick();
    const first = h.fetches[0];
    assert.strictEqual(PLAN[0][1][0], 1);
    first.send(8); // record header (6) plus 2 of the 16 payload bytes: frame 1 is not whole yet
    await tick();
    assert.deepStrictEqual(h.draws, []);
    assert.ok(!h.els.film.classList.contains("has-frame"));
    first.send(14); // the rest of frame 1, and the bundle is still open
    await tick();
    assert.deepStrictEqual(h.draws, [1]);
    assert.ok(!first.ended);
  },
  async "a fast scroll draws the nearest loaded frame"() {
    const h = run({ width: 390, dpr: 3 });
    await tick();
    h.fetches[0].send(); // the first pass only: every 8th frame plus the last
    for (let i = 0; i < 5; i++) await tick();
    const loaded = new Set(PLAN[0][1]);
    assert.ok(loaded.has(97) && loaded.has(105) && !loaded.has(102));
    assert.deepStrictEqual(h.draws, [1]);
    // 105 is never passed on the way to 102, so only a nearest-frame lookup draws it.
    h.scrollToFrame(102);
    await settle();
    assert.strictEqual(h.draws[h.draws.length - 1], 105, "nearest loaded frame to 102");
    assert.ok(h.draws.every((f) => loaded.has(f)), "only loaded frames are drawn");
    assert.ok(h.draws.length < 20, "frames in between are skipped, not stepped through");
  },
  async "a failed bundle or a frame that will not decode does not blank the canvas"() {
    const h = run({ width: 390, dpr: 3, badFrames: [9] });
    await tick();
    h.fetches[0].send();
    for (let i = 0; i < 5; i++) await tick();
    assert.strictEqual(h.draws[0], 1);
    assert.ok(h.els.film.classList.contains("has-frame"));
    const w = h.els["film-canvas"].width;
    h.scrollToFrame(9);
    assert.ok(!h.draws.includes(9), "the frame that failed to decode is never drawn");
    h.fetches[1].fail(503);
    for (let i = 0; i < 5; i++) await tick();
    assert.strictEqual(h.clears, 0);
    assert.strictEqual(h.els["film-canvas"].width, w);
    assert.ok(h.els.film.classList.contains("has-frame"));
    assert.strictEqual(h.fetches.length, CFG.parallel_loads + 2, "loading carries on after the failure");
  },
  async "decoded bitmaps are held only near the current frame, and drawn when present"() {
    const W = CFG.bitmap_window;
    assert.ok(W > 0);
    const h = run({ width: 1440, dpr: 1, bitmaps: true });
    const drawn = [];
    h.global.document.getElementById("film-canvas").getContext().drawImage = (im) => drawn.push(im);
    await h.loadAll();
    const near = (f) => [...h.bitmaps].every((b) => Math.abs(b.frame - f) <= W);
    assert.ok(h.bitmaps.size > 0 && h.bitmaps.size <= 2 * W + 1 && near(1));
    for (const f of [60, 61, 140, 240, 3]) {
      h.scrollToFrame(f);
      for (let i = 0; i < 3; i++) await tick();
      assert.ok(h.bitmaps.size <= 2 * W + 1, "at most one window of decoded frames");
      assert.ok(near(f), "bitmaps far from frame " + f + " are closed");
    }
    h.scrollToFrame(10); // inside the window held around frame 3
    await settle();
    assert.strictEqual(drawn[drawn.length - 1].frame, 10);
    assert.ok(drawn[drawn.length - 1].close, "a held bitmap is drawn instead of the <img>");
  },
  async "the parser takes records split at any byte and keeps frames whole"() {
    const h = run({ width: 1440, dpr: 1, bytes: { [CFG.bundles[0]]: new Uint8Array(Buffer.concat([1, 9, 17].map((n) => record(n)))) } });
    await tick();
    const f = h.fetches[0];
    while (!f.ended) { f.send(7); await tick(); }
    await settle();
    h.scrollToFrame(17);
    await settle();
    assert.strictEqual(h.draws[h.draws.length - 1], 17);
  },
  async "a record with a too-short length fails the bundle instead of desynchronising the rest"() {
    const bad = new Uint8Array(Buffer.concat([record(1), record(9, -2), record(17)]));
    const h = run({ width: 1440, dpr: 1, bytes: { [CFG.bundles[0]]: bad } });
    await tick();
    h.fetches[0].send();
    await settle();
    h.scrollToFrame(9);
    await settle();
    h.scrollToFrame(17);
    await settle();
    assert.deepStrictEqual(h.draws, [1], "frames 9 and 17 are not trusted");
    assert.strictEqual(h.fetches.length, CFG.parallel_loads + 1, "loading carries on with the next bundle");
  },
  async "a final record longer than the bytes that arrive is rejected, the whole ones before it kept"() {
    const bad = new Uint8Array(Buffer.concat([record(1), record(9), record(17, 50)].map((b, i) => (i < 2 ? b : b.subarray(0, b.length - 50)))));
    const h = run({ width: 1440, dpr: 1, bytes: { [CFG.bundles[0]]: bad } });
    await tick();
    h.fetches[0].send();
    await settle();
    h.scrollToFrame(17);
    await settle();
    assert.ok(h.draws.includes(1) && !h.draws.includes(17), "frame 17 is incomplete: " + h.draws);
    assert.strictEqual(h.fetches.length, CFG.parallel_loads + 1);
  },
  async "a record claiming more than max_frame_bytes is rejected at its header"() {
    const bad = new Uint8Array(Buffer.concat([record(1), record(9, CFG.max_frame_bytes), record(17)]));
    const h = run({ width: 1440, dpr: 1, bytes: { [CFG.bundles[0]]: bad } });
    await tick();
    h.fetches[0].send();
    await settle();
    assert.ok(h.fetches[0].ended);
    h.scrollToFrame(17);
    await settle();
    assert.ok(!h.draws.includes(9) && !h.draws.includes(17));
    assert.strictEqual(h.fetches.length, CFG.parallel_loads + 1);
  },
  async "object URLs are released once decoded, and no more than the window is decoded"() {
    const h = run({ width: 1440, dpr: 1 });
    await h.loadAll();
    await settle();
    assert.strictEqual(h.live.size, 0, "no object URL stays alive after decoding");
    const W = CFG.bitmap_window;
    const wide = run({ width: 1440, dpr: 1, bitmaps: true });
    await wide.loadAll();
    for (const f of [1, 130, 240, 90]) { wide.scrollToFrame(f); await settle(); }
    assert.ok(wide.peakLive <= 2 * W + 1 + CFG.max_pending_decodes, "peak decoded " + wide.peakLive);
    assert.ok(wide.calls < N, "not every frame is decoded: " + wide.calls);
  },
  async "a fast fling with a slow decoder never holds more than the window plus the pending cap"() {
    const W = CFG.bitmap_window, cap = CFG.max_pending_decodes;
    const h = run({ width: 1440, dpr: 1, bitmaps: true, slowDecode: true });
    await h.loadAll();
    for (const f of [20, 60, 100, 140, 180, 220, 240, 120]) { h.scrollToFrame(f); await settle(); }
    assert.ok(h.peakPending <= cap, "pending decodes " + h.peakPending);
    assert.ok(h.calls <= cap, "no work started for windows already left: " + h.calls);
    await h.finishDecodes();
    assert.ok(h.peakLive <= 2 * W + 1 + cap, "peak decoded " + h.peakLive);
    assert.ok([...h.bitmaps].every((b) => Math.abs(b.frame - 120) <= W), "stale decodes are closed");
    assert.ok(h.bitmaps.size > 0 && h.draws[h.draws.length - 1] !== undefined);
    assert.ok(Math.abs(h.draws[h.draws.length - 1] - 120) <= W, "settles on a frame near the target");
  },
  async "one caption per stage, switched on the stage's first frame, with copy on screen throughout"() {
    const h = run({ width: 1440, dpr: 1 });
    await h.loadAll();
    const chaps = CFG.chapters.map((_, i) => h.global.document.getElementById("film-stage").querySelectorAll()[i]);
    const on = () => chaps.map((c, i) => (c.classList.contains("on") ? i : -1)).filter((i) => i >= 0);
    for (let f = 1; f <= N; f++) {
      h.scrollToFrame(f);
      const want = CFG.chapters.findIndex((c, i) => f >= c.start && (i === CFG.chapters.length - 1 || f < c.end));
      assert.deepStrictEqual(on(), want < 0 ? [] : [want], "frame " + f);
      const opening = Number(h.els["film-open"].style.opacity);
      assert.ok(want >= 0 || opening >= 0.5, "frame " + f + ": neither a caption nor the opening copy is readable");
    }
    for (const c of CFG.chapters.slice(1)) {
      h.scrollToFrame(c.start - 1);
      const before = on();
      h.scrollToFrame(c.start);
      assert.notDeepStrictEqual(on(), before, "the caption changes exactly at frame " + c.start);
    }
  },
  async "each part gets its configured label once it has landed, and loses it as the part goes back in"() {
    const h = run({ width: 1440, dpr: 1 });
    await tick();
    assert.strictEqual(h.anchorFetches.length, 0, "the anchors wait for the first bundle");
    await h.loadAll();
    assert.deepStrictEqual(h.anchorFetches, [CFG.base_url + "/" + CFG.anchors + "?v=" + CFG.version]);
    assert.deepStrictEqual(h.labels().map((l) => l.name), CFG.parts.map((p) => p.name));
    for (let f = 1; f <= N; f++) {
      h.scrollToFrame(f);
      const want = CFG.parts.map((p) => f >= p.frame + CFG.label_land * CFG.part_rise && f < p.leave);
      assert.deepStrictEqual(h.labels().map((l) => l.on), want, "frame " + f);
      assert.deepStrictEqual(h.labels().map((l) => l.lineOn), want, "frame " + f + " leader lines");
    }
    h.scrollToFrame(Math.min(...CFG.parts.map((p) => p.leave)) - 1);
    assert.ok(h.labels().every((l) => l.on), "the exploded stack is fully labelled");
    h.scrollToFrame(N);
    assert.ok(h.labels().every((l) => !l.on && !l.lineOn), "the reassembled laptop carries no labels");
    const last = CFG.chapters.length - 1;
    assert.ok(h.global.document.getElementById("film-stage").querySelectorAll()[last].classList.contains("on"), "the closing caption stays to the last frame");
  },
  async "labels stay readable on screen, apart, off the stack, each line from its part"() {
    for (const [width, height, sides] of [[1440, 900, ["right"]], [390, 844, ["left", "right"]]]) {
      const h = run({ width, height, dpr: 1 });
      await h.loadAll();
      const layout = h.els.film.getAttribute("data-layout");
      assert.deepStrictEqual([...new Set(h.labels().map((l) => l.side))].sort(), sides, width + ": label sides");
      for (let f = 1; f <= N; f++) {
        h.scrollToFrame(f);
        const box = artBox(layout, width, height), at = ANCHORS[f];
        const on = h.labels().filter((l) => l.on);
        for (const l of on) {
          const where = width + " frame " + f + " " + l.name;
          assert.ok(l.left >= CFG.label_edge_px - 0.5 && l.right <= width - CFG.label_edge_px + 0.5, where + ": off screen");
          assert.ok(l.maxWidth >= 56, where + ": column too narrow to read (" + l.maxWidth + " px)");
          assert.ok(l.top - l.h / 2 >= 0 && l.top + l.h / 2 <= height, where + ": outside the stage");
          // Never over the stack: the whole column is past the outermost point of every labelled part.
          for (const v of Object.values(at)) {
            if (l.side === "right") assert.ok(l.left > box.left + v[0] * box.width, where + ": over the stack");
            else assert.ok(l.right < box.left + v[2] * box.width, where + ": over the stack");
          }
          const p = CFG.parts.find((q) => q.name === l.name), v = at[p.id], k = l.side === "right" ? 0 : 2;
          assert.ok(Math.abs(l.line[0] - (box.left + v[k] * box.width)) < 0.06 && Math.abs(l.line[1] - (box.top + v[k + 1] * box.height)) < 0.06, where + ": line start");
          assert.ok(Math.abs(l.line[3] - l.top) < 0.06, where + ": line end at the label");
        }
        for (const side of sides) {
          const col = on.filter((l) => l.side === side).sort((a, b) => a.top - b.top);
          for (let i = 1; i < col.length; i++) {
            const gap = col[i].top - col[i - 1].top;
            assert.ok(gap >= CFG.label_spacing_px - 0.06 && gap >= (col[i].h + col[i - 1].h) / 2, width + " frame " + f + ": " + col[i - 1].name + " and " + col[i].name + " overlap");
          }
        }
      }
    }
  },
  async "a failing anchors file is retried with growing waits, then the page goes back to the static layout"() {
    const broken = JSON.parse(JSON.stringify(ANCHORS));
    delete broken[N][CFG.parts[0].id];
    for (const anchors of ["missing", broken, { 1: "x" }]) {
      const h = run({ width: 1440, dpr: 1, anchors });
      await h.loadAll();
      await settle();
      assert.strictEqual(h.anchorFetches.length, 1);
      const waits = await h.waits();
      assert.deepStrictEqual(waits, Array.from({ length: CFG.load_retries }, (_, k) => CFG.load_backoff_ms * 2 ** k));
      assert.strictEqual(h.anchorFetches.length, 1 + CFG.load_retries);
      assert.ok(!h.els.film.classList.contains("is-live"), "static layout again");
      assert.ok(!h.els.film.classList.contains("has-frame"));
      assert.strictEqual(h.els.film.style.height, "");
    }
  },
  async "one anchors failure is retried and the labels come"() {
    const h = run({ width: 1440, dpr: 1, anchorFails: 1 });
    await h.loadAll();
    await settle();
    await h.waits();
    assert.strictEqual(h.anchorFetches.length, 2);
    assert.ok(h.els.film.classList.contains("is-live"));
    h.scrollToFrame(CFG.parts[0].frame + CFG.part_rise);
    await settle();
    assert.ok(h.labels().some((l) => l.on), "labelled after the retry");
  },
  async "a bundle that fails is tried again and its frames arrive"() {
    const h = run({ width: 1440, dpr: 1 });
    await tick();
    h.fetches[0].fail(503);
    await settle();
    const before = h.fetches.length;
    assert.deepStrictEqual(await h.waits(), [CFG.load_backoff_ms]);
    assert.ok(h.fetches.length > before && h.fetches.some((f, i) => i >= before && f.name === PLAN[0][0]));
    await h.loadAll();
    assert.ok(h.els.film.classList.contains("is-live"));
    assert.ok(h.draws.length > 0);
  },
  async "a bundle that stays unavailable puts the page back to the static layout and stops loading"() {
    const h = run({ width: 1440, dpr: 1 });
    await tick();
    for (let i = 0; i < 60 && h.els.film.classList.contains("is-live"); i++) {
      h.open().forEach((f) => { if (f.name === PLAN[0][0]) f.fail(503); else f.send(); });
      await settle();
      await h.waits();
    }
    assert.ok(!h.els.film.classList.contains("is-live"), "back to the readable static layout");
    assert.strictEqual(h.els.film.style.height, "");
    assert.strictEqual(h.fetches.filter((f) => f.name === PLAN[0][0]).length, 1 + CFG.load_retries, "bounded tries");
    const n = h.fetches.length;
    await h.waits();
    assert.strictEqual(h.fetches.length, n, "nothing more is fetched");
  },
  async "stop() cancels every pending retry and response, and releases the frame blobs"() {
    const h = run({ width: 1440, dpr: 1 });
    await tick();
    const [a, b] = h.fetches;
    b.send(20); // a response part-way in: its bundle is still pending
    h.fetches[2].send(); // a whole bundle: its frames are stored as blobs
    a.fail(503);
    await settle(); // a's retry is now waiting in a timer
    assert.ok(h.timers.length >= 1, "a retry is waiting");
    for (let i = 0; i < CFG.load_retries && h.timers.length; i++) { /* walk a's bundle to permanent failure */
      await h.waits();
      h.open().filter((f) => f.name === a.name).forEach((f) => f.fail(503));
      await settle();
    }
    assert.ok(!h.els.film.classList.contains("is-live"), "fell back to the static layout");
    assert.ok(h.aborts.every((s) => s.aborted), "in-flight fetches are aborted");
    assert.ok(b.signal && b.signal.aborted, "the pending bundle carried the abort signal");
    assert.strictEqual(h.timers.length, 0, "no retry timer is left");
    const n = h.fetches.length, an = h.anchorFetches.length;
    b.send(); // a late response for the pending bundle
    await h.waits();
    await settle();
    assert.strictEqual(h.fetches.length, n, "no bundle fetch after the fallback");
    assert.strictEqual(h.anchorFetches.length, an, "first()/anchors() never ran after the fallback");
    assert.strictEqual(h.live.size, 0, "no object URL is held");
    assert.ok(h.blobRefs.length > 0, "frames were stored");
    h.gc();
    assert.strictEqual(h.blobRefs.filter((r) => r.deref()).length, 0, "no frame Blob is retained");
  },
  async "a response without a stream reader that lands after stop() is ignored"() {
    const h = run({ width: 1440, dpr: 1, plain: true });
    await tick();
    const [a, b] = h.fetches;
    for (let i = 0; i <= CFG.load_retries; i++) {
      h.open().filter((f) => f.name === a.name).forEach((f) => f.fail(503));
      await settle();
      await h.waits();
    }
    assert.ok(!h.els.film.classList.contains("is-live"), "fell back to the static layout");
    const an = h.anchorFetches.length;
    const allocs = h.allocs;
    b.finish(); // the delayed whole-body response resolves after stop()
    await h.waits();
    await settle();
    assert.strictEqual(h.allocs, allocs, "nothing was allocated or parsed after stop()");
    assert.strictEqual(h.blobRefs.length, 0, "nothing was stored after stop()");
    assert.strictEqual(h.anchorFetches.length, an, "first()/anchors() never ran");
    assert.strictEqual(h.live.size, 0);
  },
  async "easing within one frame writes no styles"() {
    const h = run({ width: 1440, dpr: 1 });
    await h.loadAll();
    h.scrollToFrame(100);
    const before = h.styleWrites;
    // A scroll too small to change the frame: the eased loop runs but the copy stays as it is.
    h.scroll += 1;
    (h.listeners.scroll || []).forEach((cb) => cb());
    h.flush();
    assert.strictEqual(h.styleWrites, before);
  },
};

(async () => {
  let passed = 0;
  for (const [name, fn] of Object.entries(scenarios)) {
    try { await fn(); passed++; } catch (e) { console.error("FAIL " + name + "\n" + (e.stack || e)); process.exitCode = 1; }
  }
  console.log(`film.js harness: ${passed}/${Object.keys(scenarios).length} passed`);
})();
