/* Home scroll film: draws the pre-rendered teardown frames on a canvas, picked by scroll.
   Every tunable comes from #film-config (config/film.toml). Reduced motion or Save-Data leaves the
   static layout (poster plus the chapters as blocks) and loads no frames. */
(function () {
  var root = document.getElementById("film");
  var cfgEl = document.getElementById("film-config");
  if (!root || !cfgEl) return;
  var cfg = JSON.parse(cfgEl.textContent);
  var conn = navigator.connection;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || (conn && conn.saveData)) return;

  var stage = document.getElementById("film-stage");
  var canvas = document.getElementById("film-canvas");
  var poster = document.getElementById("film-poster");
  var open = document.getElementById("film-open");
  var art = document.getElementById("film-art");
  var labelsEl = document.getElementById("film-labels");
  var chaps = Array.prototype.slice.call(stage.querySelectorAll("[data-chap]"));
  var ctx = canvas.getContext("2d");
  var N = cfg.frames;

  // One frame set for the page's life: by css width, or by device pixels.
  var vw = window.innerWidth, dpr = window.devicePixelRatio || 1;
  var set = cfg.sets.filter(function (s) {
    return (s.max_css_width && vw <= s.max_css_width) || (s.max_device_width && vw * dpr <= s.max_device_width);
  })[0] || cfg.sets[cfg.sets.length - 1];

  root.style.setProperty("--film-phone-width", cfg.phone_art_width_vw + "vw");
  root.style.setProperty("--film-phone-feather", cfg.phone_feather_y_pct + "%");
  root.style.setProperty("--film-open-lift", cfg.opening_lift_px + "px");
  root.style.setProperty("--film-caption-fade", cfg.caption_fade_ms + "ms");
  // Composition by css width alone (density only picks frames), so a phone layout always has the phone frames.
  var shape = cfg.sets.filter(function (s) { return s.max_css_width && vw <= s.max_css_width; })[0] || cfg.sets[cfg.sets.length - 1];
  root.setAttribute("data-layout", shape.layout);
  root.classList.add("is-live");
  root.style.height = cfg.height_vh + "vh";

  // Frames in memory: the compressed bytes of every frame that has arrived (a Blob, small), and decoded pixels
  // only for the frames within bitmap_window of the current one (bmp). A frame is drawable once decoded.
  var blobs = {}, bmp = {}, pending = 0, bad = {}, busy = {}, W = cfg.bitmap_window;
  var drawn = 0, shown = false, dead = 0;
  function nearest(f) {
    for (var d = 0; d < N; d++) {
      if (f - d >= 1 && bmp[f - d]) return f - d;
      if (f + d <= N && bmp[f + d]) return f + d;
    }
    return 0;
  }
  function draw(f) {
    if (dead) return;
    var src = nearest(f);
    if (!src || src === drawn) return;
    var im = bmp[src], w = im.naturalWidth || im.width;
    if (canvas.width !== w) { canvas.width = w; canvas.height = im.naturalHeight || im.height; }
    ctx.drawImage(im, 0, 0);
    drawn = src;
    if (!shown) { shown = true; root.classList.add("has-frame"); }
  }

  // A canvas decodes an <img> again on the main thread when it paints (a 1600 px frame took up to 60 ms at 4x CPU,
  // ), so frames are decoded once, off the main thread, to an ImageBitmap (an <img> where that is missing,
  // its object URL released as soon as it has decoded). At most max_pending_decodes run at once, nearest frame
  // first; the list is rebuilt from the current frame on every call, so work for an old position is never started.
  function release(b) { if (b.close) b.close(); }
  function decode(n) {
    if (window.createImageBitmap) return createImageBitmap(blobs[n]);
    var im = new Image(), u = URL.createObjectURL(blobs[n]);
    im.src = u;
    var free = function () { if (URL.revokeObjectURL) URL.revokeObjectURL(u); };
    return im.decode().then(function () { free(); return im; }, function (e) { free(); throw e; });
  }
  function warm(f) {
    if (dead) return;
    for (var k in bmp) if (Math.abs(k - f) > W) { release(bmp[k]); delete bmp[k]; }
    var want = [];
    for (var n = Math.max(1, f - W); n <= Math.min(N, f + W); n++) if (blobs[n] && !bmp[n] && !bad[n] && !busy[n]) want.push(n);
    want.sort(function (a, b) { return Math.abs(a - f) - Math.abs(b - f); });
    while (pending < cfg.max_pending_decodes && want.length) {
      (function (n) {
        busy[n] = 1;
        pending++;
        decode(n).then(function (b) {
          busy[n] = 0; pending--;
          if (dead || Math.abs(n - frame()) > W) release(b); else bmp[n] = b;
          warm(frame());
          draw(frame());
        }, function () { busy[n] = 0; pending--; bad[n] = 1; warm(frame()); });
      })(want.shift());
    }
  }

  // Frames arrive in a few bundle files (tools/film_frames.py), not one request each: at about 1 s per request
  // from far away, 260 single frames took 46 s. Bundles come in pass order (every 8th frame, every 4th,
  // then all); each is read as a stream of records (uint16 LE frame, uint32 LE length, webp bytes), so a frame
  // shows as soon as its bytes are in. A record that does not fit (frame out of range, length over
  // max_frame_bytes, payload not a RIFF/WEBP of exactly that length, or a stream that ends inside a record)
  // fails the bundle: bytes after a bad length cannot be trusted.
  function store(n, bytes) {
    if (dead || blobs[n]) return;
    blobs[n] = new Blob([bytes], { type: "image/webp" });
    if (Math.abs(n - frame()) <= W) warm(frame());
  }
  function tag(b, o, s) { for (var i = 0; i < s.length; i++) if (b[o + i] !== s.charCodeAt(i)) return false; return true; }
  function read(u) {
    return fetch(u, ctl ? { signal: ctl.signal } : undefined).then(function (r) {
      if (dead) { if (r.body && r.body.cancel) r.body.cancel(); return; }
      if (!r.ok) throw new Error(u + ": " + r.status);
      var buf = new Uint8Array(0);
      function eat(chunk) {
        var joined = new Uint8Array(buf.length + chunk.length);
        joined.set(buf);
        joined.set(chunk, buf.length);
        buf = joined;
        while (buf.length >= 6) {
          var n = buf[0] + buf[1] * 256, len = buf[2] + buf[3] * 256 + buf[4] * 65536 + buf[5] * 16777216;
          if (n < 1 || n > N || len < 12 || len > cfg.max_frame_bytes) throw new Error(u + ": bad record header");
          if (buf.length < 6 + len) break;
          var p = buf.subarray(6, 6 + len);
          if (!tag(p, 0, "RIFF") || !tag(p, 8, "WEBP") || p[4] + p[5] * 256 + p[6] * 65536 + p[7] * 16777216 + 8 !== len) {
            throw new Error(u + ": record " + n + " is not a webp of its stated length");
          }
          store(n, p);
          buf = buf.subarray(6 + len);
        }
      }
      function end() { if (buf.length) throw new Error(u + ": truncated record"); }
      if (!r.body || !r.body.getReader) return r.arrayBuffer().then(function (b) { if (dead) return; eat(new Uint8Array(b)); end(); });
      var rd = r.body.getReader();
      return rd.read().then(function step(x) {
        if (dead) { if (rd.cancel) rd.cancel(); return; }
        if (x.done) return end();
        try { eat(x.value); } catch (e) { if (rd.cancel) rd.cancel(); throw e; }
        return rd.read().then(step);
      });
    });
  }
  // A failed bundle or the anchors file is retried up to load_retries times, waiting load_backoff_ms x 2^try between
  // tries. Whatever stays unavailable after that (a bundle's frames, or the anchors the labels need) puts the page back
  // to the readable static layout (poster plus the chapters as blocks): stop() below.
  // stop() is the one cancellation barrier: it clears the waiting retries and aborts what is in flight, and every
  // completion path below checks dead before it acts.
  var queue = cfg.bundles.slice(), running = 0, asked = 0, timers = [], ctl = window.AbortController ? new AbortController() : null;
  function retry(job, n, err) {
    if (dead) return;
    if (n >= cfg.load_retries) return stop();
    var t = setTimeout(function () {
      timers.splice(timers.indexOf(t), 1);
      if (!dead) job(n + 1);
    }, cfg.load_backoff_ms * Math.pow(2, n));
    timers.push(t);
  }
  function bundle(name, n) {
    running++;
    read(cfg.base_url + "/" + set.dir + "/" + name + "?v=" + cfg.version).then(function () {
      running--;
      if (dead) return;
      pump(); first();
    }, function (e) {
      running--;
      if (dead) return;
      retry(function (k) { bundle(name, k); }, n, e);
      pump();
    });
  }
  function first() { if (!dead && !asked) { asked = 1; anchors(0); } }
  function pump() {
    while (!dead && running < cfg.parallel_loads && queue.length) bundle(queue.shift(), 0);
  }
  function stop() {
    if (dead) return;
    dead = 1;
    if (ctl) ctl.abort();
    timers.forEach(clearTimeout);
    timers = [];
    queue = [];
    blobs = {};
    cancelAnimationFrame(raf);
    removeEventListener("scroll", onScroll);
    removeEventListener("resize", onResize);
    for (var k in bmp) release(bmp[k]);
    bmp = {};
    root.classList.remove("is-live");
    root.classList.remove("has-frame");
    root.style.height = "";
    labelsEl.textContent = "";
  }

  function ss(a, b, x) { x = Math.min(1, Math.max(0, (x - a) / (b - a))); return x * x * (3 - 2 * x); }
  function frame() { return 1 + Math.round(cur * (N - 1)); }
  var cur = 0, tgt = 0, raf = 0;
  function progress() {
    var r = root.getBoundingClientRect();
    return Math.min(1, Math.max(0, -r.top / (r.height - innerHeight)));
  }
  // Part labels: each [[parts]] name on a thin leader line to its part. Where each part sits in each frame
  // comes from the render (anchors.json, {frame: {id: [rx, ry, lx, ly]}}: its rightmost and leftmost points as
  // fractions of the frame, y down), fetched once the first bundle is in. A label shows once its part is label_land
  // through its rise and goes when the part starts back in (part.leave). Labels sit in one column per side, fixed for the whole film (the stack's
  // outermost point over every frame a label shows, plus label_gap_px), so they never cover the model; only their height follows
  // the part, spread at least label_spacing_px apart. A missing or malformed file leaves the film without labels.
  var SVG = "http://www.w3.org/2000/svg", A = null, reach = { right: 0, left: 1 };
  var lines = document.createElementNS(SVG, "svg");
  labelsEl.appendChild(lines);
  var marks = cfg.parts.map(function (p, i) {
    var m = { part: p, side: shape.labels === "alternate" && i % 2 ? "left" : "right", h: 0, on: false };
    m.text = document.createElement("span");
    m.text.classList.add("film-label");
    m.text.classList.add("is-" + m.side);
    m.text.textContent = p.name;
    m.line = document.createElementNS(SVG, "line");
    lines.appendChild(m.line);
    labelsEl.appendChild(m.text);
    return m;
  });
  function shows(m, f) { return f >= m.part.frame + cfg.label_land * cfg.part_rise && f < m.part.leave; }
  var from = Math.min.apply(null, marks.map(function (m) { for (var f = 1; f <= N && !shows(m, f); f++); return f; }));
  var until = Math.max.apply(null, marks.map(function (m) { return m.part.leave; }));
  function anchors(n) {
    if (dead) return;
    fetch(cfg.base_url + "/" + cfg.anchors + "?v=" + cfg.version, ctl ? { signal: ctl.signal } : undefined).then(function (r) {
      if (dead) return null;
      if (!r.ok) throw new Error(cfg.anchors + ": " + r.status);
      return r.json();
    }).then(function (a) {
      if (dead) return;
      for (var f = 1; f <= N; f++) {
        for (var i = 0; i < marks.length; i++) {
          var v = a && a[f] && a[f][marks[i].part.id];
          if (!v || v.length !== 4 || !v.every(function (x) { return typeof x === "number" && isFinite(x); })) throw new Error("bad anchors");
          if (f < from || f >= until) continue;
          reach.right = Math.max(reach.right, v[0]);
          reach.left = Math.min(reach.left, v[2]);
        }
      }
      A = a;
      layout();
      painted = 0;
      paint();
    }).catch(function (e) { if (!dead) retry(function (k) { anchors(k); }, n, e); });
  }
  // Where the art sits in the stage, in css px (read before any write in a paint).
  function box() {
    var s = labelsEl.getBoundingClientRect(), r = art.getBoundingClientRect();
    return { x: r.left - s.left, y: r.top - s.top, w: r.width, h: r.height, W: s.width, H: s.height };
  }
  function column(b, side) {
    return side === "right" ? b.x + reach.right * b.w + cfg.label_gap_px : b.x + reach.left * b.w - cfg.label_gap_px;
  }
  // Columns and widths (labels wrap rather than run off the screen), then each label's height: on load and resize.
  function layout() {
    if (!A) return;
    var b = box();
    marks.forEach(function (m) {
      var x = column(b, m.side), edge = cfg.label_edge_px;
      if (m.side === "right") { m.text.style.left = x + "px"; m.text.style.maxWidth = Math.max(0, b.W - edge - x) + "px"; }
      else { m.text.style.right = b.W - x + "px"; m.text.style.maxWidth = Math.max(0, x - edge) + "px"; }
    });
    marks.forEach(function (m) { m.h = m.text.offsetHeight; });
  }
  function labels(f) {
    if (!A) return;
    var b = box(), at = A[f], edge = cfg.label_edge_px;
    ["right", "left"].forEach(function (side) {
      var x = column(b, side), k = side === "right" ? 0 : 2;
      var mine = marks.filter(function (m) { return m.side === side; });
      var on = mine.filter(function (m) { return shows(m, f); });
      on.forEach(function (m) { var v = at[m.part.id]; m.px = b.x + v[k] * b.w; m.py = b.y + v[k + 1] * b.h; m.y = m.py; });
      on.sort(function (p, q) { return p.py - q.py; });
      function need(p, q) { return Math.max(cfg.label_spacing_px, (p.h + q.h) / 2 + 2); }
      for (var i = 0; i < on.length; i++) {
        on[i].y = Math.max(on[i].y, edge + on[i].h / 2, i ? on[i - 1].y + need(on[i - 1], on[i]) : -Infinity);
      }
      for (i = on.length - 1; i >= 0; i--) {
        on[i].y = Math.min(on[i].y, b.H - edge - on[i].h / 2, i < on.length - 1 ? on[i + 1].y - need(on[i], on[i + 1]) : Infinity);
      }
      mine.forEach(function (m) {
        var show = on.indexOf(m) >= 0;
        if (show) {
          m.text.style.top = m.y.toFixed(1) + "px";
          m.line.setAttribute("x1", m.px.toFixed(1));
          m.line.setAttribute("y1", m.py.toFixed(1));
          m.line.setAttribute("x2", (side === "right" ? x - 4 : x + 4).toFixed(1));
          m.line.setAttribute("y2", m.y.toFixed(1));
        }
        if (show !== m.on) { m.on = show; m.text.classList.toggle("on", show); m.line.setAttribute("class", show ? "on" : ""); }
      });
    });
  }
  var painted = 0;
  function paint() {
    var f = frame();
    if (f === painted) return;  // the copy depends on the frame only: no style writes while easing within one
    painted = f;
    labels(f);
    warm(f);
    draw(f);
    var op = 1 - ss(cfg.opening_fade[0], cfg.opening_fade[1], f);
    open.style.opacity = op.toFixed(2);
    open.classList.toggle("is-off", f > cfg.opening_fade[1]);
    // One caption per stage, switched exactly at the stage's first frame (the chapters tile the film after the opening).
    var act = 0;
    cfg.chapters.forEach(function (c, i) { if (f >= c.start && (i === cfg.chapters.length - 1 || f < c.end)) act = i + 1; });
    chaps.forEach(function (el, i) { el.classList.toggle("on", i === act - 1); });
  }
  function tick() {
    cur += (tgt - cur) * cfg.smoothing;
    if (Math.abs(tgt - cur) < 0.0005) cur = tgt;
    paint();
    raf = cur === tgt ? 0 : requestAnimationFrame(tick);
  }
  function onScroll() { tgt = progress(); if (!raf) raf = requestAnimationFrame(tick); }
  addEventListener("scroll", onScroll, { passive: true });
  function onResize() { layout(); painted = 0; onScroll(); }
  addEventListener("resize", onResize);
  tgt = cur = progress();
  paint();
  pump();
})();
