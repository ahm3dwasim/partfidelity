// Waiting screen: poll the job-status JSON, play the person's own
// frames in a large viewer, trace part outlines as stages finish, and stack
// finished step cards beside it. Works live (statusUrl) or as a gallery demo
// (fakeScript: [{at: seconds, status: {...}}]). Needs GSAP before this file;
// without it every change still applies, instantly.
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var POLL_MS = 1500;

  function reducedMotion() {
    return window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function fmtClock(ms) {
    var s = Math.max(0, Math.floor(ms / 1000));
    var m = Math.floor(s / 60);
    var rest = String(s % 60);
    if (rest.length < 2) rest = "0" + rest;
    return m + ":" + rest;
  }

  function el(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  function pointsToStr(polygon) {
    var out = [];
    for (var i = 0; i < polygon.length; i++) {
      out.push(polygon[i][0] + "," + polygon[i][1]);
    }
    return out.join(" ");
  }

  function WaitVideo(root, opts) {
    this.root = root;
    this.opts = opts || {};
    this.frame = root.querySelector("#wv-frame");
    this.imgs = [
      root.querySelector("#wv-img-a"),
      root.querySelector("#wv-img-b"),
    ];
    this.front = 0;
    this.overlay = root.querySelector("#wv-overlay");
    this.loading = root.querySelector("#wv-loading");
    this.clock = root.querySelector("#wv-clock");
    this.stages = root.querySelector("#wv-stages");
    this.steps = root.querySelector("#wv-steps");
    this.note = root.querySelector("#wv-note");
    this.cancelBtn = root.querySelector("#wv-cancel");
    this.guideLink = root.querySelector("#wv-guide");
    this.replayBtn = root.querySelector("#wv-replay");
    this.frameId = null;
    this.drawn = {};
    this.shownSteps = {};
    this.stageStates = {};
    this.startedAt = Date.now();
    this.terminal = false;
    this.timers = [];
    this.tick = null;
    this.pollSeq = 0;
    this.pollAbort = null;
    this.pendingFrameId = null;
    this.lastRevision = 0;
    var self = this;
    this.cancelBtn.addEventListener("click", function () { self.cancel(); });
    if (this.replayBtn) {
      this.replayBtn.addEventListener("click", function () { self.replay(); });
    }
  }

  WaitVideo.prototype.gsap = function () {
    if (reducedMotion()) return null;
    return window.gsap || null;
  };

  WaitVideo.prototype.startClock = function () {
    var self = this;
    if (this.tick) window.clearInterval(this.tick);
    this.tick = window.setInterval(function () {
      if (!self.terminal) self.clock.textContent = fmtClock(Date.now() - self.startedAt);
    }, 250);
  };

  WaitVideo.prototype.start = function () {
    var self = this;
    this.startClock();
    if (!this.opts.cancelUrl) this.cancelBtn.classList.add("is-hidden");
    if (this.opts.fakeScript) {
      this.playScript();
      if (this.replayBtn) this.replayBtn.classList.remove("is-hidden");
    } else if (this.opts.statusUrl) {
      this.pollOnce();
      this.poller = window.setInterval(function () { self.pollOnce(); }, POLL_MS);
    } else {
      this.say("Waiting for the job to start…");
    }
  };

  WaitVideo.prototype.playScript = function () {
    var self = this;
    this.opts.fakeScript.forEach(function (entry) {
      self.timers.push(window.setTimeout(function () {
        self.apply(entry.status);
      }, entry.at * 1000));
    });
  };

  WaitVideo.prototype.replay = function () {
    var self = this;
    this.timers.forEach(function (t) { window.clearTimeout(t); });
    this.timers = [];
    this.frameId = null;
    this.drawn = {};
    this.shownSteps = {};
    this.stageStates = {};
    this.terminal = false;
    this.overlay.textContent = "";
    this.steps.textContent = "";
    this.stages.textContent = "";
    this.guideLink.classList.add("is-hidden");
    this.cancelBtn.disabled = false;
    this.startedAt = Date.now();
    this.pendingFrameId = null;
    this.lastRevision = 0;
    this.startClock();
    this.playScript();
  };

  WaitVideo.prototype.pollOnce = function () {
    var self = this;
    this.pollSeq += 1;
    var seq = this.pollSeq;
    if (this.pollAbort) this.pollAbort.abort();
    var ctrl = new AbortController();
    this.pollAbort = ctrl;
    window.fetch(this.opts.statusUrl, { credentials: "same-origin", signal: ctrl.signal })
      .then(function (res) {
        if (!res.ok) throw new Error("status " + res.status);
        return res.json();
      })
      .then(function (status) {
        if (seq !== self.pollSeq) return; // superseded: a newer poll is in flight
        self.apply(status);
      })
      .catch(function (err) {
        if (err && err.name === "AbortError") return;
        if (seq !== self.pollSeq) return;
        if (!self.terminal) self.say("Still working — a status check just failed.");
      });
  };

  WaitVideo.prototype.say = function (text) {
    this.note.textContent = text;
  };

  WaitVideo.prototype.apply = function (status) {
    if (!status || this.terminal) return;
    var rev = typeof status.revision === "number" ? status.revision : 0;
    if (rev < this.lastRevision) return; // older snapshot on a newer request
    this.lastRevision = rev;
    if (status.stages) this.renderStages(status.stages);
    if (status.frame && status.frame.id !== this.frameId &&
        status.frame.id !== this.pendingFrameId) this.showFrame(status.frame);
    var self = this;
    (status.outlines || []).forEach(function (o) { self.trace(o); });
    (status.steps || []).forEach(function (s) { self.addStep(s); });
    if (status.state === "done") this.finish(status, "Your guide is ready.");
    else if (status.state === "failed") this.finish(status, "This one failed — your footage is still yours; try again.");
    else if (status.state === "cancelled") this.finish(status, "Cancelled — the footage was deleted.");
  };

  WaitVideo.prototype.renderStages = function (stages) {
    var self = this;
    var changed = stages.some(function (st) {
      return self.stageStates[st.id] !== st.state;
    });
    if (!changed && this.stages.children.length) return;
    this.stageStates = {};
    this.stages.textContent = "";
    stages.forEach(function (st) {
      self.stageStates[st.id] = st.state;
      var li = document.createElement("li");
      li.textContent = st.label;
      if (st.state === "active") li.classList.add("is-active");
      if (st.state === "done") li.classList.add("is-done");
      self.stages.appendChild(li);
    });
  };

  WaitVideo.prototype.showFrame = function (frame) {
    if (reducedMotion() && this.frameId !== null) return; // one still frame
    this.pendingFrameId = frame.id;
    var next = this.imgs[1 - this.front];
    var prev = this.imgs[this.front];
    var self = this;
    next.onload = function () {
      next.onload = null;
      if (frame.id !== self.pendingFrameId) return; // superseded while loading
      self.pendingFrameId = null;
      self.frameId = frame.id;
      self.loading.classList.add("is-hidden");
      var g = self.gsap();
      next.classList.add("is-on");
      if (g) {
        g.fromTo(next, { opacity: 0, scale: 1.06 }, {
          opacity: 1, scale: 1, duration: 0.6, ease: "power1.out",
          onComplete: function () { prev.classList.remove("is-on"); },
        });
      } else {
        prev.classList.remove("is-on");
      }
      self.front = 1 - self.front;
      self.syncOverlays();
    };
    next.alt = frame.alt || "Frame from your clip";
    next.src = frame.src;
    // If the image is cached, onload still fires; if it errors, keep the old
    // frame and release the pending id so a later poll can retry it.
    next.onerror = function () {
      next.onerror = null;
      if (frame.id === self.pendingFrameId) self.pendingFrameId = null;
    };
  };

  WaitVideo.prototype.trace = function (outline) {
    var key = outline.frame_id + ":" + outline.part;
    if (this.drawn[key]) return;
    var group = el("g", { "data-frame": outline.frame_id, "data-part": outline.part }, this.overlay);
    var poly = el("polygon", {
      points: pointsToStr(outline.polygon),
      fill: "rgba(240,164,0,0.12)",
      stroke: "#f0a400",
      "stroke-width": "6",
      "stroke-linejoin": "round",
    }, group);
    var label = outline.label || outline.polygon[0];
    var dir = label[0] > 950 ? -1 : 1; // keep the chip inside the frame
    var lx = label[0] + dir * 120;
    var ly = Math.max(50, label[1] - 70);
    var line = el("line", {
      x1: label[0], y1: label[1], x2: lx, y2: ly,
      stroke: "#f0a400", "stroke-width": "4",
    }, group);
    var text = el("text", {
      x: lx + dir * 10, y: ly,
      fill: "#f0a400", "font-size": "44", "font-family": "Oswald, sans-serif",
      "letter-spacing": "2", "text-anchor": dir < 0 ? "end" : "start",
    }, group);
    text.textContent = outline.part.toUpperCase();
    this.drawn[key] = group;
    group.style.display = outline.frame_id === this.frameId ? "" : "none";
    var g = this.gsap();
    if (!g) return;
    var len = poly.getTotalLength ? poly.getTotalLength() : 3000;
    var tl = g.timeline();
    tl.fromTo(poly, { strokeDasharray: len, strokeDashoffset: len },
      { strokeDashoffset: 0, duration: 1.2, ease: "power2.inOut" });
    tl.fromTo([line, text], { opacity: 0 }, { opacity: 1, duration: 0.4 }, "-=0.3");
  };

  WaitVideo.prototype.syncOverlays = function () {
    for (var key in this.drawn) {
      var group = this.drawn[key];
      group.style.display = group.getAttribute("data-frame") === this.frameId ? "" : "none";
    }
  };

  WaitVideo.prototype.addStep = function (step) {
    if (this.shownSteps[step.n]) return;
    this.shownSteps[step.n] = true;
    var li = document.createElement("li");
    li.className = "wv-step";
    var img = document.createElement("img");
    img.src = step.thumb;
    img.alt = "Step " + step.n + ": " + step.title;
    img.loading = "lazy";
    var h = document.createElement("h3");
    h.textContent = step.n + ". " + step.title;
    var p = document.createElement("p");
    p.textContent = "starts " + step.at;
    li.appendChild(img);
    li.appendChild(h);
    li.appendChild(p);
    this.steps.appendChild(li);
    var g = this.gsap();
    if (g) g.from(li, { x: 60, opacity: 0, duration: 0.5, ease: "power2.out" });
  };

  WaitVideo.prototype.finish = function (status, message) {
    this.terminal = true;
    this.say(message);
    this.cancelBtn.disabled = true;
    if (this.poller) window.clearInterval(this.poller);
    if (this.tick) { window.clearInterval(this.tick); this.tick = null; }
    if (this.pollAbort) this.pollAbort.abort();
    // Only same-app targets become links; anything else stays hidden.
    if (status.state === "done" && /^(\/(?!\/)|#)/.test(status.guide_url || "")) {
      this.guideLink.href = status.guide_url;
      this.guideLink.classList.remove("is-hidden");
    }
  };

  WaitVideo.prototype.cancel = function () {
    var url = this.opts.cancelUrl;
    if (!url) return; // no endpoint: the control stays hidden, nothing to post
    this.say("Cancelling…");
    this.cancelBtn.disabled = true;
    var self = this;
    window.fetch(url, { method: "POST", credentials: "same-origin" })
      .then(function (res) {
        if (!res.ok) throw new Error("cancel " + res.status);
        return res.json();
      })
      .then(function (body) {
        if (!body || body.state !== "cancelled") throw new Error("cancel not terminal");
        self.finish({ state: "cancelled" }, "Cancelled — the footage was deleted.");
      })
      .catch(function () {
        if (self.terminal) return; // done/failed/cancelled already landed
        self.say("Cancel failed — try again.");
        self.cancelBtn.disabled = false;
      });
  };

  WaitVideo.mount = function (root, opts) {
    var wv = new WaitVideo(root, opts || {});
    wv.start();
    return wv;
  };

  window.WaitVideo = WaitVideo;

  document.addEventListener("DOMContentLoaded", function () {
    var root = document.getElementById("wait-video");
    if (!root || root.dataset.mounted) return;
    root.dataset.mounted = "1";
    // Gallery pages pass a fake script and mount explicitly; a live page
    // only needs the data-status-url / data-cancel-url attributes.
    if (window.__WV_FAKE__) return;
    WaitVideo.mount(root, {
      statusUrl: root.getAttribute("data-status-url") || null,
      cancelUrl: root.getAttribute("data-cancel-url") || null,
    });
  });
})();
