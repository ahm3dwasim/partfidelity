// Fixture for tests/test_ui_loading.py: executes the REAL loading.js and
// upload.js under node with a tiny DOM stub, proving the busy button never
// sticks after a client-side rejection and locks only on the accepted path.
// Covers too-large (sync) and too-long (async probe) rejections, the valid
// path, a double submit during probing, both listener orders, and the
// no-PFLoading fallback. Usage:
//   node ui_loading_busy_order.js <loading.js> <upload.js>
"use strict";

const fs = require("fs");

const MAX_BYTES = 100;
const MAX_DURATION = 60;
const MESSAGE = "Trim it and try again.";

function fail(message) {
  console.error("BUSY-ORDER-FAIL " + message);
  process.exit(1);
}

function check(cond, message) {
  if (!cond) fail(message);
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function makeEl() {
  const listeners = {};
  return {
    attrs: {},
    dataset: {},
    disabled: false,
    hidden: false,
    focused: false,
    textContent: "",
    files: null,
    duration: 0,
    focus() {
      this.focused = true;
    },
    closest() {
      return null;
    },
    setAttribute(k, v) {
      this.attrs[k] = v;
    },
    getAttribute(k) {
      return Object.prototype.hasOwnProperty.call(this.attrs, k)
        ? this.attrs[k]
        : null;
    },
    addEventListener(t, fn) {
      (listeners[t] = listeners[t] || []).push(fn);
    },
    _fire(t, ev) {
      (listeners[t] || []).forEach((fn) => fn(ev));
    }
  };
}

/* One fresh page. evalOrder "upload-first" matches the real page (sync
   upload.js registers DOMContentLoaded before defer loading.js);
   "loading-first" proves the pair is order-independent. */
function buildWorld(loadingSrc, uploadSrc, evalOrder, withLoading) {
  const button = makeEl();
  const form = makeEl();
  const input = makeEl();
  const msg = makeEl();
  const box = makeEl();
  const zone = makeEl();
  zone.classList = {
    _set: {},
    add(c) {
      this._set[c] = true;
    },
    remove(c) {
      delete this._set[c];
    },
    toggle(c, force) {
      if (force) this._set[c] = true;
      else delete this._set[c];
    }
  };
  const world = {
    button: button,
    form: form,
    input: input,
    msg: msg,
    box: box,
    zone: zone,
    submits: 0,
    busyAtSubmit: null,
    videoDuration: 0,
    deferProbe: false,
    pendingVideos: []
  };
  form.querySelector = (sel) => (sel === '[type="submit"]' ? button : null);
  form.submit = function () {
    world.submits += 1;
    world.busyAtSubmit = button.disabled;
  };
  box.querySelector = (sel) => (sel === ".pf-state-body" ? msg : null);
  box.hidden = true;
  const dcl = [];
  global.window = {
    PARTFIDELITY_LIMITS: {
      maxBytes: MAX_BYTES,
      maxDuration: MAX_DURATION,
      message: MESSAGE,
      probeTimeoutMs: 50
    }
  };
  global.document = {
    readyState: "loading",
    addEventListener: (t, fn) => {
      if (t === "DOMContentLoaded") dcl.push(fn);
    },
    getElementById: (id) =>
      id === "upload-form"
        ? form
        : id === "clip"
          ? input
          : id === "client-error"
            ? box
            : id === "drop-zone"
              ? zone
              : null,
    querySelectorAll: (sel) => (sel === "form[data-pf-busy]" ? [form] : []),
    createElement: (tag) => {
      if (tag !== "video") fail("unexpected createElement " + tag);
      const v = makeEl();
      v.preload = "";
      v.onloadedmetadata = null;
      v.onerror = null;
      let src = "";
      Object.defineProperty(v, "src", {
        get: () => src,
        set: (u) => {
          src = u;
          if (world.deferProbe) {
            world.pendingVideos.push(v);
            return;
          }
          v.duration = world.videoDuration;
          if (typeof v.onloadedmetadata === "function") v.onloadedmetadata();
        }
      });
      return v;
    }
  };
  global.URL = {
    createObjectURL: () => "blob:stub",
    revokeObjectURL: () => {}
  };
  if (withLoading) {
    if (evalOrder === "upload-first") {
      eval(uploadSrc);
      eval(loadingSrc);
    } else {
      eval(loadingSrc);
      eval(uploadSrc);
    }
  } else {
    eval(uploadSrc);
  }
  dcl.forEach((fn) => fn());
  return world;
}

function submit(world) {
  const ev = {
    defaultPrevented: false,
    preventDefault() {
      this.defaultPrevented = true;
    }
  };
  world.form._fire("submit", ev);
  return ev;
}

function expectReleased(world, label) {
  check(world.button.disabled === false, label + ": button stuck disabled");
  check(
    world.button.attrs["aria-busy"] === "false" ||
      world.button.attrs["aria-busy"] === undefined,
    label + ": aria-busy stuck true"
  );
  check(world.submits === 0, label + ": rejected path submitted");
  check(world.box.hidden === false, label + ": error box not shown");
  check(world.box.focused === true, label + ": error box not focused");
  check(world.msg.textContent === MESSAGE, label + ": wrong error message");
}

async function scenarioTooLarge(loadingSrc, uploadSrc, evalOrder, withLoading) {
  const label = `too-large/${evalOrder}` + (withLoading ? "" : "/no-pf");
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, withLoading);
  world.input.files = [{ size: MAX_BYTES + 1 }];
  const ev = submit(world);
  check(ev.defaultPrevented === true, label + ": submit not prevented");
  expectReleased(world, label);
}

async function scenarioTooLong(loadingSrc, uploadSrc, evalOrder, withLoading) {
  const label = `too-long/${evalOrder}` + (withLoading ? "" : "/no-pf");
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, withLoading);
  world.input.files = [{ size: 10 }];
  world.videoDuration = MAX_DURATION + 1;
  submit(world);
  await tick();
  await tick();
  expectReleased(world, label);
}

async function scenarioValid(loadingSrc, uploadSrc, evalOrder, withLoading) {
  const label = `valid/${evalOrder}` + (withLoading ? "" : "/no-pf");
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, withLoading);
  world.input.files = [{ size: 10 }];
  world.videoDuration = MAX_DURATION - 1;
  submit(world);
  await tick();
  await tick();
  check(world.submits === 1, label + ": valid path did not submit");
  check(world.busyAtSubmit === true, label + ": busy not set before submit");
  check(world.button.disabled === true, label + ": button not locked");
  check(
    world.button.attrs["aria-busy"] === "true",
    label + ": aria-busy not true"
  );
  check(world.box.hidden === true, label + ": error box wrongly shown");
}

async function scenarioDoubleSubmit(loadingSrc, uploadSrc, evalOrder) {
  const label = `double-submit/${evalOrder}`;
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, true);
  world.input.files = [{ size: 10 }];
  world.videoDuration = MAX_DURATION - 1;
  submit(world);
  submit(world);
  await tick();
  await tick();
  check(world.submits === 1, label + ": probing submitted twice");
}

/* A probe for file A that resolves after file B was picked must
   neither submit B nor judge it; B is probed and judged on its own. */
async function scenarioStaleProbe(loadingSrc, uploadSrc, evalOrder) {
  const label = `stale-probe/${evalOrder}`;
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, true);
  world.deferProbe = true;
  world.videoDuration = MAX_DURATION - 1;
  world.input.files = [{ size: 10 }];
  submit(world);
  check(world.pendingVideos.length === 1, label + ": A probe not started");
  world.input.files = [{ size: 10 }];
  world.input._fire("change", {});
  submit(world);
  check(world.pendingVideos.length === 2, label + ": B probe not started");
  const [videoA, videoB] = world.pendingVideos.splice(0, 2);
  videoA.duration = MAX_DURATION - 1;
  videoA.onloadedmetadata();
  await tick();
  await tick();
  check(world.submits === 0, label + ": stale probe submitted the new file");
  check(world.box.hidden === true, label + ": stale probe showed an error");
  videoB.duration = MAX_DURATION + 1;
  videoB.onloadedmetadata();
  await tick();
  await tick();
  check(world.submits === 0, label + ": overlong file submitted");
  check(world.box.hidden === false, label + ": overlong file not rejected");
  check(world.msg.textContent === MESSAGE, label + ": wrong error message");
  check(!("probing" in world.form.dataset), label + ": probing flag stuck");
}

/* r4: dropping file B during A's probe invalidates A exactly like the
   change handler — B is probed on submit, the stale callback judges
   nothing, and the button is never left disabled. */
async function scenarioStaleDrop(loadingSrc, uploadSrc, evalOrder) {
  const label = `stale-drop/${evalOrder}`;
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, true);
  world.deferProbe = true;
  world.videoDuration = MAX_DURATION - 1;
  world.input.files = [{ size: 10 }];
  submit(world);
  check(world.pendingVideos.length === 1, label + ": A probe not started");
  world.zone._fire("drop", {
    dataTransfer: { files: [{ size: 10 }] },
    preventDefault() {}
  });
  submit(world);
  check(world.pendingVideos.length === 2, label + ": B probe not started");
  const [videoA, videoB] = world.pendingVideos.splice(0, 2);
  videoA.duration = MAX_DURATION - 1;
  videoA.onloadedmetadata();
  await tick();
  await tick();
  check(world.submits === 0, label + ": stale probe submitted the new file");
  check(world.box.hidden === true, label + ": stale probe showed an error");
  videoB.duration = MAX_DURATION + 1;
  videoB.onloadedmetadata();
  await tick();
  await tick();
  expectReleased(world, label);
  check(!("probing" in world.form.dataset), label + ": probing flag stuck");
}

/* A probe that never resolves settles through the timeout so a
   later submit is judged instead of silently no-op'ing. */
async function scenarioProbeTimeout(loadingSrc, uploadSrc, evalOrder) {
  const label = `probe-timeout/${evalOrder}`;
  const world = buildWorld(loadingSrc, uploadSrc, evalOrder, true);
  world.deferProbe = true;
  world.input.files = [{ size: 10 }];
  submit(world);
  await sleep(150);
  await tick();
  check(world.submits === 1, label + ": timed-out probe never settled");
  check(!("probing" in world.form.dataset), label + ": probing flag stuck");
}

async function main() {
  const loadingSrc = fs.readFileSync(process.argv[2], "utf8");
  const uploadSrc = fs.readFileSync(process.argv[3], "utf8");
  for (const order of ["upload-first", "loading-first"]) {
    await scenarioTooLarge(loadingSrc, uploadSrc, order, true);
    await scenarioTooLong(loadingSrc, uploadSrc, order, true);
    await scenarioValid(loadingSrc, uploadSrc, order, true);
    await scenarioDoubleSubmit(loadingSrc, uploadSrc, order);
    await scenarioStaleProbe(loadingSrc, uploadSrc, order);
    await scenarioStaleDrop(loadingSrc, uploadSrc, order);
  }
  await scenarioProbeTimeout(loadingSrc, uploadSrc, "upload-first");
  // upload.js alone (PFLoading not yet loaded): same outcomes, no throw.
  await scenarioTooLarge(loadingSrc, uploadSrc, "upload-first", false);
  await scenarioTooLong(loadingSrc, uploadSrc, "upload-first", false);
  await scenarioValid(loadingSrc, uploadSrc, "upload-first", false);
  console.log("BUSY-ORDER-OK");
}

main().catch((err) => fail("threw: " + (err && err.stack)));
