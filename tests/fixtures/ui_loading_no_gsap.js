// Fixture for tests/test_ui_loading.py: executes the real loading.js under
// node with a tiny DOM stub and NO gsap, proving the fallback path sets
// final state instead of throwing. Usage: node ui_loading_no_gsap.js <loading.js>.
"use strict";

const fs = require("fs");

function fail(message) {
  console.error("NO-GSAP-FAIL " + message);
  process.exit(1);
}

function stubEl(dataStage) {
  const toggles = {};
  return {
    attrs: dataStage === undefined ? {} : { "data-stage": dataStage },
    style: {},
    textContent: "",
    toggles: toggles,
    classList: {
      toggle: (name, on) => {
        toggles[name] = !!on;
      }
    },
    setAttribute: function (k, v) {
      this.attrs[k] = v;
    },
    getAttribute: function (k) {
      return Object.prototype.hasOwnProperty.call(this.attrs, k)
        ? this.attrs[k]
        : null;
    },
    removeAttribute: function (k) {
      delete this.attrs[k];
    }
  };
}

const src = fs.readFileSync(process.argv[2], "utf8");
global.window = {};
global.document = {
  readyState: "complete",
  querySelectorAll: () => [],
  addEventListener: () => {}
};
eval(src);

const PFLoading = global.window.PFLoading;
if (!PFLoading) fail("PFLoading not exposed");
if (PFLoading.STAGES.join(",") !== "upload,extract,locate,segment,readback") {
  fail("STAGES wrong: " + PFLoading.STAGES.join(","));
}

const root = stubEl();
const fill = stubEl();
const label = stubEl();
const items = ["upload", "extract", "locate", "segment", "readback"].map(stubEl);
root.querySelector = (sel) =>
  sel === ".pf-progress-fill" ? fill : sel === ".pf-progress-label" ? label : null;
root.querySelectorAll = () => items;

if (PFLoading.progress(root, "locate") !== true) fail("locate rejected");
if (root.attrs["data-pf-stage"] !== "locate") fail("stage attr not set");
if (root.attrs["aria-valuenow"] !== "60") fail("aria-valuenow wrong");
if (fill.style.transform !== "scaleX(0.6)") fail("fill fallback not set");
if (label.textContent !== "Finding parts — step 3 of 5") {
  fail("label wrong: " + label.textContent);
}
if (items[0].toggles["is-done"] !== true) fail("upload not done");
if (items[1].toggles["is-done"] !== true) fail("extract not done");
if (items[2].toggles["is-current"] !== true) fail("locate not current");
if (items[2].attrs["aria-current"] !== "step") fail("aria-current missing");
if (items[3].toggles["is-current"] === true) fail("segment wrongly current");

const before = JSON.stringify(root.attrs);
if (PFLoading.progress(root, "nope") !== false) fail("unknown stage accepted");
if (JSON.stringify(root.attrs) !== before) fail("unknown stage mutated root");

const button = stubEl();
PFLoading.busy(button, true);
if (button.attrs["aria-busy"] !== "true" || button.disabled !== true) {
  fail("busy(true) incomplete");
}
PFLoading.busy(button, false);
if (button.attrs["aria-busy"] !== "false" || button.disabled !== false) {
  fail("busy(false) incomplete");
}

console.log("NO-GSAP-OK");
