// Scripted disagreement replay: no network, no model. State (mode and label) lives in localStorage only.
(function () {
  var live = document.getElementById("replay-live");
  if (!live) return;
  var d = live.dataset, key = d.storage;
  var labelEl = document.querySelector("[data-replay-label]");
  var status = document.querySelector("[data-replay-status]");
  var flagged = live.querySelector("[data-replay-flagged]");
  var done = live.querySelector("[data-replay-done]");
  var persist = document.querySelector("[data-replay-persist]"), persistOk = persist && persist.textContent;
  var error = live.querySelector("[data-replay-error]");
  var state = { mode: "flagged", label: d.label };

  function load() {
    try {
      var s = JSON.parse(localStorage.getItem(key));
      if (s && (s.mode === "confirmed" || (s.mode === "renamed" && typeof s.label === "string" && s.label))) state = s;
    } catch (e) {}
  }
  function save() {
    var ok = true;
    try { if (state.mode === "flagged") localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(state)); } catch (e) { ok = false; }
    if (persist) persist.textContent = ok ? persistOk : d.persistFailed;
  }
  function show(focus) {
    var renamed = state.mode === "renamed";
    labelEl.textContent = renamed ? state.label : d.label;
    status.textContent = state.mode === "flagged" ? d.flagged : renamed ? d.renamed.replace("{label}", function () { return state.label; }) : d.confirmed;
    flagged.hidden = state.mode !== "flagged";
    done.hidden = state.mode === "flagged";
    error.textContent = "";
    if (focus) status.focus();
  }
  function set(next) { state = next; save(); show(true); }

  live.querySelector("[data-replay-confirm]").addEventListener("click", function () { set({ mode: "confirmed" }); });
  live.querySelector("[data-replay-undo]").addEventListener("click", function () { set({ mode: "flagged" }); });
  live.querySelector("[data-replay-rename]").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var value = ev.target.elements.label.value.trim();
    if (!value) { error.textContent = d.empty; return; }
    ev.target.reset();
    set({ mode: "renamed", label: value });
  });
  load();
  show(false);
  live.hidden = false;
})();
