// Refuse over-limit clips in the browser before upload, with the settled message.
(function () {
  function limits() {
    var l = window.PARTFIDELITY_LIMITS || {};
    return {
      maxBytes: Number(l.maxBytes || 104857600),
      maxDuration: Number(l.maxDuration || 300),
      probeTimeoutMs: Number(l.probeTimeoutMs || 10000),
      message: String(
        l.message ||
          "Trim it to 300 seconds or less and try again. A 720p phone video is usually under 100 MB."
      ),
    };
  }

  function showError(text) {
    var box = document.getElementById("client-error");
    if (!box) return;
    /* fill the shared pf-state body when present. */
    var msg = box.querySelector(".pf-state-body") || box;
    msg.textContent = text;
    box.hidden = !text;
    // Move keyboard and screen-reader focus to the announced error.
    if (text) box.focus();
  }

  /* Loading state: lock/release the submit button through
     PFLoading when it has loaded, else set the same attributes
     directly. Kept in this one block; the validation below is the
     upload page's. */
  function setBusy(form, on) {
    var btn = form.querySelector('[type="submit"]');
    if (!btn) return;
    if (window.PFLoading && typeof window.PFLoading.busy === "function") {
      window.PFLoading.busy(btn, on);
    } else {
      btn.setAttribute("aria-busy", on ? "true" : "false");
      btn.disabled = !!on;
    }
  }

  /* A probe that never resolves (neither metadata nor error)
     settles as unmeasurable after probeTimeoutMs instead of leaving
     probing=1 forever; the server still enforces the limits. */
  function probeDuration(file, timeoutMs) {
    return new Promise(function (resolve) {
      var done = false;
      var url = null;
      var timer = null;
      function finish(d) {
        if (done) return;
        done = true;
        if (timer) clearTimeout(timer);
        try {
          if (url) URL.revokeObjectURL(url);
        } catch (e) {}
        resolve(d);
      }
      timer = setTimeout(function () {
        finish(0);
      }, timeoutMs);
      try {
        url = URL.createObjectURL(file);
        var v = document.createElement("video");
        v.preload = "metadata";
        v.onloadedmetadata = function () {
          var d = Number(v.duration);
          finish(Number.isFinite(d) ? d : 0);
        };
        v.onerror = function () {
          finish(0);
        };
        v.src = url;
      } catch (e) {
        finish(0);
      }
    });
  }

  function markSending(form) {
    var panel = form.closest(".pf-panel");
    if (panel) panel.classList.add("is-sending");
    var btn = form.querySelector('button[type="submit"]');
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Sending…";
    }
  }

  function showPicked(input, zone) {
    var picked = document.getElementById("picked-file");
    var file = input.files && input.files[0];
    if (zone) zone.classList.toggle("has-file", !!file);
    if (!picked) return;
    picked.textContent = file ? file.name : "";
  }

  /* Every selection change (picker or drop-zone)
     invalidates in-flight probes, so a stale callback can neither submit
     nor judge the newly picked file. */
  var probeGen = 0;

  function wireDropZone(form, input) {
    var zone = document.getElementById("drop-zone");
    if (!zone) return;
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (ev) {
        ev.preventDefault();
        zone.classList.add("is-over");
      });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function (ev) {
        ev.preventDefault();
        zone.classList.remove("is-over");
      });
    });
    zone.addEventListener("drop", function (ev) {
      var files = ev.dataTransfer && ev.dataTransfer.files;
      if (!files || !files.length) return;
      try {
        input.files = files;
      } catch (e) {
        return;
      }
      showError("");
      probeGen += 1;
      delete form.dataset.probing;
      delete form.dataset.durationChecked;
      setBusy(form, false);
      showPicked(input, zone);
    });
    showPicked(input, zone);
  }

  document.addEventListener("DOMContentLoaded", function () {
    var form = document.getElementById("upload-form");
    var input = document.getElementById("clip");
    if (!form || !input) return;
    wireDropZone(form, input);
    form.addEventListener("submit", function (ev) {
      var file = input.files && input.files[0];
      if (!file) return;
      var lim = limits();
      if (file.size > lim.maxBytes) {
        ev.preventDefault();
        setBusy(form, false);
        showError(lim.message);
        return;
      }
      // Duration needs async probing: stop this submit, then re-submit if OK.
      if (!form.dataset.durationChecked) {
        ev.preventDefault();
        /* one probe at a time; the button stays enabled
           until validation passes, so a second submit while probing is a
           no-op instead of a duplicate job. */
        if (form.dataset.probing) return;
        form.dataset.probing = "1";
        setBusy(form, false);
        showError("");
        var gen = probeGen;
        probeDuration(file, lim.probeTimeoutMs).then(function (d) {
          var current = input.files && input.files[0];
          if (gen !== probeGen || current !== file) {
            /* Stale probe: a newer selection owns the form. Only the
               newest probe may clear the flag a newer submit may rely on
               (a same-generation swap is only possible with no probe
               started after it, since starting needs the flag clear). */
            if (gen === probeGen) delete form.dataset.probing;
            return;
          }
          delete form.dataset.probing;
          if (d > lim.maxDuration) {
            setBusy(form, false);
            showError(lim.message);
            return;
          }
          form.dataset.durationChecked = "1";
          setBusy(form, true);
          markSending(form);
          form.submit();
        });
      } else {
        markSending(form);
      }
    });
    input.addEventListener("change", function () {
      showError("");
      probeGen += 1;
      delete form.dataset.probing;
      delete form.dataset.durationChecked;
      showPicked(input, document.getElementById("drop-zone"));
    });
  });
})();
