/* Waiting screen: the clock ticks from the server's own start
   time (never a finish estimate), and while the job runs the page asks the
   same URL for fresh state every data-refresh seconds and swaps in the live
   parts. When the guide is ready the server redirects and so does this page.
   Without script the <noscript> refresh does the same job. */
(function () {
  "use strict";
  var root = document.querySelector(".pf-wait");
  if (!root) {
    return;
  }

  function clockText(s) {
    s = Math.max(0, Math.floor(s));
    var h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
    var mm = h ? String(m).padStart(2, "0") : String(m);
    return (h ? h + ":" : "") + mm + ":" + String(r).padStart(2, "0");
  }

  function tick() {
    var clock = document.querySelector(".pf-wait-clock");
    var out = document.getElementById("pf-wait-clock");
    if (!clock || !out) {
      return;
    }
    var start = Date.parse(clock.getAttribute("data-start") || "");
    var endAttr = clock.getAttribute("data-end");
    if (!isFinite(start)) {
      return;
    }
    var end = endAttr ? Date.parse(endAttr) : Date.now();
    out.textContent = clockText((end - start) / 1000);
  }

  /* Run-log times arrive in UTC; show them in the viewer's own clock. */
  function localTimes() {
    var rows = document.querySelectorAll(".pf-wait-log time[datetime]");
    for (var i = 0; i < rows.length; i++) {
      var d = new Date(rows[i].getAttribute("datetime"));
      if (!isNaN(d)) {
        rows[i].textContent = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
      }
    }
  }

  tick();
  localTimes();
  if (!root.hasAttribute("data-running")) {
    return;
  }
  var timer = setInterval(tick, 1000);
  var every = Math.max(1, parseInt(root.getAttribute("data-refresh"), 10) || 3) * 1000;

  function swap(id, doc) {
    var next = doc.getElementById(id);
    var here = document.getElementById(id);
    if (next && here) {
      here.replaceWith(next);
    }
  }

  function poll() {
    fetch(window.location.href, { headers: { Accept: "text/html" }, cache: "no-store" })
      .then(function (res) {
        if (res.redirected) {
          /* Only ever our own pages (the finished guide); anything else
             falls back to a plain reload, which the server decides. */
          if (new URL(res.url).origin === window.location.origin) {
            window.location.assign(res.url);
          } else {
            window.location.reload();
          }
          return null;
        }
        return res.ok ? res.text() : null;
      })
      .then(function (text) {
        if (!text) {
          return;
        }
        var doc = new DOMParser().parseFromString(text, "text/html");
        var next = doc.querySelector(".pf-wait");
        if (!next) {
          return;
        }
        if (!next.hasAttribute("data-running")) {
          /* The job stopped: load the final page whole (actions change). */
          window.location.reload();
          return;
        }
        swap("pf-wait-live", doc);
        swap("pf-wait-frames", doc);
        tick();
        localTimes();
      })
      .catch(function () { /* Offline for a moment: try again next round. */ })
      .then(function () { setTimeout(poll, every); });
  }
  setTimeout(poll, every);
  window.addEventListener("pagehide", function () { clearInterval(timer); });
})();
