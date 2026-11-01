// Step time controls: ±5s nudges and set-start-here from the paused video.
// No network, no drag control (out of scope by brief), no waiting.
// Guide phases are contiguous: moving step N's start also moves step
// N-1's end to the same instant, so no gap or overlap can open.
(function () {
  "use strict";

  function fmt(total) {
    total = Math.max(0, Math.round(total));
    return Math.floor(total / 60) + ":" + String(total % 60).padStart(2, "0");
  }

  function fmtWindow(start, end) {
    return "~" + fmt(start) + "–" + fmt(end);
  }

  function round1(value) {
    return Math.round(value * 10) / 10;
  }

  function paintWindow(root) {
    var window = root.querySelector("[data-window]");
    if (window) {
      window.textContent = fmtWindow(
        parseFloat(root.getAttribute("data-start") || "0"),
        parseFloat(root.getAttribute("data-end") || "0")
      );
    }
  }

  // The previous step's controls in card order, or null for step one.
  // Cards render in step order on every page carrying this partial.
  function previousRoot(root) {
    var cards = document.querySelectorAll(".step-time");
    var index = Array.prototype.indexOf.call(cards, root);
    return index > 0 ? cards[index - 1] : null;
  }

  // Clamp rule: start never drops below zero, never passes the step end,
  // and never passes below the previous step's start (that would invert
  // the previous window once its end follows). The previous step's end
  // always follows the new start, keeping phases contiguous.
  function setStart(root, value) {
    var end = parseFloat(root.getAttribute("data-end") || "0");
    var prev = previousRoot(root);
    var floor = 0;
    if (prev) {
      floor = parseFloat(prev.getAttribute("data-start") || "0");
    }
    var next = round1(Math.min(Math.max(floor, value), end));
    root.setAttribute("data-start", String(next));
    paintWindow(root);
    if (prev) {
      prev.setAttribute("data-end", String(next));
      paintWindow(prev);
    }
    document.dispatchEvent(new CustomEvent("pf:step-time-change"));
    return next;
  }

  function currentStart(root) {
    return parseFloat(root.getAttribute("data-start") || "0");
  }

  document.addEventListener("click", function (event) {
    var nudge = event.target.closest("[data-nudge]");
    var setter = event.target.closest("[data-set-start]");
    if (!nudge && !setter) {
      return;
    }
    var root = event.target.closest(".step-time");
    if (!root) {
      return;
    }
    if (nudge) {
      setStart(root, currentStart(root) + parseFloat(nudge.getAttribute("data-nudge")));
      return;
    }
    var video = document.getElementById(root.getAttribute("data-video") || "");
    var time = video && isFinite(video.currentTime) ? video.currentTime : 0;
    setStart(root, time);
  });
})();
