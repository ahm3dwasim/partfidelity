/* PartFidelity reader toggle (the reader module). No animation, no network: the page
   works identically with this file missing, minus the Simple view. */
(function () {
  "use strict";
  var box = document.getElementById("simple-view");
  if (!box) {
    return;
  }
  box.addEventListener("change", function () {
    document.body.classList.toggle("simple", box.checked);
  });
})();

/* Desktop two-column guide: clicking a step card seeks the video to that
   step's start and marks the card active. Clicks on controls stay with
   the controls; while the video plays, the active card follows its time.
   The now-playing lines under the video mirror the active card's title
   and time range. No video or no cards is a no-op. No network, no timers. */
(function () {
  "use strict";
  var CARDS = ".card, .step-card, #main .step";

  function stepStart(card) {
    var controls = card.querySelector(".step-time");
    if (!controls) {
      return null;
    }
    var value = parseFloat(controls.getAttribute("data-start") || "");
    return isFinite(value) ? value : null;
  }

  function cardTitle(card) {
    var head = card.querySelector("h2");
    var title = head ? head.textContent.trim() : "";
    var action = card.querySelector(".action");
    if (action) {
      title += ": " + action.textContent.trim();
    }
    return title;
  }

  function cardWindow(card) {
    var win = card.querySelector("[data-window]");
    if (win) {
      return win.textContent.trim();
    }
    var fallback = card.querySelector(".time");
    return fallback ? fallback.textContent.trim() : "";
  }

  function mark(cards, active) {
    for (var i = 0; i < cards.length; i++) {
      if (cards[i] === active) {
        cards[i].classList.add("is-active");
      } else {
        cards[i].classList.remove("is-active");
      }
    }
    if (active) {
      updateNow(active);
      document.dispatchEvent(new CustomEvent("pf:active-step", { detail: { card: active } }));
    }
  }

  var video = document.getElementById("guide-video");
  var cards = document.querySelectorAll(CARDS);
  if (!video || !cards.length) {
    return;
  }
  var nowTitle = document.getElementById("guide-now-title");
  var nowTime = document.getElementById("guide-now-time");

  function updateNow(card) {
    if (!nowTitle || !nowTime) {
      return;
    }
    nowTitle.textContent = cardTitle(card);
    nowTime.textContent = cardWindow(card);
  }

  /* guide-page.js calls this when scrolling moves its active step. */
  window.pfShowNow = updateNow;

  function follow() {
    var now = video.currentTime;
    if (!isFinite(now)) {
      return;
    }
    var current = null;
    for (var i = 0; i < cards.length; i++) {
      var at = stepStart(cards[i]);
      if (at !== null && at <= now) {
        current = cards[i];
      }
    }
    mark(cards, current);
  }

  document.addEventListener("click", function (event) {
    var card = event.target.closest(CARDS);
    if (!card) {
      return;
    }
    if (event.target.closest("button, a, input, label, summary")) {
      return;
    }
    var at = stepStart(card);
    if (at === null) {
      return;
    }
    try {
      video.currentTime = at;
    } catch (error) {
      /* No media yet: the card still marks active. */
    }
    mark(cards, card);
  });
  video.addEventListener("timeupdate", follow);
  follow();
})();

/* After "I checked" or Undo the page reloads at #flag-<label>; move keyboard
   focus to that note so the next Tab reaches its button again. */
(function () {
  "use strict";
  var id = window.location.hash.slice(1);
  if (!id) {
    return;
  }
  var note = document.getElementById(id);
  if (note && note.classList.contains("second-check-flag")) {
    note.focus();
  }
})();
