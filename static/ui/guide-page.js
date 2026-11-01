/* Guide page: the stage on the left follows the active step.
   It shows that step's real frame, outlined (the labelled copy) or the
   original, an outline toggle, Play from the step's start when the guide
   has its clip, and a Done check per step (kept in this browser only).
   Works on the server-rendered cards; missing markup is a no-op. */
(function () {
  "use strict";
  var root = document.querySelector(".guide-cols");
  var stage = document.getElementById("gd-stage");
  var cards = document.querySelectorAll(".guide .card");
  if (!root || !stage || !cards.length) {
    return;
  }
  var $ = function (id) { return document.getElementById(id); };
  var frame = $("gd-frame");
  var video = $("guide-video");
  var ctl = stage.querySelector(".gd-ctl");
  var outlinesBtn = $("gd-outlines");
  var playBtn = $("gd-play");
  var toastBox = $("gd-toast");
  var hasClip = !!(video && video.getAttribute("src"));
  var outlines = true;
  var intros = document.querySelectorAll(".guide .card.gd-intro").length;
  var active = 0;
  var part = 0;
  var key = "pf-guide-done:" + (root.getAttribute("data-guide") || "");

  function mmss(s) {
    s = Math.max(0, Math.round(s || 0));
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  function toast(text) {
    if (!toastBox) {
      return;
    }
    toastBox.textContent = text;
    toastBox.classList.add("is-on");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(function () { toastBox.classList.remove("is-on"); }, Number(root.dataset.toastMs));
  }

  function times(card) {
    var t = card.querySelector(".step-time");
    var start = t ? parseFloat(t.getAttribute("data-start")) : NaN;
    var end = t ? parseFloat(t.getAttribute("data-end")) : NaN;
    return { start: start, end: end };
  }

  /* The step's frame: its first labelled part shot when outlines are on,
     otherwise the original frame. Both are the server's own image URLs. */
  function frameFor(card) {
    var shot = card.querySelector(".shot img");
    var all = card.querySelectorAll(".part-shot img[data-labelled]");
    var labelled = all.length ? all[Math.min(part, all.length - 1)] : null;
    var original = shot ? shot.getAttribute("data-original") || shot.getAttribute("src") : "";
    if (outlines && labelled) {
      return { src: labelled.getAttribute("data-labelled"), alt: labelled.getAttribute("alt"), outlined: true };
    }
    return { src: original, alt: shot ? shot.getAttribute("alt") : "", outlined: false };
  }

  /* A step with several labelled parts gets one button per part under the
     frame (the cards' own part images are hidden behind the stage). */
  var partsBox = document.createElement("div");
  partsBox.className = "gd-parts";
  partsBox.setAttribute("role", "group");
  partsBox.setAttribute("aria-label", "Parts in this step");
  partsBox.hidden = true;
  var cap = stage.querySelector(".gd-cap");
  if (cap) {
    cap.parentNode.insertBefore(partsBox, cap);
  }
  function renderParts(card) {
    var all = card.querySelectorAll(".part-shot img[data-labelled]");
    partsBox.textContent = "";
    partsBox.hidden = !(outlines && all.length > 1);
    if (partsBox.hidden) {
      return;
    }
    for (var i = 0; i < all.length; i++) {
      var b = document.createElement("button");
      var name = (all[i].getAttribute("alt") || "").replace(/^Step \d+ /, "").replace(/: labelled frame$/, "");
      b.type = "button";
      b.textContent = (i + 1) + " " + name;
      b.setAttribute("aria-pressed", String(i === Math.min(part, all.length - 1)));
      b.addEventListener("click", (function (n) {
        return function () { part = n; render(); partsBox.querySelectorAll("button")[n].focus(); };
      })(i));
      partsBox.appendChild(b);
    }
  }

  function lastEnd() {
    var max = 0;
    for (var i = 0; i < cards.length; i++) {
      var end = times(cards[i]).end;
      if (isFinite(end) && end > max) {
        max = end;
      }
    }
    if (video && isFinite(video.duration) && video.duration > max) {
      max = video.duration;
    }
    return max;
  }

  function render() {
    var card = cards[active];
    var f = frameFor(card);
    if (frame) {
      if (f.src) {
        if (frame.getAttribute("src") !== f.src) {
          frame.setAttribute("src", f.src);
        }
        frame.alt = f.alt || "";
        frame.hidden = false;
      } else {
        frame.hidden = true;
      }
    }
    renderParts(card);
    var note = card.querySelector(".gd-step-note");
    var noteBox = $("gd-stage-note");
    if (noteBox) {
      noteBox.hidden = !note;
      noteBox.textContent = note ? note.getAttribute("data-note-stage") : "";
    }
    if (window.pfShowNow) {
      window.pfShowNow(card);
    }
    stage.classList.toggle("is-plain", !f.outlined);
    $("gd-badge").textContent = f.outlined ? "Outlined copy of your frame" : "Original frame";
    $("gd-cap").textContent = f.src ? (f.outlined ? "Labels drawn on a copy of the frame" : "Original frame from your clip") : "No frame for this step";
    $("gd-count").textContent = cards[active].classList.contains("gd-intro") ? "Intro" : "Step " + (active + 1 - intros) + " / " + (cards.length - intros);
    var t = times(card);
    var total = lastEnd();
    if (isFinite(t.start)) {
      $("gd-start").textContent = mmss(t.start);
      $("gd-end").textContent = mmss(isFinite(t.end) ? t.end : total);
      if (total > 0) {
        var track = $("gd-track");
        track.style.setProperty("--gd-from", (t.start / total * 100) + "%");
        track.style.setProperty("--gd-span", (((isFinite(t.end) ? t.end : total) - t.start) / total * 100) + "%");
      }
    }
    for (var i = 0; i < cards.length; i++) {
      cards[i].classList.toggle("is-active", i === active);
    }
    placePins();
  }

  /* On wide screens the card frames are hidden behind the stage, so the
     active step's pins move onto the stage frame (same ids, so the legend's
     hover still lights them) and go home when the step changes. Only over
     the original frame: the outlined copy has its labels drawn in already,
     and pins on top of it would number each part twice. */
  var pinBox = $("gd-pins");
  function placePins() {
    if (!pinBox) {
      return;
    }
    var moved = pinBox.querySelectorAll(".pf-pin");
    for (var m = 0; m < moved.length; m++) {
      var home = moved[m].pfHome;
      if (home) {
        home.appendChild(moved[m]);
      }
    }
    if (window.innerWidth <= 900 || (frame && frame.hidden) || !stage.classList.contains("is-plain")) {
      return;
    }
    var pins = cards[active].querySelectorAll(".shot .pf-pin");
    for (var p = 0; p < pins.length; p++) {
      pins[p].pfHome = pins[p].parentNode;
      pinBox.appendChild(pins[p]);
    }
  }
  window.addEventListener("resize", function () { placePins(); });

  function go(i) {
    if (i === active || i < 0 || i >= cards.length) {
      return;
    }
    active = i;
    if (!video || video.paused) {
      stage.classList.remove("is-playing");
      stage.classList.add("is-scrubbed");
    }
    part = 0;
    render();
  }

  /* The CSS that hides the cards' own images and the player behind the
     stage waits for this class, so without script the cards keep them. */
  root.classList.add("has-stage");
  stage.classList.add("has-stage");

  /* Card shots hidden behind the stage never load while lazy, which would
     leave their loading state on; load them now (the stage shows the same
     URLs, so they come from cache). */
  var shots = document.querySelectorAll(".guide .card .shot img[loading='lazy']");
  for (var s = 0; s < shots.length; s++) {
    if (shots[s].closest(".shot").offsetParent === null) {
      shots[s].loading = "eager";
    }
  }

  if (ctl) {
    ctl.hidden = false;
  }
  stage.classList.toggle("has-clip", hasClip);
  if (playBtn) {
    playBtn.hidden = !hasClip;
    playBtn.addEventListener("click", function () {
      var t = times(cards[active]);
      stage.classList.add("is-playing");
      stage.classList.remove("is-scrubbed");
      try {
        if (isFinite(t.start)) {
          video.currentTime = t.start;
        }
        var p = video.play();
        if (p && p.catch) {
          p.catch(function () {});
        }
      } catch (error) {
        /* No media yet: the stage still switches to the player. */
      }
      toast("Playing from " + mmss(t.start));
    });
  }
  if (outlinesBtn) {
    outlinesBtn.addEventListener("click", function () {
      outlines = !outlines;
      outlinesBtn.setAttribute("aria-pressed", String(outlines));
      outlinesBtn.textContent = outlines ? "On" : "Off";
      stage.classList.remove("is-playing");
      if (video && !video.paused) {
        video.pause();
      }
      render();
    });
  }

  /* The card in the reading band becomes active on scroll; a click
     on a card (not its controls) makes it active anywhere. */
  var index = new Map();
  /* The observer also fires when a click or a seek scrolls a card into view,
     just after reader.js marked the step the video is at; that mark wins until the hold ends. */
  var heldUntil = 0;
  var held;
  for (var c = 0; c < cards.length; c++) {
    index.set(cards[c], c);
  }
  if ("IntersectionObserver" in window) {
    var io;
    function observeCards() {
      if (io) { io.disconnect(); }
      io = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && (!video || video.paused)) {
            var wait = heldUntil - performance.now();
            if (wait <= 0) {
              go(index.get(e.target));
            } else {  /* a scroll this soon after the mark settles once the hold ends */
              clearTimeout(held);
              held = setTimeout(function () {
                if (!video || video.paused) { go(index.get(e.target)); }
              }, wait);
            }
          }
        });
      }, { rootMargin: "-" + (window.innerHeight * Number(root.dataset.observerTop) / 100) + "px 0px -" + (window.innerHeight * Number(root.dataset.observerBottom) / 100) + "px 0px" });
      cards.forEach(function (card) { io.observe(card); });
    }
    observeCards();
    window.addEventListener("resize", observeCards);
  }
  root.addEventListener("click", function (event) {
    var card = event.target.closest(".card");
    if (!card || event.target.closest("button, a, input, label, summary, details")) {
      return;
    }
    go(index.get(card));
  });
  /* Switching view stops playback and puts the step's frame back: the
     fixing view has no player to show. */
  var viewBox = $("simple-view");
  if (viewBox) {
    viewBox.addEventListener("change", function () {
      stage.classList.remove("is-playing");
      if (video && !video.paused) {
        video.pause();
      }
      render();
    });
  }
  /* reader.js follows the video's time and marks the card it reaches; the
     stage follows that same card, so both always show one active step. */
  document.addEventListener("pf:active-step", function (event) {
    heldUntil = performance.now() + Number(root.dataset.observerHold);
    clearTimeout(held);
    var i = index.get(event.detail && event.detail.card);
    if (i !== undefined) {
      go(i);
    }
  });
  document.addEventListener("pf:step-time-change", render);
  if (video) {
    video.addEventListener("play", function () {
      clearTimeout(held);
      stage.classList.add("is-playing");
      stage.classList.remove("is-scrubbed");
    });
  }

  /* Done: a per-step check the person ticks as they work. */
  var done = {};
  try {
    done = JSON.parse(localStorage.getItem(key) || "{}") || {};
  } catch (error) {
    done = {};
  }
  cards.forEach(function (card) {
    var row = document.createElement("div");
    row.className = "gd-done-row";
    var label = document.createElement("label");
    label.className = "gd-done";
    var box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!done[card.id];
    card.classList.toggle("is-done", box.checked);
    label.appendChild(box);
    label.appendChild(document.createTextNode(" Done"));
    row.appendChild(label);
    box.addEventListener("change", function () {
      done[card.id] = box.checked;
      card.classList.toggle("is-done", box.checked);
      try {
        localStorage.setItem(key, JSON.stringify(done));
      } catch (error) {
        /* Private mode: the tick still shows for this visit. */
      }
      if (box.checked) {
        var next = index.get(card) + 1;
        toast(next < cards.length ? "Step " + (next + 1 - intros) + " is next" : "All steps done");
      }
    });
    card.appendChild(row);
  });

  render();
})();
