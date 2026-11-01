/* Guide step card enhancement. No network, no
   timers, no animation: legend/pin highlight is instant, and badge classes
   are derived from the server-rendered status text. Safe to include on any
   page; every lookup is guarded and missing markup is a no-op. */
(function () {
  "use strict";

  function each(list, fn) {
    for (var i = 0; i < list.length; i++) {
      fn(list[i]);
    }
  }

  /* Legend item <-> pin highlight within one card. */
  function linkCard(card) {
    var items = card.querySelectorAll("[data-pin]");
    if (!items.length) {
      return;
    }
    each(items, function (item) {
      var pin = document.getElementById(item.getAttribute("data-pin"));
      if (!pin) {
        return;
      }
      function on() {
        pin.classList.add("on");
      }
      function off() {
        pin.classList.remove("on");
      }
      item.addEventListener("mouseenter", on);
      item.addEventListener("mouseleave", off);
      item.addEventListener("focus", on);
      item.addEventListener("blur", off);
    });
  }

  /* Reader badges share one class for both statuses; mark them so the
     calm proof style can tell "Not checked" from a passed label. */
  function markBadges(root) {
    each(root.querySelectorAll(".badges-list .badge"), function (badge) {
      var text = (badge.textContent || "").toLowerCase();
      if (text.indexOf("not checked") !== -1) {
        badge.classList.add("is-pending");
      } else {
        badge.classList.add("is-ok");
      }
    });
  }

  /* The loading pulse runs only under .is-loading; drop it once the frame
     img decodes (load fires post-decode) or errors, or at once when the
     img already finished. No timers: cached and broken imgs settle here. */
  function clearLoading(frame) {
    var img = frame.querySelector("img");
    if (!img || img.complete) {
      frame.classList.remove("is-loading");
      return;
    }
    function done() {
      frame.classList.remove("is-loading");
    }
    img.addEventListener("load", done);
    img.addEventListener("error", done);
  }

  function init() {
    each(
      document.querySelectorAll(".pf-card, .guide .card, #main .step"),
      linkCard
    );
    each(document.querySelectorAll(".pf-frame"), clearLoading);
    markBadges(document);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
