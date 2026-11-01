/* Live reader stop. Without this script every step and warning stays shown; with it,
 * each later step folds to its number, title and warnings until the button is pressed. */
(function () {
  "use strict";
  document.documentElement.classList.add("pf-js");
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("[data-stop-ack]").forEach(function (button) {
      var stop = button.closest("[data-stop]");
      var after = stop && stop.nextElementSibling;
      if (!after) return;
      after.id = "after-stop-" + stop.getAttribute("data-stop");
      button.setAttribute("aria-controls", after.id);
      button.setAttribute("aria-expanded", "false");
    });
  });
  document.addEventListener("click", function (event) {
    var button = event.target.closest && event.target.closest("[data-stop-ack]");
    if (!button) return;
    var stop = button.closest("[data-stop]");
    var after = stop && stop.nextElementSibling;
    if (!after || !after.classList.contains("after-stop")) return;
    stop.classList.add("is-acked");
    after.classList.add("is-open");
    var done = stop.querySelector("[data-stop-done]");
    if (done) done.hidden = false;
    button.setAttribute("aria-expanded", "true");
    button.hidden = true;
    var next = after.querySelector("article");
    if (next) {
      next.setAttribute("tabindex", "-1");
      next.focus();
    }
    document.dispatchEvent(new CustomEvent("pf:stop-open", { detail: { stop: stop } }));
  });
})();
