/* the hazard module reader hard stop: acknowledgement is server-owned.
 *
 * The native form POST is the only transition. This script may remember that
 * a submission is pending, keyed by immutable guide revision plus stop id, so
 * the button can expose honest busy state. It never changes
 * data-acknowledged, never reveals steps, and never treats browser storage as
 * proof. Missing revision or stop identity fails closed.
 */
(function () {
  "use strict";

  function oneValue(form, name) {
    var field = form.querySelector('[name="' + name + '"]');
    return field && field.value ? field.value : "";
  }

  function keyFor(revision, stopId) {
    if (!revision || !stopId) {
      return null;
    }
    return (
      "partfidelity-hard-stop-pending-" +
      encodeURIComponent(revision) +
      ":" +
      encodeURIComponent(stopId)
    );
  }

  function pendingKey(form) {
    return keyFor(
      oneValue(form, "guide_revision"),
      oneValue(form, "stop_id")
    );
  }

  function rememberPending(form, key) {
    try {
      window.localStorage.setItem(key, "1");
    } catch (error) {
      /* Private storage failure cannot weaken the native server POST. */
    }
    form.setAttribute("aria-busy", "true");
    form.setAttribute("data-acknowledgement-pending", "");
  }

  function clearStoredPending(key) {
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      /* Server-rendered state remains authoritative. */
    }
  }

  function init() {
    var acknowledged = document.querySelectorAll(
      '[data-hard-stop][data-acknowledged="true"]'
    );
    for (var a = 0; a < acknowledged.length; a++) {
      var acknowledgedKey = keyFor(
        acknowledged[a].getAttribute("data-guide-revision"),
        acknowledged[a].getAttribute("data-stop-id")
      );
      if (acknowledgedKey) {
        clearStoredPending(acknowledgedKey);
      }
    }
    var forms = document.querySelectorAll("[data-hard-stop-form]");
    for (var i = 0; i < forms.length; i++) {
      (function (form) {
        var key = pendingKey(form);
        if (!key) {
          form.addEventListener("submit", function (event) {
            event.preventDefault();
          });
          return;
        }
        form.addEventListener("submit", function () {
          rememberPending(form, key);
        });
      })(forms[i]);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
