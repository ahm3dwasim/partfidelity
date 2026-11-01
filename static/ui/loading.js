/* PartFidelity loading set. GSAP core drives the motion:
   progress fills tween with overwrite:"auto" (values computed from the real
   stage list), skeleton groups enter with a stagger, images fade in with
   autoAlpha, and gsap.matchMedia honours prefers-reduced-motion. Every call
   falls back to its instant final state when GSAP is absent, so the set
   works as plain CSS + vanilla JS too. */
(function () {
  "use strict";

  /* Real model-call stages, in pipeline order. Must stay identical to
     STAGE_VOCABULARY in app/evidence.py; tests/test_ui_loading.py enforces
     the match. Labels are plain English. */
  var STAGES = ["upload", "extract", "locate", "segment", "readback"];
  var LABELS = {
    upload: "Uploading your video",
    extract: "Finding steps",
    locate: "Finding parts",
    segment: "Outlining parts",
    readback: "Checking labels"
  };

  function reducedMotion() {
    return (
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function gsapIfMotion() {
    if (reducedMotion()) return null;
    return typeof window.gsap === "object" && window.gsap ? window.gsap : null;
  }

  /* Move one progress root to a real stage name. Returns false for an
     unknown stage and leaves the root untouched. */
  function progress(root, stage) {
    if (!root) return false;
    var index = STAGES.indexOf(stage);
    if (index === -1) return false;
    var fill = root.querySelector(".pf-progress-fill");
    var label = root.querySelector(".pf-progress-label");
    var items = root.querySelectorAll(".pf-stages li[data-stage]");
    var frac = (index + 1) / STAGES.length;
    root.setAttribute("data-pf-stage", stage);
    root.setAttribute("aria-valuenow", String(Math.round(frac * 100)));
    if (label) {
      label.textContent =
        LABELS[stage] + " \u2014 step " + (index + 1) + " of " + STAGES.length;
    }
    Array.prototype.forEach.call(items, function (li) {
      var at = STAGES.indexOf(li.getAttribute("data-stage"));
      li.classList.toggle("is-done", at !== -1 && at < index);
      li.classList.toggle("is-current", at === index);
      if (at === index) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });
    if (fill) {
      var gsap = gsapIfMotion();
      if (gsap) {
        gsap.to(fill, {
          scaleX: frac,
          duration: 0.18,
          ease: "power2.out",
          overwrite: "auto"
        });
      } else {
        fill.style.transform = "scaleX(" + frac + ")";
      }
    }
    return true;
  }

  /* Lock or release a submit button with its spinner state. */
  function busy(button, on) {
    if (!button) return;
    button.setAttribute("aria-busy", on ? "true" : "false");
    button.disabled = !!on;
  }

  /* Fade one image in once its bytes are ready. Broken or cached images
     resolve to visible; nothing stays hidden. */
  function fade(img) {
    if (!img) return;
    var gsap = gsapIfMotion();
    function show() {
      if (gsap) {
        gsap.fromTo(
          img,
          { autoAlpha: 0 },
          {
            autoAlpha: 1,
            duration: 0.6,
            ease: "power1.out",
            overwrite: "auto",
            clearProps: "all"
          }
        );
      } else {
        img.style.opacity = "";
        img.style.visibility = "";
      }
    }
    if (img.complete && img.naturalWidth !== 0) {
      show();
      return;
    }
    if (gsap) gsap.set(img, { autoAlpha: 0 });
    else img.style.opacity = "0";
    img.addEventListener("load", show, { once: true });
    img.addEventListener(
      "error",
      function () {
        if (gsap) gsap.set(img, { clearProps: "all" });
        else img.style.opacity = "";
      },
      { once: true }
    );
  }

  function wireStagger(gsap) {
    if (!gsap || typeof gsap.matchMedia !== "function") return;
    var mm = gsap.matchMedia();
    mm.add("(prefers-reduced-motion: no-preference)", function () {
      Array.prototype.forEach.call(
        document.querySelectorAll("[data-pf-stagger]"),
        function (box) {
          gsap.from(box.children, {
            autoAlpha: 0,
            y: 8,
            duration: 0.18,
            ease: "power2.out",
            stagger: 0.06,
            overwrite: "auto",
            clearProps: "all"
          });
        }
      );
    });
  }

  function init(root) {
    var scope = root || document;
    if (
      typeof window.gsap === "object" &&
      window.gsap &&
      typeof window.gsap.defaults === "function"
    ) {
      window.gsap.defaults({ duration: 0.18, ease: "power2.out" });
    }
    Array.prototype.forEach.call(
      scope.querySelectorAll("[data-pf-progress]"),
      function (el) {
        var stage = el.getAttribute("data-pf-stage") || STAGES[0];
        progress(el, stage);
      }
    );
    Array.prototype.forEach.call(
      scope.querySelectorAll("form[data-pf-busy]"),
      function (form) {
        form.addEventListener("submit", function (ev) {
          /* Validators (upload.js) run first and preventDefault on
             rejection: never lock the button down a rejected path. */
          if (ev && ev.defaultPrevented) return;
          busy(form.querySelector('[type="submit"]'), true);
        });
      }
    );
    Array.prototype.forEach.call(
      scope.querySelectorAll("main[data-pf-fade]"),
      function (el) {
        var gsap = gsapIfMotion();
        if (gsap) {
          gsap.from(el, {
            autoAlpha: 0,
            duration: 0.18,
            ease: "power1.out",
            overwrite: "auto",
            clearProps: "all"
          });
        }
      }
    );
    Array.prototype.forEach.call(
      scope.querySelectorAll("img[data-pf-fade]"),
      fade
    );
    wireStagger(gsapIfMotion());
  }

  window.PFLoading = {
    STAGES: STAGES,
    LABELS: LABELS,
    progress: progress,
    busy: busy,
    fade: fade,
    init: init
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      init(document);
    });
  } else {
    init(document);
  }
})();
