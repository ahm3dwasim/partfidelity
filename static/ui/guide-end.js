/* PartFidelity guide end. Progressive enhancement for
   the guide top and end: persisted tools checkboxes and the share copy
   button. The server renders the title summary, side plate, checklist,
   and share section, so the page reads identically with this file
   missing, minus the enhancements. The copy control only ever uses the
   server-issued expiring URL from the nearest data-share-url scope
   (main on reader pages); without one the script fails closed
   (download only) and never falls back to the page URL. No network,
   no timers, no animation. Every DOM lookup is guarded so unknown
   markup is left alone instead of throwing. */
(function () {
  "use strict";

  var STORAGE_PREFIX = "pf-guide-end-tools:";
  var TIME_RE = /(\d+):(\d\d)\D+(\d+):(\d\d)/;

  function ready(fn) {
    if (document.readyState !== "loading") {
      fn();
    } else {
      document.addEventListener("DOMContentLoaded", fn);
    }
  }

  function text(el) {
    return el ? (el.textContent || "").replace(/\s+/g, " ").trim() : "";
  }

  function toSeconds(minutes, seconds) {
    return parseInt(minutes, 10) * 60 + parseInt(seconds, 10);
  }

  /* "3 steps · about 1 minute · Easy" from the rendered steps, or "" when
     the times cannot be parsed (the meta line is then left out). */
  function summaryText(main) {
    var steps = main.querySelectorAll("article.step");
    if (!steps.length) {
      return "";
    }
    // Steps hidden behind an unacknowledged stop are not in the DOM; the
    // locked note names their range, so the count stays truthful.
    var total = steps.length;
    var locked = main.querySelector(".locked");
    if (locked) {
      var range = /Steps\s+(\d+)\s*[–-]\s*(\d+)/.exec(text(locked));
      if (range) {
        total = Math.max(total, parseInt(range[2], 10));
      }
    }
    var badge = main.querySelector('.badges > [class^="badge-"]');
    var difficulty = text(badge);
    var times = main.querySelectorAll("article.step .time");
    var span = "";
    if (times.length) {
      var first = TIME_RE.exec(text(times[0]));
      var last = TIME_RE.exec(text(times[times.length - 1]));
      if (first && last) {
        var minutes = Math.max(
          1,
          Math.round(
            (toSeconds(last[3], last[4]) - toSeconds(first[1], first[2])) / 60
          )
        );
        span =
          "about " + minutes + (minutes === 1 ? " minute" : " minutes");
      }
    }
    var parts = [
      total + (total === 1 ? " step" : " steps"),
      span,
      difficulty,
    ].filter(function (part) {
      return part !== "";
    });
    return parts.join(" · ");
  }

  function buildTitle(main) {
    var heading = main.querySelector("h1");
    var badges = main.querySelector(".badges");
    if (!heading || !badges || main.querySelector(".ge-title")) {
      return;
    }
    var summary = summaryText(main);
    var meta = null;
    if (summary !== "") {
      meta = document.createElement("p");
      meta.className = "ge-meta";
      meta.setAttribute("data-copy", "guide.meta");
      meta.textContent = summary;
      badges.parentNode.insertBefore(meta, badges.nextSibling);
    }
    var wrap = document.createElement("div");
    wrap.className = "ge-title";
    heading.parentNode.insertBefore(wrap, heading);
    var copy = document.createElement("div");
    copy.className = "ge-title-text";
    copy.appendChild(heading);
    copy.appendChild(badges);
    if (meta) {
      copy.appendChild(meta);
    }
    wrap.appendChild(copy);
    // Side plate: the guide's own first frame, already on the page, shown
    // again up top on wide screens (CSS hides it on phones).
    var firstImg = main.querySelector("article.step img[src]");
    if (firstImg && firstImg.getAttribute("src")) {
      var figure = document.createElement("figure");
      figure.className = "ge-plate";
      figure.setAttribute("aria-hidden", "true");
      var plate = document.createElement("img");
      plate.setAttribute("src", firstImg.getAttribute("src"));
      plate.setAttribute("alt", "");
      var caption = document.createElement("figcaption");
      caption.setAttribute("data-copy", "guide.plate");
      caption.textContent = "First frame of this guide.";
      figure.appendChild(plate);
      figure.appendChild(caption);
      wrap.appendChild(figure);
    }
  }

  function loadChecks(key) {
    try {
      var raw = window.localStorage.getItem(key);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      return [];
    }
  }

  function saveChecks(key, names) {
    try {
      window.localStorage.setItem(key, JSON.stringify(names));
    } catch (err) {
      /* Private mode: the list still ticks, it just forgets. */
    }
  }

  function checkedNames(list) {
    var names = [];
    Array.prototype.forEach.call(
      list.querySelectorAll(".ge-check"),
      function (row) {
        var tick = row.querySelector("input");
        if (tick && tick.checked) {
          names.push(text(row));
        }
      }
    );
    return names;
  }

  function restoreChecks(list, key) {
    var saved = loadChecks(key);
    Array.prototype.forEach.call(
      list.querySelectorAll(".ge-check input"),
      function (box) {
        var row = box.parentNode;
        if (saved.indexOf(text(row)) !== -1) {
          box.checked = true;
        }
        box.addEventListener("change", function () {
          saveChecks(key, checkedNames(list));
        });
      }
    );
  }

  function enhanceTools(main) {
    var list = main.querySelector(".tool-list");
    if (!list) {
      return;
    }
    var key = STORAGE_PREFIX + location.pathname;
    if (list.querySelector(".ge-check input")) {
      // Server-rendered baseline: only restore persisted ticks and wire
      // saving; the checkboxes themselves work without this script.
      restoreChecks(list, key);
      return;
    }
    var saved = loadChecks(key);
    var items = list.querySelectorAll("li");
    Array.prototype.forEach.call(items, function (item) {
      var name = text(item);
      if (name === "") {
        return;
      }
      item.textContent = "";
      var label = document.createElement("label");
      label.className = "ge-check";
      label.setAttribute("data-copy", "guide.tools.item");
      var box = document.createElement("input");
      box.type = "checkbox";
      box.checked = saved.indexOf(name) !== -1;
      var word = document.createElement("span");
      word.textContent = name;
      box.addEventListener("change", function () {
        saveChecks(key, checkedNames(list));
      });
      label.appendChild(box);
      label.appendChild(word);
      item.appendChild(label);
    });
    list.classList.add("ge-checklist");
  }

  function downloadName(src, index) {
    var ext = "png";
    var data = /^data:image\/([a-z0-9+]+)/i.exec(src);
    if (data) {
      var kind = data[1].toLowerCase();
      ext =
        kind === "svg+xml"
          ? "svg"
          : kind === "jpeg"
            ? "jpg"
            : kind === "png" || kind === "webp"
              ? kind
              : "png";
    } else {
      var match = /\.([a-z0-9]{2,4})(?:[?#]|$)/i.exec(src);
      if (match) {
        ext = match[1].toLowerCase();
      }
    }
    return "partfidelity-step-" + (index + 1) + "." + ext;
  }

  function wireShare(section, main) {
    var downloadBtn = section.querySelector("[data-ge-download]");
    var status = section.querySelector("[data-ge-status]");
    if (!downloadBtn || !status) {
      return;
    }
    var scope = section.closest("[data-share-url]") || main;
    var attr = scope.getAttribute("data-share-url");
    // Trimmed like the server normalizes: None, empty, and
    // whitespace-only all fail closed, identically with and without JS.
    var shareUrl = attr ? attr.trim() : "";
    var consent = section.querySelector("[data-ge-consent]");
    var link = section.querySelector("[data-ge-link]");
    var copyBtn = section.querySelector("[data-ge-copy]");
    if (shareUrl === "") {
      // Fail closed: without a server-issued expiring URL there is no
      // copy control, no consent, and no 24-hour promise; the download
      // stays. Never falls back to the page URL.
      if (link) {
        link.remove();
      }
      if (copyBtn) {
        copyBtn.remove();
      }
      if (consent) {
        var label = consent.parentNode;
        if (label && label.tagName === "LABEL") {
          label.remove();
        } else {
          consent.remove();
        }
      }
      downloadBtn.disabled = false;
    } else {
      if (link) {
        link.value = shareUrl;
      }
      if (consent) {
        consent.addEventListener("change", function () {
          var on = consent.checked;
          if (copyBtn) {
            copyBtn.disabled = !on;
          }
          downloadBtn.disabled = !on;
          status.textContent = on
            ? ""
            : "Tick the box to share your own repair.";
        });
      }
      if (copyBtn && link) {
        copyBtn.addEventListener("click", function () {
          function done(ok) {
            status.textContent = ok
              ? "Link copied."
              : "Copy failed — long-press the link to copy it.";
          }
          if (
            navigator.clipboard &&
            typeof navigator.clipboard.writeText === "function"
          ) {
            navigator.clipboard.writeText(link.value).then(
              function () {
                done(true);
              },
              function () {
                done(false);
              }
            );
          } else {
            link.select();
            try {
              done(document.execCommand("copy"));
            } catch (err) {
              done(false);
            }
          }
        });
      }
    }
    downloadBtn.addEventListener("click", function () {
      var targets = main.querySelectorAll(
        "article.step img[src], img[data-ge-download]"
      );
      if (!targets.length) {
        status.textContent = "No images on this page yet.";
        return;
      }
      Array.prototype.forEach.call(targets, function (img, index) {
        var src = img.getAttribute("src") || "";
        if (src === "") {
          return;
        }
        var anchor = document.createElement("a");
        anchor.href = src;
        anchor.download = downloadName(src, index);
        document.body.appendChild(anchor);
        anchor.click();
        anchor.remove();
      });
      status.textContent =
        "Downloading " +
        targets.length +
        (targets.length === 1 ? " image…" : " images…");
    });
  }

  function enhanceShare(main) {
    // The server renders the share section; the script only wires it.
    // Without one there is nothing to share, so nothing is created.
    // Each section reads its nearest data-share-url scope (main on
    // reader pages).
    Array.prototype.forEach.call(
      main.querySelectorAll(".ge-share"),
      function (section) {
        wireShare(section, main);
      }
    );
  }

  ready(function () {
    var main = document.getElementById("main");
    if (!main) {
      return;
    }
    buildTitle(main);
    enhanceTools(main);
    enhanceShare(main);
  });
})();
