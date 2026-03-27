/*
 * slider_marks.js
 * ---------------
 * Shows only the mark labels at the two currently selected handle positions
 * for the admdate-slider and age-slider.
 *
 * Approach:
 *   1. CSS hides ALL .rc-slider-mark-text elements by default (in style.css).
 *   2. This script finds the marks at exactly the left% position of each
 *      active handle and makes only those two visible.
 *
 * Fires on every slider value-change via a MutationObserver on the slider
 * containers (handles update their left% when dragged).
 */

(function () {
  "use strict";

  /* ── helpers ────────────────────────────────────────────────────── */

  function pct(value, min, max) {
    if (max === min) return 0;
    return ((value - min) / (max - min)) * 100;
  }

  /**
   * Refresh visible marks for one slider container.
   *
   * @param {HTMLElement} container  - the .rc-slider root element
   * @param {number[]}    values     - [lowHandle, highHandle]
   * @param {number}      min
   * @param {number}      max
   */
  function refreshMarks(container, values, min, max) {
    if (!container) return;

    var marks = container.querySelectorAll(".rc-slider-mark-text");
    if (!marks.length) return;

    // Calculate target left% for each active handle (round to 2 dp)
    var targets = values.map(function (v) {
      return Math.round(pct(v, min, max) * 100) / 100;
    });

    marks.forEach(function (el) {
      // The mark's inline style is "left: X%" — parse it
      var leftStr = (el.style.left || "").replace("%", "").trim();
      var leftVal = Math.round(parseFloat(leftStr) * 100) / 100;

      // Show if this mark's position matches either active handle
      var show = targets.some(function (t) {
        return Math.abs(t - leftVal) < 0.2; // 0.2% tolerance
      });

      el.style.opacity   = show ? "1" : "0";
      el.style.visibility = show ? "visible" : "hidden";
    });
  }

  /* ── slider descriptor table ─────────────────────────────────────── */
  // Each entry describes one slider we want to manage.
  // min/max/sliderMin/sliderMax are read live from the component props
  // injected by Dash into the DOM via data attributes (Dash ≥ 2.x).

  var SLIDERS = [
    { id: "admdate-slider" },
    { id: "age-slider" },
    { id: "admdate-slider-modal" },
    { id: "age-slider-modal" },
  ];

  /* ── main refresh logic ─────────────────────────────────────────── */

  function refreshSlider(sliderId) {
    // Dash renders a wrapping div whose id is the component id,
    // then the .rc-slider inside it.
    var wrapper = document.getElementById(sliderId);
    if (!wrapper) return;

    var slider = wrapper.querySelector(".rc-slider");
    if (!slider) return;

    // Read current handle positions from the handle elements
    var handles = slider.querySelectorAll(".rc-slider-handle");
    if (handles.length < 2) return;

    // Extract left% from each handle's inline style
    var leftPcts = Array.from(handles).map(function (h) {
      return parseFloat((h.style.left || "0").replace("%", ""));
    });

    // Convert left% back to mark index values so we can find the marks
    // The marks are positioned at the same left% as the handles when
    // the handle is at that mark's value — so we just match by left%.
    var marks = slider.querySelectorAll(".rc-slider-mark-text");
    marks.forEach(function (el) {
      var markLeft = Math.round(parseFloat((el.style.left || "0").replace("%", "")) * 100) / 100;
      var show = leftPcts.some(function (hp) {
        return Math.abs(Math.round(hp * 100) / 100 - markLeft) < 0.5;
      });
      el.style.opacity    = show ? "1" : "0";
      el.style.visibility = show ? "visible" : "hidden";
      el.style.transition = "opacity 0.15s ease";
    });
  }

  function refreshAll() {
    SLIDERS.forEach(function (s) { refreshSlider(s.id); });
  }

  /* ── MutationObserver: watch for handle moves ────────────────────── */

  var observer = new MutationObserver(function (mutations) {
    // Debounce slightly to avoid thrashing during drag
    clearTimeout(observer._timer);
    observer._timer = setTimeout(refreshAll, 30);
  });

  function attachObservers() {
    SLIDERS.forEach(function (s) {
      var wrapper = document.getElementById(s.id);
      if (!wrapper) return;
      observer.observe(wrapper, {
        attributes: true,
        subtree: true,
        attributeFilter: ["style", "aria-valuenow"],
      });
    });
  }

  /* ── initialise after Dash mounts the layout ─────────────────────── */

  function init() {
    attachObservers();
    refreshAll();
  }

  // Wait for Dash to render its layout (the #_dash-app-content div fills in)
  var initObserver = new MutationObserver(function () {
    // Re-attach whenever sliders appear in the DOM (e.g. accordion opens)
    attachObservers();
    refreshAll();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      init();
      initObserver.observe(document.body, { childList: true, subtree: true });
    });
  } else {
    init();
    initObserver.observe(document.body, { childList: true, subtree: true });
  }
})();
