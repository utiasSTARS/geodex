// Page behavior for the geodex docs.
//
// A recorded viser scene loads its player only when the reader asks for it, so a page with
// several scenes stays light until one is opened. Without JavaScript the still stays.
document.documentElement.classList.add("geodex-js");

document.addEventListener("click", (event) => {
  const button = event.target.closest(".geodex-scene-open");
  if (!button) return;
  const scene = button.closest(".geodex-scene");
  const frame = document.createElement("iframe");
  frame.src = scene.dataset.src;
  frame.title = scene.dataset.title || "Recorded 3D scene";
  frame.setAttribute("allowfullscreen", "");
  scene.replaceChildren(frame);
  scene.classList.add("geodex-scene-live");
});

// plotly sizes a figure to its container when it first draws, which can happen before the
// theme's layout settles, so every figure resizes once the page has loaded.
window.addEventListener("load", () => {
  if (!window.Plotly) return;
  document.querySelectorAll(".geodex-plotly .js-plotly-plot").forEach((plot) => {
    window.Plotly.Plots.resize(plot);
  });
});

// A reader who asks for reduced motion sees the poster of a landing-page video.
document.addEventListener("DOMContentLoaded", () => {
  if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  document.querySelectorAll("video.landing-media").forEach((video) => {
    video.removeAttribute("autoplay");
    video.pause();
  });
});

// Inline math keeps the punctuation that follows it on the same line.
window.addEventListener("load", () => {
  const glue = () => {
    document.querySelectorAll("mjx-container:not([display='true'])").forEach((container) => {
      // Sphinx wraps each formula in a span.math, and the punctuation follows that span.
      const math = container.closest("span.math") || container;
      const next = math.nextSibling;
      if (!next || next.nodeType !== Node.TEXT_NODE) return;
      const match = next.textContent.match(/^[,.;:)]+/);
      if (!match) return;
      const span = document.createElement("span");
      span.className = "geodex-nowrap";
      math.replaceWith(span);
      span.append(math, match[0]);
      next.textContent = next.textContent.slice(match[0].length);
    });
  };
  const startup = window.MathJax && window.MathJax.startup;
  if (startup && startup.promise) startup.promise.then(glue);
  else glue();
});
