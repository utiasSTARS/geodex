// Force every mermaid SVG to render at its intrinsic viewBox width.
//
// Mermaid emits width="100%" on the SVG element, even when its config is
// told useMaxWidth=false (the option is not honoured by every diagram type).
// That makes diagrams with smaller viewBoxes scale up more than diagrams
// with bigger viewBoxes, so two class diagrams on the same page render at
// visibly different text sizes. Pinning width/height to the viewBox makes
// every diagram render at 1 viewBox unit = 1 CSS pixel, which gives a
// consistent per-character pixel size across the page. Wider diagrams stay
// inside the column thanks to `max-width: 100%` in custom.css.
(function () {
    function viewBoxOf(svg) {
        var viewBox = svg.getAttribute("viewBox");
        if (!viewBox) return null;
        var parts = viewBox.split(/\s+|,/).map(parseFloat);
        if (parts.length !== 4 || parts.some(isNaN)) return null;
        return { w: parts[2], h: parts[3] };
    }

    function pinAll() {
        var svgs = Array.from(
            document.querySelectorAll("pre.mermaid svg, .mermaid svg")
        );
        if (svgs.length === 0) return;

        // Compute the maximum intrinsic width across every mermaid SVG on
        // the page, and the available container width (use the first
        // mermaid SVG's offset parent as the column).
        var maxIntrinsicW = 0;
        svgs.forEach(function (svg) {
            var vb = viewBoxOf(svg);
            if (vb && vb.w > maxIntrinsicW) maxIntrinsicW = vb.w;
        });
        if (maxIntrinsicW === 0) return;

        var container = svgs[0].parentElement;
        // Walk up to find an element with a real width (the .mermaid <pre>
        // itself may be width:auto and report 0 before we pin its child).
        var containerW = 0;
        var node = container;
        while (node && containerW === 0) {
            containerW = node.clientWidth || 0;
            node = node.parentElement;
        }
        if (containerW === 0) containerW = 720; // sane default

        // One factor for the whole page: scale every diagram by the same
        // amount so per-character pixel size is identical. The widest
        // diagram fits exactly inside the column, narrower diagrams stay
        // proportional and look smaller (which is correct, they have
        // less content to show).
        var factor = Math.min(1, containerW / maxIntrinsicW);

        svgs.forEach(function (svg) {
            var vb = viewBoxOf(svg);
            if (!vb) return;
            var w = vb.w * factor;
            var h = vb.h * factor;
            svg.setAttribute("width", w);
            svg.setAttribute("height", h);
            svg.style.setProperty("width", w + "px", "important");
            svg.style.setProperty("height", h + "px", "important");
            svg.style.setProperty("max-width", "100%", "important");
        });
    }

    // Mermaid renders client-side after DOMContentLoaded. Watch for new
    // SVGs being added inside .mermaid blocks and pin them as they appear.
    var observer = new MutationObserver(function () { pinAll(); });
    function start() {
        observer.observe(document.body, { childList: true, subtree: true });
        pinAll();
    }
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
