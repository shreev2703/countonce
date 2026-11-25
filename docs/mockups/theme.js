// Mockup-only light/dark/auto toggle. No framework; the real frontend is gated on approval.
(function () {
  var root = document.documentElement;
  var order = ["auto", "light", "dark"];
  var saved = null;
  try { saved = localStorage.getItem("co-theme"); } catch (e) {}
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.getElementById("theme-toggle");
    if (!btn) return;
    function label() { btn.textContent = "Theme: " + (root.getAttribute("data-theme") || "auto"); }
    label();
    btn.addEventListener("click", function () {
      var cur = root.getAttribute("data-theme") || "auto";
      var next = order[(order.indexOf(cur) + 1) % order.length];
      if (next === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", next);
      try { localStorage.setItem("co-theme", next); } catch (e) {}
      label();
    });
  });
})();
