// Live motion for the /how-it-works figure: an exploded concept phone beside six stage cards. Scrolling picks the stage
// (the card that crosses a line 45% down the visible window, so up and down both work); the phone lifts and lights that
// stage's layer, and the stepper and "now" line follow. Nothing advances on its own: the tour runs only after "Play the
// tour" and stops on the first wheel, touch, key or click. Calm eased follow; stops offscreen; still under reduced motion.
(() => {
/*GEOM*/
const root = document.querySelector("[data-how]");
if (!root) return;
const cfg = JSON.parse(document.getElementById("how-config").textContent);
const still = matchMedia("(prefers-reduced-motion: reduce)");
const narrow = matchMedia("(max-width: 59.99rem)");
const svg = root.querySelector("svg.how-svg");
const layers = [...svg.querySelectorAll("[data-layer]")].sort((a, b) => a.dataset.layer - b.dataset.layer);
const cards = [...root.querySelectorAll("[data-card]")];
const tabs = [...root.querySelectorAll(".how-stepper [data-stage]")];
const stick = root.querySelector(".how-stick");
const readout = root.querySelector("[data-readout]");
const nowN = root.querySelector("[data-now-n]");
const nowT = root.querySelector("[data-now-t]");
const playBtn = document.querySelector("[data-play]");
const COUNT = cards.length;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const lift = (el, z) => el.setAttribute("transform", "translate(" + (z * VZ[0]).toFixed(2) + " " + (z * VZ[1]).toFixed(2) + ")");
let p = 0, goal = 0, shown = -1, last = 0, visible = false, frame = 0, tour = false, timer = 0;

function bar() { return parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--pf-bar")) || 0; }
// the sticky strip sits at top 0 once stuck (phone); measure its height, not where it is before it sticks
function topClear() { return narrow.matches ? stick.offsetHeight : 0; }
// The stage whose card has crossed the reading line; the first one above the cards, the last one below them.
function fromScroll() {
  const top = topClear();
  const line = top + (innerHeight - bar() - top) * cfg.reading_position;
  let s = 0;
  cards.forEach((c, i) => { if (c.getBoundingClientRect().top <= line) s = i; });
  return s;
}
function draw() {
  layers.forEach((el, i) => {
    const on = clamp(1 - Math.abs(p - i), 0, 1);
    lift(el, (N - 1 - i) * GAP + on * cfg.lift);
    el.querySelector("[data-geometry]").style.filter = "brightness(" + (cfg.brightness_inactive + (cfg.brightness_active - cfg.brightness_inactive) * on).toFixed(3) + ")";
    el.querySelector("[data-glow]").style.opacity = on.toFixed(3);
  });
  if (goal !== shown) {
    shown = goal;
    const s = cfg.stages[goal];
    readout.textContent = s.readout;
    nowN.textContent = String(goal + 1).padStart(2, "0");
    nowT.textContent = s.title;
    layers.forEach((el, i) => el.toggleAttribute("data-on", i === goal));
    tabs.forEach((t, i) => { if (i === goal) t.setAttribute("aria-current", "step"); else t.removeAttribute("aria-current"); });
    cards.forEach((el, i) => el.toggleAttribute("data-on", i === goal));
  }
}
function tick(now) {
  const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 1 / 60;
  last = now;
  p = still.matches ? goal : lerp(p, goal, 1 - Math.exp(-dt / cfg.follow_s));
  if (Math.abs(p - goal) < 0.002) p = goal;
  draw();
  frame = visible && p !== goal ? requestAnimationFrame(tick) : 0;
}
const run = () => { if (!frame) { last = 0; frame = requestAnimationFrame(tick); } };
function follow() { const s = fromScroll(); if (s !== goal) { goal = s; draw(); } run(); }
// Bring a stage's card to the reading position, clear of the sticky figure (phone) and the fixed bottom bar.
function goTo(i) {
  const c = cards[clamp(i, 0, COUNT - 1)];
  const y = scrollY + c.getBoundingClientRect().top - topClear() - cfg.clearance;
  scrollTo({ top: y, behavior: still.matches ? "auto" : "smooth" });
}
function setTour(on) {
  on = on && !still.matches;
  tour = on; clearTimeout(timer); timer = 0;
  if (playBtn) { playBtn.setAttribute("aria-pressed", String(on)); playBtn.textContent = on ? cfg.pause : cfg.play; }
  if (on) step();
}
function step() {
  if (!tour) return;
  if (goal >= COUNT - 1 && timer) { setTour(false); return; }
  goTo(timer ? goal + 1 : (goal >= COUNT - 1 ? 0 : goal));
  timer = setTimeout(step, cfg.dwell_s * 1000);
}
tabs.forEach((t) => t.addEventListener("click", () => { setTour(false); goTo(Number(t.dataset.stage)); }));
root.querySelector(".how-stepper").addEventListener("keydown", (e) => {
  const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (!d) return;
  e.preventDefault();
  const i = clamp(tabs.indexOf(document.activeElement) + d, 0, COUNT - 1);
  tabs[i].focus(); setTour(false); goTo(i);
});
if (playBtn) playBtn.addEventListener("click", () => setTour(!tour));
// any reading gesture hands control back to the reader
for (const ev of ["wheel", "touchstart", "keydown", "pointerdown"]) {
  addEventListener(ev, (e) => {
    const onPlay = e.target.closest && e.target.closest("[data-play]");
    // on the Play button only its own activation keys and clicks are exempt; scrolling keys still hand control back
    if (tour && !(onPlay && ((ev === "pointerdown" || ev === "touchstart") || (ev === "keydown" && (e.key === "Enter" || e.key === " "))))) setTour(false);
  }, { passive: true });
}
addEventListener("scroll", follow, { passive: true });
addEventListener("resize", follow, { passive: true });
new IntersectionObserver((e) => { visible = e[e.length - 1].isIntersecting; if (visible) follow(); }, { rootMargin: "120px 0px" }).observe(root);
still.addEventListener("change", () => { if (still.matches) setTour(false); p = goal; draw(); });
goal = fromScroll(); p = goal; draw();
})();
