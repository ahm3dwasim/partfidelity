// Builds the /how-it-works phone: an exploded concept phone (glass, display, frame, battery, board, back with camera), one
// labelled layer per pipeline stage, drawn with the anatomy kit's solids and tinted by material. The phone is the picture of
// the idea, not the sample device: no stage processes hardware. Label words are numbered placeholders (%%L<n>%%, from 0) the route fills
// from config/how_it_works.toml.
// Run: node tools/how_figure/build.mjs
// Writes templates/partials/ui/how-figure.svg (static art), static/ui/how-figure.js (live motion; hand-written in
// tools/how_figure/live.js with the camera numbers filled in here) and static/ui/how-iso.css (the kit's styles).
// The kit is the anatomy skill's iso-kit (MIT), vendored beside this file.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as k from "./vendor/iso-kit.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = join(HERE, "..", "..");
const ART_W = 330; // the phone
const LABEL_W = 165; // the label column to its right
const WIDTH = ART_W + LABEL_W;
const HEIGHT = 440;

// ---- world: x across the phone, y along it, z up; about 1 unit per mm of a real phone, a little fat for the eye ----
const PH = { x: -34, y: -70, w: 68, d: 140, r: 11 };
const GAP = 46; // layer spacing when exploded
const N = 6;
const TH = [2.4, 2.8, 5.4, 4.6, 3.4, 2.6]; // layer thickness: glass, display, frame, battery, board, back

const P = k.fitProjection(
  [...k.boxCorners(PH, 0, 0), ...k.boxCorners(PH, (N - 1) * GAP + TH[0] + 6, (N - 1) * GAP + TH[0] + 6)],
  ART_W,
  HEIGHT,
  { pad: 12, azimuth: 36, elevation: 27 },
);
const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (pts, style) => k.dotsSvg(pts, style);
// o.mat names a material class (how-page.css tints its faces: glass, oled, alu, cell, pcb, chip, lens...)
const mat = (o) => (o.mat ? "m-" + o.mat : undefined);
const slab = (plan, z, h, o = {}) => S(k.slabOf(plan, z, h, P, o.steps ?? 6, o.bevel ?? 0.5), { tone: o.tone ?? "mid", crease: o.crease, lit: o.lit, flat: o.flat, className: mat(o) });
const cyl = (x, y, r, z, h, o = {}) => S(k.cylinder(x, y, r, z, h, P, o.steps ?? 28, o.bevel ?? 0), { tone: o.tone ?? "mid", crease: o.crease, lit: o.lit, className: mat(o) });
const box = (cx, cy, w, d, r = 1) => ({ x: cx - w / 2, y: cy - d / 2, w, d, r });
const outline = (plan, z, o = {}) => L(k.planOutline(plan, z, P, o.steps ?? 6), { tone: o.tone ?? "faint" });
const screws = (pts, z, o = {}) => L(pts.map(([x, y]) => k.ring(x, y, o.r ?? 1.5, z, P, 14)).join(""), { tone: "lo" }) + D(pts.map(([x, y]) => at([x, y, z])), { size: 0.42 });

// ---------------- layers (each drawn at local z = 0; assemble lifts it to its exploded height) ----------------
function glass() {
  const t = TH[0];
  const o = [];
  o.push(slab(PH, 0, t, { tone: "hi", bevel: 0.9, steps: 10, mat: "glass" }));
  o.push(outline(k.insetPlan(PH, 3.2), t, { steps: 10 }));
  o.push(outline(box(0, 0, 54, 118, 5), t, { tone: "lo" })); // the visible area under the glass
  o.push(slab(box(0, 63, 16, 2.6, 1.2), t, 0.5, { tone: "lo", crease: "none", steps: 3, bevel: 0 })); // speaker slot
  o.push(cyl(-11, 63, 1.4, t, 0.5, { tone: "lo", crease: "none", steps: 12 })); // front camera
  o.push(L(k.lineOnTop([-9, -63.5], [9, -63.5], t, P), { tone: "mid" })); // gesture bar
  // a soft reflection across the glass, fading at both ends
  o.push(k.fadeLineSvg([[-22, -30], [-4, 42]].map(([x, y]) => at([x, y, t])), { fade: [0.9, 0.9], tone: "hi" }));
  o.push(k.fadeLineSvg([[-12, -42], [8, 28]].map(([x, y]) => at([x, y, t])), { fade: [0.9, 0.9], tone: "mid" }));
  o.push(D([[-30, -64], [30, -64], [-30, 64], [30, 64]].map(([x, y]) => at([x, y, t])), { size: 0.4, tone: "lo" }));
  return o.join("");
}

function display() {
  const t = TH[1];
  const o = [];
  o.push(slab(PH, 0, t, { tone: "mid", bevel: 0.6, steps: 10, mat: "bezel" }));
  const screen = box(0, 0, 58, 124, 6);
  o.push(S(k.slabOf(screen, t, 0.5, P, 8, 0.3), { tone: "lo", flat: true, crease: "none", className: "m-oled" }));
  o.push(outline(k.insetPlan(screen, 2), t + 0.5, { tone: "lo" }));
  // pixel grid, faded toward the edges by thinning the rows
  const rows = [];
  for (let j = 0; j < 12; j++) rows.push(...k.dotGrid(-22, -52 + j * 9, 12, 1, 4, t + 0.5, P));
  o.push(D(rows, { size: 0.35, tone: "lo" }));
  // the notch for the camera and speaker
  o.push(slab(box(0, 59, 22, 5, 2.2), t + 0.5, 0.5, { tone: "faint", crease: "none", steps: 4, bevel: 0 }));
  // flex ribbon that leaves the panel at the bottom and ends in a connector
  o.push(slab(box(0, -68, 24, 14, 1), t - 1.4, 1.4, { tone: "lo", steps: 3, bevel: 0, mat: "flex" }));
  o.push(slab(box(0, -73.5, 18, 3.2, 0.6), t - 1.2, 2.6, { tone: "hi", steps: 3, bevel: 0, mat: "metal" }));
  o.push(L(k.lineOnTop([-8, -66], [8, -66], t - 0.2, P), { tone: "faint" }));
  o.push(screws([[-28, 62], [28, 62], [-28, -62], [28, -62]], t, { r: 1.2 }));
  return o.join("");
}

function frame() {
  const t = TH[2];
  const o = [];
  o.push(slab(PH, 0, t, { tone: "mid", bevel: 1.1, steps: 10, mat: "alu" }));
  const well = box(0, 0, 58, 124, 6);
  o.push(S(k.slabOf(well, t - 0.9, 0.9, P, 8, 0.2), { tone: "faint", flat: true, crease: "none", className: "m-alu-dark" }));
  o.push(outline(k.insetPlan(well, 1.4), t - 0.9, { tone: "lo" }));
  // antenna splits across both ends, side keys, charging port and SIM tray
  for (const y of [-58, 58]) o.push(L(k.lineOnTop([-34, y], [34, y], t, P), { tone: "lo" }));
  for (const [y, h] of [[34, 10], [20, 10]]) o.push(slab({ x: -37.4, y: y - h / 2, w: 3.4, d: h, r: 1.2 }, 1.1, t - 2.2, { tone: "hi", steps: 3, bevel: 0.2, mat: "alu" })); // volume keys
  o.push(slab({ x: 34, y: 26, w: 3.4, d: 16, r: 1.4 }, 1.1, t - 2.2, { tone: "hi", steps: 3, bevel: 0.2, mat: "alu" })); // power key
  o.push(slab(box(0, -66, 14, 3.2, 1.4), t - 0.6, 0.7, { tone: "faint", crease: "none", steps: 4, bevel: 0 })); // charging port
  o.push(L(k.lineOnTop([34, -14], [34, 4], t - 1.4, P), { tone: "lo" }));
  // mid-frame ribs and the screw bosses
  const bosses = [[-26, 56], [26, 56], [-26, -54], [26, -54], [-26, 2], [26, 2]];
  for (const [x, y] of bosses) o.push(cyl(x, y, 3, t - 0.9, 0.9, { tone: "lo", steps: 14, crease: "none", mat: "metal" }));
  o.push(screws(bosses, t - 0.9, { r: 1.5 }));
  return o.join("");
}

function battery() {
  const t = TH[3];
  const o = [];
  const cell = box(0, -8, 54, 92, 4);
  o.push(slab(box(0, 0, 60, 126, 7), 0, 1.4, { tone: "lo", steps: 8, bevel: 0.3, mat: "alu-dark" })); // tray
  o.push(slab(cell, 1.4, t - 1.4, { tone: "mid", steps: 6, bevel: 0.7, mat: "cell" }));
  o.push(outline(k.insetPlan(cell, 2.4), t, { tone: "faint", steps: 6 }));
  // label plate with three rule lines and the plus and minus marks
  o.push(slab(box(0, -8, 38, 24, 2), t, 0.5, { tone: "hi", steps: 4, bevel: 0, mat: "label" }));
  for (const dy of [-5, 0, 5]) o.push(L(k.lineOnTop([-14, -8 + dy], [14 - (dy === 5 ? 10 : 0), -8 + dy], t + 0.5, P), { tone: "lo" }));
  o.push(L(k.lineOnTop([-22, 30], [-17, 30], t, P) + k.lineOnTop([-19.5, 27.5], [-19.5, 32.5], t, P), { tone: "hi" }));
  o.push(L(k.lineOnTop([17, 30], [22, 30], t, P), { tone: "hi" }));
  // pull tabs of the adhesive strips
  for (const x of [-12, 12]) {
    o.push(slab(box(x, -58, 7, 12, 1), t - 0.6, 0.6, { tone: "hi", steps: 3, bevel: 0, mat: "tab" }));
    o.push(L(k.lineOnTop([x, -54], [x, -38], t - 0.6, P), { tone: "faint" }));
  }
  // the battery's flex and connector
  o.push(slab(box(0, 56, 14, 8, 1), 1.4, 1.1, { tone: "lo", steps: 3, bevel: 0, mat: "flex" }));
  o.push(slab(box(0, 59.5, 12, 3, 0.6), 1.4, 2.6, { tone: "hi", steps: 3, bevel: 0, mat: "metal" }));
  o.push(D([at([-4, 59.5, 4]), at([0, 59.5, 4]), at([4, 59.5, 4])], { size: 0.4, tone: "lo" }));
  o.push(screws([[-27, 60], [27, 60], [-27, -60], [27, -60]], 1.4, { r: 1.3 }));
  return o.join("");
}

function board() {
  const t = TH[4];
  const o = [];
  o.push(slab(box(0, 20, 58, 92, 5), 0, t, { tone: "mid", steps: 6, bevel: 0.5, mat: "pcb" }));
  o.push(slab(box(0, -50, 38, 20, 3), 0, t, { tone: "mid", steps: 5, bevel: 0.5, mat: "pcb" })); // lower board
  o.push(slab(box(0, -34, 10, 14, 0.6), t - 1.8, 1.8, { tone: "lo", steps: 3, bevel: 0, mat: "flex" })); // joining flex
  o.push(outline(box(0, 20, 54, 88, 3), t, { tone: "faint" }));
  const chip = (cx, cy, w, d, h, tone = "hi") => {
    o.push(slab(box(cx, cy, w, d, 1.2), t, h, { tone, steps: 3, bevel: 0.3, mat: "chip" }));
    o.push(outline(k.insetPlan(box(cx, cy, w, d, 1.2), 1.1), t + h, { tone: "lo", steps: 3 }));
  };
  chip(8, 44, 16, 16, 1.8); // main chip
  chip(-12, 44, 14, 12, 1.4); // memory
  chip(-8, 22, 12, 10, 1.2);
  chip(14, 24, 8, 8, 1.0, "mid");
  chip(-14, -2, 9, 9, 1.0, "mid");
  chip(14, 2, 12, 7, 1.0, "mid");
  // shield can with a grid of vent dots
  o.push(slab(box(0, -20, 30, 16, 1.4), t, 2.2, { tone: "mid", steps: 4, bevel: 0.4, mat: "metal" }));
  o.push(D(k.dotGrid(-11, -23, 8, 2, 3, t + 2.2, P), { size: 0.35, tone: "lo" }));
  // rows of passives
  o.push(D(k.dotGrid(-24, 6, 3, 6, 2.6, t, P), { size: 0.3, tone: "lo" }));
  o.push(D(k.dotGrid(12, 12, 4, 2, 2.6, t, P), { size: 0.3, tone: "lo" }));
  // camera modules on the board: two lens barrels
  o.push(slab(box(-14, 60, 22, 22, 3), t, 2.2, { tone: "lo", steps: 5, bevel: 0.4, mat: "chip" }));
  o.push(cyl(-14, 60, 7.4, t + 2.2, 2.8, { tone: "hi", steps: 28, mat: "metal" }));
  o.push(L(k.ring(-14, 60, 4.8, t + 5, P, 24), { tone: "lo" }));
  o.push(cyl(-14, 60, 3.2, t + 5, 0.5, { tone: "lo", steps: 18, crease: "none", mat: "lens" }));
  // connectors and the corner screws
  o.push(slab(box(24, 66, 6, 8, 0.8), t, 2, { tone: "hi", steps: 3, bevel: 0, mat: "metal" }));
  o.push(slab(box(0, -64, 12, 4, 0.8), t, 2, { tone: "hi", steps: 3, bevel: 0, mat: "metal" }));
  o.push(screws([[-24, 62], [24, -2], [-24, -2], [20, -58], [-20, -58]], t, { r: 1.4 }));
  return o.join("");
}

function back() {
  const t = TH[5];
  const o = [];
  o.push(slab(PH, 0, t, { tone: "lo", bevel: 1.0, steps: 10, mat: "back" }));
  o.push(outline(k.insetPlan(PH, 3), t, { steps: 10 }));
  // the charging coil: concentric rings, flat on the back
  for (const r of [10, 14, 18]) o.push(L(k.ring(0, -14, r, t, P, 40), { tone: "faint" }));
  // the camera island with two lenses and the flash
  o.push(slab(box(-12, 49, 30, 36, 8), t, 2.0, { tone: "mid", steps: 8, bevel: 0.8, mat: "glass" }));
  for (const [cx, cy] of [[-17, 56], [-17, 42]]) {
    o.push(cyl(cx, cy, 6.6, t + 2, 2.2, { tone: "hi", steps: 30, bevel: 0.4, mat: "metal" }));
    o.push(L(k.ring(cx, cy, 4.4, t + 4.2, P, 24), { tone: "lo" }));
    o.push(cyl(cx, cy, 2.8, t + 4.2, 0.4, { tone: "lo", steps: 16, crease: "none", mat: "lens" }));
  }
  o.push(cyl(-4.6, 56, 2.2, t + 2, 0.8, { tone: "hi", steps: 14, crease: "none", mat: "flash" })); // flash
  o.push(cyl(-4.6, 44, 1.4, t + 2, 0.8, { tone: "lo", steps: 12, crease: "none" })); // microphone
  o.push(L(k.lineOnTop([-26, -50], [-12, -50], t, P), { tone: "faint" }));
  return o.join("");
}

const layers = [glass, display, frame, battery, board, back];

// ---------------- assemble ----------------
// Each layer rests at its exploded height (so the figure reads without script) and carries its own label: a leader from
// the layer's outermost right corner to a tag in the label column. The page lifts and lights one layer per stage.
const vz = at([0, 0, 1]).map((v, n) => v - at([0, 0, 0])[n]);
const lift = (z) => `translate(${(z * vz[0]).toFixed(2)} ${(z * vz[1]).toFixed(2)})`;
const LX = ART_W + 14; // label column x
function labelOf(i) {
  const t = TH[i];
  const right = k.boxCorners(PH, t, t).map(at).reduce((a, b) => (b[0] > a[0] ? b : a));
  const [x0, y0] = [right[0] + 3, right[1]];
  const y = y0 - 6;
  return `<g class="how-lbl" data-label="${i}"><path class="how-lead" d="M${x0.toFixed(1)} ${y0.toFixed(1)}L${(x0 + 10).toFixed(1)} ${y.toFixed(1)}H${LX - 4}"/>` +
    `<circle class="how-lead-dot" cx="${x0.toFixed(1)}" cy="${y0.toFixed(1)}" r="1.6"/>` +
    `<text class="how-lbl-n" x="${LX}" y="${(y - 7).toFixed(1)}">0${i + 1}</text>` +
    `<text class="how-lbl-t" x="${LX}" y="${(y + 14).toFixed(1)}">%%L${i}%%</text></g>`;
}
const body = [];
body.push(`<path class="iso-halo" d="${k.haloOf(PH, -6, 4, P)}" style="filter:blur(6px)"/>`);
// back to front: the lowest layer first
for (let i = N - 1; i >= 0; i--) {
  body.push(`<g data-layer="${i}" transform="${lift((N - 1 - i) * GAP)}"><g data-geometry>${layers[i]()}<path data-glow class="iso-line" data-tone="lit" style="stroke-width:1.2" d="${k.planOutline(PH, TH[i], P, 10)}"/></g>${labelOf(i)}</g>`);
}
const svg = k.figureSvg({ width: WIDTH, height: HEIGHT, label: "", body, className: "how-svg" })
  .replace('role="img" aria-label=""', 'aria-hidden="true" focusable="false"');
writeFileSync(join(ROOT, "templates/partials/ui/how-figure.svg"), svg + "\n");

// ---------------- live motion ----------------
const geom = `const GAP = ${GAP}, N = ${N}, VZ = [${vz.map((v) => v.toFixed(4)).join(",")}];`;
const live = readFileSync(join(HERE, "live.js"), "utf8").replace("/*GEOM*/", geom);
writeFileSync(join(ROOT, "static/ui/how-figure.js"), "// Generated by tools/how_figure/build.mjs from tools/how_figure/live.js. Do not edit.\n" + live);

const iso = k.ISO_CSS.trim() + "\n";
writeFileSync(join(ROOT, "static/ui/how-iso.css"), "/* Generated by tools/how_figure/build.mjs from the anatomy kit (MIT, see tools/how_figure/vendor). */\n" + iso);
console.log("figure", Math.round(svg.length / 1024) + " KB");
