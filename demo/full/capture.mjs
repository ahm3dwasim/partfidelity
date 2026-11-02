// Gather the full cut's inputs into the scratch dir (never committed): read-only captures of the public live site,
// their mark rects, and the sample footage as frames. Usage: node demo/full/capture.mjs
// Values come from config/demo_full.toml through demo/full/config.py (fail-loud loader).
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync, readdirSync, renameSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, commit, discard, ffmpeg, hold, home, pathOf, root, run, track } from '../preview/capture.mjs';

// Paths resolve as in the preview: the [path_env] variable wins, "~/" expands, relative means beneath the repo root.
export { chromium, commit, discard, ffmpeg, hold, home, pathOf, root, run, track };
export function loadConfig() {
  return JSON.parse(execFileSync('python3', ['-m', 'demo.full.config'], { cwd: root, encoding: 'utf8' }));
}

// One mark -> [x, y, w, h] in viewport CSS px: a CSS selector (all matches unioned), "largest:<css>", or "xpath:<expr>".
const MARK = `(spec) => {
  const box = (el) => { const r = el.getBoundingClientRect(); return [r.x, r.y, r.right, r.bottom]; };
  let els;
  if (spec.startsWith('xpath:')) {
    const r = document.evaluate(spec.slice(6), document, null, XPathResult.ORDERED_NODE_SNAPSHOT_TYPE, null);
    els = Array.from({ length: r.snapshotLength }, (_, i) => r.snapshotItem(i));
  } else if (spec.startsWith('largest:')) {
    els = [...document.querySelectorAll(spec.slice(8))].sort((a, b) => (box(b)[2] - box(b)[0]) * (box(b)[3] - box(b)[1]) - (box(a)[2] - box(a)[0]) * (box(a)[3] - box(a)[1])).slice(0, 1);
  } else els = [...document.querySelectorAll(spec)];
  if (!els.length) return null;
  const bs = els.map(box);
  const x0 = Math.min(...bs.map((b) => b[0])), y0 = Math.min(...bs.map((b) => b[1]));
  return [x0, y0, Math.max(...bs.map((b) => b[2])) - x0, Math.max(...bs.map((b) => b[3])) - y0];
}`;

async function scrollTo(page, spec) {
  await page.evaluate(`((spec) => {
    const mark = ${MARK};
    const r = mark(spec);
    if (!r) throw new Error('no element for scroll ' + spec);
    window.scrollBy(0, r[1] - 60);
  })(${JSON.stringify(spec)})`);
}

async function captureUi(cfg, out) {
  const s = cfg.source, rects = {};
  const browser = await chromium(cfg).launch();
  const closeBrowser = hold(() => browser.close());   // registered at once: a failure below or a signal still closes it
  try {
  for (const c of cfg.capture) {
    const [w, h] = s[c.viewport];
    const page = await browser.newPage({ viewport: { width: w, height: h }, deviceScaleFactor: s.capture_scale });
    await page.goto(s.site + c.path, { waitUntil: 'networkidle' });
    if (c.ack) await page.click('button.ack');   // the page's own "Show the next steps" button: local state only
    if (c.scroll) await scrollTo(page, c.scroll);
    await page.waitForTimeout(s.settle_ms);
    await page.screenshot({ path: track(join(out, `${c.name}.png`)) });
    rects[c.name] = { size: [w, h] };
    for (const [k, spec] of Object.entries(c.marks)) {
      const r = await page.evaluate(`(${MARK})(${JSON.stringify(spec)})`);
      if (!r) throw new Error(`${c.name}.${k}: nothing matches ${spec}`);
      rects[c.name][k] = r;
    }
    await page.close();
  }
  // Replay page (the recorded probe table and the scripted review card), as in the approved preview.
  const page = await browser.newPage({ viewport: { width: s.desktop[0], height: s.desktop[1] }, deviceScaleFactor: s.capture_scale });
  await page.goto(s.site + s.replay_path, { waitUntil: 'networkidle' });
  rects.replay = {};
  for (const [name, sel] of [['table', '.replay-table'], ['card', '.replay-card']]) {
    const el = await page.$(sel);
    if (!el) throw new Error(`${s.replay_path}: no ${sel}`);
    await el.evaluate((e) => e.scrollIntoView({ block: 'center' }));
    await page.waitForTimeout(300);
    await el.screenshot({ path: track(join(out, `${name}.png`)) });
    rects.replay[name] = await el.evaluate((e) => { const r = e.getBoundingClientRect(); return [r.width, r.height]; });
  }
  rects.replay.note = await page.$eval('.replay-card', (c) => {
    const a = c.getBoundingClientRect(), b = c.querySelector('.replay-note').getBoundingClientRect();
    return [b.x - a.x, b.y - a.y, b.width, b.height];
  });
  rects.replay.row = await page.$eval('.replay-table', (t) => {
    const row = [...t.querySelectorAll('tr')].find((r) => /keyboard/.test(r.innerText) && /disagree/.test(r.innerText));
    if (!row) throw new Error('no recorded disagreement row');
    const a = t.getBoundingClientRect(), b = row.getBoundingClientRect();
    return [b.x - a.x, b.y - a.y, b.width, b.height];
  });
  writeFileSync(track(join(out, 'rects.json')), JSON.stringify(rects, null, 1));
  } finally {
    await closeBrowser();
  }
}

// Footage: one frame folder per footage scene, covering the scene's source range at the output frame rate.
async function captureFootage(cfg, scratch) {
  const src = join(scratch, 'src'), out = join(scratch, 'footage'), part = join(scratch, 'src.partial'), s = cfg.source, info = {};
  // Extract into a tracked temp dir, then move the one file in; src is tracked too, so no sample.mp4 survives a failure.
  const zip = pathOf(cfg, 'footage_zip'), bin = ffmpeg(cfg);
  if (!existsSync(zip)) throw new Error(`footage zip not found (${zip}): set ${cfg.path_env.footage_zip}`);
  rmSync(track(part), { recursive: true, force: true });
  mkdirSync(part, { recursive: true });
  await run('unzip', ['-o', '-q', zip, cfg.paths.footage_member, '-d', part]);
  rmSync(track(src), { recursive: true, force: true });   // tracked from before the move to commit(): a failure or signal removes it
  mkdirSync(src, { recursive: true });
  renameSync(join(part, cfg.paths.footage_member), join(src, cfg.paths.footage_member));
  rmSync(part, { recursive: true, force: true });
  rmSync(out, { recursive: true, force: true });
  track(out);
  for (const sc of cfg.scene.filter((x) => x.kind === 'footage')) {
    const t = sc.map.map((p) => p[1]), lo = Math.min(...t), hi = Math.max(...t) + 0.1;
    if (hi > s.latest_s) throw new Error(`${sc.id}: footage past ${s.latest_s} s`);
    const dir = join(out, sc.id);
    mkdirSync(dir, { recursive: true });
    await run(bin, ['-v', 'error', '-ss', String(lo), '-to', String(hi), '-i', join(src, cfg.paths.footage_member),
      '-vf', `fps=${s.footage_fps},scale=${s.footage_width}:-2`, '-q:v', '3', join(dir, '%04d.jpg')]);
    info[sc.id] = { start: lo, frames: readdirSync(dir).filter((n) => n.endsWith('.jpg')).length };
  }
  return info;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const cfg = loadConfig();
  const scratch = join(root, cfg.paths.scratch), ui = join(scratch, 'ui');
  mkdirSync(ui, { recursive: true });
  try {
    await captureUi(cfg, ui);
    const info = await captureFootage(cfg, scratch);
    writeFileSync(track(join(scratch, 'footage.json')), JSON.stringify(info));
    commit();
    console.log(`ui captures in ${ui}; footage frames: ${JSON.stringify(info)}`);
  } catch (e) {
    await discard();   // no half-written captures left for render.mjs to pick up
    console.error(e.message);
    process.exitCode = 1;
  }
}
