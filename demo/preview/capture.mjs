// Gather the preview's inputs into the scratch dir (never committed): real UI captures of the public
// live site and a few seconds of the sample footage. Usage: node demo/preview/capture.mjs
// Values come from config/demo_preview.toml through demo/preview/config.py (module demo.preview.config, the fail-loud loader).
import { execFileSync, spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, writeFileSync, readdirSync, renameSync, rmSync } from 'node:fs';
import { createRequire } from 'node:module';
import { homedir } from 'node:os';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
export const home = (p) => p.replace(/^~(?=\/)/, homedir());
// A configured path: the env var named in [path_env] wins; "~/" expands; relative means beneath the repo root.
export function pathOf(cfg, key) {
  const v = home(process.env[cfg.path_env[key]] || cfg.paths[key]);
  return isAbsolute(v) ? v : join(root, v);
}
// ffmpeg: a bare name is looked up on PATH; fails loudly if it cannot run.
export function ffmpeg(cfg) {
  const v = process.env[cfg.path_env.ffmpeg] || cfg.paths.ffmpeg;
  const bin = v.includes('/') ? pathOf(cfg, 'ffmpeg') : v;
  if (spawnSync(bin, ['-version'], { stdio: 'ignore' }).status !== 0) {
    throw new Error(`ffmpeg not found (${bin}): put it on PATH or set ${cfg.path_env.ffmpeg}`);
  }
  return bin;
}
export function loadConfig() {
  return JSON.parse(execFileSync('python3', ['-m', 'demo.preview.config'], { cwd: root, encoding: 'utf8' }));
}

// Cleanup registry. hold(close) returns an idempotent closer that memoizes the in-flight close, so a second caller (or the
// signal handler) waits for the first close to finish. Callers run it in a finally block; SIGINT/SIGTERM runs every
// closer still open, deletes the partial outputs given to track(), and exits non-zero.
const open = new Set(), partials = new Set();
let leaving = false, installed = false;
async function release() {
  for (const close of [...open].reverse()) await close().catch(() => {});
  for (const p of partials) rmSync(p, { recursive: true, force: true });
  partials.clear();
}
export function hold(close) {
  let closing = null;
  const run = () => (closing ??= (open.delete(run), Promise.resolve().then(close)));
  open.add(run);
  if (!installed) {
    installed = true;
    for (const [sig, code] of [['SIGINT', 130], ['SIGTERM', 143]]) {
      process.on(sig, async () => {
        if (leaving) return;
        leaving = true;
        setTimeout(() => process.exit(code), 5000).unref();   // a closer that hangs must not keep us alive
        await release();
        process.exit(code);
      });
    }
  }
  return run;
}
export const track = (path) => { partials.add(path); return path; };   // removed on failure or signal
export const commit = () => partials.clear();                          // the run finished: keep the outputs
export const discard = release;

// Run a child to completion without blocking the event loop (so a signal is handled); killed if we leave early.
export function run(bin, args) {
  const child = spawn(bin, args, { stdio: ['ignore', 'inherit', 'inherit'] });
  const stop = hold(async () => { child.kill('SIGKILL'); });
  return new Promise((ok, fail) => {
    child.on('error', fail);
    child.on('close', (c) => (c === 0 ? ok() : fail(new Error(`${bin} exit ${c}`))));
  }).finally(stop);
}

export function chromium(cfg) {
  return createRequire(join(pathOf(cfg, 'playwright_from'), 'package.json'))('playwright').chromium;
}

async function captureUi(cfg, out) {
  const s = cfg.source;
  let closeBrowser = null;
  try {
    const browser = await chromium(cfg).launch();
    closeBrowser = hold(() => browser.close());
    const page = await browser.newPage({ viewport: { width: s.capture_width, height: s.capture_height }, deviceScaleFactor: s.capture_scale });
    const rects = {};

    // Guide: open the folded steps (the page's own button, local state only), bring the step to the top.
    await page.goto(s.site + s.guide_path, { waitUntil: 'networkidle' });
    await page.click('button.ack');
    await page.evaluate((n) => { document.getElementById(`step-${n}`).scrollIntoView({ block: 'start' }); window.scrollBy(0, -60); }, s.guide_step);
    await page.waitForTimeout(s.settle_ms);
    await page.screenshot({ path: track(join(out, 'guide.png')) });
    rects.guide = await page.evaluate((n) => {
      const box = (el) => { const r = el.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; };
      // The visible media (the still or the video, whichever is showing) is the largest box inside the screen.
      const screen = [...document.querySelectorAll('.gd-screen img, .gd-screen video, .gd-screen canvas')]
        .sort((a, b) => box(b)[2] * box(b)[3] - box(a)[2] * box(a)[3])[0];
      return { frame: box(screen), step: box(document.getElementById(`step-${n}`)) };
    }, s.guide_step);

    // Replay page: the recorded probe table and the scripted review card, each as its own element capture.
    await page.goto(s.site + s.replay_path, { waitUntil: 'networkidle' });
    for (const [name, sel] of [['table', '.replay-table'], ['card', '.replay-card']]) {
      const el = await page.$(sel);
      if (!el) throw new Error(`${s.replay_path}: no ${sel}`);
      await el.evaluate((e) => e.scrollIntoView({ block: 'center' }));
      await page.waitForTimeout(300);
      await el.screenshot({ path: track(join(out, `${name}.png`)) });
      rects[name] = await el.evaluate((e) => { const r = e.getBoundingClientRect(); return [r.width, r.height]; });
    }
    rects.note = await page.$eval('.replay-card', (c) => {
      const a = c.getBoundingClientRect(), b = c.querySelector('.replay-note').getBoundingClientRect();
      return [b.x - a.x, b.y - a.y, b.width, b.height];
    });
    rects.row = await page.$eval('.replay-table', (t) => {
      const row = [...t.querySelectorAll('tr')].find((r) => /keyboard/.test(r.innerText) && /disagree/.test(r.innerText));
      if (!row) throw new Error('no recorded disagreement row');
      const a = t.getBoundingClientRect(), b = row.getBoundingClientRect();
      return [b.x - a.x, b.y - a.y, b.width, b.height];
    });
    writeFileSync(track(join(out, 'rects.json')), JSON.stringify(rects, null, 1));
  } finally {
    await closeBrowser?.();
  }
}

async function captureFootage(cfg, scratch) {
  const f = cfg.footage, src = join(scratch, 'src'), out = join(scratch, 'footage'), part = join(scratch, 'src.partial');
  const footageZip = pathOf(cfg, 'footage_zip');
  if (!existsSync(footageZip)) throw new Error(`footage zip not found (${footageZip}): set ${cfg.path_env.footage_zip}`);
  // Extract into a tracked temp dir, then move the one file in: a failure or signal leaves no partial sample.mp4.
  rmSync(track(part), { recursive: true, force: true });
  mkdirSync(part, { recursive: true });
  await run('unzip', ['-o', '-q', footageZip, cfg.paths.footage_member, '-d', part]);
  rmSync(track(src), { recursive: true, force: true });   // tracked from before the move to commit(): a failure or signal removes it
  mkdirSync(src, { recursive: true });
  renameSync(join(part, cfg.paths.footage_member), join(src, cfg.paths.footage_member));
  rmSync(part, { recursive: true, force: true });
  rmSync(out, { recursive: true, force: true });
  mkdirSync(track(out), { recursive: true });
  await run(ffmpeg(cfg), ['-v', 'error', '-ss', String(f.start_s), '-to', String(f.end_s), '-i', join(src, cfg.paths.footage_member),
    '-vf', `fps=${cfg.output.fps}`, '-q:v', '2', join(out, '%04d.jpg')]);
  return readdirSync(out).filter((n) => n.endsWith('.jpg')).length;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const cfg = loadConfig();
  const scratch = join(root, cfg.paths.scratch), ui = join(scratch, 'ui');
  mkdirSync(ui, { recursive: true });
  try {
    await captureUi(cfg, ui);
    const n = await captureFootage(cfg, scratch);
    writeFileSync(track(join(scratch, 'footage.json')), JSON.stringify({ frames: n }));
    commit();
    console.log(`ui captures in ${ui}; ${n} footage frames`);
  } catch (e) {
    await discard();   // no half-written captures left for render.mjs to pick up
    console.error(e.message);
    process.exitCode = 1;
  }
}
