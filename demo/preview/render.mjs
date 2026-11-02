// Render the preview: serve the composition locally, seek the paused timeline frame by frame in Playwright
// Chromium, pipe each PNG straight into ffmpeg (no frame files on a tight disk), then make the contact sheet.
// Usage: node demo/preview/render.mjs            full render to <scratch>/preview.mp4 and contact.png
//        node demo/preview/render.mjs --at 2,5.5   spot frames to <scratch>/qc/ for review
// Run demo/preview/capture.mjs first.
import { spawn } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import { extname, join, normalize, sep } from 'node:path';
import { chromium, commit, discard, ffmpeg, hold, loadConfig, root, run, track } from './capture.mjs';

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.woff2': 'font/woff2' };

function serve(allowed) {
  // Read-only, loopback only, and only beneath the listed folders of the worktree.
  const server = http.createServer((req, res) => {
    const rel = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^[/\\]+/, '');
    const file = join(root, rel);
    if (req.method !== 'GET' || !allowed.some((d) => rel.startsWith(d + sep)) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end(); return;
    }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' });
    createReadStream(file).pipe(res);
  });
  return new Promise((ok) => server.listen(0, '127.0.0.1', () => ok(server)));
}

// Launches the page; the closer is registered (and handed to the caller as `held.close`) the moment Chromium exists, so
// the caller's finally and the signal handler close it even if navigation or readiness fails below.
async function open(cfg, port, held) {
  const browser = await chromium(cfg).launch();
  held.close = hold(() => browser.close());
  const page = await browser.newPage({ viewport: { width: cfg.output.width, height: cfg.output.height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`http://127.0.0.1:${port}/demo/preview/index.html?build=/${cfg.paths.scratch}/build.json`);
  await page.waitForFunction(() => window.previewReady || window.previewError, null, { timeout: 60000 });
  const err = await page.evaluate(() => window.previewError);
  if (err) throw new Error(err);
  return page;
}

const shot = async (page, t) => { await page.evaluate((x) => window.seek(x), t); return page.screenshot({ type: 'png' }); };

async function contact(cfg, mp4, out) {
  const o = cfg.output, n = Math.round(cfg.timing.duration_s * o.fps), k = o.contact_tiles;
  const picks = Array.from({ length: k }, (_, i) => Math.round((i + 0.5) * n / k));
  await run(ffmpeg(cfg), ['-v', 'error', '-y', '-i', mp4, '-vf',
    `select='${picks.map((p) => `eq(n\\,${p})`).join('+')}',scale=${o.contact_tile_width}:-1,tile=${k}x1`,
    '-fps_mode', 'passthrough', '-frames:v', '1', out]);
  return picks;
}

const cfg = loadConfig();
const scratch = join(root, cfg.paths.scratch);
const footage = JSON.parse(readFileSync(join(scratch, 'footage.json'), 'utf8'));
writeFileSync(join(scratch, 'build.json'), JSON.stringify({
  cfg, rects: JSON.parse(readFileSync(join(scratch, 'ui', 'rects.json'), 'utf8')),
  ui: `/${cfg.paths.scratch}/ui`, footage: { frames: footage.frames, url: `/${cfg.paths.scratch}/footage` },
}));
// Every handle starts null and is closed in the finally below (or by the signal handler in capture.mjs).
let server = null, closeServer = null, closeFfmpeg = null, ff = null;
const held = { close: null };
try {
  server = await serve(['demo', 'static', normalize(cfg.paths.scratch)]);
  closeServer = hold(() => new Promise((ok) => { server.close(ok); server.closeAllConnections(); }));
  const page = await open(cfg, server.address().port, held);
  const atArg = process.argv.indexOf('--at');
  if (atArg > 0) {
    const qc = join(scratch, 'qc');
    mkdirSync(qc, { recursive: true });
    for (const t of process.argv[atArg + 1].split(',').map(Number)) {
      writeFileSync(track(join(qc, `t${t.toFixed(2).padStart(5, '0')}.png`)), await shot(page, t));
    }
    console.log(`spot frames in ${qc}`);
  } else {
    const o = cfg.output, n = Math.round(cfg.timing.duration_s * o.fps), mp4 = track(join(scratch, 'preview.mp4'));
    ff = spawn(ffmpeg(cfg), ['-v', 'error', '-y', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(o.fps), '-i', '-',
      // Tagged BT.709 with the matching matrix, so QuickTime shows the palette as rendered.
      '-vf', 'scale=out_color_matrix=bt709:out_range=tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
      '-c:v', 'libx264', '-preset', o.preset, '-crf', String(o.crf), '-pix_fmt', 'yuv420p', '-r', String(o.fps),
      '-movflags', '+faststart', mp4], { stdio: ['pipe', 'inherit', 'inherit'] });
    closeFfmpeg = hold(async () => { ff.stdin.destroy(); ff.kill('SIGKILL'); });
    ff.stdin.on('error', () => {});   // a dead ffmpeg surfaces through `done`, not as an unhandled EPIPE
    const done = new Promise((ok, fail) => {
      ff.on('error', fail);
      ff.on('close', (c) => (c === 0 ? ok() : fail(new Error(`ffmpeg exit ${c}`))));
    });
    done.catch(() => {});
    for (let i = 0; i < n; i++) {
      const png = await shot(page, i / o.fps);
      if (!ff.stdin.write(png)) await Promise.race([new Promise((ok) => ff.stdin.once('drain', ok)), done]);
    }
    ff.stdin.end();
    await done;
    const mb = statSync(mp4).size / 1048576;
    if (mb > o.max_mb) throw new Error(`preview.mp4 is ${mb.toFixed(1)} MB, over max_mb ${o.max_mb}`);
    const picks = await contact(cfg, mp4, track(join(scratch, 'contact.png')));
    console.log(`${n} frames -> ${mp4} (${mb.toFixed(2)} MB); contact.png frames ${picks.join(',')}`);
  }
  commit();
} catch (e) {
  console.error(e.message);
  process.exitCode = 1;
  await discard();   // closes whatever is still open and removes partial media
} finally {
  await closeFfmpeg?.();
  await held.close?.();
  await closeServer?.();
}
