// Render the full cut: serve the composition locally, seek the paused timeline frame by frame in Playwright Chromium,
// pipe each PNG straight into ffmpeg (no frame files on a tight disk), then mux the optional audio track and make the
// 1 fps contact sheets. Usage: node demo/full/render.mjs               full render to <scratch>/final.mp4 + contact-NN.png
//        node demo/full/render.mjs --at 2,5.5      spot frames to <scratch>/qc/ for review
// Run demo/full/capture.mjs first.
import { spawn } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import { extname, join, normalize, sep } from 'node:path';
import { chromium, commit, discard, ffmpeg, hold, home, loadConfig, root, run, track } from './capture.mjs';

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.ttf': 'font/ttf' };

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

// The browser closer is registered (as `held.close`) the moment Chromium exists, so the caller's finally and the signal
// handler close it even if navigation or readiness fails below.
async function open(cfg, port, held) {
  const browser = await chromium(cfg).launch();
  held.close = hold(() => browser.close());
  const page = await browser.newPage({ viewport: { width: cfg.output.width, height: cfg.output.height }, deviceScaleFactor: 1 });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`http://127.0.0.1:${port}/demo/full/index.html?build=/${cfg.paths.scratch}/build.json`);
  await page.waitForFunction(() => window.previewReady || window.previewError, null, { timeout: 120000 });
  const err = await page.evaluate(() => window.previewError);
  if (err) throw new Error(err);
  return page;
}

const shot = async (page, t) => { await page.evaluate((x) => window.seek(x), t); return page.screenshot({ type: 'png' }); };

async function contact(cfg, mp4, dir) {
  const o = cfg.output;
  for (const n of readdirSync(dir).filter((x) => /^contact-\d+\.png$/.test(x))) rmSync(join(dir, n));
  // Tracked before ffmpeg starts, as a pattern of the sheets it will write: a failure or signal removes any it left.
  const sheets = () => readdirSync(dir).filter((x) => /^contact-\d+\.png$/.test(x));
  const stop = hold(async () => { for (const n of sheets()) rmSync(join(dir, n)); });
  try {
    await run(FFMPEG, ['-v', 'error', '-y', '-i', mp4, '-vf',
      `fps=1,scale=${o.contact_tile_width}:-1,tile=${o.contact_cols}x${o.contact_rows}`, join(dir, 'contact-%02d.png')]);
  } catch (e) { await stop(); throw e; }
  return sheets().length;
}

const cfg = loadConfig();
const FFMPEG = ffmpeg(cfg);
const scratch = join(root, cfg.paths.scratch);
writeFileSync(join(scratch, 'build.json'), JSON.stringify({
  cfg, rects: JSON.parse(readFileSync(join(scratch, 'ui', 'rects.json'), 'utf8')), ui: `/${cfg.paths.scratch}/ui`,
  footage: { ...JSON.parse(readFileSync(join(scratch, 'footage.json'), 'utf8')), url: `/${cfg.paths.scratch}/footage` },
}));
// Every handle starts null and is closed in the finally below (or by the signal handler in preview/capture.mjs).
let server = null, closeServer = null, closeFfmpeg = null;
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
      writeFileSync(track(join(qc, `t${t.toFixed(2).padStart(6, '0')}.png`)), await shot(page, t));
    }
    console.log(`spot frames in ${qc}`);
  } else {
    const o = cfg.output, n = Math.round(cfg.total_s * o.fps), mp4 = track(join(scratch, 'video.mp4')), final = track(join(scratch, 'final.mp4'));
    const ff = spawn(FFMPEG, ['-v', 'error', '-y', '-f', 'image2pipe', '-c:v', 'png', '-framerate', String(o.fps), '-i', '-',
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
      if (i % 300 === 0) console.log(`frame ${i}/${n}`);
    }
    ff.stdin.end();
    await done;
    const a = cfg.audio;
    if (a.track) {   // optional music or voice: aac, volume from config, trimmed to the film
      await run(FFMPEG, ['-v', 'error', '-y', '-i', mp4, '-i', home(a.track), '-map', '0:v', '-map', '1:a', '-c:v', 'copy',
        '-c:a', 'aac', '-af', `volume=${a.volume}`, '-shortest', '-movflags', '+faststart', final]);
      rmSync(mp4);
    } else renameSync(mp4, final);
    const mb = statSync(final).size / 1048576;
    if (mb > o.max_mb) throw new Error(`final.mp4 is ${mb.toFixed(1)} MB, over max_mb ${o.max_mb}`);
    console.log(`${n} frames -> ${final} (${mb.toFixed(2)} MB); ${await contact(cfg, final, scratch)} contact sheets`);
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
