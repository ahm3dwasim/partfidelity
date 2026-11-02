// The preview as one paused GSAP timeline. render.mjs writes build.json (checked config, capture rects, footage
// frame count) and passes its URL as ?build=; then it calls window.seek(t) once per output frame.
(async () => {
  const q = new URLSearchParams(location.search);
  const build = await (await fetch(q.get('build'))).json();
  const { cfg, rects } = build;
  const W = cfg.output.width, H = cfg.output.height, FPS = cfg.output.fps;
  const M = cfg.layout.margin_px, T = cfg.timing, L = cfg.layout, src = cfg.source;
  const $ = (id) => document.getElementById(id);
  const at = (beat, f) => T[beat][0] + f * (T[beat][1] - T[beat][0]);
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  const css = document.documentElement.style;
  const MO = cfg.motion, R = cfg.ring, TG = cfg.tag;
  const vars = { '--w': W, '--h': H, '--radius': L.radius_px, '--ring-border': R.border_px, '--ring-radius': R.radius_px,
    '--ring-tint': R.tint_pct, '--link-stroke': R.stroke_px, '--panel-tag-px': TG.size_px, '--panel-tag-weight': TG.weight,
    '--panel-tag-spacing': TG.letter_spacing_em, '--panel-tag-gap': TG.gap_px, '--dim-radius': MO.dim_radius_px };
  for (const [k, v] of Object.entries(vars)) css.setProperty(k, v);

  const load = (img, url) => new Promise((ok, fail) => {
    img.onload = ok; img.onerror = () => fail(new Error(`missing ${url}`)); img.src = url;
  }).then(() => img.decode());
  await Promise.all([['guide-img', 'guide.png'], ['table-img', 'table.png'], ['card-img', 'card.png']]
    .map(([id, name]) => load($(id), `${build.ui}/${name}`)));
  const frames = await Promise.all(Array.from({ length: build.footage.frames }, (_, i) =>
    fetch(`${build.footage.url}/${String(i + 1).padStart(4, '0')}.jpg`).then((r) => {
      if (!r.ok) throw new Error(`footage frame ${i + 1}: ${r.status}`);
      return r.blob();
    }).then(createImageBitmap)));
  const CB = cfg.caption_box;
  await Promise.all([`${CB.text_px}px Roboto`, `600 ${CB.tag_px}px Roboto`].map((f) => document.fonts.load(f)));

  // Bands: one stage box (the same in every beat), a constant gap, then the caption block; nothing inside the margin.
  // The block is top-anchored at capTop: badge line, then up to text_lines lines of text.
  const tagH = Math.ceil(CB.tag_px * 1.2) + 2 * CB.tag_pad_px[0] + 2;
  const capH = tagH + CB.tag_gap_px + Math.ceil(CB.text_lines * CB.text_px * CB.line_height);
  const capTop = H - M - capH, contentTop = M, contentBottom = capTop - CB.stage_gap_px;
  const set = { '--tag-px': CB.tag_px, '--tag-pad-v': CB.tag_pad_px[0], '--tag-pad-h': CB.tag_pad_px[1], '--text-px': CB.text_px,
    '--line-height': CB.line_height, '--max-chars': CB.max_chars, '--tag-gap': CB.tag_gap_px, '--mark-px': CB.mark_px,
    '--cap-items': CB.align === 'center' ? 'center' : 'flex-start', '--cap-align': CB.align === 'center' ? 'center' : 'left' };
  for (const [k, v] of Object.entries(set)) css.setProperty(k, v);

  // Guide window (the camera's viewport, same aspect as the capture) fills the stage box's height.
  const C = cfg.camera;
  const winH = contentBottom - contentTop, winW = Math.round(winH * src.capture_width / src.capture_height);
  const win = { x: (W - winW) / 2, y: contentTop, width: winW, height: winH };
  const capX = CB.align === 'center' ? M : win.x;
  for (const p of document.querySelectorAll('.caption')) {
    const beat = p.dataset.beat;
    p.querySelector('.text').textContent = cfg.captions[beat];
    p.querySelector('.tag').textContent = cfg.captions[`${beat}_tag`];
    gsap.set(p, { x: capX, y: capTop, width: W - 2 * capX, autoAlpha: 0 });
  }
  $('recorded-tag').textContent = cfg.captions.recorded_tag;
  $('replay-tag').textContent = cfg.captions.replay_tag;
  gsap.set('#badge', { x: W - M - CB.mark_px, y: capTop + (capH - CB.mark_px) / 2, autoAlpha: 0 });

  // Footage card: large, centred, card_aspect kept.
  const cardH = contentBottom - contentTop, cardW = Math.round(cardH * L.card_aspect[0] / L.card_aspect[1]);
  const card = { x: (W - cardW) / 2, y: contentTop + (contentBottom - contentTop - cardH) / 2, width: cardW, height: cardH };
  gsap.set('#footage', { ...card, borderRadius: L.radius_px, autoAlpha: 0 });

  // The guide's camera; k0 maps capture CSS px to
  // output px at camera scale 1. The slot is the guide's own "Original frame" image, where the footage card lands.
  const k0 = win.width / src.capture_width;
  gsap.set('#guide', { ...win, autoAlpha: 0 });
  $('guide-img').style.width = `${src.capture_width * k0}px`;
  const [fx, fy, fw, fh] = rects.guide.frame;
  const slot = { x: win.x + 1 + fx * k0, y: win.y + 1 + fy * k0, width: fw * k0, height: fh * k0 };
  // Fit-to-rect: the step card is scaled to the largest size that keeps margin_frac of the window free on every
  // side, then centred. Nothing is clamped, so the card is always fully framed.
  const [cx0, cy0, cw0, ch0] = C.target;
  const kFit = Math.min(win.width * (1 - 2 * C.margin_frac) / cw0, win.height * (1 - 2 * C.margin_frac) / ch0);
  const camEnd = {
    scale: kFit / k0,
    x: win.width / 2 - (cx0 + cw0 / 2) * kFit,
    y: win.height / 2 - (cy0 + ch0 / 2) * kFit,
  };
  // Scrim over everything but the card (lives inside the camera, so it follows the move); hides neighbouring text.
  Object.assign($('dim').style, { left: `${cx0 * k0}px`, top: `${cy0 * k0}px`, width: `${cw0 * k0}px`, height: `${ch0 * k0}px`,
    boxShadow: `0 0 0 ${MO.scrim_spread_px}px color-mix(in srgb, var(--pf-ink) ${C.dim * 100}%, transparent)` });
  gsap.set('#dim', { autoAlpha: 0 });

  // Match panels: recorded probe table left, scripted review card right, centred as a pair.
  const ms = L.match_scale, gap = L.connector_gap_px;
  const tw = rects.table[0] * ms, th = rects.table[1] * ms, cw = rects.card[0] * ms, ch = rects.card[1] * ms;
  const mid = (contentTop + contentBottom) / 2, left = (W - tw - gap - cw) / 2;
  const tablePos = { x: left, y: mid - th / 2 }, cardPos = { x: left + tw + gap, y: mid - ch / 2 };
  $('table-img').style.width = `${tw}px`; $('card-img').style.width = `${cw}px`;
  gsap.set('#table-panel', { ...tablePos, autoAlpha: 0 });
  gsap.set('#card-panel', { ...cardPos, autoAlpha: 0 });
  const [, ry, , rh] = rects.row, [nx, ny, nw, nh] = rects.note;
  const [rowX, rowShrink] = R.row_inset_px, [padX, padY] = R.note_pad_px;
  gsap.set('#row-ring', { x: rowX, y: ry * ms, width: tw - rowShrink, height: rh * ms, scaleX: 0, transformOrigin: '0% 50%' });
  gsap.set('#note-ring', { x: nx * ms - padX, y: ny * ms - padY, width: nw * ms + 2 * padX, height: nh * ms + 2 * padY, autoAlpha: 0 });
  const a = { x: tablePos.x + tw - R.anchor_inset_px, y: tablePos.y + 1 + (ry + rh / 2) * ms };
  const b = { x: cardPos.x + 1 + nx * ms - padX, y: cardPos.y + 1 + (ny + nh / 2) * ms };
  const svg = $('link'), path = $('link-path');
  svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  path.setAttribute('d', `M${a.x},${a.y} C${a.x + gap},${a.y} ${b.x - gap},${b.y} ${b.x},${b.y}`);
  const len = path.getTotalLength();
  gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });

  const tl = gsap.timeline({ paused: true, defaults: { ease: 'power2.out' } });
  const span = (beat, f0, f1) => [at(beat, f0), at(beat, f1) - at(beat, f0)];
  const put = (target, vars, beat, f0, f1) => { const [s, d] = span(beat, f0, f1); tl.to(target, { ...vars, duration: d }, s); };
  const cap = (beat) => `.caption[data-beat="${beat}"]`;
  const fade = CB.fade_s;

  // 1. Footage card rises in; the footage plays inside it.
  tl.fromTo('#footage', { autoAlpha: 0, y: card.y + MO.footage_rise_px }, { autoAlpha: 1, y: card.y, duration: at('footage', MO.footage_rise), ease: 'power3.out' }, 0);
  tl.to(cap('footage'), { autoAlpha: 1, duration: fade, ease: 'none' }, MO.caption_in_s);
  put('#badge', { autoAlpha: 1 }, 'footage', ...MO.badge_in);
  // 2. The card flies into the guide's own frame slot; the guide comes up around it, then the card lets go.
  put('#footage', { ...slot, borderRadius: MO.slot_radius_px, borderColor: 'rgba(0,0,0,0)', boxShadow: '0 0 0 rgba(0,0,0,0)', ease: 'power2.inOut' }, 'to_guide', ...MO.fly);
  put('#guide', { autoAlpha: 1, ease: 'none' }, 'to_guide', ...MO.guide_in);
  put('#footage', { autoAlpha: 0, ease: 'none' }, 'to_guide', ...MO.dissolve);   // the dissolve into the real UI still at the same spot
  // Caption cuts: a short crossfade at each cut, between moves, so the text never changes mid-move.
  tl.to(cap('footage'), { autoAlpha: 0, duration: fade, ease: 'none' }, CB.cuts_s[0]);
  tl.to(cap('guide'), { autoAlpha: 1, duration: fade, ease: 'none' }, CB.cuts_s[0]);
  // 3. One camera move onto the step card, then it holds still (push_s ends at least hold_min_s before the beat does).
  tl.to('.cam', { ...camEnd, ease: 'power2.inOut', duration: C.push_s[1] - C.push_s[0] }, C.push_s[0]);
  tl.to('#dim', { autoAlpha: 1, ease: 'power2.inOut', duration: C.push_s[1] - C.push_s[0] }, C.push_s[0]);
  // 4. Hand over to the Image match moment.
  put('#guide', { x: M, autoAlpha: 0, ease: 'power2.in' }, 'to_match', ...MO.slide_out);  // slides out, never into the margin
  tl.to(cap('guide'), { autoAlpha: 0, duration: fade, ease: 'none' }, CB.cuts_s[1]);
  tl.to(cap('match'), { autoAlpha: 1, duration: fade, ease: 'none' }, CB.cuts_s[1]);
  tl.fromTo('#table-panel', { autoAlpha: 0, y: tablePos.y + MO.panel_rise_px }, { autoAlpha: 1, y: tablePos.y, duration: at('match', MO.row_ring[0]) - at('to_match', MO.table_in_start) }, at('to_match', MO.table_in_start));
  // 5. The recorded disagreement row, a line to the same label in the review card, the card's note.
  put('#row-ring', { scaleX: 1, ease: 'power3.out' }, 'match', ...MO.row_ring);
  put(path, { strokeDashoffset: 0, ease: 'power1.inOut' }, 'match', ...MO.link);
  tl.fromTo('#card-panel', { autoAlpha: 0, y: cardPos.y + MO.panel_rise_px }, { autoAlpha: 1, y: cardPos.y, duration: at('match', MO.card_in[1]) - at('match', MO.card_in[0]) }, at('match', MO.card_in[0]));
  put('#note-ring', { autoAlpha: 1 }, 'match', ...MO.note_ring);
  // 6. PF badge corner hold (no logo reveal).
  put(['#table-panel', '#card-panel', '#link'], { autoAlpha: MO.badge_hold_alpha, ease: 'power1.inOut' }, 'badge', ...MO.badge_hold);
  put('#badge', { scale: MO.badge_hold_scale, ease: 'power2.inOut' }, 'badge', ...MO.badge_hold);
  tl.set({}, {}, T.duration_s);

  const canvas = $('footage'), fx0 = at('footage', MO.footage_play), ctx = canvas.getContext('2d');
  canvas.width = frames[0].width; canvas.height = frames[0].height;
  function draw(t) {
    const i = clamp(Math.floor((t - fx0) * FPS), 0, frames.length - 1);
    ctx.drawImage(frames[i], 0, 0);
  }
  window.seek = (t) => {
    tl.time(t, false); draw(t);
    return new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok)));
  };
  window.previewReady = { duration: T.duration_s };
  if (q.has('play')) {
    const t0 = performance.now();
    const loop = () => { const t = ((performance.now() - t0) / 1000) % T.duration_s; tl.time(t); draw(t); requestAnimationFrame(loop); };
    loop();
  } else {
    await window.seek(0);
  }
})().catch((e) => { window.previewError = String(e && e.stack || e); });
