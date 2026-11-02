// The full cut as one paused GSAP timeline built from config/demo_full.toml. render.mjs writes build.json (checked config,
// capture rects, footage frame counts) and passes its URL as ?build=; then it calls window.seek(t) once per output frame.
(async () => {
  const q = new URLSearchParams(location.search);
  const build = await (await fetch(q.get('build'))).json();
  const { cfg, rects, footage } = build;
  const W = cfg.output.width, H = cfg.output.height, FPS = cfg.output.fps;
  const L = cfg.layout, C = cfg.composition, T = cfg.textcard, M = L.margin_px, CB = cfg.caption_box, fade = CB.fade_s, scenes = cfg.scene, total = cfg.total_s;
  const $ = (id) => document.getElementById(id);
  const mk = (tag, cls, parent, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text) e.textContent = text; if (parent) parent.appendChild(e); return e; };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const stage = $('stage'), css = document.documentElement.style;
  css.setProperty('--w', W); css.setProperty('--h', H); css.setProperty('--radius', L.radius_px);

  const load = (img, url) => new Promise((ok, fail) => { img.onload = ok; img.onerror = () => fail(new Error(`missing ${url}`)); img.src = url; }).then(() => img.decode());
  await Promise.all([`${CB.text_px}px Roboto`, `${CB.tag_font_weight} ${CB.tag_px}px Roboto`, `${T.line_px}px "David Libre"`].map((f) => document.fonts.load(f)));

  // Bands (as in the approved preview): one stage box, a constant gap, then the caption block; nothing inside the margin.
  const tagH = Math.ceil(CB.tag_px * 1.2) + 2 * CB.tag_pad_px[0] + 2;
  const capH = tagH + CB.tag_gap_px + Math.ceil(CB.text_lines * CB.text_px * CB.line_height);
  const capTop = H - M - capH, top = M, bottom = capTop - CB.stage_gap_px, boxH = bottom - top, boxW = W - 2 * M;
  const vars = { '--tag-px': CB.tag_px, '--caption-tag-weight': CB.tag_font_weight, '--tag-pad-v': CB.tag_pad_px[0], '--tag-pad-h': CB.tag_pad_px[1], '--text-px': CB.text_px,
    '--line-height': CB.line_height, '--max-chars': CB.max_chars, '--tag-gap': CB.tag_gap_px, '--mark-px': CB.mark_px,
    '--cap-items': 'center', '--cap-align': 'center',
    // The shared preview.css reads these for the match panels, the connector line and the scrim cut-out.
    '--ring-border': cfg.panel.ring_border_px, '--ring-radius': cfg.panel.ring_radius_px, '--ring-tint': cfg.panel.ring_tint_pct,
    '--link-stroke': cfg.panel.link_stroke_px, '--panel-tag-px': cfg.panel.tag_px, '--panel-tag-weight': cfg.panel.tag_weight,
    '--panel-tag-spacing': cfg.panel.tag_letter_spacing_em, '--panel-tag-gap': cfg.panel.tag_gap_px, '--dim-radius': cfg.panel.dim_radius_px,
    '--bar-inset': C.bar_inset_px, '--line-px': T.line_px, '--line-small-px': T.line_small_px, '--line-url-px': T.line_url_px,
    '--logo-icon-px': T.logo_icon_px, '--logo-word-px': T.logo_word_px, '--card-gap': T.card_gap_px, '--logo-gap': T.logo_gap_px,
    '--url-weight': T.url_weight, '--bar-bottom': C.bar_bottom_px, '--bar-height': C.bar_height_px, '--bar-radius': C.bar_radius_px,
    '--dot-px': C.dot_px, '--dot-rise': C.dot_rise_px, '--time-px': C.time_px, '--time-weight': C.time_weight, '--time-bottom': C.time_bottom_px,
    '--veil-px': C.veil_px, '--veil-weight': C.veil_weight, '--veil-spacing': C.veil_letter_spacing_em, '--veil-radius': C.veil_radius_px };
  for (const [k, v] of Object.entries(vars)) css.setProperty(k, v);

  const tl = gsap.timeline({ paused: true, defaults: { ease: 'power2.out' } });
  const footageScenes = [];
  const secs = scenes.map((sc, i) => {
    const sec = mk('section', `scene k-${sc.kind}`, stage);
    sec.dataset.id = sc.id;
    gsap.set(sec, { autoAlpha: i === 0 ? 1 : 0 });
    return sec;
  });
  stage.insertBefore($('captions'), null); stage.appendChild($('badge'));

  // ---- window layout (also the match-cut slot) ----
  function layoutWindow(sc) {
    const gap = L.world_gap_px;
    let x = 0; const place = {};
    for (const n of sc.images) { place[n] = x; x += rects[n].size[0] + gap; }
    const ww = x - gap, wh = Math.max(...sc.images.map((n) => rects[n].size[1]));
    const winH = Math.min(boxH, boxW * wh / ww), winW = Math.round(winH * ww / wh);
    const win = { x: (W - winW) / 2, y: top, width: winW, height: Math.round(winH) };
    return { win, place, ww, wh, k0: win.width / ww };
  }
  const lay = scenes.map((sc) => (sc.kind === 'window' ? layoutWindow(sc) : null));
  // A stop is "capture.mark" or a number rect; ":y0-y1" and "@x0-x1" cut it to a fraction of the mark.
  const worldRect = (sc, l, stop) => {
    if (Array.isArray(stop)) return stop;
    const m = /^(\w+)\.(\w+)(?::([\d.]+)-([\d.]+))?(?:@([\d.]+)-([\d.]+))?$/.exec(stop);
    const r = rects[m[1]][m[2]], [y0, y1] = m[3] ? [+m[3], +m[4]] : [0, 1], [x0, x1] = m[5] ? [+m[5], +m[6]] : [0, 1];
    return [l.place[m[1]] + r[0] + x0 * r[2], r[1] + y0 * r[3], (x1 - x0) * r[2], (y1 - y0) * r[3]];
  };

  // ---- scene builders ----
  scenes.forEach((sc, i) => {
    const sec = secs[i], t0 = sc.start, next = scenes[i + 1];
    const matchNext = next && next.kind === 'window' && next.from_footage;
    // Crossfade between scenes: this one in over fade_s, the previous one out over the same time.
    if (i > 0 && !(sc.kind === 'window' && sc.from_footage)) tl.to(sec, { autoAlpha: 1, duration: fade, ease: 'none' }, t0);
    if (next && !matchNext) tl.to(sec, { autoAlpha: 0, duration: fade, ease: 'none' }, next.start);

    if (sc.kind === 'footage') {
      const cardH = boxH, cardW = Math.round(cardH * C.footage_aspect[0] / C.footage_aspect[1]);
      const card = { x: (W - cardW) / 2, y: top, width: cardW, height: cardH };
      const el = mk('div', 'fcard', sec); const canvas = mk('canvas', '', el);
      gsap.set(el, { ...card, borderRadius: L.radius_px });
      let bar = null;
      if (sc.scrub) {
        mk('div', 'shade', el); const b = mk('div', 'bar', el);
        bar = { fill: mk('div', 'fill', b), dot: mk('div', 'dot', b), time: mk('div', 'time', b), w: cardW - 2 * C.bar_inset_px };
      }
      if (i === 0) tl.fromTo(el, { autoAlpha: 0, y: card.y + C.open_rise_px }, { autoAlpha: 1, y: card.y, duration: C.open_rise_s, ease: 'power3.out' }, 0);
      if (matchNext) {
        const l = lay[i + 1], fr = rects[next.images[0]].frame, s = next.start;
        const slot = { x: l.win.x + 1 + fr[0] * l.k0, y: l.win.y + 1 + fr[1] * l.k0, width: fr[2] * l.k0, height: fr[3] * l.k0 };
        const fl = L.flight_s;
        tl.to(el, { ...slot, borderRadius: C.slot_radius_px, borderColor: 'rgba(0,0,0,0)', boxShadow: '0 0 0 rgba(0,0,0,0)', ease: 'power2.inOut', duration: fl * C.flight_move_frac }, s);
        tl.to(sec, { autoAlpha: 0, duration: fl * (1 - C.flight_move_frac), ease: 'none' }, s + fl * C.flight_move_frac);
      }
      const info = footage[sc.id], ctx = canvas.getContext('2d');
      footageScenes.push({ sc, canvas, ctx, bar, info, last: -1, end: sc.start + sc.dur + (matchNext ? L.flight_s : 0) });
    }

    if (sc.kind === 'window') {
      const l = lay[i], { win, k0 } = l;
      const w = mk('div', 'window', sec), cam = mk('div', 'cam', w);
      gsap.set(w, { ...win });
      for (const n of sc.images) {
        const img = mk('img', '', cam); img.style.position = 'absolute';
        img.style.left = `${l.place[n] * k0}px`; img.style.top = '0px'; img.style.width = `${rects[n].size[0] * k0}px`;
        img.dataset.src = `${build.ui}/${n}.png`;
      }
      for (const cp of cfg.capture.filter((c) => sc.images.includes(c.name))) for (const m of cp.veil) {
        const r = rects[cp.name][m], v = mk('div', 'veil', cam, C.veil_label);
        Object.assign(v.style, { left: `${(l.place[cp.name] + r[0] - C.veil_pad_px) * k0}px`, top: `${(r[1] - C.veil_pad_px) * k0}px`, width: `${(r[2] + 2 * C.veil_pad_px) * k0}px`, height: `${(r[3] + 2 * C.veil_pad_px) * k0}px` });
      }
      const dim = mk('div', 'dim', cam);
      const stops = sc.stops.map((s) => worldRect(sc, l, s));
      const place = (r) => ({ left: r[0] * k0, top: r[1] * k0, width: r[2] * k0, height: r[3] * k0 });
      Object.assign(dim.style, { boxShadow: `0 0 0 4000px color-mix(in srgb, var(--pf-ink) ${L.dim * 100}%, transparent)` });
      gsap.set(dim, { ...place(stops[0]), autoAlpha: 0 });
      const camFor = (r) => {
        const k = Math.min(win.width * (1 - 2 * L.camera_margin) / r[2], win.height * (1 - 2 * L.camera_margin) / r[3], k0 * L.max_zoom);
        return { scale: k / k0, x: win.width / 2 - (r[0] + r[2] / 2) * k, y: win.height / 2 - (r[1] + r[3] / 2) * k };
      };
      const hold = (sc.dur - L.lead_s) / stops.length - L.push_s, step = L.push_s + hold;
      stops.forEach((r, k) => {
        const p0 = t0 + L.lead_s + k * step;
        tl.to(cam, { ...camFor(r), ease: 'power2.inOut', duration: L.push_s }, p0);
        tl.to(dim, { ...place(r), autoAlpha: 1, ease: 'power2.inOut', duration: L.push_s }, p0);
      });
      if (sc.from_footage) tl.to(sec, { autoAlpha: 1, duration: C.from_footage_fade[1], ease: 'none' }, t0 + C.from_footage_fade[0]);
    }

    if (sc.kind === 'match') {
      const R = rects.replay, ms = L.match_scale, gap = L.match_gap_px;
      const tw = R.table[0] * ms, th = R.table[1] * ms, cw = R.card[0] * ms, ch = R.card[1] * ms;
      const mid = top + boxH / 2, left = (W - tw - gap - cw) / 2;
      const tp = { x: left, y: mid - th / 2 }, cp = { x: left + tw + gap, y: mid - ch / 2 };
      const panel = (id, tag, img, p, w) => {
        const f = mk('figure', '', sec), c = mk('figcaption', 'tag', f, tag), s = mk('div', 'shot', f), im = mk('img', '', s);
        im.style.width = `${w}px`; im.dataset.src = `${build.ui}/${img}.png`; gsap.set(f, { ...p, autoAlpha: 0 }); return { f, s };
      };
      const tab = panel('t', C.table_label, 'table', tp, tw), crd = panel('c', C.card_label, 'card', cp, cw);
      const NP = C.note_pad_px, [, ry, , rh] = R.row, [nx, ny, nw, nh] = R.note;
      const rr = mk('div', 'ring', tab.s), nr = mk('div', 'ring', crd.s);
      gsap.set(rr, { x: C.row_ring_inset_px[0], y: ry * ms, width: tw - C.row_ring_inset_px[1], height: rh * ms, scaleX: 0, transformOrigin: '0% 50%' });
      gsap.set(nr, { x: nx * ms - NP[0], y: ny * ms - NP[1], width: nw * ms + 2 * NP[0], height: nh * ms + 2 * NP[1], autoAlpha: 0 });
      const a = { x: tp.x + tw - C.link_anchor_inset_px, y: tp.y + 1 + (ry + rh / 2) * ms }, b = { x: cp.x + 1 + nx * ms - NP[0], y: cp.y + 1 + (ny + nh / 2) * ms };
      const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg'), path = document.createElementNS(NS, 'path');
      svg.id = 'link'; sec.appendChild(svg); svg.appendChild(path);
      svg.setAttribute('width', W); svg.setAttribute('height', H); svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
      path.setAttribute('d', `M${a.x},${a.y} C${a.x + gap},${a.y} ${b.x - gap},${b.y} ${b.x},${b.y}`);
      const len = path.getTotalLength(); gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
      const span = (k) => [t0 + sc[k][0], sc[k][1] - sc[k][0]];
      tl.fromTo(tab.f, { autoAlpha: 0, y: tp.y + C.panel_rise_px }, { autoAlpha: 1, y: tp.y, duration: sc.table_in[1] - sc.table_in[0] }, t0 + sc.table_in[0]);
      tl.to(rr, { scaleX: 1, ease: 'power3.out', duration: span('ring')[1] }, span('ring')[0]);
      tl.to(path, { strokeDashoffset: 0, ease: 'power1.inOut', duration: span('link')[1] }, span('link')[0]);
      tl.fromTo(crd.f, { autoAlpha: 0, y: cp.y + C.panel_rise_px }, { autoAlpha: 1, y: cp.y, duration: span('card')[1] }, span('card')[0]);
      tl.to(nr, { autoAlpha: 1, duration: span('note')[1] }, span('note')[0]);
    }

    if (sc.kind === 'card') {
      const wrap = mk('div', 'textcard', sec); gsap.set(wrap, { x: 0, y: top, width: W, height: boxH });
      const items = [];
      if (sc.logo) {
        const lg = mk('div', 'logo', wrap), im = mk('img', '', lg); im.src = '/static/ui/icon.svg';
        const word = mk('span', 'word', lg, T.logo_wordmark);
        tl.fromTo(im, { autoAlpha: 0, scale: T.logo_icon_scale }, { autoAlpha: 1, scale: 1, duration: T.logo_icon_s }, t0 + T.logo_icon_start_s);
        tl.fromTo(word, { clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0% 0 0)', duration: T.logo_word_s, ease: 'power2.inOut' }, t0 + T.logo_word_start_s);
      }
      const lines = sc.lines.filter((s) => !(s.includes('{repo}') && !cfg.links.repo_url))
        .map((s) => s.replace('{live}', cfg.links.live_label).replace('{repo}', cfg.links.repo_url));
      lines.forEach((s, k) => {
        const el = mk('p', 'line' + (k === 0 && lines.length > 1 ? ' small' : '') + (s === cfg.links.live_label ? ' url' : ''), wrap, s);
        el.style.margin = '0';
        tl.fromTo(el, { autoAlpha: 0, y: T.line_rise_px }, { autoAlpha: 1, y: 0, duration: T.line_s }, t0 + sc.reveal_start_s + k * sc.reveal_gap_s);
      });
      // So the card never sits still: a slow drift over its whole length.
      tl.fromTo(wrap, { scale: 1 }, { scale: 1 + L.drift, ease: 'none', duration: sc.dur }, t0);
    }
  });

  // ---- captions: one fixed block, a crossfade at every cut; the badge from the first beat, gone on the end card ----
  const caps = [];
  scenes.forEach((sc) => sc.caption.forEach((c) => caps.push({ ...c, t: sc.start + c.at })));
  const capX = M;
  caps.forEach((c, k) => {
    const p = mk('p', 'caption', $('captions'));
    mk('span', 'tag', p, c.tag); mk('span', 'text', p, c.text);
    gsap.set(p, { x: capX, y: capTop, width: W - 2 * capX, autoAlpha: 0 });
    tl.to(p, { autoAlpha: 1, duration: fade, ease: 'none' }, k === 0 ? C.open_caption_s : c.t);
    if (caps[k + 1]) tl.to(p, { autoAlpha: 0, duration: fade, ease: 'none' }, caps[k + 1].t);
  });
  gsap.set('#badge', { x: W - M - CB.mark_px, y: capTop + (capH - CB.mark_px) / 2, autoAlpha: 0 });
  tl.to('#badge', { autoAlpha: 1, duration: C.badge_in_s, ease: 'none' }, C.open_caption_s);
  const lastLogo = scenes[scenes.length - 1].logo ? scenes[scenes.length - 1] : null;
  if (lastLogo) tl.to('#badge', { autoAlpha: 0, duration: fade, ease: 'none' }, lastLogo.start);
  tl.set({}, {}, total);

  // ---- images, loaded once; footage frames fetched per frame, so memory stays flat ----
  await Promise.all([...document.querySelectorAll('img[data-src]')].map((im) => load(im, im.dataset.src)));
  const srcAt = (map, s) => {
    if (s <= map[0][0]) return map[0][1];
    for (let k = 1; k < map.length; k++) if (s <= map[k][0]) { const [a, sa] = map[k - 1], [b, sb] = map[k]; return sa + (sb - sa) * (s - a) / (b - a); }
    return map[map.length - 1][1];
  };
  const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  async function drawFootage(t) {
    for (const f of footageScenes) {
      if (t < f.sc.start - 0.5 || t > f.end) continue;
      const src = srcAt(f.sc.map, t - f.sc.start);
      const idx = clamp(Math.round((src - f.info.start) * FPS), 0, f.info.frames - 1) + 1;
      if (idx !== f.last) {
        const r = await fetch(`${footage.url}/${f.sc.id}/${String(idx).padStart(4, '0')}.jpg`);
        if (!r.ok) throw new Error(`footage ${f.sc.id} frame ${idx}: ${r.status}`);
        const bmp = await createImageBitmap(await r.blob());
        if (f.canvas.width !== bmp.width) { f.canvas.width = bmp.width; f.canvas.height = bmp.height; }
        f.ctx.drawImage(bmp, 0, 0); bmp.close(); f.last = idx;
      }
      if (f.bar) {
        const x = f.bar.w * clamp(src / cfg.source.source_len_s, 0, 1);
        gsap.set(f.bar.fill, { width: x }); gsap.set(f.bar.dot, { x }); f.bar.time.textContent = mmss(src);
        gsap.set(f.bar.time, { x: clamp(x - 22, 0, f.bar.w - 70) });
      }
    }
  }
  window.seek = async (t) => {
    tl.time(t, false); await drawFootage(t);
    return new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok)));
  };
  window.previewReady = { duration: total };
  if (q.has('play')) {
    const t0 = performance.now();
    const loop = async () => { await window.seek(((performance.now() - t0) / 1000) % total); requestAnimationFrame(loop); };
    loop();
  } else await window.seek(0);
})().catch((e) => { window.previewError = String(e && e.stack || e); });
