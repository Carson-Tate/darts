'use strict';

/* Board geometry -- mirrors darts/board.py. If you change the radii there for a
   non-regulation board, change them here too; the picker is only a UI and does
   not need to match the *camera* calibration, but it should match the rules
   engine so that a tapped D16 is the same D16 the server scores. */
const SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5];
const R = { innerBull: 6.35, outerBull: 15.9, tripleIn: 99, tripleOut: 107, doubleIn: 162, doubleOut: 170 };

const VIEW = 420, CX = 210, CY = 210, SCALE = 196 / (R.doubleOut + 16);
const COL = { black: '#161616', yellow: '#e8bd2e', miss: '#2a2f3a', wire: '#8d93a3' };

let ws = null;
let state = null;
let correctingIndex = null;

/* ----------------------------------------------------------------- speech */

/* Callouts are synthesised here rather than streamed from the Pi. The server
   sends ~20 bytes of text instead of a 55KB WAV, which is the difference
   between instant and unusable on a slow link -- and the phone can say a real
   player name, which the pre-rendered clips cannot. */
let speechOn = false;
let lastSpokenSeq = null;
const canSpeak = 'speechSynthesis' in window;

function speak(text) {
  if (!speechOn || !canSpeak || !text) return;
  // Cancel anything still queued: a stale "sixty" arriving over the top of the
  // next dart is worse than dropping it.
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 1.05;
  u.lang = 'en-GB';
  speechSynthesis.speak(u);
}

/* iOS and Android both refuse to speak until synthesis has been triggered
   inside a real user gesture, so the first tap primes it with a silent
   utterance. Without this the toggle appears to work and stays mute. */
function toggleSpeech() {
  if (!canSpeak) return;
  speechOn = !speechOn;
  if (speechOn) {
    const prime = new SpeechSynthesisUtterance('');
    prime.volume = 0;
    speechSynthesis.speak(prime);
    // Don't replay whatever was last called before sound was switched on.
    lastSpokenSeq = state && state.speech ? state.speech.seq : 0;
  } else {
    speechSynthesis.cancel();
  }
  renderSpeechButton();
}

function renderSpeechButton() {
  const el = document.getElementById('btn-sound');
  if (!el) return;
  if (!canSpeak) {
    el.textContent = 'no audio';
    el.disabled = true;
    return;
  }
  el.textContent = speechOn ? '\u{1F50A} on' : '\u{1F507} off';
  el.classList.toggle('active', speechOn);
}

function handleSpeech(s) {
  if (!s) return;
  // First snapshot after a (re)connect establishes the baseline instead of
  // blurting out the last thing that happened before we were listening.
  if (lastSpokenSeq === null) { lastSpokenSeq = s.seq; return; }
  if (s.seq > lastSpokenSeq) {
    lastSpokenSeq = s.seq;
    speak(s.text);
  }
}

/* ------------------------------------------------------------------ board */

const px = (mm) => mm * SCALE;
function pt(rmm, degCW) {
  const a = (degCW * Math.PI) / 180;
  return [CX + px(rmm) * Math.sin(a), CY - px(rmm) * Math.cos(a)];
}

function bandPath(r1, r2, a1, a2) {
  const [x1, y1] = pt(r2, a1), [x2, y2] = pt(r2, a2);
  const [x3, y3] = pt(r1, a2), [x4, y4] = pt(r1, a1);
  const R2 = px(r2), R1 = px(r1);
  return `M${x1},${y1} A${R2},${R2} 0 0 1 ${x2},${y2} L${x3},${y3} A${R1},${R1} 0 0 0 ${x4},${y4} Z`;
}

function buildBoard() {
  const parts = [];
  const seg = (d, fill, label) =>
    parts.push(`<path class="seg" d="${d}" fill="${fill}" stroke="${COL.wire}" stroke-width=".6" data-label="${label}"/>`);

  // Everything outside the doubles scores nothing.
  parts.push(
    `<circle class="seg" cx="${CX}" cy="${CY}" r="${px(R.doubleOut + 15)}" fill="${COL.miss}" data-label="MISS"/>`
  );

  for (let i = 0; i < 20; i++) {
    const a1 = i * 18 - 9, a2 = i * 18 + 9, n = SECTORS[i];
    // Parity matches render_reference() in calibrate.py: odd index = yellow single.
    const singleFill = i % 2 === 1 ? COL.yellow : COL.black;
    const ringFill = i % 2 === 1 ? COL.black : COL.yellow;
    seg(bandPath(R.outerBull, R.tripleIn, a1, a2), singleFill, `S${n}`);
    seg(bandPath(R.tripleIn, R.tripleOut, a1, a2), ringFill, `T${n}`);
    seg(bandPath(R.tripleOut, R.doubleIn, a1, a2), singleFill, `S${n}`);
    seg(bandPath(R.doubleIn, R.doubleOut, a1, a2), ringFill, `D${n}`);
  }

  parts.push(
    `<circle class="seg" cx="${CX}" cy="${CY}" r="${px(R.outerBull)}" fill="#2f7d4f" stroke="${COL.wire}" stroke-width=".6" data-label="25"/>`,
    `<circle class="seg" cx="${CX}" cy="${CY}" r="${px(R.innerBull)}" fill="#c0392b" stroke="${COL.wire}" stroke-width=".6" data-label="BULL"/>`
  );

  for (let i = 0; i < 20; i++) {
    const [x, y] = pt(R.doubleOut + 9, i * 18);
    parts.push(
      `<text x="${x}" y="${y}" fill="#e8eaf0" font-size="13" font-weight="600" ` +
      `text-anchor="middle" dominant-baseline="central">${SECTORS[i]}</text>`
    );
  }

  return `<svg viewBox="0 0 ${VIEW} ${VIEW}" role="group" aria-label="Dartboard entry">${parts.join('')}</svg>`;
}

/* ------------------------------------------------------------------- net */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onmessage = (e) => { state = JSON.parse(e.data); handleSpeech(state.speech); render(); };
  ws.onclose = () => { setPill('err', 'offline'); setTimeout(connect, 1500); };
  ws.onerror = () => ws.close();
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? null : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    console.warn(path, res.status, detail);
    return null;
  }
  return res.json();
}

/* ---------------------------------------------------------------- render */

function setPill(cls, text) {
  const el = document.getElementById('vision-pill');
  el.className = `pill ${cls}`;
  el.textContent = text;
}

function render() {
  if (!state) return;
  const g = state.game;

  document.querySelector('.bar h1').textContent = g.config.start_score;

  // players
  const host = document.getElementById('players');
  host.className = `players${g.players.length >= 2 ? ' two' : ''}`;
  host.innerHTML = g.players.map((p, i) => {
    const cls = ['player'];
    if (i === g.current && g.winner === null) cls.push('active');
    if (g.winner === i) cls.push('won');
    const checkout = p.checkout ? p.checkout.join(' → ') : '';
    const sub = g.winner === i ? 'WINNER' : (checkout || `avg ${p.average}`);
    return `<div class="${cls.join(' ')}">
      <div class="name"><span>${escapeHtml(p.name)}</span><span>${p.darts} darts</span></div>
      <div class="score">${p.score}</div>
      <div class="sub">${escapeHtml(sub)}</div>
    </div>`;
  }).join('');

  // current turn
  const slots = [0, 1, 2].map((i) => {
    const d = g.turn[i];
    if (!d) return `<div class="dart empty">–</div>`;
    const shaky = state.last_detection
      && state.last_detection.source === 'camera'
      && state.last_detection.confidence < 0.75
      && i === g.turn.length - 1;
    return `<div class="dart${shaky ? ' low-confidence' : ''}" data-index="${i}">${d.label}</div>`;
  });
  document.getElementById('turn-darts').innerHTML = slots.join('');
  document.getElementById('turn-total').textContent = g.turn_score;

  const det = state.last_detection;
  document.getElementById('turn-hint').textContent =
    det ? `${det.source}${det.source === 'camera' ? ` · ${Math.round(det.confidence * 100)}%` : ''}` : '';

  // banner
  const banner = document.getElementById('banner');
  if (g.winner !== null) {
    banner.className = 'banner win';
    banner.textContent = `${g.players[g.winner].name} wins!`;
  } else if (g.turn_end === 'bust') {
    // Say *why*. "Bust" alone is the most confusing message in darts: hitting
    // your exact remaining score and losing the turn for it looks like a
    // broken scoreboard unless you already know the double-out rule did it.
    const why = {
      not_a_double: 'Bust — you hit exactly zero, but double out is on, so the '
        + 'last dart has to be a double (or the bull). Turn Double out off in '
        + 'Setup if you’d rather any dart could win it.',
      overshot: 'Bust — that went past zero. No score this turn.',
      left_one: 'Bust — that would leave 1, and 1 cannot be finished on a double.',
    }[g.bust_reason] || 'Bust — no score this turn';
    banner.className = 'banner bust';
    banner.textContent = why;
  } else {
    banner.className = 'banner hidden';
  }

  document.getElementById('btn-undo').disabled = !g.can_undo;
  document.getElementById('btn-next').textContent =
    g.turn_end ? 'Next Player' : `Next Player (${g.darts_left} left)`;

  // vision
  const v = state.vision || {};
  const labels = {
    disabled: ['', 'no camera'], starting: ['warn', 'starting'],
    calibrating: ['warn', 'calibrating'], 'calibration-failed': ['err', 'not calibrated'],
    calibrated: ['ok', 'ready'], idle: ['ok', 'watching'],
    settling: ['warn', 'dart landing'], hand: ['warn', 'at the board'],
  };
  const [cls, text] = labels[v.state] || ['', v.state || '—'];
  const nCal = (v.calibrated || []).length;
  setPill(cls, nCal > 1 ? `${text} · ${nCal} cams` : text);

  renderCameras(v);

  // correction mode
  const head = document.querySelector('.entry-head');
  if (correctingIndex !== null && !g.turn[correctingIndex]) correctingIndex = null;
  head.classList.toggle('correcting', correctingIndex !== null);
  document.getElementById('entry-label').textContent =
    correctingIndex !== null
      ? `Correcting dart ${correctingIndex + 1} — tap the right spot`
      : 'Tap the board to enter a dart';
  document.getElementById('btn-cancel-correct').classList.toggle('hidden', correctingIndex === null);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* --------------------------------------------------------------- actions */

async function enterDart(label) {
  if (correctingIndex !== null) {
    await post('/api/correct', { index: correctingIndex, label });
    correctingIndex = null;
  } else {
    await post('/api/throw', { label });
  }
}

function wire() {
  document.getElementById('board-host').innerHTML = buildBoard();

  document.getElementById('board-host').addEventListener('click', (e) => {
    const label = e.target.getAttribute && e.target.getAttribute('data-label');
    if (label) enterDart(label);
  });

  document.getElementById('turn-darts').addEventListener('click', (e) => {
    const chip = e.target.closest('.dart[data-index]');
    if (!chip) return;
    correctingIndex = Number(chip.dataset.index);
    render();
    document.getElementById('board-host').scrollIntoView({ behavior: 'smooth', block: 'center' });
  });

  document.getElementById('btn-cancel-correct').onclick = () => { correctingIndex = null; render(); };
  document.getElementById('btn-miss').onclick = () => enterDart('MISS');

  const sound = document.getElementById('btn-sound');
  sound.onclick = () => {
    toggleSpeech();
    // Speak on the enabling tap so it's obvious the phone can talk -- and so
    // the gesture that unlocks synthesis produces audible proof it worked.
    if (speechOn) speak('Sound on');
  };
  renderSpeechButton();
  document.getElementById('btn-next').onclick = () => post('/api/next');
  document.getElementById('btn-undo').onclick = () => post('/api/undo');

  document.getElementById('btn-settings').onclick = () => showSetup(true);
  document.getElementById('btn-cancel-setup').onclick = () => showSetup(false);
  document.getElementById('btn-add-player').onclick = () => addNameRow('');
  document.getElementById('btn-start').onclick = startGame;

  document.getElementById('vision-pill').onclick = () =>
    document.getElementById('camera-panel')
      .scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  document.getElementById('btn-recalibrate').onclick = () => post('/api/vision/recalibrate');
  document.getElementById('btn-rebaseline').onclick = () => post('/api/vision/rebaseline');
  document.getElementById('btn-forget').onclick = () => post('/api/vision/forget-orientation');
  document.getElementById('cfg-fullview').onchange = (e) => {
    fullView = e.target.checked;
    refreshTiles(tileNames);
  };
}

/* ----------------------------------------------------------------- setup */

function addNameRow(value) {
  const list = document.getElementById('name-list');
  const row = document.createElement('div');
  row.className = 'name-row';
  row.innerHTML = `<input type="text" placeholder="Name" value="${escapeHtml(value)}" maxlength="16"><button aria-label="Remove">&times;</button>`;
  row.querySelector('button').onclick = () => {
    if (list.children.length > 1) row.remove();
  };
  list.appendChild(row);
}

function showSetup(on) {
  document.getElementById('screen-setup').classList.toggle('hidden', !on);
  document.getElementById('screen-game').classList.toggle('hidden', on);
  if (!on || !state) return;

  const g = state.game;
  document.getElementById('cfg-start').value = String(g.config.start_score);
  document.getElementById('cfg-double-out').checked = g.config.double_out;
  document.getElementById('cfg-double-in').checked = g.config.double_in;
  document.getElementById('cfg-auto').checked = g.config.auto_advance;
  document.getElementById('name-list').innerHTML = '';
  g.players.forEach((p) => addNameRow(p.name));
}

async function startGame() {
  const names = [...document.querySelectorAll('#name-list input')]
    .map((i) => i.value.trim())
    .filter((n, idx, arr) => n !== '' || arr.length === 1);
  await post('/api/game/new', {
    names: names.length ? names : ['Player 1'],
    start_score: Number(document.getElementById('cfg-start').value),
    double_out: document.getElementById('cfg-double-out').checked,
    double_in: document.getElementById('cfg-double-in').checked,
    auto_advance: document.getElementById('cfg-auto').checked,
  });
  showSetup(false);
}

/* ---------------------------------------------------------------- camera */

/* Both cameras stream continuously, so the wire cost is doubled on a link that
   has been the slow part of this setup throughout. A tile is requested at
   roughly the size it is drawn -- about 14KB a frame at 3fps -- and only the
   one you tap is asked for at a size worth looking closely at. */
const TILE = { w: 360, q: 55, fps: 3 };
const BIG = { w: 960, q: 78, fps: 6 };

let enlarged = null;      // camera name shown large, or null
let tileNames = [];       // what's currently built, to avoid pointless rebuilds
let fullView = false;     // show the whole frame instead of just the board

/* Cropping to the board is a *display* choice. It does not change what the
   detector sees -- that is already confined to the board by the calibrated ROI
   -- and it cannot change exposure, which the sensor meters across the whole
   frame whatever we crop afterwards. What it buys is a board big enough on a
   phone to tell whether the overlay is sitting on the rings. Full view is for
   aiming a camera, where the surroundings are the point. */
function streamUrl(cam, big) {
  const o = big ? BIG : TILE;
  return `/api/vision/stream.mjpg?camera=${encodeURIComponent(cam)}` +
         `&w=${o.w}&q=${o.q}&fps=${o.fps}&crop=${fullView ? 0 : 1}`;
}

function buildTiles(cams) {
  const host = document.getElementById('camera-tiles');
  host.innerHTML = cams.map((c) => `
    <figure class="tile" data-cam="${escapeHtml(c)}">
      <img alt="Live view from the ${escapeHtml(c)} camera, calibration grid overlaid">
      <figcaption>
        <span class="tile-name">${escapeHtml(c)}</span>
        <span class="tile-badge"></span>
        <button class="ctl tiny tile-confirm hidden" title="Remember this orientation so it survives recalibration">Looks right</button>
        <button class="ctl tiny tile-rotate" title="Rotate this camera's grid by one sector">Rotate &#8635;</button>
      </figcaption>
    </figure>`).join('');

  host.querySelectorAll('.tile').forEach((tile) => {
    const cam = tile.dataset.cam;
    // Rotate is per camera: the two resolve the board's 36-degree symmetry
    // independently, so one can be right while the other is two sectors out.
    tile.querySelector('.tile-rotate').onclick = (e) => {
      e.stopPropagation();
      post(`/api/vision/rotate?sectors=1&camera=${encodeURIComponent(cam)}`);
    };
    // Zero sectors changes nothing and saves the template anyway -- the way to
    // confirm an orientation the system got right but isn't confident about.
    tile.querySelector('.tile-confirm').onclick = (e) => {
      e.stopPropagation();
      post(`/api/vision/rotate?sectors=0&camera=${encodeURIComponent(cam)}`);
    };
    const img = tile.querySelector('img');
    img.onclick = () => {
      enlarged = enlarged === cam ? null : cam;
      refreshTiles(cams);
    };
    img.onload = () => markAlive(img);
    img.onerror = () => { img.dataset.seen = '0'; };  // let the watchdog retry
  });
  tileNames = cams.slice();
  refreshTiles(cams);
}

/* Point each <img> at the size it is actually being shown at. Reassigning src
   restarts the MJPEG stream, so only do it when the URL really changed --
   otherwise every state broadcast would tear down and rebuild both streams. */
function refreshTiles(cams) {
  document.querySelectorAll('#camera-tiles .tile').forEach((tile) => {
    const cam = tile.dataset.cam;
    const big = enlarged === cam;
    tile.classList.toggle('big', big);
    tile.classList.toggle('shrunk', enlarged !== null && !big);
    const img = tile.querySelector('img');
    const want = streamUrl(cam, big);
    if (img.getAttribute('src') !== want) restart(img, want);
  });
}

/* ---- keeping the streams alive ----
   An MJPEG stream is one long-lived response, and when it stalls it stalls
   silently: the <img> keeps showing the last frame it got and nothing ever
   asks again. On a link that drops as often as this one does, that reads as
   "the camera died" -- which is exactly how it was reported, while both
   cameras were in fact running and scoring darts.

   So watch for frames arriving and re-request a stream that has gone quiet.
   Each restart needs a fresh URL or the browser may serve the dead connection
   back from cache, hence the cache-buster. */
const STALL_MS = 12000;

function restart(img, url) {
  img.dataset.base = url;
  img.dataset.seen = String(Date.now());
  img.src = `${url}&_=${Date.now()}`;
}

function watchStreams() {
  const now = Date.now();
  document.querySelectorAll('#camera-tiles img').forEach((img) => {
    if (!img.dataset.base) return;
    if (now - Number(img.dataset.seen || 0) < STALL_MS) return;
    console.warn('camera stream stalled, restarting', img.dataset.base);
    restart(img, img.dataset.base);
  });
}

/* `load` fires once per frame for a multipart stream in the browsers this runs
   on, which makes it a usable heartbeat. `error` means the connection is
   already gone, so restart on the next tick rather than hammering it. */
function markAlive(img) {
  img.dataset.seen = String(Date.now());
}

function renderCameras(v) {
  const cams = v.cameras || [];
  const panel = document.getElementById('camera-panel');
  panel.classList.toggle('hidden', cams.length === 0);
  if (!cams.length) return;

  const same = cams.length === tileNames.length && cams.every((c, i) => c === tileNames[i]);
  if (!same) buildTiles(cams);

  // Say which camera needs attention, not just that something does -- with two
  // of them "orientation is unsure" isn't actionable until you know which one.
  const per = v.per_camera || {};
  const unsure = [];
  document.querySelectorAll('#camera-tiles .tile').forEach((tile) => {
    const info = per[tile.dataset.cam] || {};
    const badge = tile.querySelector('.tile-badge');
    let cls = 'tile-badge ok', text = info.remembered ? 'remembered' : 'locked';
    if (!info.calibrated) {
      cls = 'tile-badge err'; text = 'not calibrated';
    } else if (info.rotation_confident === false) {
      cls = 'tile-badge warn'; text = 'check rotation';
      unsure.push(tile.dataset.cam);
    }
    badge.className = cls;
    badge.textContent = text;
    // Offer to remember an orientation that isn't already remembered -- whether
    // the system is unsure of it or merely hasn't been told it's right.
    tile.querySelector('.tile-confirm')
      .classList.toggle('hidden', !info.calibrated || !!info.remembered);
  });

  const warn = document.getElementById('rotation-warning');
  warn.classList.toggle('hidden', unsure.length === 0);
  if (unsure.length) {
    warn.textContent =
      `${unsure.join(' and ')}: the orientation wasn't a confident lock. Check the ` +
      `numbers on that view line up with the real board — tap its Rotate until they ` +
      `do, then "Looks right" to remember it.`;
  }
}

wire();
connect();
setInterval(watchStreams, 4000);
