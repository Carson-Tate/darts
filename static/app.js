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
let cameraOn = false;

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
    banner.className = 'banner bust';
    banner.textContent = 'Bust — no score this turn';
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
  setPill(cls, v.calibrated && v.calibrated.length > 1 ? `${text} · 2 cams` : text);

  const pick = document.getElementById('camera-pick');
  if (v.cameras && pick.options.length !== v.cameras.length) {
    pick.innerHTML = v.cameras.map((c) => `<option value="${c}">${c}</option>`).join('');
  }

  const calibrated = (v.calibrated || []).length > 0;
  document.getElementById('rotation-warning')
    .classList.toggle('hidden', !calibrated || v.rotation_confident !== false);

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

  document.getElementById('vision-pill').onclick = () => toggleCamera();
  document.getElementById('btn-toggle-camera').onclick = () => toggleCamera();
  document.getElementById('btn-recalibrate').onclick = () => post('/api/vision/recalibrate');
  document.getElementById('btn-rebaseline').onclick = () => post('/api/vision/rebaseline');
  document.getElementById('btn-rotate').onclick = () => post('/api/vision/rotate?sectors=1');
  document.getElementById('btn-forget').onclick = () => post('/api/vision/forget-orientation');
  document.getElementById('camera-pick').onchange = () => { if (cameraOn) setCameraSrc(); };
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

function setCameraSrc() {
  const cam = document.getElementById('camera-pick').value;
  const q = cam ? `?camera=${encodeURIComponent(cam)}` : '';
  document.getElementById('camera-view').src = `/api/vision/stream.mjpg${q}`;
}

function toggleCamera() {
  cameraOn = !cameraOn;
  const panel = document.getElementById('camera-panel');
  panel.classList.toggle('hidden', !cameraOn);
  const view = document.getElementById('camera-view');
  if (cameraOn) {
    setCameraSrc();
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } else {
    view.removeAttribute('src');  // stop pulling the MJPEG stream
  }
  document.getElementById('btn-toggle-camera').textContent =
    cameraOn ? 'Hide camera preview' : 'Show camera preview';
}

wire();
connect();
