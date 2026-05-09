/* Scene 2 — All the way to chaos.
 *
 * Pedagogical goal: x_T is essentially Gaussian noise, no matter how the
 * trajectory started. The schedule (β_t and ᾱ_t) is the supporting
 * apparatus — students should feel its shape.
 *
 * Two stacked panes (2D letter-M + 28×28 MNIST) driven from a precomputed
 * trajectory cache (T+1 snapshots) so animation is a pure index lookup.
 *
 * Step engine (cursor 0..3):
 *   0  t = 0. Both panes show clean state. Play button visible.
 *   1  Autoplay: a RAF loop sweeps t = 0 → T-1 in ~2 seconds.
 *   2  Scrubber active. Drag to any t.
 *   3  Pixel-histogram overlay (toggle).
 *
 * Cold entry: rebuilds trajectory from DATA on each onEnter (deterministic).
 */

window.scenes.scene2 = function (root) {
  const DATA = window.DATA;
  const M = window.DiffusionMath;
  const NN = window.DiffusionNN;

  const SEED_2D = 7;
  const SEED_PX = 13;
  const T = DATA.T;                      // 200
  const T_2D_PTS = DATA.letterM.points.length;   // 300
  const T_PX = 28 * 28;                  // 784
  const STEPS = 3;                       // cursor max
  const SWEEP_MS = 2000;
  const DEFAULT_DIGIT = 3;

  /* ----- DOM scaffolding ---------------------------------------------------- */

  root.innerHTML = '';
  root.classList.add('scene-s2');

  const layout = document.createElement('div');
  layout.className = 'scene-layout';
  root.appendChild(layout);

  /* viz column */
  const vizCol = document.createElement('div');
  vizCol.className = 's2-viz-col';
  layout.appendChild(vizCol);

  // Top pane — 2D
  const topPane = document.createElement('div');
  topPane.className = 's2-top-pane viz-wrap';
  vizCol.appendChild(topPane);

  const topLabel = document.createElement('div');
  topLabel.className = 's2-pane-label';
  topLabel.textContent = 'Letter M — 300 points in 2D';
  topPane.appendChild(topLabel);

  const svg2D = d3.select(topPane).append('svg')
    .attr('class', 's2-svg2d')
    .attr('viewBox', '0 0 100 100')
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const gGrid    = svg2D.append('g').attr('class', 's2-grid');
  const gPoints  = svg2D.append('g').attr('class', 's2-points');

  gGrid.selectAll('line').data([
    { x1: 0,  y1: 50, x2: 100, y2: 50 },
    { x1: 50, y1: 0,  x2: 50,  y2: 100 },
  ]).enter().append('line')
    .attr('class', 's2-grid-line')
    .attr('x1', d => d.x1).attr('y1', d => d.y1)
    .attr('x2', d => d.x2).attr('y2', d => d.y2);

  // Bottom pane — MNIST + histogram
  const botPane = document.createElement('div');
  botPane.className = 's2-bot-pane';
  vizCol.appendChild(botPane);

  const botLabel = document.createElement('div');
  botLabel.className = 's2-pane-label';
  botLabel.textContent = 'MNIST — 28×28 pixels';
  botPane.appendChild(botLabel);

  const botRow = document.createElement('div');
  botRow.className = 's2-bot-row';
  botPane.appendChild(botRow);

  const canvas = document.createElement('canvas');
  canvas.className = 'mnist-pane s2-mnist-canvas';
  canvas.width  = 112;
  canvas.height = 112;
  botRow.appendChild(canvas);
  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.imageSmoothingEnabled = false;

  // Off-screen 28×28 source for upscale
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  const offCtx = off.getContext('2d');
  const offImg = offCtx.createImageData(28, 28);

  // Histogram inset (toggleable)
  const histWrap = document.createElement('div');
  histWrap.className = 's2-hist-wrap';
  histWrap.innerHTML = `
    <div class="s2-hist-label">pixel intensities</div>
    <canvas class="s2-hist-canvas" width="160" height="80"></canvas>
  `;
  botRow.appendChild(histWrap);
  const histCanvas = histWrap.querySelector('.s2-hist-canvas');
  const histCtx = histCanvas.getContext('2d');

  // Digit selector
  const selectorWrap = document.createElement('div');
  selectorWrap.className = 's2-selector';
  selectorWrap.innerHTML =
    `<span class="s2-selector-label">digit</span>` +
    `<select class="s2-digit-select" aria-label="MNIST digit"></select>`;
  botRow.appendChild(selectorWrap);
  const digitSelect = selectorWrap.querySelector('.s2-digit-select');
  for (let d = 0; d < 10; d++) {
    const opt = document.createElement('option');
    opt.value = String(d);
    opt.textContent = String(d);
    if (d === DEFAULT_DIGIT) opt.selected = true;
    digitSelect.appendChild(opt);
  }

  /* text column */
  const textCol = document.createElement('div');
  textCol.className = 'text-col s2-text';
  layout.appendChild(textCol);

  const stepPill = document.createElement('div');
  stepPill.className = 'step-pill s2-step-pill';
  textCol.appendChild(stepPill);

  const heading = document.createElement('h2');
  heading.textContent = 'All the way to chaos.';
  textCol.appendChild(heading);

  const tagline = document.createElement('p');
  tagline.className = 's2-tagline';
  tagline.innerHTML = '<em>Two hundred steps. From M to noise.</em>';
  textCol.appendChild(tagline);

  // Time panel — counter + chart
  const timePanel = document.createElement('div');
  timePanel.className = 's2-time-panel';
  textCol.appendChild(timePanel);

  const counter = document.createElement('div');
  counter.className = 's2-counter';
  counter.innerHTML = `
    <span class="s2-counter-label">t</span>
    <span class="s2-counter-value">
      <span data-role="t-now">0</span><span class="s2-counter-sep">/</span><span data-role="t-max">${T - 1}</span>
    </span>
  `;
  timePanel.appendChild(counter);

  const chartWrap = document.createElement('div');
  chartWrap.className = 's2-chart-wrap';
  timePanel.appendChild(chartWrap);

  const chartSvg = d3.select(chartWrap).append('svg')
    .attr('class', 's2-chart-svg')
    .attr('viewBox', '0 0 320 120')
    .attr('preserveAspectRatio', 'none');

  // Controls: play/pause + scrubber
  const controls = document.createElement('div');
  controls.className = 's2-controls';
  controls.innerHTML = `
    <button class="btn s2-play-btn" type="button">Play</button>
    <input type="range" class="s2-scrubber" min="0" max="${T - 1}" step="1" value="0" disabled>
  `;
  textCol.appendChild(controls);
  const playBtn = controls.querySelector('.s2-play-btn');
  const scrubber = controls.querySelector('.s2-scrubber');

  // Histogram toggle
  const histToggleWrap = document.createElement('label');
  histToggleWrap.className = 's2-hist-toggle';
  histToggleWrap.innerHTML = `
    <input type="checkbox" class="s2-hist-checkbox">
    <span>show pixel histogram</span>
  `;
  textCol.appendChild(histToggleWrap);
  const histCheckbox = histToggleWrap.querySelector('.s2-hist-checkbox');

  const tail = document.createElement('p');
  tail.className = 'muted s2-tail';
  textCol.appendChild(tail);

  /* ----- state -------------------------------------------------------------- */

  const state = {
    cursor: 0,
    t: 0,
    digit: DEFAULT_DIGIT,
    showHist: false,
    playing: false,
    raf: null,
    playStartTs: null,
    // caches:
    traj2D: null,    // Array<Float32Array(600)>, length T+1
    trajPx: null,    // Array<Float32Array(784)>, length T+1
  };

  /* ----- trajectory builders ------------------------------------------------ */

  function build_x0_2D() {
    const x0 = new Float32Array(T_2D_PTS * 2);
    for (let i = 0; i < T_2D_PTS; i++) {
      x0[2 * i]     = DATA.letterM.points[i][0];
      x0[2 * i + 1] = DATA.letterM.points[i][1];
    }
    return x0;
  }

  function build_y0(labelIdx) {
    const src = DATA.mnistSamples[labelIdx].pixels;
    const y0 = new Float32Array(T_PX);
    for (let i = 0; i < T_PX; i++) y0[i] = src[i];
    return y0;
  }

  function rebuildTrajectories() {
    const x0 = build_x0_2D();
    const y0 = build_y0(state.digit);
    const rng2D = M.mulberry32(SEED_2D);
    const rngPx = M.mulberry32(SEED_PX);
    state.traj2D = NN.forwardTrajectory(x0, DATA.betas, rng2D).trajectory;
    state.trajPx = NN.forwardTrajectory(y0, DATA.betas, rngPx).trajectory;
  }

  /* ----- viz renderers ------------------------------------------------------ */

  function xToSvg(x) { return ((x + 1.7) / 3.4) * 100; }
  function yToSvg(y) { return ((1.7 - y) / 3.4) * 100; }

  function render2DAtT(t) {
    const x = state.traj2D[t];
    const data = [];
    for (let i = 0; i < T_2D_PTS; i++) {
      data.push({ i, x: x[2 * i], y: x[2 * i + 1] });
    }
    const sel = gPoints.selectAll('circle.point').data(data, d => d.i);
    sel.enter()
      .append('circle')
      .attr('class', 'point cluster-1')
      .attr('r', 0.7)
      .merge(sel)
      .attr('cx', d => xToSvg(d.x))
      .attr('cy', d => yToSvg(d.y));
    sel.exit().remove();
  }

  function renderMnistAtT(t) {
    const y = state.trajPx[t];
    const data = offImg.data;
    for (let i = 0; i < T_PX; i++) {
      let v = (y[i] + 1) * 127.5;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      const j = i * 4;
      data[j]     = v;
      data[j + 1] = v;
      data[j + 2] = v;
      data[j + 3] = 255;
    }
    offCtx.putImageData(offImg, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function renderCounter() {
    const tNow = counter.querySelector('[data-role="t-now"]');
    if (tNow) tNow.textContent = String(state.t);
  }

  /* ----- chart (β_t and ᾱ_t curves + cursor line) -------------------------- */

  // Layout in viewBox (320 × 120):
  //   left margin 36, right margin 14, top 8, bottom 22.
  //   x ∈ [0, T-1]   →   [36, 306]
  //   y ∈ [0, 1]     →   [8, 98]   (β shown on its own scale; ᾱ on [0,1])
  const CH = { L: 36, R: 14, T: 8, B: 22, W: 320, H: 120 };
  function chx(t) { return CH.L + (t / (T - 1)) * (CH.W - CH.L - CH.R); }

  // β has range [1e-4, 0.02] in this schedule; we plot it on its own scale
  // for legibility (else it is an indistinguishable horizontal line at the
  // foot of the chart). ᾱ uses the natural [0,1] axis.
  const BETA_MAX = Math.max.apply(null, DATA.betas);
  function chy_beta(b)   { return CH.T + (1 - b / BETA_MAX) * (CH.H - CH.T - CH.B); }
  function chy_alpha(a)  { return CH.T + (1 - a)            * (CH.H - CH.T - CH.B); }

  function renderChartStatic() {
    chartSvg.selectAll('*').remove();

    // Plot bg
    chartSvg.append('rect')
      .attr('class', 's2-chart-bg')
      .attr('x', CH.L).attr('y', CH.T)
      .attr('width',  CH.W - CH.L - CH.R)
      .attr('height', CH.H - CH.T - CH.B);

    // y-axis labels (left = β scale, right = ᾱ scale)
    chartSvg.append('text').attr('class', 's2-chart-y-l')
      .attr('x', CH.L - 4).attr('y', CH.T + 9).attr('text-anchor', 'end')
      .text(BETA_MAX.toFixed(2));
    chartSvg.append('text').attr('class', 's2-chart-y-l')
      .attr('x', CH.L - 4).attr('y', CH.H - CH.B).attr('text-anchor', 'end')
      .text('0');

    chartSvg.append('text').attr('class', 's2-chart-y-r')
      .attr('x', CH.W - CH.R + 4).attr('y', CH.T + 9).attr('text-anchor', 'start')
      .text('1');
    chartSvg.append('text').attr('class', 's2-chart-y-r')
      .attr('x', CH.W - CH.R + 4).attr('y', CH.H - CH.B).attr('text-anchor', 'start')
      .text('0');

    // x-axis labels
    chartSvg.append('text').attr('class', 's2-chart-x-l')
      .attr('x', CH.L).attr('y', CH.H - 6).attr('text-anchor', 'start')
      .text('t = 0');
    chartSvg.append('text').attr('class', 's2-chart-x-l')
      .attr('x', CH.W - CH.R).attr('y', CH.H - 6).attr('text-anchor', 'end')
      .text(`t = ${T - 1}`);

    // β line (red, cluster-2)
    const betaLine = d3.line()
      .x((_, i) => chx(i))
      .y(d => chy_beta(d));
    chartSvg.append('path')
      .attr('class', 's2-chart-beta stroke-cluster-2')
      .attr('d', betaLine(DATA.betas));

    // ᾱ line (blue, cluster-1)
    const alphaLine = d3.line()
      .x((_, i) => chx(i))
      .y(d => chy_alpha(d));
    chartSvg.append('path')
      .attr('class', 's2-chart-alpha stroke-cluster-1')
      .attr('d', alphaLine(DATA.alphaBars));

    // legend (text labels at the right end of each curve)
    chartSvg.append('text').attr('class', 's2-chart-legend cluster-2')
      .attr('x', CH.W - CH.R - 2).attr('y', chy_beta(DATA.betas[T - 1]) - 4)
      .attr('text-anchor', 'end')
      .text('βₜ');
    chartSvg.append('text').attr('class', 's2-chart-legend cluster-1')
      .attr('x', CH.W - CH.R - 2).attr('y', chy_alpha(DATA.alphaBars[T - 1]) + 12)
      .attr('text-anchor', 'end')
      .text('ᾱₜ');

    // cursor group (re-rendered on t change)
    chartSvg.append('g').attr('class', 's2-chart-cursor');
  }

  function renderChartCursor() {
    const g = chartSvg.select('g.s2-chart-cursor');
    g.selectAll('*').remove();
    const cx = chx(state.t);
    g.append('line')
      .attr('class', 's2-chart-cursor-line stroke-cluster-3')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', CH.T).attr('y2', CH.H - CH.B);
    // dots on each curve
    g.append('circle')
      .attr('class', 's2-chart-dot cluster-2')
      .attr('cx', cx).attr('cy', chy_beta(DATA.betas[state.t]))
      .attr('r', 2.2);
    g.append('circle')
      .attr('class', 's2-chart-dot cluster-1')
      .attr('cx', cx).attr('cy', chy_alpha(DATA.alphaBars[state.t]))
      .attr('r', 2.2);
  }

  /* ----- histogram --------------------------------------------------------- */

  function renderHist() {
    const w = histCanvas.width, h = histCanvas.height;
    histCtx.clearRect(0, 0, w, h);
    if (!state.showHist) return;

    const NB = 32;
    const bins = new Uint32Array(NB);
    // intensities expected in roughly [-3, 3] at high t; clip to that range.
    const lo = -3, hi = 3;
    const y = state.trajPx[state.t];
    let total = 0;
    for (let i = 0; i < y.length; i++) {
      let v = y[i];
      if (v < lo) v = lo;
      else if (v > hi) v = hi;
      const b = Math.min(NB - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * NB)));
      bins[b]++;
      total++;
    }
    let maxBin = 1;
    for (let i = 0; i < NB; i++) if (bins[i] > maxBin) maxBin = bins[i];

    // Read theme tokens for the bar fill — using c1 (data hue).
    const cs = getComputedStyle(document.documentElement);
    const c1 = cs.getPropertyValue('--c1').trim() || '#2f6cb1';
    const ruleColor = cs.getPropertyValue('--rule').trim() || '#d8d4ca';

    // baseline
    histCtx.strokeStyle = ruleColor;
    histCtx.lineWidth = 1;
    histCtx.beginPath();
    histCtx.moveTo(0, h - 0.5);
    histCtx.lineTo(w, h - 0.5);
    histCtx.stroke();

    // bars
    const bw = w / NB;
    histCtx.fillStyle = c1;
    for (let i = 0; i < NB; i++) {
      const bh = (bins[i] / maxBin) * (h - 4);
      histCtx.fillRect(i * bw, h - bh, bw - 0.5, bh);
    }

    // zero line marker
    const zeroX = ((0 - lo) / (hi - lo)) * w;
    histCtx.strokeStyle = ruleColor;
    histCtx.beginPath();
    histCtx.moveTo(zeroX, 0);
    histCtx.lineTo(zeroX, h);
    histCtx.stroke();
  }

  /* ----- text chrome ------------------------------------------------------- */

  function renderTextChrome() {
    stepPill.textContent = `Step ${state.cursor} of ${STEPS}`;

    if (state.cursor === 0) {
      tail.innerHTML = 'Press <strong>Play</strong> to run all 200 forward steps.';
    } else if (state.cursor === 1) {
      if (state.playing) {
        tail.innerHTML = '<em>Sweeping…</em> one forward step at a time.';
      } else if (state.t === T - 1) {
        tail.innerHTML = `At <span class="mono">t = ${T - 1}</span>, the cloud is a Gaussian blob and the digit is gone. The schedule is what got us here.`;
      } else {
        tail.innerHTML = 'Press <strong>Play</strong> to sweep again, or scrub.';
      }
    } else if (state.cursor === 2) {
      tail.innerHTML = 'Drag the slider. Watch ᾱ<sub>t</sub> fall and β<sub>t</sub> grow.';
    } else if (state.cursor === 3) {
      tail.innerHTML = 'Histogram visible. At t = 0, the digit is bimodal (black + white). At t = T-1, it is Gaussian.';
    }
  }

  /* ----- master render ----------------------------------------------------- */

  function render() {
    render2DAtT(state.t);
    renderMnistAtT(state.t);
    renderCounter();
    renderChartCursor();
    renderHist();
    renderTextChrome();

    // Control state
    scrubber.value = String(state.t);
    scrubber.disabled = state.cursor < 1;
    playBtn.disabled = state.playing;
    playBtn.textContent = state.playing ? 'Playing…' : (state.t === T - 1 ? 'Replay' : 'Play');

    histToggleWrap.classList.toggle('disabled', state.cursor < 3);
    histCheckbox.disabled = state.cursor < 3;
    histWrap.classList.toggle('visible', state.showHist);
  }

  /* ----- autoplay (RAF) ---------------------------------------------------- */

  function startPlay() {
    if (state.playing) return;
    state.playing = true;
    state.t = 0;
    state.playStartTs = null;
    render();
    state.raf = requestAnimationFrame(playTick);
  }

  function playTick(ts) {
    if (!state.playing) return;
    if (state.playStartTs == null) state.playStartTs = ts;
    const elapsed = ts - state.playStartTs;
    const frac = Math.min(1, elapsed / SWEEP_MS);
    state.t = Math.min(T - 1, Math.floor(frac * (T - 1)));
    render();
    if (frac >= 1) {
      state.t = T - 1;
      state.playing = false;
      state.raf = null;
      render();
      return;
    }
    state.raf = requestAnimationFrame(playTick);
  }

  function stopPlay() {
    if (state.raf != null) cancelAnimationFrame(state.raf);
    state.raf = null;
    state.playing = false;
  }

  /* ----- step engine ------------------------------------------------------- */

  // setCursor moves between abstract phases. Cursor 1 triggers autoplay,
  // cursor 2 unlocks the scrubber, cursor 3 unlocks the histogram.
  function setCursor(c) {
    if (c < 0 || c > STEPS) return false;
    if (c === state.cursor && c !== 1) return false;
    if (c === 1) {
      // Always (re)start the sweep when entering cursor 1.
      state.cursor = 1;
      stopPlay();
      startPlay();
      return true;
    }
    state.cursor = c;
    if (c === 0) {
      stopPlay();
      state.t = 0;
    } else if (c === 2) {
      // Pin at end-of-sweep so scrubbing starts from a meaningful point.
      stopPlay();
      if (state.t < T - 1) state.t = T - 1;
    } else if (c === 3) {
      stopPlay();
      state.showHist = true;
      histCheckbox.checked = true;
    }
    render();
    return true;
  }

  /* ----- input handlers ---------------------------------------------------- */

  playBtn.addEventListener('click', () => {
    if (state.cursor === 0) {
      setCursor(1);
    } else {
      // From cursor ≥ 1: replay
      stopPlay();
      state.t = 0;
      state.cursor = 1;
      startPlay();
    }
  });

  scrubber.addEventListener('input', () => {
    if (state.cursor < 1) return;
    stopPlay();
    if (state.cursor < 2) state.cursor = 2;
    state.t = parseInt(scrubber.value, 10) | 0;
    render();
  });

  digitSelect.addEventListener('change', () => {
    state.digit = parseInt(digitSelect.value, 10);
    rebuildTrajectories();
    render();
  });

  histCheckbox.addEventListener('change', () => {
    state.showHist = histCheckbox.checked;
    if (state.showHist && state.cursor < 3) state.cursor = 3;
    render();
  });

  function onThemeChange() {
    renderChartStatic();
    renderChartCursor();
    renderHist();
  }
  window.addEventListener('theme-change', onThemeChange);

  /* ----- test hook (headless verification) -------------------------------- */
  // ?test=cursor=N (with optional &t=K) jumps straight to a state.
  // For animation-blocked states (cursor 1), we emulate "post-sweep": we set
  // t to T-1 and freeze playing=false.
  function readTestParams() {
    const src = (window.location.search || '') + (window.location.hash || '');
    const cm = src.match(/test=cursor=(\d+)/);
    const tm = src.match(/[?&#]t=(\d+)/);
    const out = {};
    if (cm) {
      const c = parseInt(cm[1], 10);
      if (Number.isFinite(c) && c >= 0 && c <= STEPS) out.cursor = c;
    }
    if (tm) {
      const tt = parseInt(tm[1], 10);
      if (Number.isFinite(tt) && tt >= 0 && tt < T) out.t = tt;
    }
    return out;
  }

  function applyTestParams() {
    const p = readTestParams();
    if (p.cursor == null) return;
    stopPlay();
    if (p.cursor === 1) {
      // "post-sweep" state — show t = T-1 frozen, no autoplay.
      state.cursor = 1;
      state.t = (p.t != null) ? p.t : (T - 1);
      state.playing = false;
    } else if (p.cursor === 2) {
      state.cursor = 2;
      state.t = (p.t != null) ? p.t : (T - 1);
    } else if (p.cursor === 3) {
      state.cursor = 3;
      state.t = (p.t != null) ? p.t : (T - 1);
      state.showHist = true;
      histCheckbox.checked = true;
    } else if (p.cursor === 0) {
      state.cursor = 0;
      state.t = 0;
    }
    render();
  }

  /* ----- initial paint ---------------------------------------------------- */

  rebuildTrajectories();
  renderChartStatic();
  render();
  applyTestParams();

  return {
    onEnter() {
      stopPlay();
      state.cursor = 0;
      state.t = 0;
      state.showHist = false;
      histCheckbox.checked = false;
      rebuildTrajectories();
      renderChartStatic();
      render();
      applyTestParams();
    },
    onLeave() {
      stopPlay();
    },
    onNextKey() {
      // ArrowRight: advance the abstract step engine.
      if (state.cursor < STEPS) {
        return setCursor(state.cursor + 1);
      }
      return false;
    },
    onPrevKey() {
      if (state.cursor === 0) return false;
      stopPlay();
      return setCursor(state.cursor - 1);
    },
  };
};
