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

  // Two stacked charts (β_t on top, ᾱ_t below) — each with its own y-axis,
  // each fills the panel width. Far easier to read than the dual-axis variant.
  const chartWrap = document.createElement('div');
  chartWrap.className = 's2-chart-wrap';
  timePanel.appendChild(chartWrap);

  const chartBetaSvg = d3.select(chartWrap).append('svg')
    .attr('class', 's2-chart-svg s2-chart-beta-svg')
    .attr('viewBox', '0 0 320 110')
    .attr('preserveAspectRatio', 'none');

  const chartAlphaSvg = d3.select(chartWrap).append('svg')
    .attr('class', 's2-chart-svg s2-chart-alpha-svg')
    .attr('viewBox', '0 0 320 110')
    .attr('preserveAspectRatio', 'none');

  // Controls: scrubber only (play removed — direct scrubbing is the lesson)
  const controls = document.createElement('div');
  controls.className = 's2-controls';
  controls.innerHTML = `
    <input type="range" class="s2-scrubber" min="0" max="${T - 1}" step="1" value="0">
  `;
  textCol.appendChild(controls);
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

  /* ----- chart (two stacked panels: β_t on top, ᾱ_t on bottom) ----------- */

  // Each chart viewBox is 320 × 110. Each has L/R margins for axis labels.
  const CH = { L: 44, R: 22, T: 14, B: 26, W: 320, H: 110 };
  const PLOT_W = CH.W - CH.L - CH.R;
  const PLOT_H = CH.H - CH.T - CH.B;
  function chx(t) { return CH.L + (t / (T - 1)) * PLOT_W; }

  const BETA_MAX = Math.max.apply(null, DATA.betas);
  const ALPHA_MIN = Math.min.apply(null, DATA.alphaBars);  // ≈ 0.13
  function chy_beta(b)  { return CH.T + (1 - b / BETA_MAX) * PLOT_H; }
  // ᾱ goes from 1 down to ALPHA_MIN; map to full plot height for legibility.
  function chy_alpha(a) { return CH.T + ((1 - a) / (1 - ALPHA_MIN)) * PLOT_H; }

  function _renderOneChart(svg, opts) {
    // opts: { title, color, yLabels: [topVal, botVal], series, yMap, cursorYAt }
    svg.selectAll('*').remove();

    // Title (top-left, italic muted)
    svg.append('text').attr('class', 's2-chart-title')
      .attr('x', CH.L).attr('y', CH.T - 4)
      .text(opts.title);

    // Plot area background
    svg.append('rect').attr('class', 's2-chart-bg')
      .attr('x', CH.L).attr('y', CH.T)
      .attr('width', PLOT_W).attr('height', PLOT_H);

    // y-axis labels (top + bottom)
    svg.append('text').attr('class', 's2-chart-y-lbl')
      .attr('x', CH.L - 6).attr('y', CH.T + 5).attr('text-anchor', 'end')
      .text(opts.yLabels[0]);
    svg.append('text').attr('class', 's2-chart-y-lbl')
      .attr('x', CH.L - 6).attr('y', CH.T + PLOT_H).attr('text-anchor', 'end')
      .text(opts.yLabels[1]);

    // x-axis labels
    svg.append('text').attr('class', 's2-chart-x-lbl')
      .attr('x', CH.L).attr('y', CH.H - 6).attr('text-anchor', 'start')
      .text('t = 0');
    svg.append('text').attr('class', 's2-chart-x-lbl')
      .attr('x', CH.L + PLOT_W).attr('y', CH.H - 6).attr('text-anchor', 'end')
      .text(`t = ${T - 1}`);

    // The curve
    const line = d3.line()
      .x((_, i) => chx(i))
      .y(d => opts.yMap(d));
    svg.append('path')
      .attr('class', `s2-chart-line ${opts.colorClass}`)
      .attr('d', line(opts.series));

    // Cursor group
    svg.append('g').attr('class', 's2-chart-cursor');
  }

  function _renderOneCursor(svg, opts) {
    const g = svg.select('g.s2-chart-cursor');
    g.selectAll('*').remove();
    const cx = chx(state.t);
    g.append('line')
      .attr('class', 's2-chart-cursor-line stroke-cluster-3')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', CH.T).attr('y2', CH.T + PLOT_H);
    g.append('circle')
      .attr('class', `s2-chart-dot ${opts.colorClass}`)
      .attr('cx', cx).attr('cy', opts.yMap(opts.series[state.t]))
      .attr('r', 3.2);
    // Value readout, anchored to the right of the cursor when not too close
    // to the right edge.
    const txtX = (cx > CH.L + PLOT_W * 0.85) ? cx - 6 : cx + 6;
    const anchor = (cx > CH.L + PLOT_W * 0.85) ? 'end' : 'start';
    g.append('text')
      .attr('class', `s2-chart-readout ${opts.colorClass}`)
      .attr('x', txtX).attr('y', opts.yMap(opts.series[state.t]) - 6)
      .attr('text-anchor', anchor)
      .text(opts.format(opts.series[state.t]));
  }

  function renderChartStatic() {
    _renderOneChart(chartBetaSvg, {
      title: 'β_t — fade weight per step',
      colorClass: 'stroke-cluster-2',
      yLabels: [BETA_MAX.toFixed(3), '0'],
      series: DATA.betas,
      yMap: chy_beta,
    });
    _renderOneChart(chartAlphaSvg, {
      title: 'ᾱ_t — cumulative signal retained',
      colorClass: 'stroke-cluster-1',
      yLabels: ['1', ALPHA_MIN.toFixed(2)],
      series: DATA.alphaBars,
      yMap: chy_alpha,
    });
  }

  function renderChartCursor() {
    _renderOneCursor(chartBetaSvg, {
      colorClass: 'cluster-2',
      series: DATA.betas,
      yMap: chy_beta,
      format: (b) => b.toExponential(1).replace('e+0', 'e').replace('e-0', 'e-'),
    });
    _renderOneCursor(chartAlphaSvg, {
      colorClass: 'cluster-1',
      series: DATA.alphaBars,
      yMap: chy_alpha,
      format: (a) => a.toFixed(3),
    });
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
    if (state.t === 0) {
      tail.innerHTML = 'Drag the slider. Watch ᾱ<sub>t</sub> fall and β<sub>t</sub> grow.';
    } else if (state.t === T - 1) {
      tail.innerHTML = `At <span class="mono">t = ${T - 1}</span>, the cloud is a Gaussian blob and the digit is gone. The schedule is what got us here.`;
    } else {
      tail.innerHTML = `<span class="mono">t = ${state.t}</span>. Both panes update with each tick.`;
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

    // Control state — scrubber always enabled; histogram toggle always available
    scrubber.value = String(state.t);
    scrubber.disabled = false;

    histToggleWrap.classList.remove('disabled');
    histCheckbox.disabled = false;
    histWrap.classList.toggle('visible', state.showHist);
  }

  /* ----- input handlers ---------------------------------------------------- */

  scrubber.addEventListener('input', () => {
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
    render();
  });

  function onThemeChange() {
    renderChartStatic();
    renderChartCursor();
    renderHist();
  }
  window.addEventListener('theme-change', onThemeChange);

  /* ----- test hook (headless verification) -------------------------------- */
  // ?test=t=K (or &t=K) jumps the slider to t=K so a screenshot can land on
  // a meaningful frame (e.g. t = T-1 for the "fully noisy" pane). With
  // ?test=hist also pop the histogram on.
  function readTestParams() {
    const src = (window.location.search || '') + (window.location.hash || '');
    const tm = src.match(/[?&#]t=(\d+)/);
    const out = {};
    if (tm) {
      const tt = parseInt(tm[1], 10);
      if (Number.isFinite(tt) && tt >= 0 && tt < T) out.t = tt;
    }
    out.hist = /[?&#]test=[^&]*hist/.test(src);
    return out;
  }

  function applyTestParams() {
    const p = readTestParams();
    if (p.t != null) state.t = p.t;
    if (p.hist) {
      state.showHist = true;
      histCheckbox.checked = true;
    }
    if (/[#&?]run\b/.test(window.location.hash || '')) {
      // headless &run lands at t = T-1 so the panes show the noise endpoint.
      state.t = T - 1;
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
      state.t = 0;
      state.showHist = false;
      histCheckbox.checked = false;
      rebuildTrajectories();
      renderChartStatic();
      render();
      applyTestParams();
    },
    onLeave() {
      // nothing to clean up
    },
    onNextKey() { return false; },
    onPrevKey() { return false; },
  };
};
