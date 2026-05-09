/* Scene 1 — One forward step.
 *
 * Minimal pedagogical setup:
 *   - The forward-step formula  x_{t+1} = √(1-β_t)·x_t + √β_t·ε_t
 *   - A coefficient panel showing β_t, √(1-β_t), √β_t (live).
 *   - A β slider.
 *   - A "Step" button.  Each press applies ONE forward step to both panes
 *     (letterM cloud + MNIST digit). Pressing again applies another step
 *     starting from the current state. A "Reset" button clears back to x_0.
 *
 * No internal cursor / step-engine. No fade-only or noise-only modes. The
 * focus is on the visceral feel of one step.
 */

window.scenes.scene1 = function (root) {
  const DATA = window.DATA;
  const M    = window.DiffusionMath;

  const N_2D   = DATA.letterM.points.length;     // 300
  const N_PX   = 28 * 28;                        // 784
  const SEED   = 101;
  const BETA_DEFAULT = 0.05;
  const BETA_STOPS   = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5];

  function build_x0_2D() {
    const x0 = new Float32Array(N_2D * 2);
    for (let i = 0; i < N_2D; i++) {
      x0[2 * i]     = DATA.letterM.points[i][0];
      x0[2 * i + 1] = DATA.letterM.points[i][1];
    }
    return x0;
  }
  function build_y0(labelIdx) {
    const src = DATA.mnistSamples[labelIdx].pixels;
    const y0  = new Float32Array(N_PX);
    for (let i = 0; i < N_PX; i++) y0[i] = src[i];
    return y0;
  }

  /* ---- DOM ---------------------------------------------------------------- */
  root.innerHTML = '';
  root.classList.add('scene-s1');

  const layout = document.createElement('div');
  layout.className = 'scene-layout s1-layout';
  root.appendChild(layout);

  // Left: viz column with two stacked panes
  const vizCol = document.createElement('div');
  vizCol.className = 's1-viz-col';
  layout.appendChild(vizCol);

  const topPane = document.createElement('div');
  topPane.className = 's1-top-pane viz-wrap';
  vizCol.appendChild(topPane);
  const topLabel = document.createElement('div');
  topLabel.className = 's1-pane-label';
  topLabel.textContent = 'Letter M — 300 points in 2D';
  topPane.appendChild(topLabel);

  const svg2D = d3.select(topPane).append('svg')
    .attr('class', 's1-svg2d')
    .attr('viewBox', '0 0 100 100')
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const gGrid   = svg2D.append('g').attr('class', 's1-grid');
  const gPoints = svg2D.append('g').attr('class', 's1-points');
  gGrid.selectAll('line').data([
    { x1: 0,  y1: 50, x2: 100, y2: 50 },
    { x1: 50, y1: 0,  x2: 50,  y2: 100 },
  ]).enter().append('line')
    .attr('class', 's1-grid-line')
    .attr('x1', d => d.x1).attr('y1', d => d.y1)
    .attr('x2', d => d.x2).attr('y2', d => d.y2);

  const botPane = document.createElement('div');
  botPane.className = 's1-bot-pane';
  vizCol.appendChild(botPane);
  const botLabel = document.createElement('div');
  botLabel.className = 's1-pane-label';
  botLabel.textContent = 'MNIST — 28×28 pixels';
  botPane.appendChild(botLabel);

  const botRow = document.createElement('div');
  botRow.className = 's1-bot-row';
  botPane.appendChild(botRow);

  const canvas = document.createElement('canvas');
  canvas.className = 'mnist-pane s1-mnist-canvas';
  canvas.width = 112; canvas.height = 112;
  botRow.appendChild(canvas);
  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.imageSmoothingEnabled = false;
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  const offCtx = off.getContext('2d');
  const offImg = offCtx.createImageData(28, 28);

  const selectorWrap = document.createElement('div');
  selectorWrap.className = 's1-selector';
  selectorWrap.innerHTML =
    `<span class="s1-selector-label">digit</span>` +
    `<select class="s1-digit-select" aria-label="MNIST digit"></select>`;
  botRow.appendChild(selectorWrap);
  const digitSelect = selectorWrap.querySelector('.s1-digit-select');
  for (let d = 0; d < 10; d++) {
    const opt = document.createElement('option');
    opt.value = String(d); opt.textContent = String(d);
    if (d === 3) opt.selected = true;
    digitSelect.appendChild(opt);
  }

  // Right: text column
  const textCol = document.createElement('div');
  textCol.className = 'text-col s1-text';
  layout.appendChild(textCol);

  const heading = document.createElement('h2');
  heading.textContent = 'One forward step.';
  textCol.appendChild(heading);

  const formulaBlock = document.createElement('div');
  formulaBlock.className = 'formula-block s1-formula';
  textCol.appendChild(formulaBlock);

  const formulaCaption = document.createElement('p');
  formulaCaption.className = 'muted s1-formula-caption';
  formulaCaption.innerHTML = `<em>fade</em> · <em>noise</em>.`;
  textCol.appendChild(formulaCaption);

  const coefPanel = document.createElement('div');
  coefPanel.className = 's1-coef-panel';
  coefPanel.innerHTML = `
    <div class="s1-coef-row">
      <span class="s1-coef-name">β<sub>t</sub></span>
      <span class="s1-coef-value" data-role="beta">0.0500</span>
    </div>
    <div class="s1-coef-row">
      <span class="s1-coef-name s1-coef-fade">√(1−β<sub>t</sub>)</span>
      <span class="s1-coef-value" data-role="fade-coef">0.97468</span>
    </div>
    <div class="s1-coef-row">
      <span class="s1-coef-name s1-coef-noise">√β<sub>t</sub></span>
      <span class="s1-coef-value" data-role="noise-coef">0.22361</span>
    </div>
  `;
  textCol.appendChild(coefPanel);

  // β slider
  const sliderWrap = document.createElement('div');
  sliderWrap.className = 's1-slider-wrap';
  sliderWrap.innerHTML = `
    <label class="s1-slider-label">β<sub>t</sub></label>
    <input type="range" class="s1-slider" min="0" max="${BETA_STOPS.length - 1}" step="1" value="3">
    <div class="s1-slider-stops" aria-hidden="true">
      ${BETA_STOPS.map(b => `<span class="s1-slider-stop">${formatBetaStop(b)}</span>`).join('')}
    </div>
    <p class="muted s1-slider-note">
      The schedule starts near 1e-4. Large β makes one step much more visible.
    </p>
  `;
  textCol.appendChild(sliderWrap);
  const sliderEl = sliderWrap.querySelector('.s1-slider');

  // Step / Reset buttons
  const ctrlRow = document.createElement('div');
  ctrlRow.className = 's1-ctrl-row';
  ctrlRow.innerHTML = `
    <button class="btn primary s1-step-btn" type="button">Step</button>
    <button class="btn s1-reset-btn" type="button">Reset</button>
    <span class="step-pill mono s1-step-counter" data-role="counter">step 0</span>
  `;
  textCol.appendChild(ctrlRow);
  const stepBtn  = ctrlRow.querySelector('.s1-step-btn');
  const resetBtn = ctrlRow.querySelector('.s1-reset-btn');
  const stepCounterEl = ctrlRow.querySelector('[data-role="counter"]');

  /* ---- KaTeX -------------------------------------------------------------- */
  katex.render(
    'x_{t+1} \\;=\\; \\htmlClass{s1-hl-fade}{\\sqrt{1-\\beta_t}\\,x_t} \\;+\\; \\htmlClass{s1-hl-noise}{\\sqrt{\\beta_t}\\,\\varepsilon_t}',
    formulaBlock,
    { throwOnError: false, displayMode: true, trust: ctx => ctx.command === '\\htmlClass' }
  );

  /* ---- state -------------------------------------------------------------- */
  const state = {
    digit: 3,
    beta: BETA_DEFAULT,
    rng: M.mulberry32(SEED),
    x: build_x0_2D(),         // current 2D state (mutates with each step)
    y: build_y0(3),           // current MNIST state
    nSteps: 0,
  };

  /* ---- render helpers ----------------------------------------------------- */
  function setBeta(b) {
    state.beta = b;
    coefPanel.querySelector('[data-role="beta"]').textContent = b.toFixed(4);
    coefPanel.querySelector('[data-role="fade-coef"]').textContent = Math.sqrt(1 - b).toFixed(5);
    coefPanel.querySelector('[data-role="noise-coef"]').textContent = Math.sqrt(b).toFixed(5);
  }

  function paint2D() {
    const sel = gPoints.selectAll('circle.point').data(state.x.slice ? state.x : Array.from(state.x), () => 0);
    // We use a fixed-set pattern: just upsert N_2D circles.
    let circles = gPoints.selectAll('circle.point').data(d3.range(N_2D));
    circles.exit().remove();
    circles = circles.enter().append('circle')
      .attr('class', 'point cluster-1')
      .attr('r', 0.45)
      .merge(circles);
    circles
      .attr('cx', i => 50 + 25 * state.x[2 * i])
      .attr('cy', i => 50 - 25 * state.x[2 * i + 1]);
  }

  function paintMNIST() {
    const data = offImg.data;
    for (let i = 0; i < N_PX; i++) {
      let v = (state.y[i] + 1) * 127.5;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      const j = i * 4;
      data[j] = v; data[j + 1] = v; data[j + 2] = v; data[j + 3] = 255;
    }
    offCtx.putImageData(offImg, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function repaint() {
    paint2D();
    paintMNIST();
    stepCounterEl.textContent = `step ${state.nSteps}`;
  }

  function resetState() {
    state.rng = M.mulberry32(SEED);
    state.x = build_x0_2D();
    state.y = build_y0(state.digit);
    state.nSteps = 0;
    repaint();
  }

  function applyOneForwardStep() {
    const eps2D = M.randnVector(state.rng, N_2D * 2);
    const epsPx = M.randnVector(state.rng, N_PX);
    state.x = M.forwardStep(state.x, state.beta, eps2D);
    state.y = M.forwardStep(state.y, state.beta, epsPx);
    state.nSteps += 1;
    repaint();
  }

  /* ---- listeners ---------------------------------------------------------- */
  sliderEl.addEventListener('input', () => {
    const idx = parseInt(sliderEl.value, 10);
    setBeta(BETA_STOPS[idx]);
  });

  stepBtn.addEventListener('click', applyOneForwardStep);
  resetBtn.addEventListener('click', resetState);

  digitSelect.addEventListener('change', () => {
    state.digit = parseInt(digitSelect.value, 10);
    resetState();
  });

  /* ---- init --------------------------------------------------------------- */
  setBeta(BETA_DEFAULT);
  resetState();

  /* ---- public ------------------------------------------------------------- */
  return {
    onEnter() { resetState(); },
    onLeave() { /* nothing dynamic to tear down */ },
    onNextKey() { return false; },
    onPrevKey() { return false; },
  };

  /* ---- helpers ------------------------------------------------------------ */
  function formatBetaStop(b) {
    if (b < 0.001) return '1e-4';
    if (b < 0.01)  return '1e-3';
    if (b < 0.05)  return b.toFixed(2);
    return b.toFixed(2);
  }
};
