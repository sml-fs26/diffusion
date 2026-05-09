/* Scene 1 — One forward step.
 *
 * Pedagogical goal: students see one forward DDPM step as the sum of
 *   (a) a deterministic fade  √(1-β_t) · x_t   and
 *   (b) a Gaussian kick        √β_t · ε_t .
 *
 * Two panes (2D letter-M cloud + 28×28 MNIST sample) are driven from a single
 * source-of-truth state x_t (2D, 300 floats) and y_t (784 floats). Every step
 * recomputes the visible state from x_0 / y_0 — never depends on prior step.
 *
 * Step engine (cursor 0..5):
 *   0  clean state. β_t = 0.05 (chosen so the step is visibly substantial).
 *   1  fade only:    x = √(1-β) · x_0           (highlight √(1-β_t)·x_t term)
 *   2  noise only:   x = x_0 + √β · ε           (highlight √β_t·ε_t term)
 *   3  combined:     x = √(1-β)·x_0 + √β·ε      (no formula highlight)
 *   4  β-slider becomes interactive
 *   5  dock to β = β_0 = DATA.betas[0] — the lesson lands
 *
 * Cold entry: rebuilds from DATA on every onEnter; deterministic seed.
 */

window.scenes.scene1 = function (root) {
  const DATA = window.DATA;
  const M = window.DiffusionMath;

  /* ----- constants & data --------------------------------------------------- */

  const SEED = 101;
  const T_2D = DATA.letterM.points.length;       // 300
  const T_PX = 28 * 28;                          // 784
  const STEPS = 5;                               // cursor max
  const BETA_DEFAULT = 0.05;
  const BETA_STOPS = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5];

  // x_0 (2D): flatten letterM points into a Float32Array length 600.
  function build_x0_2D() {
    const x0 = new Float32Array(T_2D * 2);
    for (let i = 0; i < T_2D; i++) {
      x0[2 * i]     = DATA.letterM.points[i][0];
      x0[2 * i + 1] = DATA.letterM.points[i][1];
    }
    return x0;
  }

  // y_0 (28×28): MNIST sample for the digit 3. The lecturer can switch via the
  // selector. mnistSamples is indexed in label order so [3] has label 3.
  function build_y0(labelIdx) {
    const src = DATA.mnistSamples[labelIdx].pixels;
    const y0 = new Float32Array(T_PX);
    for (let i = 0; i < T_PX; i++) y0[i] = src[i];
    return y0;
  }

  /* ----- DOM scaffolding ---------------------------------------------------- */

  root.innerHTML = '';
  root.classList.add('scene-s1');

  const layout = document.createElement('div');
  layout.className = 'scene-layout';
  root.appendChild(layout);

  /* viz column ------------------------------------------------------------- */
  const vizCol = document.createElement('div');
  vizCol.className = 's1-viz-col';
  layout.appendChild(vizCol);

  // Top pane — 2D
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
  const gGrid    = svg2D.append('g').attr('class', 's1-grid');
  const gPoints  = svg2D.append('g').attr('class', 's1-points');

  // Faint grid (axes through origin and ±1 frame)
  const gridLines = [
    { x1: 0,  y1: 50, x2: 100, y2: 50 },
    { x1: 50, y1: 0,  x2: 50,  y2: 100 },
  ];
  gGrid.selectAll('line').data(gridLines).enter()
    .append('line')
    .attr('class', 's1-grid-line')
    .attr('x1', d => d.x1).attr('y1', d => d.y1)
    .attr('x2', d => d.x2).attr('y2', d => d.y2);

  // Bottom pane — MNIST canvas
  const botPane = document.createElement('div');
  botPane.className = 's1-bot-pane';
  vizCol.appendChild(botPane);

  const botLabel = document.createElement('div');
  botLabel.className = 's1-pane-label';
  botLabel.textContent = 'MNIST — 28×28 pixels';
  botPane.appendChild(botLabel);

  // Digit selector + canvas
  const botRow = document.createElement('div');
  botRow.className = 's1-bot-row';
  botPane.appendChild(botRow);

  const canvas = document.createElement('canvas');
  canvas.className = 'mnist-pane s1-mnist-canvas';
  canvas.width  = 112;
  canvas.height = 112;
  botRow.appendChild(canvas);

  const ctx = canvas.getContext('2d', { alpha: false });
  ctx.imageSmoothingEnabled = false;

  // Off-screen 28×28 image we sample to upscale.
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
    opt.value = String(d);
    opt.textContent = String(d);
    if (d === 3) opt.selected = true;
    digitSelect.appendChild(opt);
  }

  /* text column ----------------------------------------------------------- */
  const textCol = document.createElement('div');
  textCol.className = 'text-col s1-text';
  layout.appendChild(textCol);

  const stepPill = document.createElement('div');
  stepPill.className = 'step-pill s1-step-pill';
  textCol.appendChild(stepPill);

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

  // β slider widget
  const sliderWrap = document.createElement('div');
  sliderWrap.className = 's1-slider-wrap';
  sliderWrap.innerHTML = `
    <label class="s1-slider-label">β<sub>t</sub></label>
    <input type="range" class="s1-slider" min="0" max="${BETA_STOPS.length - 1}" step="1" value="3" disabled>
    <div class="s1-slider-stops" aria-hidden="true">
      ${BETA_STOPS.map(b => `<span class="s1-slider-stop">${formatBetaStop(b)}</span>`).join('')}
    </div>
    <p class="muted s1-slider-note">
      The schedule starts near 1e-4. Large β makes one step much more visible.
    </p>
  `;
  textCol.appendChild(sliderWrap);

  const sliderEl = sliderWrap.querySelector('.s1-slider');

  const tail = document.createElement('p');
  tail.className = 'muted s1-tail';
  textCol.appendChild(tail);

  /* ----- helpers ---------------------------------------------------------- */

  function formatBetaStop(b) {
    if (b < 0.01) return b.toExponential(0).replace('e+0', 'e').replace('e-0', 'e-');
    return b.toFixed(2);
  }

  function fmtBeta(b) {
    if (b < 0.001) return b.toExponential(1);
    return b.toFixed(4);
  }

  function renderKatex(host, src, displayMode) {
    if (!host) return;
    host.textContent = '';
    if (window.katex) {
      try {
        window.katex.render(src, host, { displayMode: !!displayMode, throwOnError: false });
      } catch (e) {
        host.textContent = src;
      }
    } else {
      host.textContent = src;
    }
  }

  // The formula. We re-render it with optional \htmlClass markers around the
  // fade or noise term to highlight whichever step is active. KaTeX's trust
  // option lets \htmlClass through; cluster-1/cluster-2 inherit color via CSS.
  function renderFormula(highlight /* 'fade' | 'noise' | null */) {
    const fadeTerm  = '\\sqrt{1-\\beta_t}\\,x_t';
    const noiseTerm = '\\sqrt{\\beta_t}\\,\\varepsilon_t';
    let lhs = 'x_{t+1} \\;=\\; ';
    let s;
    if (highlight === 'fade') {
      s = `${lhs}\\htmlClass{s1-hl-fade}{${fadeTerm}} \\;+\\; ${noiseTerm}`;
    } else if (highlight === 'noise') {
      s = `${lhs}${fadeTerm} \\;+\\; \\htmlClass{s1-hl-noise}{${noiseTerm}}`;
    } else {
      s = `${lhs}${fadeTerm} \\;+\\; ${noiseTerm}`;
    }
    formulaBlock.textContent = '';
    if (window.katex) {
      try {
        window.katex.render(s, formulaBlock, {
          displayMode: true,
          throwOnError: false,
          trust: ctx => ctx.command === '\\htmlClass',
          strict: 'ignore'
        });
      } catch (e) {
        formulaBlock.textContent = s;
      }
    } else {
      formulaBlock.textContent = s;
    }
  }

  /* ----- state + step engine --------------------------------------------- */

  const state = {
    cursor: 0,
    digit: 3,
    beta: BETA_DEFAULT,
    x_t: null,        // Float32Array length 600
    y_t: null,        // Float32Array length 784
    eps2D: null,      // cached ε for 2D (Float32Array length 600)
    epsPx: null,      // cached ε for pixels (Float32Array length 784)
  };

  function rebuildEps() {
    const rng = M.mulberry32(SEED);
    state.eps2D = M.randnVector(rng, T_2D * 2);
    state.epsPx = M.randnVector(rng, T_PX);
  }

  function resetState() {
    state.cursor = 0;
    state.beta = BETA_DEFAULT;
    state.x_t = build_x0_2D();
    state.y_t = build_y0(state.digit);
    rebuildEps();
  }

  // Recomputes x_t / y_t from x_0 / y_0 according to (cursor, beta).
  // Step semantics:
  //   cursor 0           → identity (x_0)
  //   cursor 1           → fade only,  scale by √(1-β)
  //   cursor 2           → noise only, x_0 + √β · ε
  //   cursor ≥ 3         → both,       √(1-β)·x_0 + √β · ε
  function recomputeFromState() {
    const x0 = build_x0_2D();
    const y0 = build_y0(state.digit);
    const a = Math.sqrt(1 - state.beta);
    const b = Math.sqrt(state.beta);

    if (state.cursor === 0) {
      state.x_t = x0;
      state.y_t = y0;
      return;
    }
    if (state.cursor === 1) {
      const x = new Float32Array(x0.length);
      for (let i = 0; i < x0.length; i++) x[i] = a * x0[i];
      const y = new Float32Array(y0.length);
      for (let i = 0; i < y0.length; i++) y[i] = a * y0[i];
      state.x_t = x;
      state.y_t = y;
      return;
    }
    if (state.cursor === 2) {
      const x = new Float32Array(x0.length);
      for (let i = 0; i < x0.length; i++) x[i] = x0[i] + b * state.eps2D[i];
      const y = new Float32Array(y0.length);
      for (let i = 0; i < y0.length; i++) y[i] = y0[i] + b * state.epsPx[i];
      state.x_t = x;
      state.y_t = y;
      return;
    }
    // cursor 3, 4, 5 — combined
    const x = new Float32Array(x0.length);
    for (let i = 0; i < x0.length; i++) x[i] = a * x0[i] + b * state.eps2D[i];
    const y = new Float32Array(y0.length);
    for (let i = 0; i < y0.length; i++) y[i] = a * y0[i] + b * state.epsPx[i];
    state.x_t = x;
    state.y_t = y;
  }

  /* ----- renderers -------------------------------------------------------- */

  // 2D: x ∈ [-1.7, 1.7] → svg [0, 100] (y flipped)
  function xToSvg(x) { return ((x + 1.7) / 3.4) * 100; }
  function yToSvg(y) { return ((1.7 - y) / 3.4) * 100; }

  function render2D() {
    const data = [];
    for (let i = 0; i < T_2D; i++) {
      data.push({ i, x: state.x_t[2 * i], y: state.x_t[2 * i + 1] });
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

  // MNIST: y_t ∈ [-1, 1] (allow exceeding) → grayscale 0..255.
  function renderMnist() {
    const data = offImg.data;
    for (let i = 0; i < T_PX; i++) {
      let v = (state.y_t[i] + 1) * 127.5;
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

  function renderCoefPanel() {
    const a = Math.sqrt(1 - state.beta);
    const b = Math.sqrt(state.beta);
    coefPanel.querySelector('[data-role="beta"]').textContent = fmtBeta(state.beta);
    coefPanel.querySelector('[data-role="fade-coef"]').textContent = a.toFixed(5);
    coefPanel.querySelector('[data-role="noise-coef"]').textContent = b.toFixed(5);
  }

  function renderTextChrome() {
    stepPill.textContent = `Step ${state.cursor} of ${STEPS}`;

    const c = state.cursor;
    if (c === 0) {
      tail.textContent = 'A clean letter M, a clean digit. β is set so one step is visible.';
    } else if (c === 1) {
      tail.innerHTML = '<em>Fade only.</em> Multiply by √(1−β). The cloud shrinks toward the origin; the digit dims toward black.';
    } else if (c === 2) {
      tail.innerHTML = '<em>Noise only.</em> Add √β · ε. The cloud keeps its shape but jitters; the digit is speckled.';
    } else if (c === 3) {
      tail.innerHTML = '<em>Both at once.</em> The result is the pointwise sum of fade and noise.';
    } else if (c === 4) {
      tail.innerHTML = 'Drag the slider. Watch the panes change as β grows or shrinks.';
    } else {
      tail.innerHTML = `Docked at β<sub>0</sub> = ${DATA.betas[0].toExponential(0)}. In practice the schedule uses tiny steps. We will see why next.`;
    }
  }

  function renderFormulaForCursor() {
    if (state.cursor === 1)      renderFormula('fade');
    else if (state.cursor === 2) renderFormula('noise');
    else                         renderFormula(null);
  }

  function syncSlider() {
    const stopIdx = nearestStopIndex(state.beta);
    sliderEl.value = String(stopIdx);
    sliderEl.disabled = state.cursor < 4;
  }

  function nearestStopIndex(b) {
    let best = 0, bestDist = Infinity;
    for (let i = 0; i < BETA_STOPS.length; i++) {
      const d = Math.abs(Math.log(BETA_STOPS[i]) - Math.log(b));
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }

  function render() {
    recomputeFromState();
    render2D();
    renderMnist();
    renderCoefPanel();
    renderTextChrome();
    renderFormulaForCursor();
    syncSlider();
  }

  /* ----- step engine glue ------------------------------------------------- */

  function setCursor(c) {
    if (c < 0 || c > STEPS) return false;
    if (c === state.cursor) return false;
    state.cursor = c;
    if (c === 0) {
      state.beta = BETA_DEFAULT;
    } else if (c === 5) {
      state.beta = DATA.betas[0];
    } else if (c >= 1 && c <= 3) {
      state.beta = BETA_DEFAULT;
    }
    // cursor 4 — leave β as-is (whatever the user chose) so a sweep persists.
    render();
    return true;
  }

  /* ----- input handlers --------------------------------------------------- */

  sliderEl.addEventListener('input', () => {
    if (state.cursor < 4) return;
    const idx = parseInt(sliderEl.value, 10);
    state.beta = BETA_STOPS[Math.max(0, Math.min(BETA_STOPS.length - 1, idx))];
    render();
  });

  digitSelect.addEventListener('change', () => {
    state.digit = parseInt(digitSelect.value, 10);
    render();
  });

  function onThemeChange() { render(); }
  window.addEventListener('theme-change', onThemeChange);

  /* ----- test hook (headless verification) -------------------------------- */
  // ?test=cursor=N or #scene=1&test=cursor=N jumps directly to a step.
  function readTestCursor() {
    const src = (window.location.search || '') + (window.location.hash || '');
    const m = src.match(/test=cursor=(\d+)/);
    if (!m) return null;
    const c = parseInt(m[1], 10);
    return (Number.isFinite(c) && c >= 0 && c <= STEPS) ? c : null;
  }

  /* ----- initial paint ---------------------------------------------------- */
  resetState();
  render();
  const _initCursor = readTestCursor();
  if (_initCursor != null) setCursor(_initCursor);

  return {
    onEnter() {
      resetState();
      render();
      const c = readTestCursor();
      if (c != null) setCursor(c);
    },
    onLeave() {},
    onNextKey() {
      return setCursor(state.cursor + 1);
    },
    onPrevKey() {
      if (state.cursor === 0) return false;
      return setCursor(state.cursor - 1);
    },
  };
};
