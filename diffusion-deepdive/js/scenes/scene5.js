/* Scene 5 — Train the oracle.
 *
 * Pedagogical goal: students *watch* a small MLP learn to predict ε from
 * (x_t, t). The lesson lands when both signals move together — the loss
 * curve drops AND a vector field over the (x, y) plane develops structure
 * pointing inward toward the letter-M data manifold.
 *
 * Layout: split-eq.
 *   LEFT  — square SVG showing letter-M data (faint) under a 16×16 grid of
 *           green arrows displaying −ε̂(x, y, t/(T-1)) (the *denoise*
 *           direction). Updated as training proceeds.
 *   RIGHT — status panel · loss chart (log-y) · t-slider · TRAIN/RESET.
 *
 * Step engine cursor 0..3:
 *   0  untrained model. Vector field ≈ random. TRAIN button highlighted.
 *   1  training in progress (auto-advances when loss has dropped ≥60% from
 *      the initial level OR step ≥ 1000).
 *   2  scrub the t-slider — watch the field change with t.
 *   3  closing caption appears.
 *
 * Cold entry: always instantiates a fresh untrained TwoDModel — the user
 * came here to TRAIN, that's the lesson. After a successful run the trained
 * model is saved to window.diffusionShared.twoDModel for scene 6.
 */

window.scenes.scene5 = function (root) {

  const DATA = window.DATA;
  const M = window.DiffusionMath;
  const NN = window.DiffusionNN;

  /* ----- constants -------------------------------------------------------- */

  const T          = DATA.T;
  const SEED       = 7;
  const HIDDEN     = 64;
  const LR         = 2e-3;
  const BATCH_SIZE = 64;
  const GRID_N     = 16;            // 16×16 grid of arrows
  const VIEW_LO    = -1.7;
  const VIEW_HI    =  1.7;
  const VIEW_RANGE = VIEW_HI - VIEW_LO;
  const ARROW_TARGET_LEN = 0.15;    // longest arrow in plot units
  const REFRESH_EVERY    = 5;       // recompute field / loss curve every N steps
  const PLATEAU_DROP     = 0.6;     // 60% loss drop threshold
  const MAX_STEPS_AUTO   = 1000;
  const STEPS_TOTAL      = 4;       // cursor max+1

  /* ----- shared state ----------------------------------------------------- */

  window.diffusionShared = window.diffusionShared || {};

  /* ----- scene state ------------------------------------------------------ */

  const state = {
    cursor: 0,
    model: null,
    rng: null,
    initialLoss: null,        // smoothed initial loss for plateau detection
    sampledLoss: [],          // [{step, loss}] downsampled for chart
    training: false,
    rafId: null,
    tNorm: 1.0,               // slider value, in [0, 1]
    cappedAuto: false,        // cursor auto-advance has fired
  };

  /* ----- DOM scaffolding -------------------------------------------------- */

  root.innerHTML = '';
  root.classList.add('scene-s5');

  const layout = document.createElement('div');
  layout.className = 'scene-layout split-eq s5-layout';
  root.appendChild(layout);

  /* LEFT — vector field ---------------------------------------------------- */

  const leftCol = document.createElement('div');
  leftCol.className = 's5-left';
  layout.appendChild(leftCol);

  const leftHeader = document.createElement('div');
  leftHeader.className = 's5-pane-label';
  leftHeader.textContent = 'Vector field — what the NN thinks denoises.';
  leftCol.appendChild(leftHeader);

  const leftWrap = document.createElement('div');
  leftWrap.className = 's5-vf-wrap viz-wrap';
  leftCol.appendChild(leftWrap);

  const svgVF = d3.select(leftWrap).append('svg')
    .attr('class', 's5-vf-svg')
    .attr('viewBox', `${VIEW_LO} ${VIEW_LO} ${VIEW_RANGE} ${VIEW_RANGE}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Defs — arrowhead marker (scaled in user units; small).
  const defs = svgVF.append('defs');
  defs.append('marker')
    .attr('id', 's5-arrow-head')
    .attr('viewBox', '0 -3 6 6')
    .attr('refX', 5)
    .attr('refY', 0)
    .attr('markerWidth', 4)
    .attr('markerHeight', 4)
    .attr('orient', 'auto')
    .attr('markerUnits', 'strokeWidth')
    .append('path')
    .attr('d', 'M0,-3L6,0L0,3Z')
    .attr('class', 's5-arrow-head');

  // Layers
  const gFrame  = svgVF.append('g').attr('class', 's5-frame');
  const gPoints = svgVF.append('g').attr('class', 's5-data-points');
  const gField  = svgVF.append('g').attr('class', 's5-field');

  // Frame rectangle and origin axes (faint).
  gFrame.append('line')
    .attr('class', 's5-frame-axis')
    .attr('x1', VIEW_LO).attr('y1', 0)
    .attr('x2', VIEW_HI).attr('y2', 0);
  gFrame.append('line')
    .attr('class', 's5-frame-axis')
    .attr('x1', 0).attr('y1', VIEW_LO)
    .attr('x2', 0).attr('y2', VIEW_HI);

  // Faint letter-M data points as background reference (cluster-1, low opacity).
  const dataPoints = DATA.letterM.points;
  gPoints.selectAll('circle.point')
    .data(dataPoints)
    .enter()
    .append('circle')
    .attr('class', 'point cluster-1 s5-data-pt')
    .attr('r', 0.02)
    .attr('cx', d => d[0])
    .attr('cy', d => -d[1]);   // flip y so M is upright

  // Pre-build the grid sample positions (in plot units).
  const gridPoints = [];
  for (let iy = 0; iy < GRID_N; iy++) {
    for (let ix = 0; ix < GRID_N; ix++) {
      const x = VIEW_LO + (ix + 0.5) * (VIEW_RANGE / GRID_N);
      const y = VIEW_LO + (iy + 0.5) * (VIEW_RANGE / GRID_N);
      gridPoints.push({ x, y, idx: iy * GRID_N + ix });
    }
  }

  // Pre-create one <line> per grid point. We update x2/y2 on each refresh.
  const fieldLines = gField.selectAll('line')
    .data(gridPoints, d => d.idx)
    .enter()
    .append('line')
    .attr('class', 's5-field-arrow stroke-cluster-5')
    .attr('marker-end', 'url(#s5-arrow-head)')
    .attr('x1', d => d.x)
    .attr('y1', d => -d.y)
    .attr('x2', d => d.x)
    .attr('y2', d => -d.y);

  /* RIGHT — controls + chart ---------------------------------------------- */

  const rightCol = document.createElement('div');
  rightCol.className = 's5-right text-col';
  layout.appendChild(rightCol);

  const stepPill = document.createElement('div');
  stepPill.className = 'step-pill s5-step-pill';
  rightCol.appendChild(stepPill);

  const heading = document.createElement('h2');
  heading.textContent = 'Train the oracle.';
  rightCol.appendChild(heading);

  const subhead = document.createElement('p');
  subhead.className = 's5-subhead';
  subhead.innerHTML = 'A small MLP learns the mapping <code>(x<sub>t</sub>, t) → &epsilon;&#770;</code>. The recipe is one line.';
  rightCol.appendChild(subhead);

  const formulaBlock = document.createElement('div');
  formulaBlock.className = 'formula-block s5-formula';
  rightCol.appendChild(formulaBlock);

  // Status panel — tabular numerics
  const status = document.createElement('div');
  status.className = 's5-status';
  status.innerHTML = `
    <div class="s5-status-row">
      <span class="s5-status-label">step</span>
      <span class="s5-status-val mono" data-role="step">0</span>
    </div>
    <div class="s5-status-row">
      <span class="s5-status-label">loss</span>
      <span class="s5-status-val mono" data-role="loss">—</span>
    </div>
  `;
  rightCol.appendChild(status);

  // Loss chart (log y).
  const chartWrap = document.createElement('div');
  chartWrap.className = 's5-chart-wrap';
  rightCol.appendChild(chartWrap);

  const CHART_W = 480, CHART_H = 160;
  const CHART_M = { top: 12, right: 10, bottom: 22, left: 38 };
  const svgChart = d3.select(chartWrap).append('svg')
    .attr('class', 's5-chart')
    .attr('viewBox', `0 0 ${CHART_W} ${CHART_H}`)
    .attr('preserveAspectRatio', 'none');

  const gChart = svgChart.append('g')
    .attr('transform', `translate(${CHART_M.left}, ${CHART_M.top})`);
  const innerW = CHART_W - CHART_M.left - CHART_M.right;
  const innerH = CHART_H - CHART_M.top - CHART_M.bottom;

  const gAxisX = svgChart.append('g')
    .attr('class', 'axis s5-axis-x')
    .attr('transform', `translate(${CHART_M.left}, ${CHART_M.top + innerH})`);
  const gAxisY = svgChart.append('g')
    .attr('class', 'axis s5-axis-y')
    .attr('transform', `translate(${CHART_M.left}, ${CHART_M.top})`);

  const lossPath = gChart.append('path')
    .attr('class', 's5-loss-path stroke-cluster-2');

  // t-slider
  const tWrap = document.createElement('div');
  tWrap.className = 's5-t-wrap';
  tWrap.innerHTML = `
    <label class="s5-t-caption">View vector field at t = <span data-role="t-val" class="mono">${T - 1}</span></label>
    <input type="range" class="s5-t-slider" min="0" max="${T - 1}" step="1" value="${T - 1}" disabled>
  `;
  rightCol.appendChild(tWrap);
  const tSlider = tWrap.querySelector('.s5-t-slider');
  const tValEl  = tWrap.querySelector('[data-role="t-val"]');

  // Training controls
  const ctrls = document.createElement('div');
  ctrls.className = 's5-ctrls';
  ctrls.innerHTML = `
    <button class="btn primary s5-train-btn" data-role="train">Train</button>
    <button class="btn s5-reset-btn" data-role="reset">Reset</button>
  `;
  rightCol.appendChild(ctrls);

  const trainBtn = ctrls.querySelector('[data-role="train"]');
  const resetBtn = ctrls.querySelector('[data-role="reset"]');

  const note = document.createElement('p');
  note.className = 'muted s5-note';
  note.innerHTML = '~50 s for the loss to plateau on a laptop. The 28×28 image case is shipped pre-trained — see next scene.';
  rightCol.appendChild(note);

  const closing = document.createElement('p');
  closing.className = 'muted s5-closing s5-hidden';
  closing.innerHTML = "<em>That&rsquo;s it.</em> &epsilon;&#770; = NN(x<sub>t</sub>, t). The next scene puts it to work.";
  rightCol.appendChild(closing);

  /* ----- KaTeX renders ---------------------------------------------------- */

  function renderKatex(host, src) {
    if (!host) return;
    host.textContent = '';
    if (window.katex) {
      try {
        window.katex.render(src, host, { displayMode: true, throwOnError: false });
      } catch (e) {
        host.textContent = src;
      }
    } else {
      host.textContent = src;
    }
  }

  renderKatex(formulaBlock,
    "\\mathcal{L} \\;=\\; \\mathbb{E}\\bigl[\\,\\|\\hat{\\varepsilon}(x_t, t) - \\varepsilon\\|^2\\,\\bigr]");

  /* ----- model + training loop ------------------------------------------- */

  function instantiateModel() {
    state.model = new NN.TwoDModel({ hidden: HIDDEN, lr: LR, seed: SEED });
    state.rng = M.mulberry32(SEED + 9001);
    state.initialLoss = null;
    state.sampledLoss = [];
    state.cappedAuto = false;
  }

  // Record the trained model into shared state so scene 6 picks it up.
  function shareModel() {
    if (!state.model) return;
    window.diffusionShared.twoDModel  = state.model;
    window.diffusionShared.lossHistory = state.model.lossHistory.slice();
  }

  function startTraining() {
    if (state.training) return;
    state.training = true;
    trainBtn.textContent = 'Pause';
    trainBtn.classList.remove('primary');
    trainBtn.classList.add('s5-pause');
    trainBtn.classList.remove('s5-cta');
    if (state.cursor === 0) {
      // implicit advance to "training in progress"
      state.cursor = 1;
      renderTextChrome();
    }
    state.rafId = requestAnimationFrame(trainTick);
  }

  function pauseTraining() {
    state.training = false;
    if (state.rafId != null) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    trainBtn.textContent = 'Train';
    trainBtn.classList.add('primary');
    trainBtn.classList.remove('s5-pause');
    if (state.cursor === 0) trainBtn.classList.add('s5-cta');
  }

  function trainTick() {
    if (!state.training) return;

    // One trainBatch per RAF (cap ~30 fps; JS is single-threaded).
    const batch = NN.sample2DBatch(DATA.letterM.points, DATA.alphaBars, BATCH_SIZE, state.rng);
    const loss = state.model.trainBatch(batch);

    if (state.initialLoss == null) state.initialLoss = loss;

    if (state.model.step % REFRESH_EVERY === 0) {
      state.sampledLoss.push({ step: state.model.step, loss });
      updateStatus(state.model.step, loss);
      updateLossChart();
      updateField();
    }

    // Auto-advance the step engine once: cursor 1 → 2 when training has
    // demonstrably reduced the loss (or we hit the cap). We pause when this
    // fires so the lecture beat can land.
    if (!state.cappedAuto && state.cursor === 1) {
      const reachedDrop = state.initialLoss != null
        && loss <= state.initialLoss * (1 - PLATEAU_DROP);
      if (reachedDrop || state.model.step >= MAX_STEPS_AUTO) {
        state.cappedAuto = true;
        // Pause on the beat. User can resume from the button.
        pauseTraining();
        shareModel();
        state.cursor = 2;
        renderTextChrome();
        // Enable the t-slider now that the model has structure.
        tSlider.disabled = false;
        return;
      }
    }

    state.rafId = requestAnimationFrame(trainTick);
  }

  /* ----- vector field render --------------------------------------------- */

  function updateField() {
    if (!state.model) return;
    const tn = state.tNorm;

    // First pass — compute raw −ε̂ for each grid point.
    const raw = new Array(gridPoints.length);
    let maxLen = 1e-9;
    for (let i = 0; i < gridPoints.length; i++) {
      const p = gridPoints[i];
      const eh = state.model.predict(p.x, p.y, tn);
      // Denoise direction is the OPPOSITE of the predicted noise.
      const dx = -eh[0];
      const dy = -eh[1];
      raw[i] = { dx, dy };
      const len = Math.hypot(dx, dy);
      if (len > maxLen) maxLen = len;
    }

    // Scale so the longest arrow ≈ ARROW_TARGET_LEN.
    const scale = ARROW_TARGET_LEN / maxLen;

    fieldLines
      .each(function (d, i) {
        const r = raw[i];
        const ex = d.x + r.dx * scale;
        const ey = d.y + r.dy * scale;
        // Flip y for SVG.
        d3.select(this)
          .attr('x2', ex)
          .attr('y2', -ey);
      });
  }

  /* ----- loss chart render ----------------------------------------------- */

  function updateLossChart() {
    const data = state.sampledLoss;
    if (data.length < 1) {
      lossPath.attr('d', '');
      gAxisX.selectAll('*').remove();
      gAxisY.selectAll('*').remove();
      return;
    }

    const xMax = Math.max(50, data[data.length - 1].step);
    const lossesArr = data.map(d => d.loss).filter(v => Number.isFinite(v) && v > 0);
    const lo = Math.max(1e-3, d3.min(lossesArr) * 0.85);
    const hi = Math.max(lo * 2, d3.max(lossesArr) * 1.15);

    const xScale = d3.scaleLinear().domain([0, xMax]).range([0, innerW]);
    const yScale = d3.scaleLog().domain([lo, hi]).range([innerH, 0]);

    const lineGen = d3.line()
      .defined(d => Number.isFinite(d.loss) && d.loss > 0)
      .x(d => xScale(d.step))
      .y(d => yScale(d.loss));

    lossPath.attr('d', lineGen(data));

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5).tickSizeOuter(0);
    const yAxis = d3.axisLeft(yScale)
      .ticks(4, '~g')
      .tickSizeOuter(0);
    gAxisX.call(xAxis);
    gAxisY.call(yAxis);
  }

  /* ----- status / chrome ------------------------------------------------- */

  function updateStatus(step, loss) {
    status.querySelector('[data-role="step"]').textContent = String(step);
    status.querySelector('[data-role="loss"]').textContent =
      Number.isFinite(loss) ? loss.toFixed(3) : '—';
  }

  function renderTextChrome() {
    stepPill.textContent = `Step ${state.cursor} of ${STEPS_TOTAL - 1}`;

    if (state.cursor === 0) {
      trainBtn.classList.add('s5-cta');
      closing.classList.add('s5-hidden');
    } else if (state.cursor === 1) {
      trainBtn.classList.remove('s5-cta');
      closing.classList.add('s5-hidden');
    } else if (state.cursor === 2) {
      trainBtn.classList.remove('s5-cta');
      closing.classList.add('s5-hidden');
    } else if (state.cursor === 3) {
      trainBtn.classList.remove('s5-cta');
      closing.classList.remove('s5-hidden');
    }
  }

  /* ----- t-slider --------------------------------------------------------- */

  tSlider.addEventListener('input', () => {
    const t = parseInt(tSlider.value, 10);
    state.tNorm = (T <= 1) ? 0 : t / (T - 1);
    tValEl.textContent = String(t);
    updateField();
  });

  /* ----- buttons ---------------------------------------------------------- */

  trainBtn.addEventListener('click', () => {
    if (state.training) {
      pauseTraining();
    } else {
      startTraining();
    }
  });

  resetBtn.addEventListener('click', () => {
    pauseTraining();
    instantiateModel();
    state.cursor = 0;
    state.tNorm = 1.0;
    tSlider.value = String(T - 1);
    tValEl.textContent = String(T - 1);
    tSlider.disabled = true;
    updateStatus(0, NaN);
    updateField();
    updateLossChart();
    renderTextChrome();
  });

  /* ----- step engine glue (Prev/Next keystrokes) ------------------------- */

  function setCursor(c) {
    if (c < 0 || c >= STEPS_TOTAL) return false;
    if (c === state.cursor) return false;

    // Going forward
    if (c > state.cursor) {
      // Cursor 0 → 1: start training (if not running).
      if (state.cursor === 0 && c >= 1) {
        startTraining();
        return true;
      }
      // Cursor 1 → 2: snap to "trained" if training never ran. We just pause
      // and let the slider be useful.
      if (state.cursor === 1 && c >= 2) {
        if (state.training) pauseTraining();
        if (state.cappedAuto === false) state.cappedAuto = true;
        shareModel();
        tSlider.disabled = false;
        state.cursor = 2;
        renderTextChrome();
        if (c > 2) return setCursor(c);
        return true;
      }
      if (state.cursor === 2 && c === 3) {
        state.cursor = 3;
        renderTextChrome();
        return true;
      }
    }

    // Going back: rewind clean.
    if (c < state.cursor) {
      pauseTraining();
      // Reset to a fresh model whenever we go back to 0; otherwise just
      // demote chrome — the slider stays interactive.
      if (c === 0) {
        instantiateModel();
        state.tNorm = 1.0;
        tSlider.value = String(T - 1);
        tValEl.textContent = String(T - 1);
        tSlider.disabled = true;
        updateStatus(0, NaN);
        updateField();
        updateLossChart();
      }
      state.cursor = c;
      renderTextChrome();
      return true;
    }

    return false;
  }

  /* ----- initial paint ---------------------------------------------------- */

  function fullRender() {
    updateStatus(state.model ? state.model.step : 0, NaN);
    updateField();
    updateLossChart();
    renderTextChrome();
  }

  // Headless verification hook: ?test=trained or &run lands on cursor=2 with a
  // burst-trained model. Not a user feature.
  function shouldAutoTrain() {
    return /[#&?]run\b/.test(window.location.hash || '')
        || /[#&?]test=trained\b/.test(window.location.hash || '');
  }

  function burstTrainFor(steps) {
    if (!state.model) return;
    state.initialLoss = null;
    for (let s = 0; s < steps; s++) {
      const batch = NN.sample2DBatch(DATA.letterM.points, DATA.alphaBars, BATCH_SIZE, state.rng);
      const loss = state.model.trainBatch(batch);
      if (state.initialLoss == null) state.initialLoss = loss;
      if (state.model.step % REFRESH_EVERY === 0) {
        state.sampledLoss.push({ step: state.model.step, loss });
      }
    }
    shareModel();
    state.cursor = 2;
    state.cappedAuto = true;
    tSlider.disabled = false;
    updateStatus(state.model.step,
      state.model.lossHistory[state.model.lossHistory.length - 1]);
    updateField();
    updateLossChart();
    renderTextChrome();
  }

  instantiateModel();
  fullRender();
  if (shouldAutoTrain()) {
    // Schedule on next tick so DOM is mounted.
    setTimeout(() => burstTrainFor(800), 50);
  }

  /* ----- onEnter / onLeave ------------------------------------------------ */

  return {
    onEnter() {
      // Cold entry: always reset. The user came here to TRAIN.
      pauseTraining();
      instantiateModel();
      state.cursor = 0;
      state.tNorm = 1.0;
      tSlider.value = String(T - 1);
      tValEl.textContent = String(T - 1);
      tSlider.disabled = true;
      fullRender();
      if (shouldAutoTrain()) {
        setTimeout(() => burstTrainFor(800), 50);
      }
    },
    onLeave() {
      pauseTraining();
      // If training had progress, persist it so scene 6 can use it.
      if (state.model && state.model.step > 0) shareModel();
    },
    onNextKey() {
      return setCursor(state.cursor + 1);
    },
    onPrevKey() {
      if (state.cursor === 0) return false;
      return setCursor(state.cursor - 1);
    },
  };
};
