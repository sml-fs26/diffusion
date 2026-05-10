/* Scene 3 — The shortcut (fast-forward formula).
 *
 * Two equal columns: iterative on the left, fast-forward on the right.
 * Each column shows the same letter-M cloud arriving at x_t — the left by
 * applying β_t one step at a time (drawing a fresh ε_t every step), the
 * right in a single shot via x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε̂.
 *
 * The point: the two distributions of x_t are identical. Same destination,
 * vastly different compute. Internal step engine cursor 0..3 walks t = 0,
 * t = 50, t = T-1, then a 256-seed scatter that demonstrates the equivalence
 * visually (centroids overlap to within ~0.05 normalised units).
 */

window.scenes.scene3 = function (root) {
  const DATA = window.DATA;
  const Math_ = window.DiffusionMath;
  const NN = window.DiffusionNN;

  const T = DATA.T;
  const TMAX = T - 1;

  // Pick a digit-3 MNIST sample (same digit as scene 4 for narrative cohesion).
  const mnistSample = (DATA.mnistSamples || []).find(s => s && s.label === 3)
                   || (DATA.mnistSamples || [])[3]
                   || (DATA.mnistSamples || [])[0];

  // ---- Build static DOM ---------------------------------------------------
  root.innerHTML = '';
  root.classList.add('scene-s3');

  const wrap = document.createElement('div');
  wrap.className = 's3-wrap';
  root.appendChild(wrap);

  // Header row
  const header = document.createElement('div');
  header.className = 's3-header';
  wrap.appendChild(header);

  const title = document.createElement('h2');
  title.className = 's3-title';
  title.textContent = 'Same destination, two paths.';
  header.appendChild(title);

  // Subtitle: KaTeX-rendered, with the two formulas inline so the equivalence
  // claim names them explicitly. Escaped `\\` for the JS string literal.
  const sub = document.createElement('p');
  sub.className = 's3-sub';
  header.appendChild(sub);
  try {
    sub.innerHTML = '';
    const ITERATIVE_TEX = '\\textcolor{#2f6cb1}{x_{t+1} = \\sqrt{1-\\beta_t}\\,x_t + \\sqrt{\\beta_t}\\,\\varepsilon_t}';
    const FAST_TEX      = '\\textcolor{#7a5c8c}{x_t = \\sqrt{\\bar\\alpha_t}\\,x_0 + \\sqrt{1-\\bar\\alpha_t}\\,\\hat\\varepsilon}';
    const a = document.createElement('span'); sub.appendChild(document.createTextNode('Iterating '));
    sub.appendChild(a);
    katex.render(ITERATIVE_TEX, a, { throwOnError: false, displayMode: false, trust: true });
    sub.appendChild(document.createTextNode(' for t steps lands on the same distribution as one shot through '));
    const b = document.createElement('span'); sub.appendChild(b);
    katex.render(FAST_TEX, b, { throwOnError: false, displayMode: false, trust: true });
    sub.appendChild(document.createTextNode('. Compute is the only thing that differs.'));
  } catch (e) {
    sub.textContent = 'Iterating x_{t+1} = √(1−β_t)x_t + √β_t·ε_t for t steps lands on the same distribution as one shot through x_t = √ᾱ_t x_0 + √(1−ᾱ_t)·ε̂.';
  }

  // Split layout: two columns
  const split = document.createElement('div');
  split.className = 's3-split';
  wrap.appendChild(split);

  function makeColumn(side, headingTxt, headingClass) {
    const col = document.createElement('div');
    col.className = `s3-col s3-col-${side}`;

    const h = document.createElement('div');
    h.className = `s3-col-head ${headingClass || ''}`;
    h.textContent = headingTxt;
    col.appendChild(h);

    const fblock = document.createElement('div');
    fblock.className = 's3-formula';
    col.appendChild(fblock);

    const panes = document.createElement('div');
    panes.className = 's3-panes';
    col.appendChild(panes);

    const svgWrap = document.createElement('div');
    svgWrap.className = 's3-svg-wrap';
    panes.appendChild(svgWrap);
    const svg = d3.select(svgWrap).append('svg')
      .attr('class', `s3-svg s3-svg-${side}`)
      .attr('viewBox', '-3 -3 6 6')
      .attr('preserveAspectRatio', 'xMidYMid meet');
    const gPoints = svg.append('g').attr('class', `s3-g-points`);
    const gScatter = svg.append('g').attr('class', `s3-g-scatter`);

    const mnWrap = document.createElement('div');
    mnWrap.className = 's3-mn-wrap';
    panes.appendChild(mnWrap);
    const canvas = document.createElement('canvas');
    canvas.className = 'mnist-pane s3-mnist';
    canvas.width = 28;
    canvas.height = 28;
    mnWrap.appendChild(canvas);

    const meta = document.createElement('div');
    meta.className = 's3-meta';
    col.appendChild(meta);

    const counter = document.createElement('div');
    counter.className = 's3-counter';
    counter.innerHTML = '<span class="s3-counter-label">RNG calls (per sample)</span><span class="s3-counter-val mono">—</span>';
    meta.appendChild(counter);

    const noiseLabel = document.createElement('div');
    noiseLabel.className = 's3-noise';
    noiseLabel.innerHTML = '<span class="s3-noise-label">‖noise‖</span><span class="s3-noise-val mono">—</span>';
    meta.appendChild(noiseLabel);

    return { col, fblock, svg, gPoints, gScatter, canvas, counter, counterVal: counter.querySelector('.s3-counter-val'), noiseVal: noiseLabel.querySelector('.s3-noise-val') };
  }

  const colL = makeColumn('left',  'Iterative',     'cluster-1');
  const colR = makeColumn('right', 'Fast-forward',  'cluster-4');
  split.appendChild(colL.col);
  split.appendChild(colR.col);

  // KaTeX render
  try {
    katex.render(
      "x_{t+1} \\;=\\; \\sqrt{1-\\beta_t}\\,x_t \\;+\\; \\sqrt{\\beta_t}\\,\\varepsilon_t",
      colL.fblock,
      { throwOnError: false, displayMode: true }
    );
  } catch (e) { colL.fblock.textContent = 'x_{t+1} = √(1−β_t) x_t + √β_t · ε_t'; }
  try {
    katex.render(
      "x_t \\;=\\; \\sqrt{\\bar\\alpha_t}\\,x_0 \\;+\\; \\sqrt{1-\\bar\\alpha_t}\\,\\hat\\varepsilon",
      colR.fblock,
      { throwOnError: false, displayMode: true }
    );
  } catch (e) { colR.fblock.textContent = 'x_t = √ᾱ_t x_0 + √(1−ᾱ_t) ε̂'; }

  // Slider + seed input
  const ctlRow = document.createElement('div');
  ctlRow.className = 's3-ctl';
  wrap.appendChild(ctlRow);

  const tCtl = document.createElement('div');
  tCtl.className = 's3-tctl';
  ctlRow.appendChild(tCtl);
  const tLabel = document.createElement('label');
  tLabel.className = 's3-ctl-label';
  tLabel.innerHTML = `<span>t</span><span class="mono s3-tval">0</span><span class="muted s3-trange">/ ${TMAX}</span>`;
  tCtl.appendChild(tLabel);
  const tSlider = document.createElement('input');
  tSlider.type = 'range';
  tSlider.min = 0;
  tSlider.max = TMAX;
  tSlider.step = 1;
  tSlider.value = 0;
  tSlider.className = 's3-slider';
  tCtl.appendChild(tSlider);
  const tValEl = tLabel.querySelector('.s3-tval');

  const seedCtl = document.createElement('div');
  seedCtl.className = 's3-seed';
  ctlRow.appendChild(seedCtl);
  const seedLabel = document.createElement('label');
  seedLabel.className = 's3-ctl-label';
  seedLabel.innerHTML = '<span>seed</span>';
  seedCtl.appendChild(seedLabel);
  const seedInput = document.createElement('input');
  seedInput.type = 'number';
  seedInput.value = 42;
  seedInput.step = 1;
  seedInput.className = 's3-seed-input mono';
  seedCtl.appendChild(seedInput);

  // No step pill, no equivalence-demo overlay. Scene 3 is a single live
  // state: drag the t slider, watch the iterative cloud animate (with its
  // RNG-call counter ticking up) and the fast-forward cloud arrive instantly.
  // The whole pedagogical point is the comparison; it doesn't need a
  // multi-step walkthrough.

  // ---- Letter-M points (subsampled) --------------------------------------
  const letterPoints = (function () {
    const src = DATA.letterM && Array.isArray(DATA.letterM.points)
      ? DATA.letterM.points : [];
    const stride = Math.max(1, Math.floor(src.length / 140));
    const sub = [];
    for (let i = 0; i < src.length; i += stride) {
      const p = src[i];
      if (Array.isArray(p) && p.length >= 2) sub.push([p[0], p[1]]);
    }
    return sub;
  })();
  const N2D = letterPoints.length;
  const x0_2D = new Float32Array(2 * N2D);
  for (let i = 0; i < N2D; i++) {
    x0_2D[2 * i]     = letterPoints[i][0];
    x0_2D[2 * i + 1] = letterPoints[i][1];
  }

  const x0_mnist = new Float32Array(784);
  if (mnistSample && Array.isArray(mnistSample.pixels)) {
    for (let i = 0; i < 784; i++) x0_mnist[i] = mnistSample.pixels[i] || 0;
  }

  // ---- Compute helpers ----------------------------------------------------

  // Iterative forward: starting at x_0, apply forwardStep with fresh ε's drawn
  // from a seeded RNG. Returns { x_t, x_T_eps_aggregate (vector that maps back via
  // (x_t - √ᾱ_t·x_0)/√(1-ᾱ_t)) }.
  function iterativeForward(x0, t, seed) {
    const rng = Math_.mulberry32(seed);
    let cur = new Float32Array(x0);
    for (let s = 0; s < t; s++) {
      const e = Math_.randnVector(rng, cur.length);
      cur = Math_.forwardStep(cur, DATA.betas[s], e);
    }
    return cur;
  }

  function fastForwardOnce(x0, t, seed) {
    if (t === 0) return new Float32Array(x0);
    const rng = Math_.mulberry32(seed);
    const eps = Math_.randnVector(rng, x0.length);
    return Math_.fastForward(x0, DATA.alphaBars[t], eps);
  }

  function aggregateNoise(x_t, x0, t) {
    // Recover the implicit single-shot ε given x_t and x_0:
    //   ε = (x_t − √ᾱ_t · x_0) / √(1 − ᾱ_t)
    // This is the lemma identity that makes the two paths interchangeable
    // in distribution. Returns the L2 norm of that ε.
    if (t === 0) return 0;
    const ab = DATA.alphaBars[t];
    const sa = Math.sqrt(ab);
    const sb = Math.sqrt(1 - ab);
    let s = 0;
    for (let i = 0; i < x_t.length; i++) {
      const v = (x_t[i] - sa * x0[i]) / sb;
      s += v * v;
    }
    return Math.sqrt(s);
  }

  // ---- Rendering ----------------------------------------------------------
  function paintCloud(g, state) {
    const N = state.length / 2;
    const sel = g.selectAll('circle.s3-pt').data(d3.range(N));
    sel.exit().remove();
    const ent = sel.enter().append('circle')
      .attr('class', 'point cluster-1 s3-pt')
      .attr('r', 0.04);
    ent.merge(sel)
      .attr('cx', i => state[2 * i])
      .attr('cy', i => -state[2 * i + 1]);
  }

  function paintScatter(g, samples, klass) {
    // samples: Float32Array length 2K (interleaved). Plot all 2D points.
    const K = samples.length / 2;
    const sel = g.selectAll('circle.s3-scatter').data(d3.range(K));
    sel.exit().remove();
    const ent = sel.enter().append('circle')
      .attr('class', 'point s3-scatter')
      .attr('r', 0.045);
    ent.merge(sel)
      .attr('class', `point s3-scatter ${klass}`)
      .attr('cx', i => samples[2 * i])
      .attr('cy', i => -samples[2 * i + 1]);
  }

  function clearScatter(g) {
    g.selectAll('circle.s3-scatter').remove();
  }

  function paintMnist(canvas, state) {
    const ctx = canvas.getContext('2d');
    if (!state) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, 28, 28);
      return;
    }
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      let v = state[i];
      if (v < -1) v = -1;
      if (v >  1) v =  1;
      const g = Math.round((v + 1) * 0.5 * 255);
      img.data[4 * i + 0] = g;
      img.data[4 * i + 1] = g;
      img.data[4 * i + 2] = g;
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  function fmt(x) {
    if (!isFinite(x)) return '∞';
    if (x === 0) return '0';
    if (Math.abs(x) >= 100) return x.toFixed(0);
    if (Math.abs(x) >= 1)   return x.toFixed(2);
    if (Math.abs(x) >= 0.001) return x.toFixed(3);
    return x.toExponential(2);
  }

  // ---- State --------------------------------------------------------------
  let curT = 0;
  let curSeed = 42;
  let leftAnimToken = 0;     // cancellation token for in-progress animations

  // ---- Single-mode painters ----------------------------------------------
  // Animate the iterative cloud from x_0 to x_t. Capped to ~600ms total.
  function animateLeft(t, seed) {
    leftAnimToken++;
    const myToken = leftAnimToken;
    clearScatter(colL.gScatter);
    clearScatter(colR.gScatter);

    const rng = Math_.mulberry32(seed);
    let cur = new Float32Array(x0_2D);
    let mnCur = new Float32Array(x0_mnist);
    const rngMn = Math_.mulberry32(seed + 1009);

    paintCloud(colL.gPoints, cur);
    paintMnist(colL.canvas, mnCur);

    if (t === 0) {
      colL.counterVal.textContent = '0';
      colL.noiseVal.textContent = '0';
      return;
    }

    // For t > 0, animate via setInterval with skip rate so total ≈ 600 ms.
    const totalMs = 600;
    const ms_per_frame = 16;
    const total_frames = Math.max(1, Math.floor(totalMs / ms_per_frame));
    const stepsPerFrame = Math.max(1, Math.ceil(t / total_frames));

    let s = 0;
    function advance() {
      if (myToken !== leftAnimToken) return;
      const target = Math.min(t, s + stepsPerFrame);
      while (s < target) {
        const e2 = Math_.randnVector(rng, cur.length);
        cur = Math_.forwardStep(cur, DATA.betas[s], e2);
        const eM = Math_.randnVector(rngMn, mnCur.length);
        mnCur = Math_.forwardStep(mnCur, DATA.betas[s], eM);
        s++;
      }
      paintCloud(colL.gPoints, cur);
      paintMnist(colL.canvas, mnCur);
      colL.counterVal.textContent = String(s);
      colL.noiseVal.textContent = fmt(aggregateNoise(cur, x0_2D, s));
      if (s < t) {
        setTimeout(advance, ms_per_frame);
      } else {
        // Final paint with full agg-noise (in 784-D pixel space) for the MNIST
        // norm display — but we keep noiseVal as the 2D aggregate norm for
        // visual parity with the right column.
      }
    }
    setTimeout(advance, 0);
  }

  function paintRight(t, seed) {
    clearScatter(colR.gScatter);
    if (t === 0) {
      paintCloud(colR.gPoints, x0_2D);
      paintMnist(colR.canvas, x0_mnist);
      colR.counterVal.textContent = '0';
      colR.noiseVal.textContent = '0';
      return;
    }
    const xt = fastForwardOnce(x0_2D, t, seed);
    paintCloud(colR.gPoints, xt);
    const xtMn = fastForwardOnce(x0_mnist, t, seed + 1009);
    paintMnist(colR.canvas, xtMn);
    colR.counterVal.textContent = '1';
    // Single ε's norm: same identity as the iterative aggregate.
    colR.noiseVal.textContent = fmt(aggregateNoise(xt, x0_2D, t));
  }

  // ---- Scatter mode (the equivalence demo) -------------------------------
  // Run BOTH methods K times with different seeds, scatter the resulting x_t
  // values for each method. Two clouds visually overlap.
  function runScatter(t, baseSeed) {
    const K = 256;
    const iterSamples = new Float32Array(2 * K);
    const ffSamples   = new Float32Array(2 * K);

    // For iterative we only need the final 2D sample (just the centroid pixel).
    // We use a single 2D point (the centroid of x_0) as the seed input —
    // this isolates the noise distribution and avoids a heavy per-seed full-cloud
    // run. The marginal of x_t starting at any single x_0 is exactly the same
    // distribution we want to compare.
    const seedPoint = new Float32Array(2);
    seedPoint[0] = 0;
    seedPoint[1] = 0;
    // Use a fixed, central x_0 (a representative letter-M point — origin) so
    // both samples share x_0; the spread comes purely from the noise.

    // K iterative samples
    for (let k = 0; k < K; k++) {
      const seed = (baseSeed + 7919 * k) >>> 0;
      const cur = iterativeForward(seedPoint, t, seed);
      iterSamples[2 * k]     = cur[0];
      iterSamples[2 * k + 1] = cur[1];
    }
    // K fast-forward samples (independent seeds)
    for (let k = 0; k < K; k++) {
      const seed = (baseSeed + 7919 * k + 31) >>> 0;
      const cur = fastForwardOnce(seedPoint, t, seed);
      ffSamples[2 * k]     = cur[0];
      ffSamples[2 * k + 1] = cur[1];
    }

    // Compute centroids and normalised distance.
    function centroid(s) {
      let cx = 0, cy = 0;
      const n = s.length / 2;
      for (let i = 0; i < n; i++) { cx += s[2 * i]; cy += s[2 * i + 1]; }
      return [cx / n, cy / n];
    }
    function stdscale(s) {
      // Population std combined across both axes (z-score normalisation).
      const n = s.length / 2;
      let mx = 0, my = 0;
      for (let i = 0; i < n; i++) { mx += s[2 * i]; my += s[2 * i + 1]; }
      mx /= n; my /= n;
      let v = 0;
      for (let i = 0; i < n; i++) {
        const dx = s[2 * i] - mx, dy = s[2 * i + 1] - my;
        v += dx * dx + dy * dy;
      }
      v /= (2 * n);
      return Math.sqrt(v);
    }

    const cI = centroid(iterSamples);
    const cF = centroid(ffSamples);
    const sd = (stdscale(iterSamples) + stdscale(ffSamples)) * 0.5 || 1;
    const dx = cI[0] - cF[0], dy = cI[1] - cF[1];
    const cdist = Math.sqrt(dx * dx + dy * dy);
    const normCdist = cdist / sd;

    // Stash on shared for later inspection.
    window.diffusionShared = window.diffusionShared || {};
    window.diffusionShared.scene3Scatter = {
      t: t, K: K, baseSeed: baseSeed,
      centroidIter: cI, centroidFF: cF,
      cdist: cdist, normCdist: normCdist
    };

    // Paint both scatters on top of the cleared cloud.
    paintCloud(colL.gPoints, x0_2D);  // keep faint M reference
    paintCloud(colR.gPoints, x0_2D);
    paintScatter(colL.gScatter, iterSamples, 'cluster-1');
    paintScatter(colR.gScatter, ffSamples,   'cluster-4');
    // Hide the underlying x_0 cloud so it doesn't compete visually.
    colL.gPoints.attr('opacity', 0.08);
    colR.gPoints.attr('opacity', 0.08);

    paintMnist(colL.canvas, fastForwardOnce(x0_mnist, t, baseSeed));
    paintMnist(colR.canvas, fastForwardOnce(x0_mnist, t, baseSeed + 1009));

    colL.counterVal.textContent = `${t} × ${K}`;
    colR.counterVal.textContent = `1 × ${K}`;
    colL.noiseVal.textContent = `cdist ${fmt(cdist)} (${fmt(normCdist)} norm)`;
    colR.noiseVal.textContent = `cdist ${fmt(cdist)} (${fmt(normCdist)} norm)`;

    overlay.innerHTML = `<em>Same destination, two paths.</em><span class="s3-overlay-detail">256 samples per method · normalised centroid distance ${fmt(normCdist)}</span>`;
  }

  function leaveScatter() {
    colL.gPoints.attr('opacity', 1);
    colR.gPoints.attr('opacity', 1);
    clearScatter(colL.gScatter);
    clearScatter(colR.gScatter);
    overlay.innerHTML = '';
  }

  // ---- Master render ------------------------------------------------------
  // No mode switch, no cursor — just paint at the current (t, seed). The
  // `animate` flag controls whether the iterative cloud animates step-by-step
  // (used on slider release) or paints statically (used during drag).
  function render(animate) {
    tValEl.textContent = String(curT);
    if (animate) {
      animateLeft(curT, curSeed);
    } else {
      const xt = iterativeForward(x0_2D, curT, curSeed);
      paintCloud(colL.gPoints, xt);
      const xtMn = iterativeForward(x0_mnist, curT, curSeed + 1009);
      paintMnist(colL.canvas, xtMn);
      colL.counterVal.textContent = String(curT);
      colL.noiseVal.textContent = fmt(aggregateNoise(xt, x0_2D, curT));
    }
    paintRight(curT, curSeed);
  }

  // ---- Event wiring -------------------------------------------------------
  tSlider.addEventListener('input', () => {
    curT = parseInt(tSlider.value, 10) || 0;
    tValEl.textContent = String(curT);
    // Static paint while dragging (no animation — too jittery).
    const xt = iterativeForward(x0_2D, curT, curSeed);
    paintCloud(colL.gPoints, xt);
    const xtMn = iterativeForward(x0_mnist, curT, curSeed + 1009);
    paintMnist(colL.canvas, xtMn);
    colL.counterVal.textContent = String(curT);
    colL.noiseVal.textContent = fmt(aggregateNoise(xt, x0_2D, curT));
    paintRight(curT, curSeed);
  });
  seedInput.addEventListener('input', () => {
    const v = parseInt(seedInput.value, 10);
    if (Number.isFinite(v)) {
      curSeed = v;
      render(false);
    }
  });

  // ---- Init ---------------------------------------------------------------
  function init() {
    curT = 0;
    curSeed = parseInt(seedInput.value, 10) || 42;
    tSlider.value = 0;
    tValEl.textContent = '0';
    render(false);
  }
  init();

  // `&run` flag jumps to t = T-1 for headless capture (no cursor system).
  function shouldAutoRun() { return /[#&?]run(All)?\b/.test(window.location.hash || ''); }
  if (shouldAutoRun()) {
    curT = TMAX;
    tSlider.value = curT;
    setTimeout(() => render(false), 80);
  }

  return {
    onEnter() {
      curSeed = parseInt(seedInput.value, 10) || 42;
      init();
      if (shouldAutoRun()) {
        curT = TMAX;
        tSlider.value = curT;
        setTimeout(() => render(false), 80);
      }
    },
    onLeave() {
      leftAnimToken++;  // cancel any pending animation timers
    },
    onNextKey() { return false; },
    onPrevKey() { return false; },
  };
};
