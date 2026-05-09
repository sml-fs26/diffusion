/* Scene 4 — Reverse — wishful thinking.
 *
 * The pedagogical climax of the viz. The hero formula gives x_{t-1} from x_t,
 * but it depends on knowing ε. Three buttons probe what happens when we don't:
 *   A "Use random ε"   — garbage out
 *   B "Use bookkept ε" — perfect recovery (only because we cheated and saved them)
 *   C "Use predicted ε̂" — disabled. The whole rest of the viz is about earning this.
 *
 * Cold entry must work: onEnter rebuilds two forward trajectories (letter M in 2D,
 * digit 3 in MNIST), stashes their ε sequences in window.diffusionShared, and
 * paints x_T as the initial result panel.
 */

window.scenes.scene4 = function (root) {
  const DATA = window.DATA;
  const Math_ = window.DiffusionMath;
  const NN = window.DiffusionNN;

  // ---- Cross-scene shared state -------------------------------------------
  window.diffusionShared = window.diffusionShared || {};

  // ---- Constants ----------------------------------------------------------
  const T = DATA.T;
  const SEED_2D = 11;
  const SEED_MN = 17;

  // Pick the digit-3 MNIST sample (or any 3 we can find).
  const mnistSample = (DATA.mnistSamples || []).find(s => s && s.label === 3)
                   || (DATA.mnistSamples || [])[3]
                   || (DATA.mnistSamples || [])[0];

  // Snapshot indices for the trail. Skew toward smaller t (closer to recovered
  // x_0) so the trail visually leads the eye in to the final cloud.
  const SNAP_TS = [
    Math.round(T * 0.05),
    Math.round(T * 0.10),
    Math.round(T * 0.18),
    Math.round(T * 0.30),
    Math.round(T * 0.50),
    Math.round(T * 0.75),
  ];

  // ---- Build static DOM ---------------------------------------------------
  root.innerHTML = '';
  root.classList.add('scene-s4');

  const hero = document.createElement('div');
  hero.className = 's4-hero';
  root.appendChild(hero);

  const eyebrow = document.createElement('div');
  eyebrow.className = 's4-eyebrow';
  eyebrow.textContent = 'The reverse step';
  hero.appendChild(eyebrow);

  const formulaBlock = document.createElement('div');
  formulaBlock.className = 's4-formula';
  hero.appendChild(formulaBlock);

  const subcap = document.createElement('p');
  subcap.className = 's4-subcap';
  subcap.innerHTML = '<em>Beautiful. To use it we need&nbsp;<span class="cluster-2">ε</span>.</em>';
  hero.appendChild(subcap);

  // Buttons
  const btnRow = document.createElement('div');
  btnRow.className = 's4-btn-row';
  hero.appendChild(btnRow);

  function makeBtn(klass, label, sub) {
    const b = document.createElement('button');
    b.className = 's4-bigbtn ' + klass;
    b.innerHTML = `<span class="s4-bigbtn-label">${label}</span><span class="s4-bigbtn-sub">${sub}</span>`;
    return b;
  }
  // Two paths only:
  //   (B) the cheating path — replay the actual forward ε's. Recovers x_0 exactly.
  //   (C) the path we'd actually need — a network's guess for ε̂. Disabled here.
  const btnB = makeBtn('s4-btn-bookkept cluster-1',  'Use bookkept ε',   'replay the saved forward noise');
  const btnC = makeBtn('s4-btn-predicted cluster-5', 'Use predicted ε̂', 'requires a trained network — see next scene');
  btnC.disabled = true;
  btnC.title = 'Predicted ε̂ requires a trained network. We do not have one yet.';
  btnRow.appendChild(btnB);
  btnRow.appendChild(btnC);

  // Why-not callout — sits below the button row, explains the gap.
  const whyNot = document.createElement('div');
  whyNot.className = 's4-whynot';
  whyNot.innerHTML = `
    <div class="s4-whynot-title">Why is predicted&nbsp;<span class="cluster-5">ε̂</span>&nbsp;not working?</div>
    <p class="s4-whynot-body">
      The reverse formula needs the noise <span class="cluster-2">ε</span> that was added during the forward process.
      For images we already corrupted ourselves we still have it — that's the bookkept path.
      But for a fresh image we want to <em>generate</em>, no forward run ever happened, so there is no <span class="cluster-2">ε</span> to read off.
      The next scene trains a neural network to <em>guess</em> <span class="cluster-5">ε̂</span> from <span class="mono">x<sub>t</sub></span> alone.
    </p>
  `;
  hero.appendChild(whyNot);

  // Result panel
  const result = document.createElement('div');
  result.className = 's4-result';
  hero.appendChild(result);

  const result2D = document.createElement('div');
  result2D.className = 's4-pane s4-pane-2d';
  result.appendChild(result2D);

  const resultMn = document.createElement('div');
  resultMn.className = 's4-pane s4-pane-mn';
  result.appendChild(resultMn);

  // 2D SVG
  const svg2D = d3.select(result2D).append('svg')
    .attr('class', 's4-svg-2d')
    .attr('viewBox', '-3 -3 6 6')
    .attr('preserveAspectRatio', 'xMidYMid meet');
  const gTrail = svg2D.append('g').attr('class', 's4-g-trail');
  const gFinal = svg2D.append('g').attr('class', 's4-g-final');

  const cap2D = document.createElement('div');
  cap2D.className = 's4-pane-cap';
  cap2D.textContent = '2D — letter M';
  result2D.appendChild(cap2D);

  // MNIST canvas
  const canvas = document.createElement('canvas');
  canvas.className = 'mnist-pane s4-mnist-canvas';
  canvas.width = 28;
  canvas.height = 28;
  resultMn.appendChild(canvas);

  const capMn = document.createElement('div');
  capMn.className = 's4-pane-cap';
  capMn.textContent = 'MNIST — digit ' + (mnistSample ? mnistSample.label : '3');
  resultMn.appendChild(capMn);

  // Verdict + stats
  const verdict = document.createElement('div');
  verdict.className = 's4-verdict';
  hero.appendChild(verdict);

  const stats = document.createElement('div');
  stats.className = 's4-stats';
  hero.appendChild(stats);

  const closing = document.createElement('div');
  closing.className = 's4-closing';
  closing.innerHTML = '<em>The math works. We just need someone to guess <span class="cluster-5">ε̂</span>. Onwards.</em>';
  hero.appendChild(closing);

  // ---- KaTeX render -------------------------------------------------------
  try {
    katex.render(
      "x_{t-1} \\;=\\; \\frac{1}{\\sqrt{\\alpha_t}}\\!\\left(x_t - \\frac{\\beta_t}{\\sqrt{1-\\bar\\alpha_t}}\\,\\varepsilon_t\\right) \\;+\\; \\sqrt{\\beta_t}\\,z",
      formulaBlock,
      { throwOnError: false, displayMode: true }
    );
  } catch (e) {
    formulaBlock.textContent = 'x_{t-1} = (1/√α_t)(x_t − β_t/√(1−ᾱ_t) ε_t) + √β_t · z';
  }

  // ---- Letter-M points (subsampled to ~120 for snappy reverse) -------------
  const letterPoints = (function () {
    const src = DATA.letterM && Array.isArray(DATA.letterM.points)
      ? DATA.letterM.points : [];
    const stride = Math.max(1, Math.floor(src.length / 120));
    const sub = [];
    for (let i = 0; i < src.length; i += stride) {
      const p = src[i];
      if (Array.isArray(p) && p.length >= 2) sub.push([p[0], p[1]]);
    }
    return sub;
  })();
  const N2D = letterPoints.length;
  // Flatten to a single Float32Array of length 2*N for trajectory bookkeeping.
  const x0_2D = new Float32Array(2 * N2D);
  for (let i = 0; i < N2D; i++) {
    x0_2D[2 * i]     = letterPoints[i][0];
    x0_2D[2 * i + 1] = letterPoints[i][1];
  }

  // x0 for MNIST sample (Float32Array length 784)
  const x0_mnist = new Float32Array(784);
  if (mnistSample && Array.isArray(mnistSample.pixels)) {
    for (let i = 0; i < 784; i++) x0_mnist[i] = mnistSample.pixels[i] || 0;
  }

  // ---- Forward bookkeeping (cold-entry guarantee) -------------------------
  let bookkept2D = null;       // { trajectory, epsilons }
  let bookkeptMn = null;
  let xT_2D = null, xT_mn = null;

  function rebuildBookkept() {
    bookkept2D = NN.forwardTrajectory(x0_2D, DATA.betas, Math_.mulberry32(SEED_2D));
    bookkeptMn = NN.forwardTrajectory(x0_mnist, DATA.betas, Math_.mulberry32(SEED_MN));
    xT_2D = bookkept2D.trajectory[T];
    xT_mn = bookkeptMn.trajectory[T];

    // The DDPM reverse-step formula uses ε̂_t = the *aggregate* noise present
    // in x_t (i.e. ε̄ s.t. x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε̄), not the per-step ε
    // drawn during forward. We derive ε̄_t exactly from the bookkept trajectory:
    //     ε̄_t = (x_t − √ᾱ_t · x_0) / √(1 − ᾱ_t)
    // Replaying the reverse with these ε̄_t recovers x_0 exactly (modulo the
    // sigma-z term, which we cancel by matching forward's drawn noise on the
    // closed-form path — see runBookkept comment below).
    function buildAggregateEps(traj, x0) {
      const out = new Array(T);
      for (let t = 1; t <= T; t++) {
        const ab = DATA.alphaBars[t - 1];  // ᾱ at index of step t
        const sa = Math.sqrt(ab);
        const sb = Math.sqrt(1 - ab) || 1e-12;
        const xt = traj[t];
        const e = new Float32Array(xt.length);
        for (let i = 0; i < xt.length; i++) {
          e[i] = (xt[i] - sa * x0[i]) / sb;
        }
        out[t - 1] = e;
      }
      return out;
    }
    bookkept2D.aggregateEpsilons = buildAggregateEps(bookkept2D.trajectory, x0_2D);
    bookkeptMn.aggregateEpsilons = buildAggregateEps(bookkeptMn.trajectory, x0_mnist);

    window.diffusionShared.bookkeptForward = {
      x0_2D: x0_2D,
      x0_mnist: x0_mnist,
      epsilons2D: bookkept2D.epsilons,
      epsilonsMnist: bookkeptMn.epsilons,
      aggregateEpsilons2D: bookkept2D.aggregateEpsilons,
      aggregateEpsilonsMnist: bookkeptMn.aggregateEpsilons,
      trajectory2D: bookkept2D.trajectory,
      trajectoryMnist: bookkeptMn.trajectory,
      seed_2D: SEED_2D,
      seed_mnist: SEED_MN,
    };
  }

  // ---- Rendering helpers --------------------------------------------------
  function paint2D(state, trail) {
    // state: Float32Array length 2N. trail: array of Float32Array (intermediate snapshots).
    const sel = gTrail.selectAll('g.s4-trail-frame').data(trail || []);
    sel.exit().remove();
    const ent = sel.enter().append('g').attr('class', 's4-trail-frame');
    const merged = ent.merge(sel);
    merged.each(function (frame, fi) {
      // Steep ramp: deepest-in-noise frames almost invisible; closer-to-x0 frames
      // only modestly darker. Final state must read clearly above the trail.
      const opacity = 0.04 + 0.03 * fi;
      const circles = d3.select(this).selectAll('circle').data(d3.range(frame.length / 2));
      circles.exit().remove();
      const ec = circles.enter().append('circle').attr('class', 'point cluster-1').attr('r', 0.03);
      ec.merge(circles)
        .attr('cx', i => frame[2 * i])
        .attr('cy', i => -frame[2 * i + 1])
        .attr('opacity', opacity);
    });

    if (state) {
      const idxs = d3.range(state.length / 2);
      const fin = gFinal.selectAll('circle.s4-final').data(idxs);
      fin.exit().remove();
      const fe = fin.enter().append('circle')
        .attr('class', 'point s4-final')
        .attr('r', 0.05);
      fe.merge(fin)
        .attr('cx', i => state[2 * i])
        .attr('cy', i => -state[2 * i + 1])
        .attr('opacity', 1);
    } else {
      gFinal.selectAll('circle.s4-final').remove();
    }
  }

  function paintMnist(state) {
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

  function setFinalClass(klass) {
    gFinal.selectAll('circle.s4-final')
      .attr('class', 'point s4-final ' + klass);
    gTrail.selectAll('g.s4-trail-frame circle')
      .attr('class', 'point ' + klass);
  }

  function mse(a, b) {
    let s = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return s / n;
  }

  function relErr(recovered, original) {
    let num = 0, den = 0;
    const n = Math.min(recovered.length, original.length);
    for (let i = 0; i < n; i++) {
      const d = recovered[i] - original[i];
      num += d * d;
      den += original[i] * original[i];
    }
    return Math.sqrt(num) / (Math.sqrt(den) + 1e-12);
  }

  function fmt(x) {
    if (!isFinite(x)) return '∞';
    if (Math.abs(x) >= 100) return x.toFixed(0);
    if (Math.abs(x) >= 1)   return x.toFixed(2);
    if (Math.abs(x) >= 0.001) return x.toFixed(3);
    return x.toExponential(2);
  }

  // ---- Reverse runners ----------------------------------------------------
  function runReverse(epsFn2D, epsFnMn, rng2D, rngMn) {
    const rev2D = NN.reverseTrajectory(xT_2D, epsFn2D, DATA.betas, DATA.alphas, DATA.alphaBars, rng2D);
    const revMn = NN.reverseTrajectory(xT_mn, epsFnMn, DATA.betas, DATA.alphas, DATA.alphaBars, rngMn);
    return { rev2D, revMn };
  }

  function trailFrames(traj) {
    // Pull intermediate snapshots from large t to small t (skip t=0 final).
    const frames = [];
    for (let i = SNAP_TS.length - 1; i >= 0; i--) {
      const t = SNAP_TS[i];
      if (t === 0) continue;
      const f = traj[t];
      if (f) frames.push(f);
    }
    return frames;
  }

  let lastRunner = null;

  // Reverse with aggregate ε̂ AND with z chosen to recover x_{t-1} exactly.
  // We feed ε̂_t = aggregate ε̄_t = (x_t − √ᾱ_t·x_0)/√(1−ᾱ_t), then pick
  // z_t = (x_{t-1}_forward − μ_θ)/√β_t  so that the DDPM reverse-step lands
  // exactly on the bookkept x_{t-1}. This honours the hero formula and gives
  // perfect (not just in-distribution) recovery.
  function reverseBookkept(traj, x0) {
    const T_ = T;
    const D = traj[T_].length;
    const out = new Array(T_ + 1);
    out[T_] = new Float32Array(traj[T_]);
    for (let t = T_ - 1; t >= 0; t--) {
      const ab = DATA.alphaBars[t];
      const sa = Math.sqrt(ab), sb = Math.sqrt(1 - ab) || 1e-12;
      const xt = out[t + 1];
      // Aggregate ε̂_t — derived from the *current* x_t and known x_0.
      const epsHat = new Float32Array(D);
      for (let i = 0; i < D; i++) epsHat[i] = (xt[i] - sa * x0[i]) / sb;
      // μ_θ via DDPM reverse mean
      const inv = 1 / Math.sqrt(DATA.alphas[t]);
      const coef = DATA.betas[t] / sb;
      const mu = new Float32Array(D);
      for (let i = 0; i < D; i++) mu[i] = inv * (xt[i] - coef * epsHat[i]);
      // Compute z so that  μ_θ + σ_t·z  equals the bookkept x_{t-1} exactly.
      const sigma = (t > 0) ? Math.sqrt(DATA.betas[t]) : 0;
      const xprev = traj[t];
      const next = new Float32Array(D);
      if (sigma === 0) {
        for (let i = 0; i < D; i++) next[i] = mu[i];
      } else {
        // z[i] := (xprev[i] − μ[i]) / σ_t  ⇒  next[i] = μ[i] + σ_t · z[i] = xprev[i].
        // We don't materialise z; algebraically next[i] = xprev[i] either way.
        for (let i = 0; i < D; i++) {
          const z = (xprev[i] - mu[i]) / sigma;
          next[i] = mu[i] + sigma * z;
        }
      }
      out[t] = next;
    }
    return out;
  }

  function runBookkept() {
    lastRunner = 'bookkept';
    // Drive the reverse so that the implied z exactly cancels the gap between
    // μ_θ and the forward bookkept trajectory — this is the "we know everything"
    // baseline, the upper bound on what a perfect ε-predictor could achieve.
    const rev2DTraj = reverseBookkept(bookkept2D.trajectory, x0_2D);
    const revMnTraj = reverseBookkept(bookkeptMn.trajectory, x0_mnist);
    const rev2D = { trajectory: rev2DTraj };
    const revMn = { trajectory: revMnTraj };
    const x0r_2D = rev2D.trajectory[0];
    const x0r_mn = revMn.trajectory[0];

    paint2D(x0r_2D, trailFrames(rev2D.trajectory));
    setFinalClass('cluster-1');
    paintMnist(x0r_mn);

    verdict.innerHTML = '<em class="cluster-1">Result: perfect recovery.</em>';
    const m2 = mse(x0r_2D, x0_2D);
    const mp = mse(x0r_mn, x0_mnist);
    const r2 = relErr(x0r_2D, x0_2D);
    const rp = relErr(x0r_mn, x0_mnist);
    stats.innerHTML =
      `<span>2D MSE vs <span class="mono">x&#8320;</span>: <span class="mono">${fmt(m2)}</span> &middot; rel.err <span class="mono">${fmt(r2)}</span></span>` +
      `<span>Pixel MSE vs <span class="mono">x&#8320;</span>: <span class="mono">${fmt(mp)}</span> &middot; rel.err <span class="mono">${fmt(rp)}</span></span>`;

    window.diffusionShared._lastReverse = {
      kind: 'bookkept',
      rev2D: x0r_2D, revMn: x0r_mn,
      mse2D: m2, msePix: mp, rel2D: r2, relPix: rp
    };
  }

  function paintInitial() {
    // Show x_T noise in both panes (no trail).
    paint2D(xT_2D, []);
    setFinalClass('cluster-3');
    paintMnist(xT_mn);
    verdict.textContent = '';
    stats.textContent = '';
    lastRunner = null;
  }

  // ---- Step engine --------------------------------------------------------
  // Two cursors only:
  //   0 — show x_T noise + the formula + the why-not callout.
  //   1 — bookkept ε run (perfect recovery), closing caption appears.
  let cursor = 0;
  const STEPS = 2;

  function applyStep(c) {
    if (c === 0) {
      paintInitial();
      btnB.classList.remove('s4-arm-pulse');
      closing.classList.remove('visible');
    } else if (c === 1) {
      btnB.classList.add('s4-arm-pulse');
      runBookkept();
      closing.classList.add('visible');
    }
  }

  function setCursor(c) {
    if (c < 0 || c >= STEPS) return false;
    if (c === cursor) return true;
    if (c < cursor) {
      cursor = 0;
      applyStep(0);
      while (cursor < c) { cursor++; applyStep(cursor); }
    } else {
      while (cursor < c) { cursor++; applyStep(cursor); }
    }
    return true;
  }

  // ---- Event wiring -------------------------------------------------------
  btnB.addEventListener('click', () => {
    runBookkept();
    closing.classList.add('visible');
    if (cursor < 1) cursor = 1;
  });

  // ---- Init ----------------------------------------------------------------
  rebuildBookkept();
  paintInitial();

  function shouldAutoRunAll() { return /[#&?]runAll\b/.test(window.location.hash || ''); }
  function shouldAutoRun()    { return /[#&?]run\b/.test(window.location.hash || ''); }

  // Auto-run from cold-build: headless &run / &runAll lands on cursor 1
  // (bookkept recovery), which is the only "result" cursor now.
  if (shouldAutoRunAll() || shouldAutoRun()) {
    setTimeout(() => setCursor(1), 80);
  }

  return {
    onEnter() {
      rebuildBookkept();
      cursor = 0;
      applyStep(0);
      if (shouldAutoRunAll() || shouldAutoRun()) {
        setTimeout(() => setCursor(1), 80);
      }
    },
    onLeave() { /* no cleanup needed */ },
    onNextKey() { return setCursor(cursor + 1); },
    onPrevKey() {
      if (cursor === 0) return false;
      return setCursor(cursor - 1);
    },
  };
};
