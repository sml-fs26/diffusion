/* Scene 6 — Generate.
 *
 * Pedagogical goal: see the trained NN denoise pure noise into letter M
 * (live, using the model from scene 5) and into a digit (using the shipped
 * MNIST weights, when available).
 *
 * Layout: split-eq.
 *   LEFT  — 2D scatter (300 pts), x_T → … → x_0 reconstituting the M.
 *   RIGHT — 6 small 56×56 canvases showing snapshots from the MNIST reverse
 *           process at t ∈ {200, 160, 120, 80, 40, 0}, ordered top→bottom.
 *
 * Cold entry: if no scene-5 model, train a 200-step burst here so the 2D
 * pane still works. The MNIST model is used only when DATA.mnistModel is
 * non-null — otherwise we fall back to DATA.mnistReferenceTrajectories[0]
 * (when populated) or a "still preparing" placeholder.
 *
 * Step engine cursor 0..2:
 *   0  pre-generate. Both panels display N(0, I) noise.
 *   1  GENERATE pressed → both panels animate to completion.
 *   2  RE-ROLL enabled. Pressing → fresh seed, regenerate.
 */

window.scenes.scene6 = function (root) {

  const DATA = window.DATA;
  const M = window.DiffusionMath;
  const NN = window.DiffusionNN;

  /* ----- constants -------------------------------------------------------- */

  const T = DATA.T;
  const N_2D = DATA.letterM.points.length;       // 300
  const PX_DIM = 28 * 28;                        // 784
  const VIEW_LO = -3, VIEW_HI = 3;               // wider so x_T fits

  // Snapshot times for the MNIST strip (top → bottom: noise → digit).
  const MNIST_TS = [T, Math.round(T * 0.8), Math.round(T * 0.6),
                    Math.round(T * 0.4), Math.round(T * 0.2), 0];

  const HIDDEN     = 64;
  const LR         = 2e-3;
  const SEED_TRAIN = 7;
  const BATCH_SIZE = 64;
  const COLD_BURST = 200;

  /* ----- shared --------------------------------------------------------- */

  window.diffusionShared = window.diffusionShared || {};

  /* ----- state ----------------------------------------------------------- */

  const state = {
    cursor: 0,
    twoDModel: null,
    mnistModel: null,
    mnistMode: 'unavailable',     // 'live' | 'ref' | 'unavailable'
    seed: 1234,
    generating: false,
    rafId: null,
    // 2D
    twoDState: null,              // current Float32Array length 600
    twoDStep: T,                  // current t (T → 0)
    twoDRng: null,
    // MNIST
    mnistTrajectory: null,        // Array of Float32Array OR null
    mnistSnapshots: new Array(MNIST_TS.length).fill(null),
  };

  /* ----- DOM scaffolding ------------------------------------------------- */

  root.innerHTML = '';
  root.classList.add('scene-s6');

  const layout = document.createElement('div');
  layout.className = 'scene-layout split-eq s6-layout';
  root.appendChild(layout);

  /* LEFT — 2D ------------------------------------------------------------- */
  const leftCol = document.createElement('div');
  leftCol.className = 's6-left';
  layout.appendChild(leftCol);

  const leftLabel = document.createElement('div');
  leftLabel.className = 's6-pane-label';
  leftLabel.textContent = 'From noise to M.';
  leftCol.appendChild(leftLabel);

  const leftWrap = document.createElement('div');
  leftWrap.className = 's6-2d-wrap viz-wrap';
  leftCol.appendChild(leftWrap);

  const svg2D = d3.select(leftWrap).append('svg')
    .attr('class', 's6-2d-svg')
    .attr('viewBox', `${VIEW_LO} ${VIEW_LO} ${VIEW_HI - VIEW_LO} ${VIEW_HI - VIEW_LO}`)
    .attr('preserveAspectRatio', 'xMidYMid meet');

  // Faint axes
  const gFrame = svg2D.append('g').attr('class', 's6-frame');
  gFrame.append('line').attr('class', 's6-frame-axis')
    .attr('x1', VIEW_LO).attr('y1', 0).attr('x2', VIEW_HI).attr('y2', 0);
  gFrame.append('line').attr('class', 's6-frame-axis')
    .attr('x1', 0).attr('y1', VIEW_LO).attr('x2', 0).attr('y2', VIEW_HI);

  const gPoints = svg2D.append('g').attr('class', 's6-2d-points');

  const leftCaption = document.createElement('p');
  leftCaption.className = 's6-pane-caption muted';
  leftCaption.innerHTML = 'x<sub>T</sub> ~ N(0, I), reverse 200 steps.';
  leftCol.appendChild(leftCaption);

  // Warming-up overlay (shown during cold-entry training burst).
  const warmOverlay = document.createElement('div');
  warmOverlay.className = 's6-warm-overlay s6-hidden';
  warmOverlay.innerHTML = `
    <div class="s6-warm-card">
      <div class="s6-warm-title">Warming up…</div>
      <div class="s6-warm-sub muted">No model from scene 5 — training a 200-step burst.</div>
    </div>
  `;
  leftWrap.appendChild(warmOverlay);

  /* RIGHT — MNIST strip --------------------------------------------------- */
  const rightCol = document.createElement('div');
  rightCol.className = 's6-right';
  layout.appendChild(rightCol);

  const rightLabel = document.createElement('div');
  rightLabel.className = 's6-pane-label';
  rightLabel.textContent = 'From noise to digit.';
  rightCol.appendChild(rightLabel);

  const stripWrap = document.createElement('div');
  stripWrap.className = 's6-mnist-strip';
  rightCol.appendChild(stripWrap);

  // 6 rows: small canvas + tiny t-label.
  const mnistCanvases = [];
  MNIST_TS.forEach((tVal) => {
    const row = document.createElement('div');
    row.className = 's6-mnist-row';

    const cv = document.createElement('canvas');
    cv.className = 'mnist-pane s6-mnist-canvas';
    cv.width = 56;
    cv.height = 56;
    row.appendChild(cv);

    const label = document.createElement('span');
    label.className = 's6-mnist-tlabel mono';
    label.textContent = `t = ${tVal}`;
    row.appendChild(label);

    stripWrap.appendChild(row);
    mnistCanvases.push(cv);
  });

  const mnistStatus = document.createElement('p');
  mnistStatus.className = 's6-pane-caption muted';
  mnistStatus.textContent = '';
  rightCol.appendChild(mnistStatus);

  /* CONTROLS ------------------------------------------------------------- */

  const ctrlBar = document.createElement('div');
  ctrlBar.className = 's6-ctrls';
  ctrlBar.innerHTML = `
    <button class="btn primary s6-gen-btn" data-role="gen">Generate</button>
    <button class="btn s6-roll-btn" data-role="roll" disabled>Re-roll seed</button>
    <span class="s6-seed-pill mono muted">seed: <span data-role="seed">${state.seed}</span></span>
  `;
  root.appendChild(ctrlBar);

  const genBtn  = ctrlBar.querySelector('[data-role="gen"]');
  const rollBtn = ctrlBar.querySelector('[data-role="roll"]');
  const seedEl  = ctrlBar.querySelector('[data-role="seed"]');

  // Closing caption
  const closing = document.createElement('p');
  closing.className = 's6-closing muted';
  closing.innerHTML = 'Same NN. Same recipe. New samples on every roll.';
  root.appendChild(closing);

  /* ----- helpers --------------------------------------------------------- */

  function makeRng(seed) {
    return M.mulberry32(seed >>> 0);
  }

  function sampleNoise(rng, n) {
    return M.randnVector(rng, n);
  }

  /* ----- 2D model setup -------------------------------------------------- */

  function ensureTwoDModel() {
    // Prefer the trained model from scene 5.
    if (window.diffusionShared && window.diffusionShared.twoDModel) {
      state.twoDModel = window.diffusionShared.twoDModel;
      return Promise.resolve();
    }
    // Cold path — train a 200-step burst here.
    warmOverlay.classList.remove('s6-hidden');
    return new Promise(resolve => {
      // Defer to next frame so the overlay paints before we block.
      requestAnimationFrame(() => {
        const m = new NN.TwoDModel({ hidden: HIDDEN, lr: LR, seed: SEED_TRAIN });
        const rng = M.mulberry32(SEED_TRAIN + 9001);
        for (let s = 0; s < COLD_BURST; s++) {
          const batch = NN.sample2DBatch(DATA.letterM.points, DATA.alphaBars, BATCH_SIZE, rng);
          m.trainBatch(batch);
        }
        state.twoDModel = m;
        window.diffusionShared.twoDModel = m;
        window.diffusionShared.lossHistory = m.lossHistory.slice();
        warmOverlay.classList.add('s6-hidden');
        resolve();
      });
    });
  }

  /* ----- MNIST model setup ----------------------------------------------- */

  function ensureMnistModel() {
    if (state.mnistMode && state.mnistMode !== 'unavailable') return;

    // Preferred path: a pool of pre-recorded UNet trajectories (the production
    // setup). Each entry is { seed, snapshots: [{ t, pixels }] }. Browser
    // plays them back; "re-roll seed" cycles through the pool.
    if (Array.isArray(DATA.mnistReferenceTrajectories)
        && DATA.mnistReferenceTrajectories.length > 0
        && Array.isArray(DATA.mnistReferenceTrajectories[0].snapshots)
        && DATA.mnistReferenceTrajectories[0].snapshots.length > 0) {
      state.mnistMode = 'pool';
      state.mnistPoolIdx = 0;
      return;
    }

    // Legacy path: a live MNIST denoising model. Kept for backward-compat;
    // new builds set DATA.mnistModel = null.
    if (DATA.mnistModel && DATA.mnistModel.architecture && DATA.mnistModel.weights) {
      try {
        state.mnistModel = new NN.MNISTModel(DATA.mnistModel);
        state.mnistMode = 'live';
        return;
      } catch (e) {
        console.warn('MNISTModel failed to load:', e);
      }
    }

    state.mnistMode = 'unavailable';
  }

  // Build a length-(T+1) array indexed by t from a sparse {t, pixels}
  // snapshot list. For t values not present in the snapshots, picks the
  // nearest snapshot (no interpolation — keeps each frame a real model
  // output rather than a lerp between two).
  function densifyFromSnapshots(snapshots) {
    const out = new Array(T + 1);
    const sorted = snapshots.slice().sort((a, b) => a.t - b.t);
    for (let t = 0; t <= T; t++) {
      let best = sorted[0];
      let bd = Math.abs(t - best.t);
      for (let i = 1; i < sorted.length; i++) {
        const d = Math.abs(t - sorted[i].t);
        if (d < bd) { bd = d; best = sorted[i]; }
      }
      out[t] = Float32Array.from(best.pixels);
    }
    return out;
  }

  /* ----- 2D rendering ---------------------------------------------------- */

  // Build initial scatter shells (300 circles).
  const init2D = new Array(N_2D).fill(0).map((_, i) => ({ i, x: 0, y: 0 }));
  gPoints.selectAll('circle.point')
    .data(init2D, d => d.i)
    .enter()
    .append('circle')
    .attr('class', 'point cluster-1 s6-2d-pt')
    .attr('r', 0.04)
    .attr('cx', 0).attr('cy', 0);

  function paint2D() {
    if (!state.twoDState) return;
    const sel = gPoints.selectAll('circle.point');
    sel.attr('cx', (d, i) => state.twoDState[2 * i])
       .attr('cy', (d, i) => -state.twoDState[2 * i + 1]);   // flip y
  }

  function reset2DToNoise() {
    const rng = makeRng(state.seed);
    state.twoDState = sampleNoise(rng, N_2D * 2);
    state.twoDStep = T;
    state.twoDRng = rng;
    paint2D();
  }

  /* ----- 2D generation step (one reverse step per RAF) ------------------- */

  // εFn: row by row — for each (x, y) pair, predict via twoDModel.
  function predict2DEpsBatch(xt, t) {
    const tn = (T <= 1) ? 0 : t / (T - 1);
    const out = new Float32Array(xt.length);
    for (let i = 0; i < N_2D; i++) {
      const eh = state.twoDModel.predict(xt[2 * i], xt[2 * i + 1], tn);
      out[2 * i]     = eh[0];
      out[2 * i + 1] = eh[1];
    }
    return out;
  }

  function step2DOnce() {
    if (state.twoDStep <= 0) return false;
    // Going from x_{tIdx+1} → x_{tIdx}. tIdx is the schedule index used in
    // reverseStep (matching reverseTrajectory semantics).
    const tIdx = state.twoDStep - 1;
    const epsHat = predict2DEpsBatch(state.twoDState, tIdx);
    const z = sampleNoise(state.twoDRng, state.twoDState.length);
    const next = M.reverseStep(
      state.twoDState,
      DATA.alphas[tIdx],
      DATA.alphaBars[tIdx],
      DATA.betas[tIdx],
      epsHat,
      z,
      tIdx
    );
    state.twoDState = next;
    state.twoDStep = state.twoDStep - 1;
    return true;
  }

  /* ----- MNIST trajectory ------------------------------------------------ */

  // Compute the full reverse trajectory once per generation. ~200 forward
  // passes through a 4-layer MLP — well under a second on a laptop.
  function computeMnistTrajectory() {
    if (state.mnistMode === 'live' && state.mnistModel) {
      const rng = makeRng(state.seed + 99);
      const xT = sampleNoise(rng, PX_DIM);
      const epsFn = (xt, t) => state.mnistModel.forward(xt, t);
      const result = NN.reverseTrajectory(xT, epsFn, DATA.betas, DATA.alphas, DATA.alphaBars, rng);
      // result.trajectory[t] = x_t
      state.mnistTrajectory = result.trajectory;
    } else if (state.mnistMode === 'pool') {
      // Pool playback: pick a trajectory by index (cycles on re-roll), expand
      // its sparse snapshots into a full length-(T+1) array indexed by t.
      const pool = DATA.mnistReferenceTrajectories;
      const idx = ((state.mnistPoolIdx | 0) % pool.length + pool.length) % pool.length;
      state.mnistTrajectory = densifyFromSnapshots(pool[idx].snapshots);
    } else if (state.mnistMode === 'ref') {
      // Legacy single-trajectory path with .frames array — kept for backwards
      // compat with the original CPU-MLP build. Modern builds use 'pool'.
      const ref = DATA.mnistReferenceTrajectories[0];
      const F = ref.frames.length;
      const traj = new Array(T + 1);
      for (let t = 0; t <= T; t++) {
        const frac = (T - t) / T;
        const fi = Math.min(F - 1, Math.max(0, Math.round(frac * (F - 1))));
        traj[t] = Float32Array.from(ref.frames[fi]);
      }
      state.mnistTrajectory = traj;
    } else {
      state.mnistTrajectory = null;
    }
  }

  function drawMnistFrame(canvas, frame /* Float32Array(784), values approx in [-1, 1] */) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    if (!frame) {
      // muted black canvas
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }
    // Use a tiny offscreen 28x28 image.
    if (!drawMnistFrame._off) {
      const off = document.createElement('canvas');
      off.width = 28; off.height = 28;
      drawMnistFrame._off = off;
      drawMnistFrame._offCtx = off.getContext('2d');
      drawMnistFrame._offImg = drawMnistFrame._offCtx.createImageData(28, 28);
    }
    const off = drawMnistFrame._off;
    const offCtx = drawMnistFrame._offCtx;
    const img = drawMnistFrame._offImg;
    const data = img.data;
    // Map [-1, 1] → [0, 255]; clamp out-of-range so noise frames remain legible.
    for (let i = 0; i < 28 * 28; i++) {
      let v = (frame[i] + 1) * 127.5;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      const j = i * 4;
      data[j] = v; data[j + 1] = v; data[j + 2] = v; data[j + 3] = 255;
    }
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function paintMnistSnapshots() {
    for (let i = 0; i < MNIST_TS.length; i++) {
      drawMnistFrame(mnistCanvases[i], state.mnistSnapshots[i]);
    }
  }

  function resetMnistToNoise() {
    const rng = makeRng(state.seed + 99);
    const noise = sampleNoise(rng, PX_DIM);
    state.mnistSnapshots = MNIST_TS.map((t, i) => i === 0 ? noise : null);
    paintMnistSnapshots();
  }

  function fillMnistFromTrajectory() {
    if (!state.mnistTrajectory) {
      state.mnistSnapshots = MNIST_TS.map(() => null);
      return;
    }
    state.mnistSnapshots = MNIST_TS.map(t => {
      const tt = Math.max(0, Math.min(T, t));
      return state.mnistTrajectory[tt];
    });
  }

  function setMnistStatus() {
    if (state.mnistMode === 'live') {
      mnistStatus.innerHTML = '784-d MLP, shipped weights. Same recipe.';
    } else if (state.mnistMode === 'pool') {
      const n = (DATA.mnistReferenceTrajectories || []).length;
      mnistStatus.innerHTML = `Pre-recorded UNet samples (${n} seeds in the pool).`;
    } else if (state.mnistMode === 'ref') {
      mnistStatus.innerHTML = 'Reference trajectory (legacy).';
    } else {
      mnistStatus.innerHTML = 'MNIST model unavailable.';
    }
  }

  /* ----- generation loop ------------------------------------------------- */

  function generate() {
    if (state.generating) return;
    if (!state.twoDModel) return;

    // Reset 2D to white noise; clear MNIST strip.
    reset2DToNoise();

    // Pre-compute the entire MNIST trajectory now (cheap).
    computeMnistTrajectory();

    state.generating = true;
    genBtn.disabled = true;
    rollBtn.disabled = true;

    // We animate the 2D scatter step by step. The MNIST strip fills in one
    // canvas at a time, paced to land at the same time as the 2D animation.
    const totalSteps = T;
    let stepsTaken = 0;
    // Steps per RAF — keep ≤ 3 so animation reads cleanly. 200 steps × ~16 ms ~= 3 s.
    const stepsPerFrame = 1;
    // Reveal MNIST snapshots progressively as 2D progresses.
    const mnistRevealAt = MNIST_TS.map((t, i) => {
      // For i = 0 (t=T) we already paint at start. The remaining 5 land at
      // evenly-spaced fractions through the 2D animation.
      return Math.round(((i + 1) / MNIST_TS.length) * totalSteps);
    });
    // Pre-paint the t=T noise frame (i=0) immediately if available.
    if (state.mnistTrajectory) {
      state.mnistSnapshots[0] = state.mnistTrajectory[T];
      drawMnistFrame(mnistCanvases[0], state.mnistSnapshots[0]);
    } else {
      // unavailable
      state.mnistSnapshots[0] = null;
      drawMnistFrame(mnistCanvases[0], null);
    }

    function onStep() {
      for (let k = 0; k < stepsPerFrame && state.twoDStep > 0; k++) {
        step2DOnce();
        stepsTaken++;
        // Reveal MNIST canvases at thresholds.
        for (let i = 1; i < MNIST_TS.length; i++) {
          if (state.mnistSnapshots[i] == null && stepsTaken >= mnistRevealAt[i]) {
            const tt = Math.max(0, Math.min(T, MNIST_TS[i]));
            state.mnistSnapshots[i] = state.mnistTrajectory ? state.mnistTrajectory[tt] : null;
            drawMnistFrame(mnistCanvases[i], state.mnistSnapshots[i]);
          }
        }
      }
      paint2D();

      if (state.twoDStep > 0) {
        state.rafId = requestAnimationFrame(onStep);
      } else {
        // Final fill — make sure all MNIST snapshots are present.
        for (let i = 0; i < MNIST_TS.length; i++) {
          if (state.mnistSnapshots[i] == null) {
            const tt = Math.max(0, Math.min(T, MNIST_TS[i]));
            state.mnistSnapshots[i] = state.mnistTrajectory ? state.mnistTrajectory[tt] : null;
            drawMnistFrame(mnistCanvases[i], state.mnistSnapshots[i]);
          }
        }
        state.generating = false;
        genBtn.disabled = false;
        rollBtn.disabled = false;
        if (state.cursor < 2) {
          state.cursor = 2;
        }
      }
    }

    state.rafId = requestAnimationFrame(onStep);
  }

  function cancelGenerationLoop() {
    if (state.rafId != null) {
      cancelAnimationFrame(state.rafId);
      state.rafId = null;
    }
    state.generating = false;
    genBtn.disabled = false;
    // rollBtn enable depends on whether we've generated at least once.
  }

  /* ----- buttons --------------------------------------------------------- */

  genBtn.addEventListener('click', () => {
    if (state.generating) return;
    if (!state.twoDModel) return;
    if (state.cursor === 0) state.cursor = 1;
    generate();
  });

  rollBtn.addEventListener('click', () => {
    if (state.generating) return;
    state.seed = (state.seed * 1664525 + 1013904223) >>> 0;
    // Advance through the pre-recorded UNet pool so each click yields a
    // visibly different digit (the seed alone only re-randomises the 2D
    // pane; the MNIST pane is playback-driven).
    if (state.mnistMode === 'pool') {
      state.mnistPoolIdx = ((state.mnistPoolIdx | 0) + 1);
      // Update the seed pill to mirror the trajectory's seed, so the audience
      // sees a meaningful number rather than a derived RNG.
      const pool = DATA.mnistReferenceTrajectories;
      const i = state.mnistPoolIdx % pool.length;
      seedEl.textContent = String(pool[i].seed);
    } else {
      seedEl.textContent = String(state.seed);
    }
    generate();
  });

  /* ----- onEnter / onLeave ----------------------------------------------- */

  function paintInitialState() {
    // Paint noise on both panes for cursor 0 entry.
    reset2DToNoise();
    resetMnistToNoise();
    setMnistStatus();
  }

  // Headless verification hook.
  function shouldAutoRun() {
    return /[#&?]run\b/.test(window.location.hash || '');
  }

  // For headless: synchronously walk the entire reverse trajectory and paint
  // the final state. No animation.
  function instantGenerate() {
    if (!state.twoDModel) return;
    reset2DToNoise();
    while (state.twoDStep > 0) step2DOnce();
    paint2D();
    computeMnistTrajectory();
    fillMnistFromTrajectory();
    paintMnistSnapshots();
    state.cursor = 2;
    rollBtn.disabled = false;
  }

  function fullEnter() {
    cancelGenerationLoop();
    state.cursor = 0;
    state.seed = 1234;
    seedEl.textContent = String(state.seed);
    rollBtn.disabled = true;

    // Paint *something* immediately so the user isn't staring at an empty
    // pane during the cold-burst training.
    paintInitialState();

    ensureMnistModel();
    setMnistStatus();

    ensureTwoDModel().then(() => {
      // After cold-burst (or warm short-circuit), repaint with proper noise
      // sample.  No auto-generate — user clicks GENERATE.
      paintInitialState();
      if (shouldAutoRun()) {
        // Headless: train more for a recognizable M, then jump to final state.
        if (state.twoDModel && state.twoDModel.step < 800) {
          const rng = M.mulberry32(SEED_TRAIN + 9001 + state.twoDModel.step);
          for (let s = 0; s < 600; s++) {
            const batch = NN.sample2DBatch(DATA.letterM.points, DATA.alphaBars, BATCH_SIZE, rng);
            state.twoDModel.trainBatch(batch);
          }
        }
        instantGenerate();
      }
    });
  }

  /* ----- step engine glue ------------------------------------------------ */

  function setCursor(c) {
    if (c < 0 || c > 2) return false;
    if (c === state.cursor) return false;

    if (c > state.cursor) {
      if (state.cursor === 0 && c >= 1) {
        // Trigger Generate (button equivalent).
        if (state.twoDModel && !state.generating) {
          state.cursor = 1;
          generate();
          return true;
        }
        return false;
      }
      if (state.cursor === 1 && c >= 2) {
        // If still animating, wait; else just bump cursor.
        if (state.generating) return true;
        state.cursor = 2;
        return true;
      }
    } else {
      // Going back.
      if (c === 0) {
        cancelGenerationLoop();
        state.cursor = 0;
        rollBtn.disabled = true;
        paintInitialState();
        return true;
      }
      if (c === 1) {
        state.cursor = 1;
        return true;
      }
    }
    return false;
  }

  /* ----- initial paint --------------------------------------------------- */

  fullEnter();

  return {
    onEnter() {
      fullEnter();
    },
    onLeave() {
      cancelGenerationLoop();
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
