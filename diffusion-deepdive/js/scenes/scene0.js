/* Scene 0 — Title hero.
 * Big serif "Diffusion." with an italic tagline. Behind the title sits a
 * subtle ambient texture: a small cloud of letter-M points slowly diffusing
 * forward (DDPM forward step) and restarting every ~10 seconds. The texture
 * is opacity 0.15 so the title carries the page; if rendering fails we just
 * skip it.
 *
 * Honors the &run / shouldAutoRun convention: when the URL hash contains the
 * `run` flag we paint the final state immediately so headless screenshots
 * don't catch a mid-fade. */

window.scenes.scene0 = function (root) {

  /* ---- shouldAutoRun helper -------------------------------------------- */
  function shouldAutoRun() {
    return /[#&?]run\b/.test(window.location.hash || '');
  }

  /* ---- Build static DOM ------------------------------------------------ */
  root.classList.add('scene-hero');

  const stage = document.createElement('div');
  stage.className = 'hero-stage';
  root.appendChild(stage);

  // Background SVG: ambient diffusing-M texture.
  const bgSvg = d3.select(stage)
    .append('svg')
    .attr('class', 'hero-bg')
    .attr('preserveAspectRatio', 'xMidYMid slice')
    .attr('viewBox', '-1.4 -1.0 2.8 2.0');

  const gPoints = bgSvg.append('g').attr('class', 'g-points');

  // Foreground stack: title + tagline + CTA.
  const stack = document.createElement('div');
  stack.className = 'hero-stack';
  stage.appendChild(stack);

  const h1 = document.createElement('h1');
  h1.className = 'hero-title';
  h1.textContent = 'Diffusion.';
  stack.appendChild(h1);

  const tagline = document.createElement('p');
  tagline.className = 'hero-tagline';
  tagline.textContent = 'How a model that knows nothing learns to dream.';
  stack.appendChild(tagline);

  const cta = document.createElement('p');
  cta.className = 'hero-cta';
  cta.innerHTML = 'Press <kbd>→</kbd> to begin.';
  stack.appendChild(cta);

  /* ---- Background diffusion animation ---------------------------------- */
  // Subsample letter-M to ~180 points so the SVG stays cheap.
  let bgState = null;       // current Float32Array, length 2N (interleaved x,y)
  let bgInitial = null;     // initial state to restart from
  let bgT = 0;              // current diffusion step (0..T-1)
  let rafId = null;
  let cycleId = null;
  let frameCount = 0;
  let stepEvery = 4;        // advance one diffusion step every N frames (~15 fps)

  function buildBg() {
    if (!window.DATA || !window.DATA.letterM || !Array.isArray(window.DATA.letterM.points)) {
      return false;
    }
    const src = window.DATA.letterM.points;
    // Subsample deterministically — every 2nd point, capped at 180.
    const stride = Math.max(1, Math.floor(src.length / 180));
    const sub = [];
    for (let i = 0; i < src.length; i += stride) {
      const p = src[i];
      if (Array.isArray(p) && p.length >= 2) sub.push(p[0], p[1]);
    }
    const N = sub.length / 2;
    if (N < 8) return false;

    bgInitial = new Float32Array(sub);
    bgState = new Float32Array(bgInitial);
    bgT = 0;

    const data = new Array(N).fill(0).map((_, i) => i);
    const sel = gPoints.selectAll('circle.point').data(data);
    sel.exit().remove();
    sel.enter()
      .append('circle')
      .attr('class', 'point cluster-1')
      .attr('r', 0.012)
      .merge(sel)
      .attr('cx', i => bgState[2 * i])
      .attr('cy', i => -bgState[2 * i + 1]);  // flip y so M is upright

    return true;
  }

  function paintBg() {
    if (!bgState) return;
    const N = bgState.length / 2;
    gPoints.selectAll('circle.point')
      .attr('cx', (_, i) => bgState[2 * i])
      .attr('cy', (_, i) => -bgState[2 * i + 1]);
  }

  function stepDiffusion() {
    if (!bgState || !window.DiffusionMath || !window.DATA || !window.DATA.betas) return;
    const betas = window.DATA.betas;
    if (bgT >= betas.length) return;
    const beta_t = betas[bgT];
    // Generate fresh ε ~ N(0, I) for each point. We don't need a seeded RNG
    // here — this is ambient texture, not data the student inspects.
    const eps = new Float32Array(bgState.length);
    for (let i = 0; i < eps.length; i++) {
      // Box–Muller without the rejection-sampling guard — Math.random() never
      // returns 0 in practice; if it does we'd just emit a 0, which is fine.
      const u = Math.random() || 1e-9;
      const v = Math.random() || 1e-9;
      eps[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    bgState = window.DiffusionMath.forwardStep(bgState, beta_t, eps);
    bgT++;
  }

  function restartBg() {
    if (!bgInitial) return;
    bgState = new Float32Array(bgInitial);
    bgT = 0;
    paintBg();
  }

  function tick() {
    rafId = requestAnimationFrame(tick);
    frameCount++;
    if (frameCount % stepEvery !== 0) return;
    stepDiffusion();
    paintBg();
  }

  function startBg() {
    if (!buildBg()) return;
    paintBg();
    rafId = requestAnimationFrame(tick);
    cycleId = setInterval(restartBg, 10000);
  }

  function stopBg() {
    if (rafId != null) { cancelAnimationFrame(rafId); rafId = null; }
    if (cycleId != null) { clearInterval(cycleId); cycleId = null; }
  }

  /* ---- Foreground entrance --------------------------------------------- */
  let entranceTimers = [];

  function clearTimers() {
    entranceTimers.forEach(t => clearTimeout(t));
    entranceTimers = [];
  }

  function paintFinal() {
    h1.style.transition = 'none';
    tagline.style.transition = 'none';
    h1.style.opacity = 1;
    h1.style.transform = 'translateY(0)';
    tagline.style.opacity = 1;
    tagline.style.transform = 'translateY(0)';
    cta.classList.add('visible');
    // Force the scene container itself to its final opacity so a headless
    // screenshot doesn't catch the scene-engine's 400ms cross-fade mid-way.
    // (Dev affordance only — gated on &run.)
    root.style.transition = 'none';
    root.style.opacity = 1;
  }

  function playEntrance() {
    clearTimers();

    if (shouldAutoRun()) {
      paintFinal();
      return;
    }

    // Reset starting state for replay.
    h1.style.transition = 'none';
    tagline.style.transition = 'none';
    cta.classList.remove('visible');
    h1.style.opacity = 0;
    h1.style.transform = 'translateY(8px)';
    tagline.style.opacity = 0;
    tagline.style.transform = 'translateY(8px)';

    // Force a reflow so the next transition takes effect.
    // eslint-disable-next-line no-unused-expressions
    h1.offsetHeight;

    entranceTimers.push(setTimeout(() => {
      h1.style.transition = 'opacity 700ms ease-out, transform 700ms ease-out';
      h1.style.opacity = 1;
      h1.style.transform = 'translateY(0)';
    }, 60));

    entranceTimers.push(setTimeout(() => {
      tagline.style.transition = 'opacity 700ms ease-out, transform 700ms ease-out';
      tagline.style.opacity = 1;
      tagline.style.transform = 'translateY(0)';
    }, 320));

    entranceTimers.push(setTimeout(() => {
      cta.classList.add('visible');
    }, 1100));
  }

  // Initial paint.
  startBg();
  playEntrance();

  return {
    onEnter() {
      // Restart background and replay entrance on revisit.
      stopBg();
      startBg();
      playEntrance();
    },
    onLeave() {
      clearTimers();
      stopBg();
    },
    onNextKey() { return false; },
    onPrevKey() { return false; },
  };
};
