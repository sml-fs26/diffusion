/* Scene 7 — Takeaways.
 * Three numbered claims, large serif, with one-line muted gloss each. A
 * trailing italic note flags what was skipped. If MNIST reference
 * trajectories are available, render a small reverse-process strip; if not,
 * the strip is omitted. */

window.scenes.scene7 = function (root) {

  /* ---- Static DOM ------------------------------------------------------ */
  const layout = document.createElement('div');
  layout.className = 'scene-layout center takeaways-layout';
  root.appendChild(layout);

  const stack = document.createElement('div');
  stack.className = 'takeaways-stack';
  layout.appendChild(stack);

  const heading = document.createElement('h2');
  heading.className = 'takeaways-heading';
  heading.textContent = 'Takeaways.';
  stack.appendChild(heading);

  const list = document.createElement('ol');
  list.className = 'takeaways-list';
  stack.appendChild(list);

  const claims = [
    {
      claim: 'Forward = fade + noise, T times.',
      gloss: 'Any image becomes essentially Gaussian after 200 steps.'
    },
    {
      claim: 'Reverse has a clean formula. It just needs &epsilon;.',
      gloss: "And we don't have &epsilon; for an image we never saw."
    },
    {
      claim: 'A neural net learns to guess &epsilon; from a single corrupted step.',
      gloss: "That's the whole magic. The rest is repeating."
    }
  ];

  claims.forEach((c, i) => {
    const li = document.createElement('li');
    li.className = 'takeaway-item';

    const num = document.createElement('div');
    num.className = 'takeaway-num';
    num.textContent = String(i + 1) + '.';
    li.appendChild(num);

    const body = document.createElement('div');
    body.className = 'takeaway-body';

    const claimEl = document.createElement('p');
    claimEl.className = 'takeaway-claim';
    claimEl.innerHTML = c.claim;
    body.appendChild(claimEl);

    const glossEl = document.createElement('p');
    glossEl.className = 'takeaway-gloss';
    glossEl.innerHTML = c.gloss;
    body.appendChild(glossEl);

    li.appendChild(body);
    list.appendChild(li);
  });

  /* ---- Optional reverse-trajectory strip -------------------------------
   * Build only if DATA.mnistReferenceTrajectories[0] yields ≥4 frames.
   * Each frame is a 28x28 grayscale array; we draw it via canvas. */
  function maybeBuildStrip() {
    const D = window.DATA;
    if (!D || !Array.isArray(D.mnistReferenceTrajectories) ||
        D.mnistReferenceTrajectories.length === 0) return;

    const traj = D.mnistReferenceTrajectories[0];
    if (!traj || !Array.isArray(traj.frames) || traj.frames.length < 4) return;

    // Pick 4 frames evenly across the reverse trajectory: noise → blur → faint → digit.
    const F = traj.frames.length;
    const idxs = [F - 1, Math.floor(F * 0.66), Math.floor(F * 0.33), 0];
    const frames = idxs.map(i => traj.frames[i]).filter(f => Array.isArray(f));
    if (frames.length < 4) return;

    const strip = document.createElement('div');
    strip.className = 'takeaways-strip';

    frames.forEach((frame, i) => {
      const cv = document.createElement('canvas');
      cv.className = 'takeaways-strip-frame';
      cv.width = 28;
      cv.height = 28;
      drawFrame(cv, frame);
      strip.appendChild(cv);
      if (i < frames.length - 1) {
        const arrow = document.createElement('span');
        arrow.className = 'takeaways-strip-arrow';
        arrow.textContent = '→';
        strip.appendChild(arrow);
      }
    });

    const caption = document.createElement('p');
    caption.className = 'takeaways-strip-caption';
    caption.textContent = 'The reverse process: pure noise becomes a digit.';
    strip.appendChild(caption);

    stack.appendChild(strip);
  }

  function drawFrame(canvas, frame) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    // frame may be Float32Array-like with 784 entries in [0,1] (or roughly
    // standardized). Clamp into [0, 255] to be safe.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < frame.length; i++) {
      if (frame[i] < lo) lo = frame[i];
      if (frame[i] > hi) hi = frame[i];
    }
    const range = (hi - lo) || 1;
    for (let i = 0; i < 28 * 28; i++) {
      const v = (frame[i] - lo) / range;
      const g = Math.max(0, Math.min(255, Math.round(v * 255)));
      img.data[4 * i + 0] = g;
      img.data[4 * i + 1] = g;
      img.data[4 * i + 2] = g;
      img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  maybeBuildStrip();

  // Trailing skipped-topics note.
  const skipped = document.createElement('p');
  skipped.className = 'takeaways-skipped';
  skipped.textContent =
    'We skipped: latent diffusion, conditional generation, U-Net architecture.';
  stack.appendChild(skipped);

  return {
    onEnter() { /* static layout — nothing to refresh */ },
    onLeave() { /* nothing to clean up */ },
    onNextKey() { return false; },
    onPrevKey() { return false; },
  };
};
