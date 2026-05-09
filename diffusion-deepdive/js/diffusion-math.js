/* DDPM math utilities — shared across all scenes.
 *
 * Notation matches the lecture (board-note slides for parts 4–8):
 *   x_t        — state at time t (2D point or flattened 28×28 image)
 *   β_t        — noise schedule step
 *   α_t        — 1 - β_t
 *   ᾱ_t        — Π_{s≤t} α_s
 *   ε          — Gaussian noise added in the forward process
 *   ε̂          — predicted noise (from a NN)
 *   z          — fresh Gaussian noise added in the reverse step
 *
 * All vector ops are length-agnostic (work for 2D and 784-d).
 * Schedule arrays live in window.DATA.{betas, alphas, alphaBars}.
 */

window.DiffusionMath = (function () {

  /* ---------- seeded RNG --------------------------------------------------- */

  // Mulberry32. Returns a function rng() → uniform [0, 1).
  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6D2B79F5) | 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Box–Muller standard normal sample.
  function randn(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Length-n Float32Array of i.i.d. N(0, 1) samples.
  function randnVector(rng, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = randn(rng);
    return out;
  }

  /* ---------- diffusion step formulae -------------------------------------- */

  // Forward one step:    x_{t+1} = √(1 − β_t) · x_t + √β_t · ε
  // x and eps must have the same length. Returns a new Float32Array.
  function forwardStep(x, beta_t, eps) {
    const a = Math.sqrt(1 - beta_t);
    const b = Math.sqrt(beta_t);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = a * x[i] + b * eps[i];
    return out;
  }

  // Fast-forward (single-shot):  x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε̂
  function fastForward(x0, alphaBar_t, epsHat) {
    const a = Math.sqrt(alphaBar_t);
    const b = Math.sqrt(1 - alphaBar_t);
    const out = new Float32Array(x0.length);
    for (let i = 0; i < x0.length; i++) out[i] = a * x0[i] + b * epsHat[i];
    return out;
  }

  // Reverse one step (DDPM, σ_t = √β_t):
  //   x_{t-1} = (1/√α_t) · ( x_t − β_t/√(1−ᾱ_t) · ε̂_t )  +  σ_t · z
  //   At t=0 the σ_t·z term is dropped (no fresh noise on the final step).
  function reverseStep(x_t, alpha_t, alphaBar_t, beta_t, epsHat, z, t) {
    const inv  = 1 / Math.sqrt(alpha_t);
    const coef = beta_t / Math.sqrt(1 - alphaBar_t);
    const sigma = (t > 0) ? Math.sqrt(beta_t) : 0;
    const out = new Float32Array(x_t.length);
    for (let i = 0; i < x_t.length; i++) {
      out[i] = inv * (x_t[i] - coef * epsHat[i]) + sigma * z[i];
    }
    return out;
  }

  /* ---------- time embedding (matches the trained MNIST model) ------------- */

  // Sinusoidal embedding, half sin / half cos, log-spaced frequencies.
  // dim must be even.
  function timeEmbedding(t, dim, maxPeriod) {
    maxPeriod = maxPeriod || 10000;
    const half = dim >>> 1;
    const out = new Float32Array(dim);
    for (let i = 0; i < half; i++) {
      const freq = Math.exp(-Math.log(maxPeriod) * i / half);
      out[i]        = Math.sin(t * freq);
      out[i + half] = Math.cos(t * freq);
    }
    return out;
  }

  /* ---------- helpers used by multiple scenes ------------------------------ */

  // L2 norm of a vector.
  function norm(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
  }

  // Element-wise zero-mean, unit-variance summary.
  function meanStd(v) {
    let m = 0;
    for (let i = 0; i < v.length; i++) m += v[i];
    m /= v.length;
    let s = 0;
    for (let i = 0; i < v.length; i++) s += (v[i] - m) * (v[i] - m);
    return { mean: m, std: Math.sqrt(s / v.length) };
  }

  return {
    mulberry32, randn, randnVector,
    forwardStep, fastForward, reverseStep,
    timeEmbedding,
    norm, meanStd
  };
})();
