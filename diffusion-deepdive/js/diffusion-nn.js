/* Neural-network modules for the diffusion deep-dive.
 *
 *   TwoDModel   — small MLP for the 2D toy. Trained LIVE in the browser
 *                 in scene 5 ("Train the oracle"); used in scene 6 to
 *                 generate by reversing from white noise.
 *
 *   MNISTModel  — pure inference. Loads weights produced offline by
 *                 precompute/train_mnist_ddpm.py and embedded into
 *                 data/datasets.js. Architecture: 3-layer MLP with SiLU.
 *
 * Both use DiffusionMath for time embeddings and RNG.
 */

window.DiffusionNN = (function () {

  /* ---------- low-level ops ----------------------------------------------- */

  // Row-major linear: out[m] = b[m] + Σ_k W[m, k] · x[k]
  // W stored row-major [out_dim × in_dim]. Used by the 2D MLP (live training).
  function linear(W, b, x, in_dim, out_dim, out) {
    out = out || new Float32Array(out_dim);
    for (let m = 0; m < out_dim; m++) {
      let s = b[m];
      const row = m * in_dim;
      for (let k = 0; k < in_dim; k++) s += W[row + k] * x[k];
      out[m] = s;
    }
    return out;
  }

  // Col-major linear: out[j] = b[j] + Σ_i x[i] · W[i, j]
  // W stored col-major [in_dim × out_dim]  (W[i*out_dim + j]).
  // Used by MNISTModel — matches the convention exported by precompute/train_mnist_ddpm.py
  // (PyTorch fc.weight is [out, in]; the script transposes via .T before flattening).
  // Cache-friendlier than row-major for sequential scan over outputs.
  function linearCol(W, b, x, in_dim, out_dim, out) {
    out = out || new Float32Array(out_dim);
    for (let j = 0; j < out_dim; j++) out[j] = b[j];
    for (let i = 0; i < in_dim; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * out_dim;
      for (let j = 0; j < out_dim; j++) out[j] += xi * W[row + j];
    }
    return out;
  }

  function siluInPlace(a) {
    for (let i = 0; i < a.length; i++) {
      const v = a[i];
      a[i] = v / (1 + Math.exp(-v));
    }
    return a;
  }

  function silu(z) { return z / (1 + Math.exp(-z)); }
  function siluDeriv(z) {
    const s = 1 / (1 + Math.exp(-z));
    return s + z * s * (1 - s);
  }

  /* ---------- MNIST inference --------------------------------------------- */

  // MNISTModel — pure inference. Architecture (set by precompute/train_mnist_ddpm.py):
  //   emb = sinusoidal(t, timeEmbedDim)
  //   h   = SiLU( W1 @ concat(x_t, emb) + b1 )           (816 → H)
  //   h   = SiLU( W2 @ concat(h,   emb) + b2 )           (H+32 → H)   ← skip-time
  //   h   = SiLU( W3 @ concat(h,   emb) + b3 )           (H+32 → H)   ← skip-time
  //   ε̂  =        W4 @ h + b4                            (H → 784)
  // Weights are col-major (W[i, j] = W[i*out_dim + j]). architecture.timeSkipConditioning
  // controls whether layers 2 and 3 receive the time embedding (true = the trained 4-layer net,
  // false = the simpler 3-layer fallback used by the placeholder).
  // Forward signature: model.forward(x_t /* Float32Array length inputDim */, t /* int */) → Float32Array length inputDim.
  class MNISTModel {
    constructor(modelData) {
      const arch = modelData.architecture;
      this.inputDim = arch.inputDim;
      this.timeDim  = arch.timeEmbedDim;
      this.hidden   = arch.hiddenDim;
      this.numLayers = arch.numLayers || 3;
      this.skipTime  = arch.timeSkipConditioning === true;
      this.maxPeriod = arch.timeEmbedMaxPeriod || 10000;

      const W = modelData.weights;
      this.W1 = new Float32Array(W.W1); this.b1 = new Float32Array(W.b1);
      this.W2 = new Float32Array(W.W2); this.b2 = new Float32Array(W.b2);
      this.W3 = new Float32Array(W.W3); this.b3 = new Float32Array(W.b3);
      if (this.numLayers >= 4) {
        this.W4 = new Float32Array(W.W4); this.b4 = new Float32Array(W.b4);
      }

      // reusable buffers
      this._inEmb  = new Float32Array(this.inputDim + this.timeDim);
      this._hEmb   = new Float32Array(this.hidden + this.timeDim);
      this._h      = new Float32Array(this.hidden);
      this._h2     = new Float32Array(this.hidden);
      this._out    = new Float32Array(this.inputDim);
    }

    forward(x_t, t) {
      const inDim = this.inputDim, H = this.hidden, tDim = this.timeDim;
      const tEmb = DiffusionMath.timeEmbedding(t, tDim, this.maxPeriod);

      // Layer 1: concat(x_t, tEmb) → H
      for (let i = 0; i < inDim; i++) this._inEmb[i] = x_t[i];
      for (let i = 0; i < tDim;  i++) this._inEmb[inDim + i] = tEmb[i];
      linearCol(this.W1, this.b1, this._inEmb, inDim + tDim, H, this._h);
      siluInPlace(this._h);

      if (this.numLayers >= 4 && this.skipTime) {
        // Layer 2: concat(h, tEmb) → H
        for (let i = 0; i < H;    i++) this._hEmb[i]       = this._h[i];
        for (let i = 0; i < tDim; i++) this._hEmb[H + i]   = tEmb[i];
        linearCol(this.W2, this.b2, this._hEmb, H + tDim, H, this._h2);
        siluInPlace(this._h2);

        // Layer 3: concat(h2, tEmb) → H
        for (let i = 0; i < H;    i++) this._hEmb[i]     = this._h2[i];
        for (let i = 0; i < tDim; i++) this._hEmb[H + i] = tEmb[i];
        linearCol(this.W3, this.b3, this._hEmb, H + tDim, H, this._h);
        siluInPlace(this._h);

        // Layer 4: H → 784
        linearCol(this.W4, this.b4, this._h, H, inDim, this._out);
      } else {
        // 3-layer fallback (no skip-time)
        linearCol(this.W2, this.b2, this._h,  H, H, this._h2);
        siluInPlace(this._h2);
        linearCol(this.W3, this.b3, this._h2, H, inDim, this._out);
      }

      return this._out;
    }
  }

  /* ---------- 2D toy MLP — live training ---------------------------------- */

  // Inputs:  [x, y, t/T]              (3-d)
  // Hidden:  H units, two layers      (default H = 64)
  // Output:  [ε̂_x, ε̂_y]              (2-d)
  // Loss: 0.5 ‖ε̂ − ε‖² (MSE).
  // Optimizer: Adam.
  class TwoDModel {
    constructor(opts) {
      opts = opts || {};
      this.H = opts.hidden || 64;
      this.lr = opts.lr || 2e-3;
      this.seed = opts.seed || 7;
      this._rng = DiffusionMath.mulberry32(this.seed);

      this.W1 = this._heInit(this.H, 3);
      this.b1 = new Float32Array(this.H);
      this.W2 = this._heInit(this.H, this.H);
      this.b2 = new Float32Array(this.H);
      this.W3 = this._heInit(2, this.H);
      this.b3 = new Float32Array(2);

      this._adam = {
        W1: this._adamSlot(this.W1), b1: this._adamSlot(this.b1),
        W2: this._adamSlot(this.W2), b2: this._adamSlot(this.b2),
        W3: this._adamSlot(this.W3), b3: this._adamSlot(this.b3)
      };
      this.step = 0;
      this.lossHistory = [];
    }

    _heInit(out_dim, in_dim) {
      const arr = new Float32Array(out_dim * in_dim);
      const std = Math.sqrt(2 / in_dim);
      for (let i = 0; i < arr.length; i++) {
        arr[i] = DiffusionMath.randn(this._rng) * std;
      }
      return arr;
    }

    _adamSlot(a) {
      return { m: new Float32Array(a.length), v: new Float32Array(a.length) };
    }

    // Internal forward keeping intermediates for backprop.
    _forward(x, y, tn) {
      const inp = new Float32Array([x, y, tn]);
      const z1 = linear(this.W1, this.b1, inp, 3, this.H);
      const a1 = new Float32Array(this.H);
      for (let i = 0; i < this.H; i++) a1[i] = silu(z1[i]);
      const z2 = linear(this.W2, this.b2, a1, this.H, this.H);
      const a2 = new Float32Array(this.H);
      for (let i = 0; i < this.H; i++) a2[i] = silu(z2[i]);
      const out = linear(this.W3, this.b3, a2, this.H, 2);
      return { inp, z1, a1, z2, a2, out };
    }

    // Inference. Returns a 2-element Float32Array [ε̂_x, ε̂_y].
    predict(x, y, tn) {
      return this._forward(x, y, tn).out;
    }

    // One Adam step on a batch.
    // batch: array of [x, y, tn, eps_x, eps_y].
    // Returns mean per-sample MSE (½ ‖ε̂ − ε‖²).
    trainBatch(batch) {
      const N = batch.length;
      const H = this.H;

      const gW1 = new Float32Array(this.W1.length);
      const gb1 = new Float32Array(this.b1.length);
      const gW2 = new Float32Array(this.W2.length);
      const gb2 = new Float32Array(this.b2.length);
      const gW3 = new Float32Array(this.W3.length);
      const gb3 = new Float32Array(this.b3.length);

      let lossSum = 0;

      for (let n = 0; n < N; n++) {
        const s = batch[n];
        const x = s[0], y = s[1], tn = s[2], ex = s[3], ey = s[4];
        const f = this._forward(x, y, tn);

        const d0 = f.out[0] - ex;
        const d1 = f.out[1] - ey;
        lossSum += d0 * d0 + d1 * d1;

        // Layer 3 gradients
        const da2 = new Float32Array(H);
        gb3[0] += d0; gb3[1] += d1;
        for (let j = 0; j < H; j++) {
          gW3[0 * H + j] += d0 * f.a2[j];
          gW3[1 * H + j] += d1 * f.a2[j];
          da2[j] += this.W3[0 * H + j] * d0 + this.W3[1 * H + j] * d1;
        }

        // Layer 2 gradients
        const dz2 = new Float32Array(H);
        for (let j = 0; j < H; j++) dz2[j] = da2[j] * siluDeriv(f.z2[j]);
        const da1 = new Float32Array(H);
        for (let i = 0; i < H; i++) {
          gb2[i] += dz2[i];
          for (let j = 0; j < H; j++) {
            gW2[i * H + j] += dz2[i] * f.a1[j];
            da1[j] += this.W2[i * H + j] * dz2[i];
          }
        }

        // Layer 1 gradients
        const dz1 = new Float32Array(H);
        for (let j = 0; j < H; j++) dz1[j] = da1[j] * siluDeriv(f.z1[j]);
        for (let i = 0; i < H; i++) {
          gb1[i] += dz1[i];
          for (let k = 0; k < 3; k++) {
            gW1[i * 3 + k] += dz1[i] * f.inp[k];
          }
        }
      }

      const inv = 1 / N;
      for (let i = 0; i < gW1.length; i++) gW1[i] *= inv;
      for (let i = 0; i < gb1.length; i++) gb1[i] *= inv;
      for (let i = 0; i < gW2.length; i++) gW2[i] *= inv;
      for (let i = 0; i < gb2.length; i++) gb2[i] *= inv;
      for (let i = 0; i < gW3.length; i++) gW3[i] *= inv;
      for (let i = 0; i < gb3.length; i++) gb3[i] *= inv;

      this.step++;
      this._adamUpdate('W1', gW1);
      this._adamUpdate('b1', gb1);
      this._adamUpdate('W2', gW2);
      this._adamUpdate('b2', gb2);
      this._adamUpdate('W3', gW3);
      this._adamUpdate('b3', gb3);

      const meanLoss = (lossSum / N) * 0.5;
      this.lossHistory.push(meanLoss);
      return meanLoss;
    }

    _adamUpdate(name, grad) {
      const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
      const param = this[name];
      const state = this._adam[name];
      const t = this.step;
      const bc1 = 1 - Math.pow(beta1, t);
      const bc2 = 1 - Math.pow(beta2, t);
      const lr = this.lr;
      for (let i = 0; i < param.length; i++) {
        const g = grad[i];
        state.m[i] = beta1 * state.m[i] + (1 - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1 - beta2) * g * g;
        const mhat = state.m[i] / bc1;
        const vhat = state.v[i] / bc2;
        param[i] -= lr * mhat / (Math.sqrt(vhat) + eps);
      }
    }
  }

  /* ---------- training-batch sampler for the 2D toy ----------------------- */

  // Build a batch of size `batchSize` from `points` (array of [x, y]) using
  // the fast-forward formula. `alphaBars` is window.DATA.alphaBars (length T).
  // Returns array of [x_t.x, x_t.y, t/(T-1), eps_x, eps_y] suitable for trainBatch.
  function sample2DBatch(points, alphaBars, batchSize, rng) {
    const T = alphaBars.length;
    const out = new Array(batchSize);
    for (let n = 0; n < batchSize; n++) {
      const p = points[Math.floor(rng() * points.length)];
      const t = 1 + Math.floor(rng() * (T - 1));    // t in [1, T-1]
      const ab = alphaBars[t];
      const ex = DiffusionMath.randn(rng);
      const ey = DiffusionMath.randn(rng);
      const sa = Math.sqrt(ab);
      const sb = Math.sqrt(1 - ab);
      const xt0 = sa * p[0] + sb * ex;
      const xt1 = sa * p[1] + sb * ey;
      out[n] = [xt0, xt1, t / (T - 1), ex, ey];
    }
    return out;
  }

  /* ---------- whole-trajectory generators (used by scenes 2, 3, 6) -------- */

  // Forward iteratively from x0 for all T steps. Returns:
  //   { trajectory: [Float32Array(D)] of length T+1,    // x_0 ... x_T
  //     epsilons:   [Float32Array(D)] of length T }     // ε used at each step
  // betas, alphas, alphaBars are window.DATA arrays.
  function forwardTrajectory(x0, betas, rng) {
    const T = betas.length;
    const D = x0.length;
    const traj = new Array(T + 1);
    const eps  = new Array(T);
    traj[0] = new Float32Array(x0);
    for (let t = 0; t < T; t++) {
      const e = DiffusionMath.randnVector(rng, D);
      eps[t] = e;
      traj[t + 1] = DiffusionMath.forwardStep(traj[t], betas[t], e);
    }
    return { trajectory: traj, epsilons: eps };
  }

  // Reverse from a starting x_T using a noise-prediction function epsFn(x_t, t).
  // Returns { trajectory: [Float32Array(D)] of length T+1 } indexed so that
  // result.trajectory[t] is the state x_t (so [0] = generated, [T] = noise).
  //
  // Optional `clipRange` ([lo, hi]) — clamp each x_{t-1} into the range. The
  // MLP-without-conv MNIST backbone in this viz is intentionally weak; without
  // clipping its reverse process can diverge exponentially. Pass [-1.5, 1.5]
  // for MNIST. Default null = no clipping (correct for the well-trained 2D toy).
  function reverseTrajectory(xT, epsFn, betas, alphas, alphaBars, rng, clipRange) {
    const T = betas.length;
    const D = xT.length;
    const traj = new Array(T + 1);
    traj[T] = new Float32Array(xT);
    for (let t = T - 1; t >= 0; t--) {
      const epsHat = epsFn(traj[t + 1], t);
      const z = DiffusionMath.randnVector(rng, D);
      let next = DiffusionMath.reverseStep(traj[t + 1], alphas[t], alphaBars[t], betas[t], epsHat, z, t);
      if (clipRange) {
        const lo = clipRange[0], hi = clipRange[1];
        for (let i = 0; i < D; i++) {
          if (next[i] < lo) next[i] = lo;
          else if (next[i] > hi) next[i] = hi;
        }
      }
      traj[t] = next;
    }
    return { trajectory: traj };
  }

  return {
    linear, linearCol, siluInPlace, silu, siluDeriv,
    MNISTModel, TwoDModel,
    sample2DBatch,
    forwardTrajectory, reverseTrajectory
  };
})();
