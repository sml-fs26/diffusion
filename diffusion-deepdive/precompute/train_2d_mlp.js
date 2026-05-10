/* Pre-train the 2D letter-M denoising MLP offline (Node).
 *
 * Why offline? In-browser cold-burst training of the same network gave
 * inconsistent letter-M generation: a 1500-step model produced two vertical
 * bars without the V valley, a 3000-step run was variable, and asking the
 * browser to do >5 s of synchronous training is hostile. Better to ship the
 * trained weights in data/datasets.js the same way we ship the MNIST UNet
 * trajectories — the in-browser scene 5 still trains a fresh model live for
 * pedagogy, but scene 6's "Generate" uses the shipped weights so the M
 * always emerges cleanly.
 *
 * Output: precompute/_artifacts/twoD_model.json
 *   {
 *     architecture: { hidden, lr, seed, inputDim: 3, outputDim: 2 },
 *     weights:      { W1, b1, W2, b2, W3, b3 },
 *     trainingMeta: { steps, batchSize, lossInitial, lossFinal, evalNN, time }
 *   }
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
globalThis.window = globalThis;
function load(p) { (new Function(fs.readFileSync(path.join(ROOT, p), 'utf8')))(); }
load('js/diffusion-math.js');
load('js/diffusion-nn.js');
load('data/datasets.js');

const DATA = window.DATA;
const M    = window.DiffusionMath;
const NN   = window.DiffusionNN;

// -----------------------------------------------------------------------------
const HIDDEN = 128;        // wider than scene 5's 64 — afforded by offline budget
const LR     = 2e-3;
const SEED   = 7;
const STEPS  = 8000;       // bench: mean nearest-M dist ≈ 0.048 at this point
const BATCH  = 64;

// -----------------------------------------------------------------------------
function evalReverse(model, T, points, betas, alphas, alphaBars, evalSeed) {
  const rng = M.mulberry32(evalSeed);
  const N = points.length;
  let x = new Float32Array(2 * N);
  for (let i = 0; i < x.length; i++) x[i] = M.randn(rng);
  for (let t = T - 1; t >= 0; t--) {
    const tn = t / (T - 1);
    const epsHat = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) {
      const eh = model.predict(x[2 * i], x[2 * i + 1], tn);
      epsHat[2 * i] = eh[0];
      epsHat[2 * i + 1] = eh[1];
    }
    const z = M.randnVector(rng, 2 * N);
    x = M.reverseStep(x, alphas[t], alphaBars[t], betas[t], epsHat, z, t);
  }
  return x;
}

function meanNearestNeighbor(x, points) {
  const N = x.length / 2;
  let s = 0;
  for (let i = 0; i < N; i++) {
    let best = Infinity;
    for (const [px, py] of points) {
      const d = (x[2*i]-px)**2 + (x[2*i+1]-py)**2;
      if (d < best) best = d;
    }
    s += Math.sqrt(best);
  }
  return s / N;
}

// -----------------------------------------------------------------------------
console.log(`Training 2D MLP: hidden=${HIDDEN} lr=${LR} steps=${STEPS} batch=${BATCH} seed=${SEED}`);
const model = new NN.TwoDModel({ hidden: HIDDEN, lr: LR, seed: SEED });
const rng = M.mulberry32(SEED + 9001);
const t0 = Date.now();
const lossSeries = [];
for (let s = 1; s <= STEPS; s++) {
  const batch = NN.sample2DBatch(DATA.letterM.points, DATA.alphaBars, BATCH, rng);
  const loss = model.trainBatch(batch);
  lossSeries.push(loss);
  if (s === 1 || s % 1000 === 0) {
    const recent = lossSeries.slice(-Math.min(1000, lossSeries.length));
    const meanLoss = recent.reduce((a, b) => a + b, 0) / recent.length;
    console.log(`  step ${s.toString().padStart(5)}: loss(last1k mean)=${meanLoss.toFixed(4)}`);
  }
}
const dt = (Date.now() - t0) / 1000;
const lossInitial = lossSeries.slice(0, 1000).reduce((a,b)=>a+b,0) / Math.min(1000, lossSeries.length);
const lossFinal   = lossSeries.slice(-1000).reduce((a,b)=>a+b,0) / Math.min(1000, lossSeries.length);
console.log(`done in ${dt.toFixed(1)}s   loss: ${lossInitial.toFixed(3)} -> ${lossFinal.toFixed(3)}`);

// -----------------------------------------------------------------------------
// Eval a few seeds to see distribution of nearest-M-point distance.
const evalSeeds = [12345, 99, 777, 31415, 271828];
const evalNNs = [];
for (const es of evalSeeds) {
  const x = evalReverse(model, DATA.T, DATA.letterM.points, DATA.betas, DATA.alphas, DATA.alphaBars, es);
  const nn = meanNearestNeighbor(x, DATA.letterM.points);
  evalNNs.push(nn);
  console.log(`  reverse seed=${es}: mean nearest-M dist = ${nn.toFixed(4)}`);
}
const evalNN = evalNNs.reduce((a,b)=>a+b,0) / evalNNs.length;
console.log(`mean over ${evalSeeds.length} seeds: ${evalNN.toFixed(4)}`);

// -----------------------------------------------------------------------------
const out = {
  architecture: { hidden: HIDDEN, lr: LR, seed: SEED, inputDim: 3, outputDim: 2 },
  weights: {
    W1: Array.from(model.W1), b1: Array.from(model.b1),
    W2: Array.from(model.W2), b2: Array.from(model.b2),
    W3: Array.from(model.W3), b3: Array.from(model.b3),
  },
  trainingMeta: {
    steps: STEPS, batchSize: BATCH,
    lossInitial: lossInitial, lossFinal: lossFinal,
    evalNN: evalNN, evalSeeds: evalSeeds, evalNNs: evalNNs,
    timeSeconds: dt,
  },
};
const outPath = path.join(ROOT, 'precompute', '_artifacts', 'twoD_model.json');
fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(out));
const sizeKB = (fs.statSync(outPath).size / 1024).toFixed(1);
console.log(`\nwrote ${outPath} (${sizeKB} KB)`);
