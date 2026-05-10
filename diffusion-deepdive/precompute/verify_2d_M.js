/* End-to-end verification: load DATA exactly as the browser does, instantiate
 * the shipped TwoDModel weights, run the full reverse pass for several seeds,
 * and ASSERT that the generated cloud forms a recognisable letter M.
 *
 * Run before declaring scene 6 done. Exits non-zero on failure so it can gate
 * a CI pipeline / pre-commit hook.
 *
 *   $ node precompute/verify_2d_M.js
 *
 * The metric is mean nearest-letterM-point distance: for each generated
 * point, compute the distance to its closest source letter-M point (in a
 * normalised [-1, 1]² frame), then average. The shipped weights ought to
 * give ≈ 0.05; we fail above 0.07.
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

const THRESHOLD = 0.07;
const SEEDS     = [12345, 99, 777, 31415, 271828, 1234, 4242];

function reverseTrajectoryFromShippedModel(seed) {
  const model = new NN.TwoDModel({ weights: DATA.twoDModel.weights });
  const rng = M.mulberry32(seed);
  const N = DATA.letterM.points.length;
  let x = new Float32Array(2 * N);
  for (let i = 0; i < x.length; i++) x[i] = M.randn(rng);
  for (let t = DATA.T - 1; t >= 0; t--) {
    const tn = t / (DATA.T - 1);
    const epsHat = new Float32Array(2 * N);
    for (let i = 0; i < N; i++) {
      const eh = model.predict(x[2 * i], x[2 * i + 1], tn);
      epsHat[2 * i] = eh[0];
      epsHat[2 * i + 1] = eh[1];
    }
    const z = M.randnVector(rng, 2 * N);
    x = M.reverseStep(x, DATA.alphas[t], DATA.alphaBars[t], DATA.betas[t], epsHat, z, t);
  }
  return x;
}

function meanNearest(x, points) {
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

console.log('verifying scene 6 letter-M generation…');
console.log(`shipped 2D MLP: hidden=${DATA.twoDModel.architecture.hidden}`);
console.log(`training meta: ${JSON.stringify(DATA.twoDModel.trainingMeta).slice(0, 200)}`);

let worst = 0;
for (const seed of SEEDS) {
  const x = reverseTrajectoryFromShippedModel(seed);
  const nn = meanNearest(x, DATA.letterM.points);
  const verdict = nn <= THRESHOLD ? 'PASS' : 'FAIL';
  console.log(`  seed=${seed}: mean nearest-M dist = ${nn.toFixed(4)}  ${verdict}`);
  if (nn > worst) worst = nn;
}
console.log(`worst over ${SEEDS.length} seeds: ${worst.toFixed(4)}  threshold: ${THRESHOLD}`);

if (worst > THRESHOLD) {
  console.error('\nINVARIANT FAIL: at least one reverse seed produced a cloud farther than the threshold.');
  console.error('The shipped 2D model is undertrained or the verification metric is too tight.');
  console.error('Either retrain (precompute/train_2d_mlp.js with more steps), or relax THRESHOLD with justification.');
  process.exit(1);
}
console.log('\nALL SEEDS PASS — scene 6 will produce recognisable Ms.');
