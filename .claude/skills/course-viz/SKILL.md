---
name: course-viz
description: Build browser-only educational visualizations for university courses (math, statistics, computer science, machine learning). Use when the user asks for a "visualization", "interactive viz", "scrollytale", "explainer", or "deep-dive" tied to a course concept (PCA, k-means, sorting algorithms, probability distributions, optimization, kernels, …). Defines the editorial aesthetic, multi-file project structure, scrollytelling and click-step patterns, parallel-agent fan-out, data validation, and verification. Course-agnostic — drop into any course repo and follow the recipe.
---

# course-viz

Conventions for browser-only educational visualizations of mathematical, statistical, computer-science, and machine-learning concepts. Audience: university students at the level the course is taught (typically advanced undergrad or master's). Deliverable: a viz that runs from `file://` on the lecturer's laptop, with no build step.

These rules are derived from a small library of viz built originally for a Statistical Machine Learning course — the originating repo is referenced as `pca-kmeans-rf` in citations below, and contains three working example folders (`congress-story/`, `kmeans-deepdive/`, `movies-pca/`). The patterns are course-agnostic: the aesthetic, palette, and voice work for an algorithms course or a probability course as well as for ML. Read once, then apply throughout the build.

If a course needs a fundamentally different look (e.g. a hand-drawn whiteboard aesthetic for an intro CS course), see the sibling `teaching-viz` skill — that one defines the casual single-file pattern. This skill is the editorial multi-file pattern.

**A note on shared rules.** Several rules below — hash routing, headless screenshot, debugging discipline, color cross-talk, no-fabrication, D3 handler-on-merged-selection, KaTeX conventions — are course-agnostic and may also appear in sibling viz skills you maintain. They're the lessons; if you sharpen one of these here, sharpen it in any sibling skills too.

## Hard requirements

1. **Multi-file structure.** Each viz is one folder containing:
   ```
   <viz-name>/
     index.html
     css/
       style.css        ← theme tokens + scene engine layout + cluster classes
       sceneN.css       ← per-scene styles
     js/
       theme.js         ← light/dark toggle (CSS variables, localStorage, 't' shortcut)
       *.js             ← shared modules (math utils, scene engine, widgets)
       scenes/
         sceneN.js      ← one file per scene; registers on window.scenes
     data/
       datasets.js      ← inline data: window.DATA = {...}
     vendor/            ← d3, katex, scrollama (vendored, never CDN)
     precompute/        ← Node/Python data generators (not loaded at runtime)
   ```

2. **Browser-openable from `file://`.** No `fetch()`. No CDN imports. No relative ES-module `import()`. Plain `<script src="...">` tags only — vendored libs in `vendor/`.

3. **Light + dark themes.** Light is the lecture default (bright projection halls). Dark is for solo study. CSS-variable token swap via `data-theme` on `<html>`. Mandatory `t` keyboard shortcut for mid-lecture toggle. All cluster/categorical colors via CSS classes (`.cluster-N`) so theme switch is automatic — never inline `fill="#..."`.

4. **One viz = one sibling folder.** Don't expand existing viz with new content — create a new sibling.

5. **Verify before reporting done.** See §Verification for the parse check + browser walkthrough + headless Chrome screenshot loop.

## Aesthetic — editorial

Polished Pudding/NYT/Distill voice, not casual hand-drawn. This is the house style — opinionated and consistent across viz, regardless of course.

**Light mode (lecture default):**
- Background `#f9f7f1` warm cream
- Ink `#1a1a1a`, secondary `#6b6b6b`, hairline rule `#d8d4ca`
- Categorical palette: blue `#2f6cb1`, red `#b8323a`, amber `#c08a3e`, purple `#7a5c8c`, green `#4a8b6b`, burnt orange `#a05a2c`
- No brand pink. Accent is restrained black; emphasis comes from the categorical palette and serif typography.

**Dark mode:**
- Background `#181612` warm dark (not pure black — keeps warmth)
- Ink `#ece7d9`, secondary `#8a8579`, rule `#383530`
- Categorical palette: lighter, less saturated companions of the light values
- Yellow doesn't work on white — light mode swaps it for amber.

**Typography:**
- Headers: Iowan Old Style → Palatino → Georgia serif, weight 500
- Body: -apple-system / Inter sans, 16–17px, line-height 1.55
- Mono: SF Mono / Menlo

**Cards:** flat hairline border (`1px solid var(--rule)`), 4px radius, no offset shadows. Visual demarcation, not weight.

**Color cross-talk.** Don't reuse a group-identity color for a second encoding. If party/cluster/class is encoded in red and blue, don't *also* use red for "important" or "alert" elsewhere — readers will infer group membership where you meant emphasis. Reserve a third hue (amber, purple) for highlights that aren't about identity. (Real story: a chart in `congress-story` colored "procedural" votes in red and "substantive" in black; the red read as Republican. Switched to amber.)

**When the algorithm partitions space, draw the partition.** Voronoi cells, decision boundaries, kernel-induced regions — these are the geometric basis for the algorithm's behavior. Showing them makes the algorithm intuitive in a way no amount of prose or distance lines can match. Use semi-transparent fills (`fill-opacity: 0.08`) layered behind data points.

## Two interaction patterns

Pick one and commit. Don't mix.

### Scrollytelling

Sticky viz column on one side, prose scrolls on the other. `scrollama.min.js` triggers scene transitions on scroll.

- **Best for:** linear hero narratives. Lecturer scrolls through it as a presentation.
- **Reference:** `congress-story/` in the originating repo.
- **Limitation:** unidirectional. Can't have "click Next within scene 4 to walk through the assignment-step math." If the math is non-trivial and needs explicit decision points, use click-step.
- **CSS gotcha — sticky inside CSS Grid.** The shared shell uses `display: grid` for the prose/viz two-column layout. Grid's default `align-items: stretch` makes the viz column fill the full grid row (in a 12-scene viz, that's `12 × 100vh` = 1200vh), which silently kills sticky positioning — the column has nowhere to "stick" when it already covers everything. **Always add `align-self: start` on the viz column** alongside `position: sticky; top: 0; height: 100vh`. Symptom when you forget: viz-stage shows briefly above the hero, then disappears entirely as you scroll into the first scene.

### Click-step

Topbar (brand · scene title · theme toggle · dot pager). Stage with one `.scene` div per scene. Footer with Prev/Next. Internal step engines per scene supported.

- **Best for:** math-heavy didactic content. Bidirectional with rewind.
- **Reference:** `kmeans-deepdive/` in the originating repo.

## Scene engine pattern

Reference implementation: `kmeans-deepdive/js/main.js`.

```js
const SCENE_TITLES = ["", "Setup", "What we're minimizing", ...];

let current = -1;
const sceneNodes = [];
const sceneState = [];

function goTo(idx) {
  if (idx < 0 || idx >= SCENE_TITLES.length) return;
  if (idx === current) return;
  // fade-out old, fade-in new, update dots, prev/next disabled state
  if (!sceneNodes[idx]) {
    const node = document.createElement('div');
    node.className = 'scene';
    stage.appendChild(node);
    sceneNodes[idx] = node;
    const builder = window.scenes['scene' + idx];
    if (builder) sceneState[idx] = builder(node) || {};
  } else if (sceneState[idx]?.onEnter) {
    sceneState[idx].onEnter();
  }
  current = idx;
  setTimeout(() => sceneNodes[idx].classList.add('active'), 20);
  // ... update hash, dots, button states
}

window.addEventListener('keydown', e => {
  if (e.target?.tagName?.match(/input|textarea|select/i)) return;
  if (e.key === 'ArrowRight') {
    const handled = sceneState[current]?.onNextKey?.();
    if (!handled) goTo(current + 1);
  } else if (e.key === 'ArrowLeft') {
    const handled = sceneState[current]?.onPrevKey?.();
    if (!handled) goTo(current - 1);
  }
});
```

Each scene file at `js/scenes/sceneN.js` registers `window.scenes.sceneN(root) -> { onEnter?, onLeave?, onNextKey?, onPrevKey? }`. Returning `true` from `onNextKey` consumes the keystroke (advances an internal cursor); returning `false` lets the driver advance the scene.

### Hash-based scene routing (mandatory)

Every click-step viz must support direct deep-link to a scene via the URL hash:

```js
function readHashScene() {
  const m = (window.location.hash || '').match(/scene=(\d+)/);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  return (Number.isFinite(n) && n >= 0 && n < SCENE_TITLES.length) ? n : null;
}
window.addEventListener('hashchange', () => {
  const n = readHashScene();
  if (n != null) goTo(n);
});
goTo(readHashScene() ?? 0);
```

This is non-negotiable. It saves the lecturer (deep-linking mid-lecture) and the agent (jumping straight to scene 8 instead of clicking Prev/Next 8 times per reload).

### `&run` flag for animation scenes

Scenes that gate their main animation behind a button press should also accept a `&run` URL flag for headless verification:

```js
function shouldAutoRun() {
  return /[#&?]run\b/.test(window.location.hash);
}
if (shouldAutoRun()) setTimeout(() => projectBtn.click(), 200);
```

This is a **dev affordance, not a user feature** — the button stays the canonical interaction. The flag exists so the headless Chrome screenshot loop (see §Verification) can reach post-animation states.

If a scene has multiple gated buttons (e.g. "Train" then "Predict"), `&run` triggers only the **primary** one — the most visually prominent button, usually labelled "Run", "Go", or with a play icon. For multi-step playthrough use a separate `&runAll` flag that walks every gated button in order. Don't overload `&run`; it should mean "get me past the obvious gate."

## Step engine (within-scene Next button)

Many scenes have an internal step counter:

```js
let cursor = 0;
const STEPS = N;

function resetState() { /* mutate state to initial */ }
function applyStep(c) { /* mutate state for step c (cursor c-1 → c) */ }
function render()     { /* full re-render from state */ }

function setCursor(c, animate = true) {
  if (c < 0 || c >= STEPS) return false;
  if (c < cursor) {
    resetState();
    cursor = 0;
    while (cursor < c) { cursor++; applyStep(cursor); }
    render();
  } else {
    while (cursor < c) { cursor++; applyStep(cursor); render(); }
    if (animate) playLatestAnimation();
  }
  return true;
}

return {
  onEnter()   { resetState(); cursor = 0; render(); },
  onNextKey() { return setCursor(cursor + 1, true); },
  onPrevKey() { return setCursor(cursor - 1, false); },
};
```

**Critical rules:**
- **State is the source of truth, animations are decoration.** `render(state)` produces a correct snapshot from scratch.
- **Prev = rewind via reset+replay.** Don't write inverse mutations. Disable animations during fast-forward.
- **One DOM action per step.** Resist cramming several visual changes into one click.
- **Cold entry must work.** If the user jumps to scene 5 via dot pager without visiting scene 4, scene 5 must reconstruct its initial state from `DATA` — never depend on a prior scene having run.

## Cross-scene state

When scenes share narrative state (multi-step transformations spanning scenes):

```js
window.<vizName>Shared = window.<vizName>Shared || {
  // shared narrative state, e.g.:
  points: DATA.iterDemo.points,
  k: 3,
  assignmentsAfterAssign: null,
  centroidsAfterUpdate: null,
  // ...
};
```

Use a project-prefixed global (`kmeansShared`, `pcaShared`, `gradientShared`, …) — never a generic name like `state` or `shared` that another script in the same window might collide with. Each scene's `onEnter` reads from it; each scene's `applyStep` writes to it. Cold entry must work — if the field is empty, reconstruct from `DATA`.

## Parallel-agent fan-out

For viz with 6+ scenes, fan out to parallel agents.

**Phase 0 (sequential, you):** scene engine, theme tokens, shared math utilities, vendored libs, scene stubs that register `window.scenes.sceneN = function(root) { /* stub */ }`.

**Phase 0.5 (data-prep agent, parallel with Phase 0):** if pre-canned trajectories or precomputed data are needed:
- Node.js (or Python) script writes `data/datasets.js` from a deterministic source.
- **Mandatory verification step in the agent's brief** — assert every invariant downstream scenes will rely on (e.g. monotone J, init-sensitivity gap > N%, no NaN, no nulls). State each invariant as code, not as a final-report claim.
- Out of scope for this agent: any DOM/JS for the viz itself.

**Phase 1 (scene agents, parallel after Phase 0):** typically 4 agents, each owns 1–3 scenes.

**Bundle scenes for tonal coherence, NOT for parallelism:**
- Keep tonal triplets together (e.g. assign + update + converge → one agent). Splitting causes drift in the visual language for the most pedagogically important scenes.
- Bundle bookends (title + takeaways) — light work, similar voice.
- Bundle narrative arcs (setup → objective → init).
- Don't split a tonal triplet for parallelism's sake.

**Strict contract per scene agent:**
- Function signature: `window.scenes.sceneN = function(root) { return { onEnter?, onLeave?, onNextKey?, onPrevKey? }; }`
- Allowed globals enumerated (`d3`, `katex`, project-specific math module, `Theme`, `DATA`, …)
- CSS class allowlist
- Hard rules: no inline cluster colors, no `<style>` injection from JS, no fetch, no relative imports
- Output: one JS file per scene + optional one CSS file per scene
- Verification: parse check + browser walkthrough in both themes + headless screenshot for non-trivial layouts

### Scene agent prompt template

Copy and fill the bracketed fields:

> Build scene `[N]` of the `[VIZ-NAME]` viz. Topic: `[ONE-LINE-DESCRIPTION]`.
>
> **File location.** Write the JS to `js/scenes/scene[N].js` and any per-scene CSS to `css/scene[N].css`. Link the CSS in `index.html`.
>
> **Function contract.**
> ```js
> window.scenes.scene[N] = function(root) {
>   // build DOM into `root`
>   return { onEnter?, onLeave?, onNextKey?, onPrevKey? };
> };
> ```
> `onNextKey` returns `true` to consume the keystroke (advance internal step), `false` to let the driver advance the scene.
>
> **Allowed globals.** `d3`, `katex`, `Theme`, `DATA`, `[VIZ-SPECIFIC-MODULES]`. No others.
>
> **Hard rules.**
> - No inline cluster colors. Use `.cluster-N` classes — already defined in `css/style.css`.
> - No `<style>` injection from JS. Per-scene CSS goes in `css/scene[N].css`.
> - No fetch, no relative imports.
> - Cold entry must work — the scene must reconstruct its initial state from `DATA` if visited via dot-pager without prior scenes having run.
> - All KaTeX rendered via `katex.render(tex, host, { throwOnError: false })`. Use display mode for centred formulas.
>
> **Scene narrative** (the pedagogical arc this scene must deliver):
> `[3–5 SENTENCES — what the student should understand by the end]`
>
> **Step plan** (if multi-step):
> `[STEP-BY-STEP NARRATIVE]`
>
> **Verification.** Parse-check the file, browser-walkthrough in both themes, and headless-screenshot via `--screenshot=...#scene=[N]` (combine with `&run` for animation gates).

**Phase 2 (aggregation, you):** link scene CSS files in `index.html`, parse-check all files, browser-verify both themes, fix integration issues, spot-check agent invariant claims.

## Inline data convention

```js
// data/datasets.js
window.DATA = {
  /* the entire payload */
};
```

Loaded via `<script src="data/datasets.js">` — never `fetch()`. A companion `precompute/build-datasets.js` (Node) or `precompute/build_data.py` (Python) lives alongside for human auditing and regeneration; it's **not** loaded at runtime.

**Reproducibility.** Any precompute script that draws random numbers must use a **seeded RNG** (Mulberry32 is enough — 6 lines), with the seed pinned at the top of the file. Box-Muller for Gaussians. The same precompute run must produce byte-identical `data/datasets.js` every time, so a downstream code edit can be re-verified with `diff` instead of by re-checking every value. Pin the seed once; never tune it to make a particular sample look better.

## KaTeX usage

```js
katex.render(
  "J \\;=\\; \\sum_{k=1}^{K} \\sum_{x \\in C_k} \\|x - \\mu_k\\|^2",
  document.getElementById('formula-container'),
  { throwOnError: false, displayMode: true }
);
```

In a JS string literal, `\\` produces a single `\` for KaTeX. Use `\mathrm{name}` for function names like `dist`, `argmin`. Use `\bigl\{ … \bigr\}` for visible braces (escape with `\{`).

KaTeX uses `currentColor` for most rules — automatically theme-aware via the surrounding element's `color` property.

## Math: render every formula in KaTeX, define every symbol

Two hard rules. Both are easy to violate by accident; both are visible in any screenshot review and either are caught upfront or are caught later by a confused student.

### 1. Every formula renders in KaTeX, not in HTML pseudo-math

HTML's `<sub>`, `<sup>`, and Unicode math glyphs (`β`, `√`, `ᾱ`, `ε̂`) are *labels*, not typesetting. They look fine inline at small scale and look amateur the moment they sit next to a real KaTeX equation in the same page — different fonts, different baselines, different italicization. A student reads "the formulas in this viz are inconsistent" before they read the math.

Use KaTeX for:
- Any **displayed equation** (formula blocks, hero formulas in scene headers).
- Any **inline reference that names a symbol with subscripts, fractions, square roots, hats, bars, or accents** — e.g. `x_t`, `\bar\alpha_t`, `\sqrt{1-\beta_t}`, `\hat\varepsilon`. These belong in `katex.render(..., { displayMode: false })`.

HTML markup is fine for:
- Plain prose mentions of a symbol *as a name*, not as part of an expression — e.g. `<em>β</em>` in "The β slider sets one number".
- Chart axis labels and legends where a SVG `<text>` element is the natural host. (KaTeX-in-SVG is brittle.)
- Tabular numerics next to a KaTeX-rendered symbol.

Rendering pattern for inline KaTeX in mid-sentence (this is the clean way):

```js
const p = document.createElement('p');
p.appendChild(document.createTextNode('A small MLP learns the mapping '));
const formula = document.createElement('span');
p.appendChild(formula);
katex.render('(x_t,\\,t) \\;\\to\\; \\hat\\varepsilon', formula,
  { throwOnError: false, displayMode: false });
p.appendChild(document.createTextNode('. The recipe is one line.'));
```

Avoid `innerHTML = '...<sub>t</sub>...'` for anything that's mathematical.

### 2. Every symbol the viz uses is defined somewhere visible

If a formula contains a symbol the student hasn't seen defined, the formula is ornamental, not informative. The first scene that introduces a new symbol must either define it inline (in the surrounding prose) or reference a notation block earlier in the viz.

The cheap, durable approach is a **notation block**: a small panel near the start of whichever scene first uses the symbols, listing each symbol on its own row with an italic gloss. The diffusion deep-dive's scene 2 has one — it defines `β_t`, `α_t = 1 − β_t`, `ᾱ_t = ∏α_s`, `ε ~ N(0, I)` before any later scene puts them in a hero formula.

Pattern:

```js
const NOTATION_LINES = [
  { tex: '\\beta_t \\in [0, 1]',
    gloss: 'noise schedule (the DDPM hyperparameter)' },
  { tex: '\\alpha_t \\;=\\; 1 - \\beta_t',
    gloss: 'signal weight retained at one step' },
  { tex: '\\bar\\alpha_t \\;=\\; \\textstyle\\prod_{s\\le t}\\alpha_s',
    gloss: 'signal retained from x₀ to x_t' },
  { tex: '\\varepsilon \\sim \\mathcal{N}(0, I)',
    gloss: 'standard Gaussian noise' },
];
// Each row: KaTeX-rendered symbol on the left, italic prose on the right.
```

Style: italic muted serif for the gloss (matches `.callout-title`), tabular numerics for the symbol column, a small panel border (`1px solid var(--rule)`).

### Audit before reporting done

When the viz uses any non-trivial math, write a one-line audit:
- "Every formula I display is rendered by `katex.render`."
- "Every symbol that appears in any formula is defined somewhere a student can find — either inline at first use, or in a notation block."

If you can't write both lines, the viz isn't ready.

(Real story: the diffusion-deepdive shipped scene 4's hero formula `x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t)·ε_t) + √β_t·z` correctly KaTeX-rendered, but `ᾱ_t` was never defined anywhere in the viz. A student looking at scene 4 had no anchor for what the bar meant. The fix was a four-line notation block in scene 2; the bug was that nobody had written the audit out loud before declaring the viz done.)

## Voice and language

All UI strings, prose, captions, labels, and formula descriptions in **English**, even when the course is taught in another language. The slide deck may be bilingual; the visualizations are English-only for portability.

**Imitate this register** — short declarative sentences, present tense, the occasional rhetorical pivot:
- *"Two numbers describe American politics."*
- *"k-means is the algorithm that makes J small. Everything from here is about that."*
- *"Same data. Same algorithm. Two starting points."*
- *"The truth lives in 2 dimensions."*

**Avoid:**
- Exclamation points, emoji, em-dashes used for excitement.
- Hand-drawn-feel chattiness (*"That's PCA!"*, *"Cool, right?"*).
- Textbook hedging (*"Note that, in general, one might consider…"*).
- Marketing copy (*"Discover the power of PCA"*).

**Captions** sit in italic muted serif at ~13–14px, complete sentence, no trailing punctuation if it's a single phrase. **Headings** use the serif Iowan stack and are short — title-case avoided in favour of sentence-case with a period.

## Debugging discipline

When an interactive widget behaves wrong, **check the data layer before iterating on the widget**. The UI is almost always a faithful renderer of upstream state; the bug is almost always upstream.

Concrete steps, in this order:

1. **Print the source variable**, not the rendered output. Logging "the slider thumb is at position 0" tells you nothing; logging `currentIdx = 0` tells you everything.
2. **`curl` the data file** the UI reads from. Confirm shape, length, ordering. The most expensive bugs come from stale or truncated data files.
3. **Verify data freshness.** If you're iterating against a JSON file (during precompute work, not the viz itself), browsers silently serve cached JSON for hours unless `Cache-Control: no-store, no-cache, must-revalidate` is sent. Use this 20-line wrapper:

   ```python
   """Local dev server that disables ALL caching headers."""
   import sys
   from http.server import HTTPServer, SimpleHTTPRequestHandler

   class NoCacheHandler(SimpleHTTPRequestHandler):
       def end_headers(self):
           self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
           self.send_header("Pragma", "no-cache")
           self.send_header("Expires", "0")
           super().end_headers()

   if __name__ == "__main__":
       port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
       print(f"Serving with no-cache headers at http://127.0.0.1:{port}/")
       HTTPServer(("127.0.0.1", port), NoCacheHandler).serve_forever()
   ```

   All three headers (`Cache-Control`, `Pragma`, `Expires`) matter for hitting different browser cache layers — don't drop any.
4. Only then iterate on the widget.

If a debugging session has gone three iterations without progress, **stop and re-verify your assumptions about upstream state**. Don't keep escalating fixes on the same layer.

(Real story: a slider in `congress-story` ate ~8 hours because the displayed index variable was assumed to match the on-screen label. It didn't — the upstream JSON had been silently truncated by a partial precompute run. The slider was rewritten three times before anyone checked the data.)

## Persistent widget caveat

Persistent fixed-position widgets (J-meter, OOB-error trace, hud overlay) caused real bugs in `kmeans-deepdive`:

- An "objective J" panel pinned to the right edge overlapped the scene's right text column. Fix attempts: (1) reserve right-padding via body class, (2) remove the widget. Option 2 won.
- **If a widget isn't doing pedagogical work that can't be done inline, don't add it.** Put the value in a `.formula-block` inside the scene that needs it.
- If a widget *is* genuinely useful (e.g. a longitudinal trace where the multi-iteration view is the lesson), reserve its space with body-class layout shift up front. Don't let it overlay scene content.

## Reference implementations

The patterns above were extracted from three viz that live in the originating repo (`pca-kmeans-rf`). When you have access to that repo, these are the canonical examples — match their CSS classes, DOM shapes, and naming. When you don't, the patterns above are sufficient to start from scratch.

| Pattern | Reference path (in `pca-kmeans-rf`) |
|---|---|
| Scrollytelling carousel + sticky viz | `congress-story/` |
| Click-step scene engine + dot pager + hash routing | `kmeans-deepdive/js/main.js` |
| Internal step engine (math walkthrough) | `kmeans-deepdive/js/scenes/scene4.js` |
| Light/dark theme toggle | `kmeans-deepdive/js/theme.js` |
| Math utility module (e.g. Lloyd's, Voronoi, k-means++) | `kmeans-deepdive/js/kmeans.js` |
| Pre-canned data + Node generator + verified invariants | `kmeans-deepdive/precompute/build-datasets.js` |
| Voronoi cells reshaping with centroid motion | `kmeans-deepdive/js/scenes/scene5.js` |
| Side-by-side trajectory comparison | `kmeans-deepdive/js/scenes/scene8.js` |
| Cross-scene shared state | `kmeans-deepdive/js/scenes/scene4.js` (and 5, 6) |
| No-cache local dev server | `congress-story/serve.py` (the 20-line wrapper inlined in §Debugging discipline) |

## Verification

### 1. Parse check (every JS file)

```bash
for f in js/*.js js/scenes/*.js; do
  node -e "
    const code = require('fs').readFileSync('$f', 'utf8');
    try { new Function(code); console.log('OK: $f'); }
    catch (e) { console.log('ERR: $f →', e.message); process.exit(1); }
  " || exit 1
done
```

### 2. Browser walkthrough in both themes

`open index.html`. Click/scroll through every scene in **light mode first** (lecture default), then toggle to dark with `t` and walk through again. A scene that looks weak in one mode should be redesigned, not shipped.

### 3. Headless Chrome screenshot (mandatory for non-trivial layouts)

The agent has no eyes. When asked to "describe what you saw," it fabricates: it reads the code, infers what *should* render, and reports that. Real layout bugs (clipped labels, misaligned overlays, off-screen elements) slip past pure code review.

```bash
CHROME="$(command -v google-chrome || command -v chromium \
  || echo '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')"

# Single scene
"$CHROME" --headless --disable-gpu --virtual-time-budget=4000 \
  --window-size=1400,900 --screenshot=/tmp/check.png \
  "file:///abs/path/to/index.html#scene=4"

# All scenes in a loop
for n in 0 1 2 3 4 5 6 7 8 9; do
  "$CHROME" --headless --disable-gpu --virtual-time-budget=4000 \
    --window-size=1400,900 --screenshot=/tmp/scene_$n.png \
    "file:///abs/path/to/index.html#scene=$n"
done
```

Then `Read` the PNGs. Combine with `&run` for animation scenes:

```bash
... --screenshot=/tmp/check.png "file:///.../#scene=8&run"
```

Ignore stderr unless it contains `CONSOLE` or `JavaScript` — almost everything else is GPU/CVDisplayLink platform noise.

The flag set above works on Chrome 100+. On Chrome 109+ you can also use `--headless=new` (the modern rendering path). If a headless run gives layout that disagrees with what `open index.html` shows interactively, swap `--headless` for `--headless=new` and re-screenshot — they sometimes diverge. Add `--no-sandbox --hide-scrollbars` if running in a container or CI.

(Real example: a 16×12 heatmap in a PCA viz had column labels rotated -55° and clipped past the right edge by ~37px. Pure code review showed nothing. The screenshot showed it immediately, and one CSS edit fixed it.)

**Open the PNG. Don't skip this.** A screenshot you don't open isn't verification — it's a file. The agent has no eyes; if you don't `Read` the image, you're back to fabricating descriptions of what *should* be there. This is a hard rule. The whole point of the screenshot is the moment you look at it.

#### Multi-viewport — capture both lecture-hall and laptop sizes

Many viz pages are taught on a projector and demoed on a laptop. Layout that survives 1280×800 can break at 1920×1080 (and vice versa: a sticky widget that sits below the fold on a laptop fills the projector's screen). Take both:

```bash
for w in "1280,800" "1920,1080"; do
  "$CHROME" --headless --disable-gpu \
    --window-size=$w \
    --screenshot=/tmp/scene4_${w/,/x}.png \
    "file:///abs/path/to/index.html#scene=4"
done
```

If the layout collapses at one viewport, the fix usually goes in `style.css` (a `clamp()` on font-size, a min/max-width on a container) rather than per-scene. Don't ship until both look right.

#### What headless screenshots catch reliably

- Layout bugs: panel taller than viewport, items below the fold, sticky positioning broken.
- CSS flex/grid auto-stretching of canvases (see "Lock canvas dimensions explicitly" in §Things to never do).
- Text overflow, padding, color/contrast.
- Scene-to-scene layout drift across the eight scenes.

#### What headless screenshots catch poorly — bake a `?test=` hook in from day 1

`requestAnimationFrame` doesn't run smoothly under headless. `--virtual-time-budget=N` advances JS time but RAF callbacks land mid-frame. Animations capture mid-pour, mid-fade, or barely started. **For physics demos and animated transitions, the layout will look correct live and broken in the screenshot — this is a tooling artifact, not a real bug.**

Plain `--screenshot` also can't scroll, click, hover, or wait. If your viz drives state from `IntersectionObserver`, scroll position, or post-interaction state, the screenshot only sees the initial frame.

The fix is to add a query-param test mode to the page from day 1 — same idea as the existing `&run` flag (see §`&run` flag for animation scenes), generalized:

```js
const params = new URLSearchParams(location.search);
const test = params.get('test');
if (test === 'revealed') {
  // jump straight to the post-reveal state without simulating scroll
  state.estimates.push(...DATA.demoEstimates);
  state.truthRevealed = true;
  render();
}
if (test === 'mid-anim') {
  // freeze any t-driven animation at a known phase for capture
  state.animT = 0.6;
  render();
}
```

Then `--screenshot` against `?test=revealed` and verify the post-state without simulating user input. **Two-line cost; pays back every iteration.** Like `&run`, this is a dev affordance, not a user feature — the canonical interaction stays the click/scroll. The flag exists so headless verification can reach states it otherwise can't.

(Real story, learned painfully on a separate viz: a canvas with attribute `width="280" height="360"` was being silently stretched by its `flex-direction: column` parent's default `align-items: stretch`. The page looked perfect interactively because Chrome's regular renderer reflowed canvas content; under `--headless` the same canvas captured as a thin slice. Two days of "but it works on my machine" before someone read the PNG. Fix in §Things to never do.)

#### Scrollytelling viz: hash routing alone won't scroll the viewport

For **scrollytelling** viz (sticky viz column + scroll-driven scenes), the recipe above lands on the hero, not on the requested scene. Headless Chrome runs the page's `scrollIntoView` from hash routing, but `--screenshot=` fires before the viewport scroll commits. All five agents who built `random-forest-deepdive/` independently reinvented the same iframe-wrapper workaround — inline it here so the next one doesn't.

Save this once as `/tmp/grab.html` and screenshot through it instead of the page directly:

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>scene grab</title>
<style>html,body{margin:0;padding:0;height:100%;width:100%;overflow:hidden}iframe{width:100%;height:100%;border:0}</style>
</head><body><iframe id="f"></iframe>
<script>
  const params = new URLSearchParams(location.search);
  const scene = parseInt(params.get('s') || '1', 10);
  const theme = params.get('theme') || 'light';
  const f = document.getElementById('f');
  f.src = params.get('url');                          // pass the viz URL via ?url=
  f.onload = () => {
    const doc = f.contentDocument, win = f.contentWindow;
    doc.documentElement.setAttribute('data-theme', theme);
    let tries = 0;
    function tryScroll() {
      tries++;
      const el = doc.querySelector(`.scene[data-scene="${scene}"]`);
      const viz = doc.querySelector('.viz-column');
      if (el && viz && viz.getBoundingClientRect().height > 100) {
        el.scrollIntoView({ block: 'start' });
        win.dispatchEvent(new Event('scroll'));         // wakes scrollama
        document.title = `READY scene ${scene}`;
      } else if (tries < 40) setTimeout(tryScroll, 80);
    }
    setTimeout(tryScroll, 150);
  };
</script></body></html>
```

Then loop:

```bash
URL="file:///abs/path/to/index.html"
for n in 1 2 3 4 5 6 7 8 9 10 11 12; do
  for theme in light dark; do
    "$CHROME" --headless --disable-gpu --virtual-time-budget=10000 \
      --window-size=1400,900 \
      --screenshot=/tmp/scene_${n}_${theme}.png \
      "file:///tmp/grab.html?s=${n}&theme=${theme}&url=${URL}"
  done
done
```

If the screenshots still come back showing only the hero, the iframe-wrapper trick has hit a Chrome version that defers nested-frame layout. In that case the working alternative every time is **DevTools Protocol via `Page.captureScreenshot`** after explicitly waiting for `Page.frameStoppedLoading` and a manual `Runtime.evaluate` to scroll. That's heavier — only reach for it if the iframe wrapper truly fails.

(Real example: building `random-forest-deepdive/` in a single fan-out, all 5 scene agents wasted 2–5 minutes each rediscovering this. The skill saving it once would have saved the whole batch ~15 minutes of duplicated trial-and-error.)

### 4. When to graduate to Puppeteer / Playwright

The bare `--screenshot` flag is enough for layout checks, color checks, and `?test=`-gated state checks. Don't add a headless-browser dependency until the cost has hit you twice.

You've crossed the threshold when the verification pass needs any of:
- Clicking a button and screenshotting the result
- Scrolling to a specific position (beyond the iframe-wrapper trick)
- Waiting for animations to settle before capture
- Hover-state captures (tooltips, focus rings)
- Console-log assertions or DOM-state assertions

At that point: `npm i -g playwright` once, then a small helper that takes `(path, [(action, screenshot-name)])`. Resist building it speculatively — concrete pain twice over before adding it to the verification flow.

## Verification checklist (the order matters)

1. Edit the file.
2. **Parse-check** every JS file you touched (§1).
3. **Headless screenshot** at one or two viewports (§3).
4. **Read** the PNG. Inspect against the §Aesthetic-issue checklist below.
5. If a state needs verifying that requires interaction or scroll, add a `?test=` URL param and re-screenshot.
6. Browser walkthrough in both themes (§2) for any non-trivial change.
7. Only then commit + push.

Step 4 — the agent actually looking at the screenshot — is what makes the rest work. Don't skip it. The "trust but verify" rule applies to your own work too.

## Aesthetic-issue checklist (apply to EVERY captured screenshot)

This list is what to look for when you `Read` the PNG. Don't declare a scene done until each of these is verified clear at 1400×900 light theme — and re-checked at 1400×900 dark theme. Most of them are invisible to a code-review pass; they only show up in pixels.

**Cropping / overflow.** Is anything cut at the top, bottom, left, or right edge of the scene's stage area? Specifically:
- Hero formulae cropped under the topbar — common when a content column outgrew its scene height.
- Closing captions or stats clipped at the footer edge — same cause from the bottom.
- Long titles wrapped or sliced — chart titles, button labels, axis ticks.
- Right-column text wider than the scene-layout right cell — the right column is `minmax(320px, 0.6fr)`; long lines with no wrap word-break and overflow.

If the scene is supposed to fit the viewport, it must fit. Add a `@media (max-height: ...)` block that shrinks panes or adds `overflow: auto` on the scene container (with a clear scroll affordance), and re-screenshot.

**Chart legibility.** Charts often look fine in code but unreadable in pixels. Specifically:
- A series whose y-range is dwarfed by another on a shared axis ends up as a near-flat line at the foot of the chart (real example: β_t on [1e-4, 0.02] sharing an axis with ᾱ_t on [0, 1] was a horizontal hairline). **Fix**: split into stacked panels, each with its own y-axis. Don't overload a single dual-axis plot if the audience needs to compare both curves.
- Hairline strokes (`stroke-width: 1`) disappear on projector resolutions. Use `stroke-width: 2–3` for primary curves; reserve thin strokes for grids and rules.
- Tick labels and legends that fit at 72-DPI but don't at projector size. If the chart's legend reads "βₜ" / "ᾱₜ" in 9 px italics, lecture-hall students won't see it. Use 11 px+ and put titles ABOVE the chart, not as inline legends at curve endpoints.
- Axis labels truncated. Always reserve enough left/right margin for the longest label you'll print.

**Layout pressure.** The scene's content column may LOOK fine, but there's only so much screen height. Common smells:
- New element added (callout, slider, secondary panel) silently pushes existing content past the bottom edge — and this is the screenshot agent's blind spot, because you only see the visible portion. After any addition, re-capture and verify *every* element is still in-frame.
- A second pane was added side-by-side, and at the lecture viewport (1400×900) the right column has no air. Either drop a less-essential element or swap to a vertical stack.

**Animation mid-frame.** The scene engine fades scenes in over 400 ms via `opacity: 0 → 1`. Headless screenshots taken under `--virtual-time-budget` may capture mid-transition — the page looks ghost-faded. Symptoms: text and SVG points present but at low alpha. Defeat this by either (a) raising the budget, or (b) inside the scene, when the `&run` flag is set, force `root.style.transition = 'none'; root.style.opacity = 1;` after painting. Match the pattern in `scene0.js`'s `paintFinal()`.

**Theme contrast.** Anything that's only legible in light mode is a defect in dark mode. Specifically:
- White-on-white panels (forgot to use `var(--panel)` instead of literal `#fff`).
- Black SVG strokes (forgot to use `currentColor` / theme tokens).
- Hairline rules (`#d8d4ca` light / `#383530` dark) disappear in opposite themes if hard-coded.

**Empty state.** Scenes that gate their content behind a button often paint nothing on cursor 0. The first screenshot the user sees is empty space. Always paint *something* meaningful (a noise sample, a faint preview) at cursor 0 so the affordance is clear.

**Off-by-one / cluster grouping.** Group identity colors (the `cluster-N` palette) cross-talk if reused for emphasis. If a callout is amber but amber is also a cluster, students will infer a non-existent group identity. See the cross-talk rule above; reserve a third hue for emphasis.

### Forcing the review

Before reporting a scene as done, write the four-line review out loud:
1. "I screenshotted scene N at 1400×900 light theme."
2. "I `Read` the PNG and looked at it."
3. "I checked it against §Aesthetic-issue checklist — nothing in the list is present."
4. "I also screenshotted dark theme and the issues from §3 are not present there either."

If you can't write this out, the scene isn't done. Don't claim it is.

Real story (this skill, May 2026): a five-agent fan-out shipped six scenes that all *parsed* and all *visually drew* — and then user inspection found three different scenes with cropping issues, one chart where two curves were mutually invisible, one scene with a fade-in caught mid-transition, and a generated digit pane that looked OK in scene 5's vector field but didn't reverse to the right shape in scene 6 (the model trained 1500 steps wasn't enough — needed 3000). Every one of these was visible in a screenshot the agent took but didn't actually open. The cost was a full revisit pass. The lesson: the screenshot is only verification if you look at it, and you only know what you saw if you write it down.

## Things to never do

- **Inline cluster/categorical colors** (`fill="#..."` in SVG, `style="color: #..."` in DOM). Use CSS classes (`.cluster-N`, `.voronoi-cell.cluster-N`) so theme switch works.
- **Inject `<style>` tags from JS.** Write per-scene CSS files, link them in `index.html`.
- **Auto-run animations on `onEnter`** for interactive scenes (defeats the click-to-think loop). The `&run` hash flag is the *only* exception, and only for headless verification.
- **Display the answer before the student attempts it** — no HUD pill showing the optimal value.
- **Add legends explaining colour conventions inside the viz** — the conventions speak for themselves once the student plays.
- **Re-implement existing widgets from scratch.** Copy from the reference implementations (when available) and adapt.
- **Skip the browser visual check** in BOTH themes, and skip the headless screenshot for any scene with non-trivial layout.
- **Fabricate examples, numbers, labels, narrative anchors, *or stand-in synthetic data presented as real*.** Every stated value must come from `DATA` or be computed via a referenced math utility. Convincingly-noisy synthetic stand-ins (e.g. fake roll-call cells generated from a senator's PC1 position because the real matrix wasn't loaded yet) are still fabrication. **A sharp viewer will check.** Hold the scene back rather than ship a placeholder.
- **Add a fixed-position widget without reserving the layout space it occupies.** It will overlap the scene's right column.
- **Split a tonal triplet of scenes** (e.g. assign/update/converge) across multiple agents.
- **Skip the data-prep verification step.** The agent's invariant claim ("J monotone") must be asserted in code, not stated in a final report.
- **Trust agent reports without spot-checking** the produced files for the invariants they claim.
- **Attach D3 click handlers in `.enter()` only** when the handler can change after first render. Apply `.on('click', …)` and `.style('cursor', …)` on the **merged** selection (`enter().merge(update)`), and re-render after swapping handlers. Otherwise existing nodes silently keep the original handler — or none at all.
- **Reuse a group-identity color for a second encoding.** If category is encoded in red/blue, don't also use red for "important" elsewhere. Reserve a third hue.
- **Use `Math.random()` directly in a precompute script.** Use a seeded RNG (Mulberry32). Otherwise the data quietly drifts every regeneration.
- **Leave a `<canvas>` (or `<svg>` with intrinsic size) inside a flex/grid container without explicit CSS dimensions.** Default `align-items: stretch` will silently inflate or squash it — `width="280" height="360"` HTML attributes are *intrinsic* sizes; the browser still scales the layout box to its parent. The kicker: it often *renders fine interactively* (Chrome's main renderer reflows generously) and *captures wrong under headless* (the headless renderer respects the squashed box). Two-line fix:
  ```css
  .my-canvas {
    width: 280px;          /* lock the layout box */
    height: 360px;
    flex: 0 0 auto;        /* don't let flex stretch us */
  }
  ```
  Match the HTML attributes to the CSS values exactly. If you need a responsive canvas, pick a parent-relative `width: 100%` + `aspect-ratio` instead and pin `flex: 0 0 auto`. (Real story: a separate viz lost two days to "but it works on my machine" because the screenshot agent was capturing a thin slice while the live page looked correct — fixed by adding the three CSS lines above.)

## Recipe for a new viz

0. **Validate that the data tells the story.** Before any UI code, run a 5-line script that prints the key statistics across the dataset. If the data doesn't tell the intended story (e.g. "the cluster separation should grow in the second half of the trajectory"), you'll know before investing in scrollytelling infrastructure. The viz is a faithful renderer; the data is the story.

1. Pick the closest existing viz as a template. If your current repo has a similar viz, copy its directory. Otherwise reach into the `pca-kmeans-rf` reference repo (or whichever you've cloned) — `kmeans-deepdive/` for click-step + math-heavy, `congress-story/` for scrollytelling, `movies-pca/` for PCA-specific patterns.

2. Copy its directory to `<sibling-name>/`. Strip the body of scene-specific content; keep shell, topbar, controls, scene engine, theme toggle, hash router.

3. Inline the new data at `data/datasets.js`. If pre-canned trajectories are needed, dispatch a data-prep agent in parallel with foundation work — its **first** task is the validation from step 0 (does the data tell the intended story?), its **second** is to write `data/datasets.js`, its **third** is to assert downstream invariants in code (monotone J, init-sensitivity gap > N%, no NaN). Don't treat the agent as a black box — its invariant claims must be checkable from the produced files.

4. Plan the scene list; group scenes for tonal coherence; identify the parallel-agent fan-out.

5. Build foundation (theme, scene engine, shared utilities, scene stubs).

6. Dispatch scene agents in parallel with strict contracts (use the prompt template from §Parallel-agent fan-out).

7. Aggregate: link scene CSS, parse-check, browser-verify both themes, headless-screenshot any scene with non-trivial layout.

8. Iterate based on user feedback. Common asks:
   - Add Voronoi / partition cells where the algorithm partitions space.
   - Remove persistent widgets that overlap content.
   - Fix tonal drift if a triplet was accidentally split.
   - Replace synthetic stand-in data with real data.

9. **Pick the right iteration scale.** Targeted edits (single concept across ≤3 files): do directly, no agent. Cross-cutting additions across many scenes (e.g. add a visual layer everywhere pertinent): one focused agent with a per-scene plan, never full fan-out — the existing logic must not break. Major redesigns or new scenes: dispatch as Phase-1 agents per the parallel fan-out section. Wrong scale wastes either the agent's context or your own.

10. Don't commit. Hand back to the user for review.
