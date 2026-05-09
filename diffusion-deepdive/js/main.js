/* Scene engine + driver.
 * Scenes register builders on window.scenes.scene<N>(root) -> { onEnter?, onLeave?, onNextKey?, onPrevKey? }
 * Returning true from onNextKey/onPrevKey consumes the keystroke (advances internal step). */

(function () {
  const SCENE_TITLES = [
    "",                                  // 0: title (hero)
    "One forward step",                  // 1
    "All the way to chaos",              // 2
    "The shortcut",                      // 3
    "Reverse — wishful thinking",        // 4
    "Train the oracle",                  // 5
    "Generate",                          // 6
    "Takeaways"                          // 7
  ];

  let current = -1;
  const sceneNodes = [];
  const sceneState = [];

  function init() {
    Theme.init();

    document.getElementById('theme-toggle').addEventListener('click', () => Theme.toggle());

    const dotsEl = document.getElementById('dots');
    SCENE_TITLES.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.className = 'dot';
      dot.setAttribute('data-idx', i);
      dot.setAttribute('aria-label', `Scene ${i}`);
      dot.addEventListener('click', () => goTo(i));
      dotsEl.appendChild(dot);
    });

    document.getElementById('prev-btn').addEventListener('click', handlePrev);
    document.getElementById('next-btn').addEventListener('click', handleNext);

    window.addEventListener('keydown', e => {
      if (e.target && /input|textarea|select/i.test(e.target.tagName || '')) return;
      if (e.key === 'ArrowRight') handleNext();
      else if (e.key === 'ArrowLeft') handlePrev();
    });

    window.addEventListener('hashchange', () => {
      const n = readHashScene();
      if (n != null && n !== current) goTo(n);
    });

    goTo(readHashScene() ?? 0);
  }

  function readHashScene() {
    const m = (window.location.hash || '').match(/scene=(\d+)/);
    if (!m) return null;
    const n = parseInt(m[1], 10);
    return (Number.isFinite(n) && n >= 0 && n < SCENE_TITLES.length) ? n : null;
  }

  function writeHashScene(idx) {
    const hash = window.location.hash || '';
    const flags = hash.match(/[#&](run|runAll|test=[^&]+)/g) || [];
    const flagStr = flags.map(f => f.replace(/^#/, '&')).join('');
    const next = `#scene=${idx}${flagStr}`;
    if (next !== hash) {
      try { history.replaceState(null, '', next); } catch (e) { window.location.hash = next; }
    }
  }

  function handleNext() {
    const handled = sceneState[current] && sceneState[current].onNextKey
      ? sceneState[current].onNextKey() : false;
    if (!handled) goTo(current + 1);
  }

  function handlePrev() {
    const handled = sceneState[current] && sceneState[current].onPrevKey
      ? sceneState[current].onPrevKey() : false;
    if (!handled) goTo(current - 1);
  }

  function goTo(idx) {
    if (idx < 0 || idx >= SCENE_TITLES.length) return;
    if (idx === current) return;

    const stage = document.getElementById('stage');
    const oldNode = sceneNodes[current];
    if (oldNode) {
      oldNode.classList.remove('active');
      const oldState = sceneState[current];
      if (oldState && oldState.onLeave) oldState.onLeave();
    }

    if (!sceneNodes[idx]) {
      const node = document.createElement('div');
      node.className = 'scene';
      node.setAttribute('data-scene', idx);
      stage.appendChild(node);
      sceneNodes[idx] = node;
      const builder = window.scenes && window.scenes['scene' + idx];
      if (builder) {
        try {
          sceneState[idx] = builder(node) || {};
        } catch (err) {
          console.error(`Scene ${idx} builder threw:`, err);
          node.innerHTML = `<div class="scene-stub"><h2>Scene ${idx} — error</h2><p class="muted">${err.message}</p></div>`;
        }
      } else {
        node.innerHTML = `<div class="scene-stub"><h2>Scene ${idx}</h2><p class="muted">${SCENE_TITLES[idx] || '(title)'}</p><p class="muted" style="font-size:13px;margin-top:24px;">Builder not yet registered.</p></div>`;
      }
    } else if (sceneState[idx] && sceneState[idx].onEnter) {
      sceneState[idx].onEnter();
    }

    current = idx;
    setTimeout(() => sceneNodes[idx].classList.add('active'), 20);

    document.getElementById('scene-title').textContent = SCENE_TITLES[idx] || '';
    document.querySelectorAll('#dots .dot').forEach((d, i) => {
      d.classList.toggle('active', i === idx);
    });
    document.getElementById('prev-btn').disabled = idx === 0;
    document.getElementById('next-btn').disabled = idx === SCENE_TITLES.length - 1;

    writeHashScene(idx);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  window.SceneEngine = { goTo, getCurrent: () => current, SCENE_TITLES };
})();
