// Doza Assist: how a clip is shown in the clip library (1.0.47).
//
// Pure helpers, no DOM: the same file runs in the browser (as
// window.dozaClipText, loaded by project.html before its inline script) and
// under Node for core/tests/test_clip_titles.py.
//
// A clip has `text` (what the add wrote: the transcript fragment for brush,
// Story Brief and soundbite clips, a headline for Chat and story clips) and,
// once the titles route has run, `title` (the display title). The page shows
// the title, the first line of the transcript under it, and the whole
// transcript behind a Show text toggle.
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (root) root.dozaClipText = api;
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';

  const LEAD_MAX = 140;

  function squash(text) {
    return String(text || '').replace(/\s+/g, ' ').trim();
  }

  // The opening of a transcript as one line: the first sentence when it is
  // short enough, else a word-boundary cut with an ellipsis.
  // A lead shorter than this keeps taking sentences, so a clip that starts
  // on the tail of a sentence does not read "daybreak?" and nothing else.
  const LEAD_MIN = 36;

  function firstLine(text, maxChars) {
    const max = maxChars || LEAD_MAX;
    const t = squash(text);
    if (!t) return '';
    if (t.length <= max) return t;
    const re = /[.!?]["')\]]*(\s|$)/g;
    let end = 0, m;
    while ((m = re.exec(t)) !== null) {
      const e = m.index + m[0].length;
      if (e > max) break;
      end = e;
      if (end >= LEAD_MIN) break;
    }
    if (end >= LEAD_MIN) return t.slice(0, end).trim();
    let cut = t.lastIndexOf(' ', max);
    if (cut < max / 2) cut = max;
    return t.slice(0, cut).replace(/[\s,;:]+$/, '') + '…';
  }

  // What the card calls the clip. A generated or carried title wins; a clip
  // still waiting on the titles route shows nothing here (the caller renders
  // its pending label); otherwise the stored text, as before.
  function displayTitle(clip) {
    if (!clip) return 'Untitled clip';
    const title = squash(clip.title);
    if (title) return title;
    if (clip._titling) return '';
    return squash(clip.text) || 'Untitled clip';
  }

  // Clips the page should send to the titles route: no title yet and not
  // already in flight or given up on for this page load.
  function needsTitle(clip) {
    if (!clip) return false;
    if (squash(clip.title)) return false;
    if (clip._titling || clip._titleFailed) return false;
    return true;
  }

  // Merge one titles-route item ({start, end, title, status}) into the
  // matching section (same range within tolerance). Returns true when a
  // title was written.
  function applyTitle(sections, item, tolerance) {
    const tol = typeof tolerance === 'number' ? tolerance : 0.5;
    if (!Array.isArray(sections) || !item) return false;
    const start = Number(item.start), end = Number(item.end);
    for (let i = 0; i < sections.length; i++) {
      const s = sections[i];
      if (!s) continue;
      if (Math.abs(Number(s.start) - start) >= tol) continue;
      if (Math.abs(Number(s.end) - end) >= tol) continue;
      const title = squash(item.title);
      if (title && title !== s.title) {
        s.title = title;
        if (item.title_auto) s.title_auto = true; else delete s.title_auto;
        return true;
      }
      return false;
    }
    return false;
  }

  return {
    LEAD_MAX: LEAD_MAX,
    firstLine: firstLine,
    displayTitle: displayTitle,
    needsTitle: needsTitle,
    applyTitle: applyTitle,
  };
});
