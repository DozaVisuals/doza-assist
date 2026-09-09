/* Inline transcript correction.
 *
 * Commit 1: paragraph refresh plumbing only. Exposes
 * window.dozaTranscriptEdit.refreshParagraph(start, projectId) which fetches
 * the re-rendered paragraph partial for the paragraph containing `start`,
 * swaps it into #transcriptContainer, then re-runs the page's word index
 * (window.rebuildWordIndex, project.html) so drag-to-highlight, playback
 * highlighting and the label repaint all see the new .tw nodes. The trim
 * sheet forgets its memoized word list on its own: it watches the container
 * for childList mutations (pro/trim/static/trim.js), and replacing a
 * .para-block node is exactly that.
 */
(function () {
  'use strict';

  function _container() {
    return document.getElementById('transcriptContainer');
  }

  function _paraBlockAt(start, projectId) {
    const tc = _container();
    if (!tc) return null;
    const want = parseFloat(start);
    const blocks = tc.querySelectorAll('.para-block');
    let best = null;
    for (const b of blocks) {
      if (projectId && b.dataset.project && b.dataset.project !== projectId) continue;
      const s = parseFloat(b.dataset.start);
      if (isNaN(s)) continue;
      if (s <= want + 1e-6 && (!best || s > parseFloat(best.dataset.start))) best = b;
    }
    return best;
  }

  function _projectColor(block) {
    const badge = block && block.querySelector('.para-project-badge');
    if (!badge) return null;
    const m = /var\(--([a-z0-9-]+)\)/i.exec(badge.getAttribute('style') || '');
    return m ? m[1] : 'accent';
  }

  /**
   * Re-render the paragraph containing `start` for `projectId` and swap it
   * in place. Resolves to the new .para-block, or null when nothing was
   * swapped (no matching block, request failed).
   */
  async function refreshParagraph(start, projectId) {
    const pid = projectId || (typeof PROJECT_ID !== 'undefined' ? PROJECT_ID : null);
    if (!pid) return null;
    const block = _paraBlockAt(start, pid);
    if (!block) return null;

    const params = new URLSearchParams({ start: String(start) });
    const color = _projectColor(block);
    if (color) { params.set('multi', '1'); params.set('color', color); }

    let data;
    try {
      const resp = await fetch(`/project/${encodeURIComponent(pid)}/transcript/paragraph-html?${params}`);
      if (!resp.ok) return null;
      data = await resp.json();
    } catch (e) {
      return null;
    }
    if (!data || !data.html) return null;

    const tpl = document.createElement('template');
    tpl.innerHTML = data.html.trim();
    const fresh = tpl.content.querySelector('.para-block');
    if (!fresh) return null;

    block.replaceWith(fresh);
    if (typeof window.rebuildWordIndex === 'function') window.rebuildWordIndex();
    return fresh;
  }

  window.dozaTranscriptEdit = Object.assign(window.dozaTranscriptEdit || {}, {
    refreshParagraph,
  });
})();
