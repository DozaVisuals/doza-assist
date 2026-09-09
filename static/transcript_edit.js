/* Inline transcript correction.
 *
 * Double-click a word (.tw) to edit it in place. Text correction only: the
 * server never changes media timing. Every operation is pushed onto its own
 * undo stack (Cmd+Z / Ctrl+Z), separate from the label Undo button, and
 * undone by calling the inverse route.
 *
 * Page contract (project.html): PROJECT_ID, PROJECT, window.rebuildWordIndex,
 * runAnalysis, showToast. Everything is typeof-guarded so a host without
 * them (Pro Collections) degrades to read-only.
 *
 * After any successful op the affected paragraph(s) are re-fetched from
 * GET /project/<id>/transcript/paragraph-html and swapped in, then
 * window.rebuildWordIndex() re-runs the word index and transcriptInit, which
 * repaints labels by time overlap. The trim sheet forgets its memoized word
 * list on its own: it watches the container for childList mutations
 * (pro/trim/static/trim.js), and replacing a .para-block node is exactly that.
 */
(function () {
  'use strict';

  const CSS = `
.tw-editor { display: inline-flex; align-items: center; gap: 4px; vertical-align: baseline; }
.tw-edit-input { font: inherit; color: var(--text-primary); background: var(--bg-input);
  border: 1px solid var(--accent); border-radius: 3px; padding: 0 4px; min-width: 3ch; outline: none; }
.tw-edit-actions { display: inline-flex; gap: 2px; }
.tw-edit-actions button { font: inherit; font-size: 0.75em; line-height: 1.4; padding: 0 6px;
  border: 1px solid var(--border-light); border-radius: 3px; background: var(--bg-card);
  color: var(--text-secondary); cursor: pointer; }
.tw-edit-actions button:hover { color: var(--text-primary); border-color: var(--accent); }
.tw-edit-actions button.tw-edit-primary { color: #fff; background: var(--accent); border-color: var(--accent); }
.tw-picker { position: absolute; z-index: 60; min-width: 180px; padding: 6px; border: 1px solid var(--border-light);
  border-radius: 6px; background: var(--bg-card); box-shadow: 0 6px 24px rgba(0,0,0,0.35); font-size: 0.9em; }
.tw-picker-title { padding: 2px 6px 6px; color: var(--text-muted); font-size: 0.85em; }
.tw-picker button { display: block; width: 100%; text-align: left; font: inherit; padding: 4px 8px; border: 0;
  border-radius: 4px; background: transparent; color: var(--text-primary); cursor: pointer; }
.tw-picker button:hover, .tw-picker button:focus { background: var(--bg-hover); outline: none; }
.tw-picker button .tw-picker-raw { color: var(--text-muted); font-size: 0.85em; margin-left: 6px; }
.tw-picker button.tw-picker-new { color: var(--accent); }
.transcript-stale-banner { display: flex; align-items: center; gap: 10px; margin: 0 0 10px;
  padding: 8px 12px; border: 1px solid var(--border-light); border-left: 3px solid var(--accent);
  border-radius: 6px; background: var(--bg-card); color: var(--text-secondary); font-size: 0.9em; }
.transcript-stale-banner button { font: inherit; font-size: 0.9em; padding: 2px 10px;
  border: 1px solid var(--accent); border-radius: 4px; background: transparent; color: var(--accent); cursor: pointer; }
.transcript-stale-banner button:hover { background: var(--accent); color: #fff; }
`;

  // ── state ──────────────────────────────────────────────────────────────
  const SINGLE_CLICK_DELAY_MS = 320;   // under the macOS double-click interval
  let _pendingClick = null;  // {tw, timer} a held single click on a word
  let _editor = null;        // {wrap, input, tw, seg, w, original, pid, paraStart}
  const _undo = [];          // [{type, ...}] newest last
  let _busy = false;

  function _container() { return document.getElementById('transcriptContainer'); }
  function _cancelPendingClick() {
    if (_pendingClick) { clearTimeout(_pendingClick.timer); _pendingClick = null; }
  }
  function _pid() { return typeof PROJECT_ID !== 'undefined' ? PROJECT_ID : null; }
  function _toast(msg, isError) {
    if (typeof showToast === 'function') showToast(msg, !!isError);
  }

  function _injectCss() {
    if (document.getElementById('transcriptEditCss')) return;
    const el = document.createElement('style');
    el.id = 'transcriptEditCss';
    el.textContent = CSS;
    document.head.appendChild(el);
  }

  // ── server ─────────────────────────────────────────────────────────────
  async function _post(pid, op, body) {
    const resp = await fetch(`/project/${encodeURIComponent(pid)}/transcript/${op}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await resp.json(); } catch (e) { data = null; }
    if (!resp.ok) {
      const err = new Error((data && data.error) || `${op} failed (${resp.status})`);
      err.status = resp.status;
      err.data = data;
      throw err;
    }
    return data;
  }

  // ── paragraph refresh ──────────────────────────────────────────────────
  function _blocksFor(pid) {
    const tc = _container();
    if (!tc) return [];
    return Array.from(tc.querySelectorAll('.para-block')).filter(b =>
      !pid || !b.dataset.project || b.dataset.project === pid);
  }

  function _paraBlockAt(start, pid) {
    const want = parseFloat(start);
    let best = null;
    for (const b of _blocksFor(pid)) {
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
   * Re-render every paragraph overlapping [start, end] for `projectId` and
   * swap them in place of the blocks currently covering that range.
   * Resolves to the first fresh .para-block, or null when nothing changed.
   */
  async function refreshRange(start, end, projectId) {
    const pid = projectId || _pid();
    if (!pid) return null;
    const anchor = _paraBlockAt(start, pid);
    if (!anchor) return null;

    const params = new URLSearchParams({ start: String(start), end: String(end == null ? start : end) });
    const color = _projectColor(anchor);
    if (color) { params.set('multi', '1'); params.set('color', color); }

    let data;
    try {
      const resp = await fetch(`/project/${encodeURIComponent(pid)}/transcript/paragraph-html?${params}`);
      if (!resp.ok) return null;
      data = await resp.json();
    } catch (e) { return null; }
    if (!data || !data.html) return null;

    const tpl = document.createElement('template');
    tpl.innerHTML = data.html.trim();
    const fresh = Array.from(tpl.content.querySelectorAll('.para-block'));
    if (!fresh.length) return null;

    // Remove every current block whose start lies inside the re-rendered
    // span, then insert the fresh ones where the first of them stood.
    const lo = parseFloat(data.start) - 1e-6;
    const hi = parseFloat(data.end) - 1e-6;
    const stale = _blocksFor(pid).filter(b => {
      const s = parseFloat(b.dataset.start);
      return !isNaN(s) && s >= lo && s < hi;
    });
    if (!stale.includes(anchor)) stale.unshift(anchor);
    const marker = document.createComment('tw-refresh');
    stale[0].parentNode.insertBefore(marker, stale[0]);
    stale.forEach(b => b.remove());
    fresh.forEach(b => marker.parentNode.insertBefore(b, marker));
    marker.remove();

    if (typeof window.rebuildWordIndex === 'function') window.rebuildWordIndex();
    _applySpeakerDisplay(fresh);
    return fresh[0];
  }

  // The partial renders raw speaker labels; the page shows display names.
  // Pro's diarization module owns that mapping when present, otherwise map
  // through the project's speaker_names snapshot.
  function _applySpeakerDisplay(blocks) {
    const locked = document.querySelector('.para-speaker.diar-locked-cycle');
    if (locked) {
      blocks.forEach(b => b.querySelectorAll('.para-speaker').forEach(el => {
        el.classList.add('diar-locked-cycle');
        el.title = locked.title;
      }));
    }
    if (window.diarization && typeof window.diarization.applySpeakerNamesToDOM === 'function') {
      window.diarization.applySpeakerNamesToDOM();
      return;
    }
    const names = (typeof PROJECT !== 'undefined' && PROJECT && PROJECT.speaker_names) || null;
    if (!names) return;
    blocks.forEach(b => b.querySelectorAll('.para-speaker').forEach(el => {
      const raw = el.dataset.raw || el.textContent.trim();
      if (names[raw]) { el.dataset.raw = raw; el.textContent = names[raw]; }
    }));
  }

  function refreshParagraph(start, projectId) {
    return refreshRange(start, start, projectId);
  }

  // ── stale banner ───────────────────────────────────────────────────────
  function _showStaleBanner(hasAnalysis) {
    if (!hasAnalysis) return;
    const tc = _container();
    if (!tc || !tc.parentNode) return;
    let banner = document.getElementById('transcriptStaleBanner');
    if (!banner) {
      banner = document.createElement('div');
      banner.id = 'transcriptStaleBanner';
      banner.className = 'transcript-stale-banner';
      const msg = document.createElement('span');
      msg.textContent = 'Transcript edited since last analysis. Re-run Analysis to update soundbites and story beats.';
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = 'Re-run';
      btn.addEventListener('click', () => {
        banner.hidden = true;
        if (typeof runAnalysis === 'function') runAnalysis();
      });
      banner.appendChild(msg);
      banner.appendChild(btn);
      tc.parentNode.insertBefore(banner, tc);
    }
    banner.hidden = false;
  }

  function _initialStale() {
    if (typeof PROJECT === 'undefined' || !PROJECT) return;
    if (PROJECT.derived_stale && PROJECT.analysis) _showStaleBanner(true);
  }

  // ── inline editor ──────────────────────────────────────────────────────
  function _closeEditor(restore) {
    _closePicker();
    if (!_editor) return;
    const ed = _editor;
    _editor = null;
    if (!ed.wrap.parentNode) return;
    if (ed.insert) ed.wrap.remove();               // the word it followed never left the DOM
    else if (restore) ed.wrap.replaceWith(ed.tw);
  }

  function _button(label, primary, onClick) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = label;
    if (primary) b.className = 'tw-edit-primary';
    b.addEventListener('mousedown', e => e.preventDefault()); // keep input focus
    b.addEventListener('click', e => { e.stopPropagation(); onClick(); });
    return b;
  }

  function _openEditor(tw) {
    const seg = parseInt(tw.dataset.seg, 10);
    const w = parseInt(tw.dataset.w, 10);
    if (isNaN(seg) || seg < 0) {
      _toast('This transcript cannot be edited inline.', true);
      return;
    }
    _closeEditor(true);
    const para = tw.closest('.para-block');
    const pid = (para && para.dataset.project) || _pid();
    const original = tw.textContent.trim();

    const wrap = document.createElement('span');
    wrap.className = 'tw-editor';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'tw-edit-input';
    input.value = original;
    input.size = Math.max(3, original.length + 1);
    input.setAttribute('aria-label', 'Edit word');
    const actions = document.createElement('span');
    actions.className = 'tw-edit-actions';
    actions.appendChild(_button('Save', true, () => _saveEdit()));
    if (w >= 0) actions.appendChild(_button('Insert after', false, () => _insertAfter()));
    if (w > 0) actions.appendChild(_button('Split here', false, () => _splitHere()));
    actions.appendChild(_button('Speaker', false, () => _reassignHere()));
    wrap.appendChild(input);
    wrap.appendChild(actions);

    // Keep the brush drag, jumpTo and label context menu out of the editor.
    ['mousedown', 'mouseup', 'click', 'dblclick', 'contextmenu'].forEach(evt =>
      wrap.addEventListener(evt, e => e.stopPropagation()));
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === 'Return' || e.keyCode === 13) { e.preventDefault(); _saveEdit(); }
      else if (e.key === 'Escape' || e.key === 'Esc' || e.keyCode === 27) { e.preventDefault(); _closeEditor(true); }
      e.stopPropagation();
    });
    input.addEventListener('input', () => { input.size = Math.max(3, input.value.length + 1); });

    tw.replaceWith(wrap);
    _editor = { wrap, input, tw, seg, w, original, pid, paraStart: para ? parseFloat(para.dataset.start) : NaN };
    input.focus();
    input.select();
  }

  // "Insert after": put the word back, then open an empty input right
  // after it; Enter commits through insert-word.
  function _insertAfter() {
    if (!_editor || _busy) return;
    const ed = _editor;
    _closeEditor(true);
    const para = ed.tw.closest('.para-block');
    const wrap = document.createElement('span');
    wrap.className = 'tw-editor tw-editor-insert';
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'tw-edit-input';
    input.placeholder = 'new word';
    input.size = 8;
    input.setAttribute('aria-label', 'Insert word');
    const actions = document.createElement('span');
    actions.className = 'tw-edit-actions';
    actions.appendChild(_button('Insert', true, () => _saveInsert()));
    wrap.appendChild(input);
    wrap.appendChild(actions);
    ['mousedown', 'mouseup', 'click', 'dblclick', 'contextmenu'].forEach(evt =>
      wrap.addEventListener(evt, e => e.stopPropagation()));
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === 'Return' || e.keyCode === 13) { e.preventDefault(); _saveInsert(); }
      else if (e.key === 'Escape' || e.key === 'Esc' || e.keyCode === 27) { e.preventDefault(); _closeEditor(true); }
      e.stopPropagation();
    });
    input.addEventListener('input', () => { input.size = Math.max(8, input.value.length + 1); });
    ed.tw.after(wrap);
    // tw stays in the DOM; on close the wrapper is simply removed.
    _editor = { wrap, input, tw: null, seg: ed.seg, w: ed.w, original: '', pid: ed.pid,
                paraStart: para ? parseFloat(para.dataset.start) : NaN, insert: true };
    input.focus();
  }

  async function _saveInsert() {
    if (!_editor || !_editor.insert || _busy) return;
    const ed = _editor;
    const text = ed.input.value.trim();
    if (!text) { _closeEditor(true); return; }
    _busy = true;
    try {
      const data = await _post(ed.pid, 'insert-word', { seg: ed.seg, after_w: ed.w, text });
      _closeEditor(false);
      _undo.push({ type: 'insert', pid: ed.pid, seg: ed.seg, w: data.w, text: data.word.word,
                   paraStart: data.paragraph_start });
      await refreshParagraph(data.paragraph_start, ed.pid);
      _showStaleBanner(data.has_analysis);
    } catch (err) {
      _closeEditor(true);
      await _handleError(err, ed.pid, ed.paraStart);
    } finally {
      _busy = false;
    }
  }

  async function _saveEdit() {
    if (!_editor || _busy || _editor.insert) return;
    const ed = _editor;
    const text = ed.input.value.trim();
    if (!text || text === ed.original) { _closeEditor(true); return; }
    const body = { seg: ed.seg, expected: ed.original, text };
    if (ed.w >= 0) body.w = ed.w;
    _busy = true;
    try {
      const data = await _post(ed.pid, 'edit-word', body);
      _closeEditor(false);
      _undo.push({ type: 'edit', pid: ed.pid, seg: ed.seg, w: ed.w, oldText: ed.original, newText: text,
                   paraStart: data.paragraph_start });
      await refreshParagraph(data.paragraph_start, ed.pid);
      _showStaleBanner(data.has_analysis);
    } catch (err) {
      _closeEditor(true);
      await _handleError(err, ed.pid, ed.paraStart);
    } finally {
      _busy = false;
    }
  }

  async function _handleError(err, pid, paraStart) {
    if (err.status === 409) {
      const cur = err.data && err.data.current;
      _toast(cur ? `That word changed since the page loaded (now "${String(cur).trim()}"). Refreshed.` : err.message, true);
      if (!isNaN(paraStart)) await refreshParagraph(paraStart, pid);
    } else {
      _toast(err.message || 'Edit failed', true);
    }
  }

  // ── segment index bookkeeping ──────────────────────────────────────────
  // A split or merge renumbers every later segment. Blocks outside the
  // re-rendered range keep their DOM, so shift their data-seg in place.
  function _shiftSegIndices(fromSeg, delta, pid) {
    _blocksFor(pid).forEach(b => b.querySelectorAll('.tw[data-seg]').forEach(tw => {
      const i = parseInt(tw.dataset.seg, 10);
      if (!isNaN(i) && i >= fromSeg) tw.dataset.seg = String(i + delta);
    }));
  }

  // ── split + speaker picker ─────────────────────────────────────────────
  let _picker = null;

  function _closePicker() {
    if (_picker) { _picker.remove(); _picker = null; }
  }

  async function _fetchSpeakers(pid) {
    try {
      const resp = await fetch(`/project/${encodeURIComponent(pid)}/transcript/speakers`);
      if (!resp.ok) return [];
      const data = await resp.json();
      return Array.isArray(data.speakers) ? data.speakers : [];
    } catch (e) { return []; }
  }

  /**
   * Speaker picker anchored under `anchorEl`: the current raw labels resolved
   * through speaker_names, plus "New speaker". Resolves to the chosen raw
   * label, '__new__', or null when dismissed.
   */
  function _pickSpeaker(anchorEl, speakers, currentRaw, titleText) {
    _closePicker();
    return new Promise(resolve => {
      const box = document.createElement('div');
      box.className = 'tw-picker';
      const title = document.createElement('div');
      title.className = 'tw-picker-title';
      title.textContent = titleText || 'Speaker for the new segment';
      box.appendChild(title);
      const done = value => { _closePicker(); document.removeEventListener('mousedown', onDoc, true); resolve(value); };
      const add = (label, raw, cls, hint = true) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.textContent = label;
        if (hint && raw && raw !== label) {
          const hint = document.createElement('span');
          hint.className = 'tw-picker-raw';
          hint.textContent = raw;
          b.appendChild(hint);
        }
        if (cls) b.classList.add(cls);
        b.addEventListener('click', e => { e.stopPropagation(); done(raw); });
        box.appendChild(b);
        return b;
      };
      add(`Keep ${_displayFor(speakers, currentRaw)}`, currentRaw === undefined ? null : currentRaw, null, false);
      speakers.filter(sp => sp.raw !== currentRaw).forEach(sp => add(sp.display, sp.raw));
      add('New speaker', '__new__', 'tw-picker-new');
      ['mousedown', 'mouseup', 'click', 'dblclick'].forEach(evt => box.addEventListener(evt, e => e.stopPropagation()));
      box.addEventListener('keydown', e => { if (e.key === 'Escape') { e.preventDefault(); done(null); } });
      const onDoc = e => { if (!box.contains(e.target)) done(null); };
      document.addEventListener('mousedown', onDoc, true);

      const r = anchorEl.getBoundingClientRect();
      box.style.left = `${Math.round(r.left + window.scrollX)}px`;
      box.style.top = `${Math.round(r.bottom + window.scrollY + 4)}px`;
      document.body.appendChild(box);
      _picker = box;
      const first = box.querySelector('button');
      if (first) first.focus();
    });
  }

  function _displayFor(speakers, raw) {
    const hit = speakers.find(sp => sp.raw === raw);
    return hit ? hit.display : (raw || 'this speaker');
  }

  function _rawSpeakerOf(block) {
    const el = block && block.querySelector('.para-speaker');
    if (!el) return undefined;
    return el.dataset.raw || el.textContent.trim();
  }

  async function _splitHere() {
    if (!_editor || _busy) return;
    const ed = _editor;
    if (ed.w <= 0) { _toast('Split at the first word is not possible.', true); return; }
    const anchor = ed.wrap;
    const block = anchor.closest('.para-block');
    const currentRaw = _rawSpeakerOf(block);
    const speakers = await _fetchSpeakers(ed.pid);
    const choice = await _pickSpeaker(anchor, speakers, currentRaw);
    if (choice === null) return;                       // dismissed
    if (!_editor || _editor !== ed) return;            // editor went away meanwhile
    _closeEditor(true);
    const body = { seg: ed.seg, at_w: ed.w };
    const changing = choice && choice !== currentRaw;
    if (changing) body.new_speaker = choice;
    _busy = true;
    try {
      const data = await _post(ed.pid, 'split-segment', body);
      _shiftSegIndices(ed.seg + 1, 1, ed.pid);
      _undo.push({ type: 'split', pid: ed.pid, seg: ed.seg, paraStart: data.paragraph_start });
      if (data.speaker_changed) {
        // Undo order: the speaker goes back first, then the halves merge.
        _undo.push({ type: 'reassign', pid: ed.pid, seg: data.new_seg, oldSpeaker: currentRaw,
                     oldManual: false, newSpeaker: data.speaker, paraStart: data.paragraph_start });
      }
      await refreshRange(data.paragraph_start, data.paragraph_end, ed.pid);
      await _afterSpeakerChange(data, ed.pid);
      _showStaleBanner(data.has_analysis);
    } catch (err) {
      await _handleError(err, ed.pid, ed.paraStart);
    } finally {
      _busy = false;
    }
  }

  async function _reassignHere() {
    if (!_editor || _busy) return;
    const ed = _editor;
    const block = ed.wrap.closest('.para-block');
    const currentRaw = _rawSpeakerOf(block);
    const speakers = await _fetchSpeakers(ed.pid);
    const choice = await _pickSpeaker(ed.wrap, speakers, currentRaw, 'Speaker for this segment');
    if (choice === null) return;
    if (!_editor || _editor !== ed) return;
    _closeEditor(true);
    if (!choice || choice === currentRaw) return;
    _busy = true;
    try {
      const data = await _post(ed.pid, 'reassign-speaker', { seg: ed.seg, speaker: choice });
      _undo.push({ type: 'reassign', pid: ed.pid, seg: ed.seg, oldSpeaker: data.previous_speaker,
                   oldManual: data.previous_manual, newSpeaker: data.speaker, paraStart: data.paragraph_start });
      await refreshRange(data.paragraph_start, data.paragraph_end, ed.pid);
      await _afterSpeakerChange(data, ed.pid);
      _showStaleBanner(data.has_analysis);
    } catch (err) {
      await _handleError(err, ed.pid, ed.paraStart);
    } finally {
      _busy = false;
    }
  }

  // A new raw label must reach the Speakers sidebar (pro) and the page's
  // speaker_names snapshot; the display fallback handles the rest.
  async function _afterSpeakerChange(data, pid) {
    if (!data || !data.speaker_changed) return;
    const d = window.diarization;
    if (!d || typeof d.loadSpeakerNames !== 'function') return;
    await d.loadSpeakerNames();
    // The Speakers sidebar (where the user renames the new label) redraws
    // only on a status poll; ask for one now.
    if (typeof d.renderSidebarSpeakers !== 'function') return;
    try {
      const resp = await fetch(`/diarization/status/${encodeURIComponent(pid || _pid())}`, { cache: 'no-store' });
      if (resp.ok) d.renderSidebarSpeakers(await resp.json());
    } catch (e) { /* sidebar catches up on its next poll */ }
  }

  // ── undo ───────────────────────────────────────────────────────────────
  async function undo() {
    if (_busy || !_undo.length) return;
    const entry = _undo.pop();
    _busy = true;
    try {
      let data;
      if (entry.type === 'edit') {
        const body = { seg: entry.seg, expected: entry.newText, text: entry.oldText };
        if (entry.w >= 0) body.w = entry.w;
        data = await _post(entry.pid, 'edit-word', body);
        await refreshParagraph(data.paragraph_start, entry.pid);
      } else if (entry.type === 'insert') {
        data = await _post(entry.pid, 'delete-word', { seg: entry.seg, w: entry.w, expected: entry.text });
        await refreshParagraph(data.paragraph_start, entry.pid);
      } else if (entry.type === 'split') {
        data = await _post(entry.pid, 'merge-segment', { seg: entry.seg });
        _shiftSegIndices(entry.seg + 2, -1, entry.pid);
        await refreshRange(data.paragraph_start, data.paragraph_end, entry.pid);
      } else if (entry.type === 'reassign') {
        data = await _post(entry.pid, 'reassign-speaker', { seg: entry.seg, speaker: entry.oldSpeaker,
                                                            speaker_manual: entry.oldManual });
        await refreshRange(data.paragraph_start, data.paragraph_end, entry.pid);
        await _afterSpeakerChange(Object.assign({ speaker_changed: true }, data), entry.pid);
      } else {
        return;
      }
      _showStaleBanner(data && data.has_analysis);
      _toast('Transcript edit undone');
    } catch (err) {
      _undo.push(entry);
      await _handleError(err, entry.pid, entry.paraStart);
    } finally {
      _busy = false;
    }
  }

  function _isTypingTarget(el) {
    if (!el) return false;
    const tag = (el.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable;
  }

  // ── wiring ─────────────────────────────────────────────────────────────
  function _wire() {
    const tc = _container();
    if (!tc || tc.dataset.twEditWired) return;
    tc.dataset.twEditWired = '1';
    _injectCss();

    // A word's inline jumpTo seeks AND plays, and the first click of a
    // double-click used to fire it before the double-click was known, so
    // opening the editor started playback. Every single click on a word is
    // now held briefly in the capture phase; a second click cancels it, so
    // a double-click never touches the player and a single click still
    // jumps (a beat later than before).
    // The click still bubbles (menus that close on an outside click keep
    // working); only the word's own inline handler is detached for the
    // duration of the event and replayed after the delay.
    tc.addEventListener('click', e => {
      const tw = e.target.closest && e.target.closest('.tw');
      if (!tw) return;
      const handler = tw.onclick;
      if (typeof handler !== 'function') return;
      tw.onclick = null;
      setTimeout(() => { if (tw.onclick === null) tw.onclick = handler; }, 0);
      _cancelPendingClick();
      if (e.detail >= 2) return;
      _pendingClick = { tw, timer: setTimeout(() => {
        _pendingClick = null;
        if (tw.isConnected) handler.call(tw, e);
      }, SINGLE_CLICK_DELAY_MS) };
    }, true);

    tc.addEventListener('dblclick', e => {
      const tw = e.target.closest && e.target.closest('.tw');
      if (!tw) return;
      e.preventDefault();
      e.stopPropagation();
      _cancelPendingClick();
      const sel = window.getSelection && window.getSelection();
      if (sel && sel.removeAllRanges) sel.removeAllRanges();
      _openEditor(tw);
    });

    document.addEventListener('mousedown', e => {
      if (_editor && !_editor.wrap.contains(e.target)) _closeEditor(true);
    });

    document.addEventListener('keydown', e => {
      if (!(e.metaKey || e.ctrlKey) || e.shiftKey || e.altKey) return;
      if (e.key.toLowerCase() !== 'z') return;
      if (_editor || _isTypingTarget(document.activeElement)) return;
      if (!_undo.length) return;
      e.preventDefault();
      undo();
    });

    _initialStale();
  }

  window.dozaTranscriptEdit = Object.assign(window.dozaTranscriptEdit || {}, {
    refreshParagraph,
    refreshRange,
    undo,
    openEditor: _openEditor,
    undoDepth: () => _undo.length,
    _pushUndo: entry => _undo.push(entry),
    shiftSegIndices: _shiftSegIndices,
    _post,
    showStaleBanner: _showStaleBanner,
  });

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', _wire);
  else _wire();
})();
