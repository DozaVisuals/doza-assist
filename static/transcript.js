/**
 * Transcript tab — color labels, drag-to-clip, save/load.
 *
 * Extracted from the inline <script> in templates/project.html (Phase
 * cleanup, no behaviour change). Loaded as a classic <script src> tag
 * so top-level `let` bindings sit in the page's shared global lexical
 * environment — other inline scripts in project.html (chat tab,
 * clips tab, export tab, suggestions tab, etc.) continue to read and
 * mutate `labelSections`, `colorLabels`, `sectionIdCounter`, and call
 * `saveLabels` / `renderAllHighlights` / `updateSelectCount` exactly
 * as before.
 *
 * Globals this file ASSUMES the host page defines:
 *   - transcriptContainer  (DOM element holding the .para-block list)
 *   - allWords             (array of every .tw word span on the page)
 *   - PROJECT_ID           (string, used by saveLabels' fetch URL)
 *   - showToast(msg)       (toast helper)
 *   - _refreshExportButtonState() (export tab refresher; tolerant of
 *                          being absent — wrapped in optional call)
 *
 * Initialization: the host page calls `transcriptInit({...})` once
 * the DOM is ready, passing the per-project state pulled from the
 * Jinja2 template. The Pro Collections Transcript tab will call this
 * again whenever the editor switches to a different interview.
 */

// ── State (shared with other inline scripts via global lexical env) ──
let activeBrush = 'blue';  // Auto-select first color
let isDragging = false;
let dragStartWord = null;
let dragCurrentWord = null;
let colorLabels = {};
let labelSections = [];
let sectionIdCounter = 0;
let segmentVectors = [];
let previewedWords = new Set();
let saveLabelTimeout = null;
// Whether transcriptInit has wired the drag handlers on the current
// transcriptContainer. Idempotent — re-init replaces state but does
// NOT re-attach handlers (we rely on event delegation on a stable
// container element).
let _transcriptHandlersWired = false;


// ── Segment-vector helpers ──

// Convert HH:MM:SS / MM:SS string to seconds (for segment vector lookup)
function _segTcToSec(tc) {
    if (typeof tc === 'number') return tc;
    if (!tc) return 0;
    const parts = String(tc).split(':');
    if (parts.length === 3) return (+parts[0]) * 3600 + (+parts[1]) * 60 + parseFloat(parts[2]);
    if (parts.length === 2) return (+parts[0]) * 60 + parseFloat(parts[1]);
    return parseFloat(tc) || 0;
}

// Find the segment vector that overlaps the given time range the most.
function findSegmentVectorForRange(start, end) {
    if (!segmentVectors || !segmentVectors.length) return null;
    let best = null;
    let bestOverlap = 0;
    for (const sv of segmentVectors) {
        const ss = _segTcToSec(sv.timecode_in);
        const se = _segTcToSec(sv.timecode_out);
        const overlap = Math.max(0, Math.min(end, se) - Math.max(start, ss));
        if (overlap > bestOverlap) {
            bestOverlap = overlap;
            best = sv;
        }
    }
    return best;
}

function renderClipBadges(start, end) {
    const sv = findSegmentVectorForRange(start, end);
    if (!sv) return '';
    let html = '';
    if (sv.narrative_score === 'high') {
        html += '<span class="seg-badge seg-badge-gold" title="High narrative score">HIGH</span>';
    } else if (sv.narrative_score === 'medium') {
        html += '<span class="seg-badge seg-badge-silver" title="Medium narrative score">MED</span>';
    }
    if (sv.memory_type) {
        html += `<span class="seg-memory seg-memory-${sv.memory_type}">${sv.memory_type}</span>`;
    }
    return html;
}


// ── Color brush ──

function toggleBrush(color) {
    const wasActive = activeBrush === color;
    document.querySelectorAll('.label-swatch').forEach(s => s.classList.remove('active'));
    if (wasActive) {
        activeBrush = null;
        document.body.classList.remove('painting-mode');
        document.body.removeAttribute('data-brush');
    } else {
        activeBrush = color;
        document.querySelector(`.label-swatch[data-color="${color}"]`).classList.add('active');
        document.body.classList.add('painting-mode');
        document.body.setAttribute('data-brush', color);
    }
}

function renameLabel(color, name) {
    colorLabels[color] = name.trim() || color;
    // Update rendered badges
    document.querySelectorAll(`.label-range-tag[data-color="${color}"] .label-range-name`).forEach(el => {
        el.textContent = colorLabels[color];
    });
    saveLabels();
}


// ── Word-level drag selection ──

// Find the closest .tw word from any element (handles clicks on child nodes, labels, etc.)
function getWordFromEvent(e) {
    // Direct hit
    let word = e.target.closest('.tw');
    if (word) return word;

    // If we hit a non-word element, find the nearest word by position
    const x = e.clientX, y = e.clientY;
    const words = transcriptContainer.querySelectorAll('.tw');
    let closest = null, closestDist = Infinity;
    for (const w of words) {
        const r = w.getBoundingClientRect();
        // Only consider words roughly on the same line (within 20px vertically)
        if (y < r.top - 20 || y > r.bottom + 20) continue;
        const dx = Math.max(r.left - x, 0, x - r.right);
        const dy = Math.max(r.top - y, 0, y - r.bottom);
        const dist = dx + dy;
        if (dist < closestDist) {
            closestDist = dist;
            closest = w;
        }
    }
    return closestDist < 50 ? closest : null;
}

function updateDragPreview() {
    const range = getWordRange(dragStartWord, dragCurrentWord);
    const newSet = new Set(range);

    // Remove preview from words no longer in range
    for (const w of previewedWords) {
        if (!newSet.has(w)) w.classList.remove('tw-drag-preview');
    }
    // Add preview to new words in range
    for (const w of newSet) {
        if (!previewedWords.has(w)) w.classList.add('tw-drag-preview');
    }
    previewedWords = newSet;
}

function clearDragPreview() {
    for (const w of previewedWords) w.classList.remove('tw-drag-preview');
    previewedWords.clear();
}

function getWordRange(startWord, endWord) {
    // Get ordered range of words between start and end
    const startIdx = allWords.indexOf(startWord);
    const endIdx = allWords.indexOf(endWord);
    if (startIdx === -1 || endIdx === -1) return [];
    const lo = Math.min(startIdx, endIdx);
    const hi = Math.max(startIdx, endIdx);
    return allWords.slice(lo, hi + 1);
}

function commitSelection(startWord, endWord, color) {
    const range = getWordRange(startWord, endWord);
    if (range.length === 0) return;

    const startTime = parseFloat(range[0].dataset.s);
    const endTime = parseFloat(range[range.length - 1].dataset.e);
    const text = range.map(w => w.textContent).join('').trim();

    // Check if clicking on an existing selection to remove it
    if (range.length <= 2) {
        const existing = findSectionAt(startTime);
        if (existing && existing.color === color) {
            removeSection(existing.id);
            return;
        }
    }

    // Clips require at least 2 words. A single-word selection (e.g. a click while
    // reading through the transcript) must NOT create a clip.
    if (range.length < 2) return;

    // Snapshot BEFORE the mutation so Undo restores the pre-add state.
    _snapshotForUndo();

    // Create new section
    const id = ++sectionIdCounter;
    labelSections.push({ id, start: startTime, end: endTime, color, text: text.substring(0, 200) });

    renderAllHighlights();
    updateSelectCount();
    saveLabels();
    _notifyCollectionClip('add', { start: startTime, end: endTime, color, text: text.substring(0, 200) });
}

function findSectionAt(time) {
    return labelSections.find(s => time >= s.start && time <= s.end);
}

function removeSection(id) {
    _snapshotForUndo();
    const removed = labelSections.find(s => s.id === id);
    labelSections = labelSections.filter(s => s.id !== id);
    renderAllHighlights();
    updateSelectCount();
    saveLabels();
    if (removed) {
        _notifyCollectionClip('remove', { start: removed.start, end: removed.end, color: removed.color });
    }
}

// Optional hook for host pages (Pro Collections) to mirror transcript
// highlights into a higher-level clip library. No-op on the standalone
// single-project page, where window.onTranscriptClip is undefined — so this
// can never regress the project transcript. The host receives the section's
// time span + color plus the source interview id (PROJECT_ID) so it can
// attribute the clip to the right file inside the collection.
function _notifyCollectionClip(action, section) {
    if (typeof window === 'undefined' || typeof window.onTranscriptClip !== 'function') return;
    try {
        window.onTranscriptClip(action, {
            start: section.start,
            end: section.end,
            color: section.color,
            text: section.text || '',
            projectId: (typeof PROJECT_ID !== 'undefined') ? PROJECT_ID : null,
        });
    } catch (e) {
        console.error('[transcript] onTranscriptClip hook failed:', e);
    }
}

function renderAllHighlights() {
    // Clear label highlights (preserve tw-active for playback)
    allWords.forEach(w => {
        w.classList.remove('tw-labeled');
        w.removeAttribute('data-label-color');
        delete w.dataset.sectionId;
    });

    // Apply each section's color to its word range
    labelSections.forEach(sec => {
        allWords.forEach(w => {
            const ws = parseFloat(w.dataset.s);
            const we = parseFloat(w.dataset.e);
            // Word overlaps with section
            if (we > sec.start && ws < sec.end) {
                w.classList.add('tw-labeled');
                w.setAttribute('data-label-color', sec.color);
                w.dataset.sectionId = sec.id;
            }
        });
    });

    // Clean up old inline tags
    document.querySelectorAll('.label-range-tag').forEach(el => el.remove());
}

function updateSelectCount() {
    const el = document.getElementById('selectCount');
    if (el) el.textContent = labelSections.length;
    const exportCountEl = document.getElementById('exportCatLabelsCount');
    if (exportCountEl) exportCountEl.textContent = labelSections.length;
    if (typeof _refreshExportButtonState === 'function') _refreshExportButtonState();
    // The project page's workflow strip reads labelSections for its Clips
    // step; hosts without the strip (Pro Collections) skip this.
    if (typeof refreshWorkflowStrip === 'function') refreshWorkflowStrip();
}

// ── Undo (multi-step history) ──
//
// Every label mutation — commitSelection, removeSection, clearAllLabels —
// pushes a snapshot of labelSections onto _undoStack BEFORE mutating. The
// Undo button (#labelUndoBtn, onclick="undoLastAction()") is enabled
// whenever the stack is non-empty; each click pops the most recent snapshot
// and restores it. No redo. The stack is reset on transcriptInit() so a
// Pro Collections interview swap can't restore the wrong interview's labels.

let _undoStack = [];

function _snapshotForUndo() {
    _undoStack.push({
        sections: labelSections.map(s => ({ ...s })),
        // Clips-tab ordering mode ('time' | 'manual') rides along so Undo of
        // a drag or "Sort by time" restores HOW the array is displayed, not
        // just its contents (restoring a manual array while the mode stays
        // 'time' would leave the restored order invisible — and the next
        // drag's normalization would destroy it). typeof-guarded: the mode
        // is a top-level `let` owned by project.html; hosts without it
        // (Pro Collections) snapshot undefined and restore nothing.
        orderMode: (typeof clipOrderMode !== 'undefined') ? clipOrderMode : undefined,
    });
    _setUndoButtonEnabled(true);
}

function undoLastAction() {
    if (_undoStack.length === 0) return;
    const snap = _undoStack.pop();
    labelSections = snap.sections.map(s => ({ ...s }));
    if (snap.orderMode !== undefined && typeof clipOrderMode !== 'undefined') {
        clipOrderMode = snap.orderMode;
    }
    renderAllHighlights();
    updateSelectCount();
    saveLabels();
    _setUndoButtonEnabled(_undoStack.length > 0);
    if (typeof showToast === 'function') showToast('Undone');
}

function clearAllLabels() {
    if (!confirm('Remove all color labels?')) return;
    _snapshotForUndo();
    labelSections = [];
    renderAllHighlights();
    updateSelectCount();
    saveLabels();
    if (typeof showToast === 'function') showToast('Labels cleared');
}

function _resetUndo() {
    _undoStack = [];
    _setUndoButtonEnabled(false);
}

function _setUndoButtonEnabled(enabled) {
    const btn = document.getElementById('labelUndoBtn');
    if (!btn) return;
    btn.disabled = !enabled;
    btn.classList.toggle('btn-undo', enabled);
}

async function _doSaveLabels() {
    // speaker is optional: the Story Brief "+" carries the moment's speaker
    // (1.0.47); brush-painted and AI Analysis clips have none and omit it.
    const sections = labelSections.map(s => {
        const out = { start: s.start, end: s.end, color: s.color, text: s.text };
        if (s.speaker) out.speaker = s.speaker;
        return out;
    });
    const body = { color_labels: colorLabels, labeled_sections: sections };
    // Clips-tab ordering mode ('time' | 'manual') is owned by the host
    // page (top-level `let clipOrderMode` in project.html — shared via
    // the global lexical environment, same as labelSections). Hosts
    // without it (Pro Collections) omit the key; the server then
    // preserves the stored value.
    if (typeof clipOrderMode !== 'undefined') {
        body.clip_order_mode = clipOrderMode;
    }
    try {
        await fetch(`/project/${PROJECT_ID}/labels`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
    } catch (err) {
        console.error('Failed to save labels:', err);
    }
}

function saveLabels() {
    clearTimeout(saveLabelTimeout);
    saveLabelTimeout = setTimeout(() => {
        saveLabelTimeout = null;
        _doSaveLabels();
    }, 500);
}

// Run any pending debounced save NOW and await it. The subset-export flow
// (project.html _persistLabelSections) calls this before temporarily
// persisting a checked subset: a still-pending full-array save landing
// after the subset write — but before the export route's server-side read —
// would make a "3 selected clips" export silently ship ALL clips (and
// cancelling instead of flushing would drop the drag's clip_order_mode
// flip, since the subset writes omit that key). No-op when nothing is
// pending.
async function flushSaveLabels() {
    if (saveLabelTimeout === null) return;
    clearTimeout(saveLabelTimeout);
    saveLabelTimeout = null;
    await _doSaveLabels();
}


// ── Initialization ──

/**
 * Initialize / re-initialize the transcript tab against the current
 * page state. Called by project.html on first load with the per-
 * project state from the Jinja2 template; will be called again by
 * the Pro Collections Transcript tab each time the editor switches
 * to a different interview.
 *
 * Options:
 *   colorLabels:     dict like { 'blue': 'Hero', 'green': 'B-roll' }
 *   labeledSections: list of { start, end, color, text? } from server
 *   segmentVectors:  list of segment vectors for narrative-score
 *                    badges (optional; pass [] when absent)
 *   preserveBrush:   bool. When true, skip the activeBrush reset and
 *                    keep whatever color is currently selected (plus
 *                    the painting-mode body class + active swatch).
 *                    Used by the Pro Collections Transcript tab on
 *                    interview swap — the editor's brush choice
 *                    persists across interviews. Default false (first
 *                    load behavior: brush starts on 'blue').
 *
 * Resets all in-memory state to the supplied values, restores swatch
 * label inputs, repaints highlights, refreshes the section count,
 * and (on first call only) wires the drag handlers to
 * transcriptContainer.
 */
function transcriptInit(opts) {
    opts = opts || {};
    colorLabels = opts.colorLabels || {};
    segmentVectors = opts.segmentVectors || [];

    // Discard any pending undo history from the previous interview so the
    // Pro Collections interview swap can't restore the wrong labels.
    if (_undoStack.length) _resetUndo();

    // Reset and reload labeled sections.
    labelSections = [];
    sectionIdCounter = 0;
    const saved = opts.labeledSections || [];
    saved.forEach(sec => {
        const id = ++sectionIdCounter;
        const entry = {
            id,
            start: sec.start,
            end: sec.end,
            color: sec.color,
            text: sec.text || '',
        };
        if (sec.speaker) entry.speaker = sec.speaker;
        labelSections.push(entry);
    });

    // Restore swatch label inputs (DOM owned by the page template).
    Object.entries(colorLabels).forEach(([color, name]) => {
        const input = document.querySelector(`.swatch-label[data-color="${color}"]`);
        if (input) input.value = name;
    });

    // Auto-activate first swatch + painting mode (preserves the
    // existing default-on UX from the inline implementation). Skipped
    // when the caller passes preserveBrush:true — the previously-
    // selected brush stays active across the call.
    if (!opts.preserveBrush) {
        activeBrush = 'blue';
        document.body.classList.add('painting-mode');
        document.body.setAttribute('data-brush', 'blue');
        document.querySelectorAll('.label-swatch').forEach(s => s.classList.remove('active'));
        const firstSwatch = document.querySelector('.label-swatch[data-color="blue"]');
        if (firstSwatch) firstSwatch.classList.add('active');
    }

    // Repaint highlights against whatever .tw words are currently in
    // the DOM (the host page is responsible for `allWords` being
    // populated against the live transcript).
    renderAllHighlights();
    updateSelectCount();

    // Wire drag handlers exactly once. transcriptContainer is the
    // stable host element; even when its inner HTML is swapped for
    // a different interview, mouse events still bubble to it.
    if (!_transcriptHandlersWired && typeof transcriptContainer !== 'undefined' && transcriptContainer) {
        _wireDragHandlers();
        _transcriptHandlersWired = true;
    }
}

function _wireDragHandlers() {
    transcriptContainer.addEventListener('mousedown', (e) => {
        if (e.button !== 0) return;  // Left click only — right-click is handled by contextmenu below.
        if (!activeBrush) return;
        const word = getWordFromEvent(e);
        if (!word) return;
        e.preventDefault();
        // Prevent text selection during drag
        document.body.style.userSelect = 'none';
        document.body.style.webkitUserSelect = 'none';
        isDragging = true;
        dragStartWord = word;
        dragCurrentWord = word;
        updateDragPreview();
    });

    transcriptContainer.addEventListener('mousemove', (e) => {
        if (!isDragging || !activeBrush) return;
        e.preventDefault();
        const word = getWordFromEvent(e);
        if (word && word !== dragCurrentWord) {
            dragCurrentWord = word;
            updateDragPreview();
        }
    });

    document.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        document.body.style.userSelect = '';
        document.body.style.webkitUserSelect = '';
        clearDragPreview();
        if (dragStartWord && dragCurrentWord && activeBrush) {
            commitSelection(dragStartWord, dragCurrentWord, activeBrush);
        }
        dragStartWord = null;
        dragCurrentWord = null;
    });

    // Right-click on a labeled word → "Remove [color] highlight" menu.
    transcriptContainer.addEventListener('contextmenu', (e) => {
        const labeled = e.target.closest('.tw-labeled[data-section-id]');
        if (!labeled) return;  // Native menu on unlabeled words.
        e.preventDefault();
        const id = parseInt(labeled.dataset.sectionId, 10);
        if (Number.isNaN(id)) return;
        const colorKey = labeled.getAttribute('data-label-color') || 'blue';
        _showLabelContextMenu(e.clientX, e.clientY, id, colorKey);
    });
}


// ── Label context menu ──
//
// Lazy single-instance menu attached to <body>. Created on first right-click
// and reused — survives Pro Collections interview swaps because it doesn't
// live inside transcriptContainer.

let _ctxMenu = null;
let _ctxDismissWired = false;

const _COLOR_VAR = {
    blue:   '--accent',
    green:  '--green',
    purple: '--purple',
    orange: '--orange',
    red:    '--red',
};

function _showLabelContextMenu(x, y, sectionId, colorKey) {
    if (!_ctxMenu) {
        _ctxMenu = document.createElement('div');
        _ctxMenu.className = 'transcript-context-menu';
        _ctxMenu.id = 'transcriptContextMenu';
        _ctxMenu.style.display = 'none';
        _ctxMenu.innerHTML = `
            <div class="transcript-context-menu-item" data-action="remove">
                <span class="transcript-context-menu-dot"></span>
                <span class="transcript-context-menu-label">Remove highlight</span>
            </div>
        `;
        document.body.appendChild(_ctxMenu);

        _ctxMenu.addEventListener('click', (e) => {
            const item = e.target.closest('.transcript-context-menu-item');
            if (!item) return;
            const id = parseInt(_ctxMenu.dataset.sectionId, 10);
            _hideLabelContextMenu();
            if (!Number.isNaN(id)) removeSection(id);
        });

        // Don't let the menu's own mousedown dismiss it before click fires.
        _ctxMenu.addEventListener('mousedown', (e) => e.stopPropagation());
    }

    if (!_ctxDismissWired) {
        document.addEventListener('mousedown', (e) => {
            if (!_ctxMenu || _ctxMenu.style.display === 'none') return;
            if (_ctxMenu.contains(e.target)) return;
            _hideLabelContextMenu();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') _hideLabelContextMenu();
        });
        window.addEventListener('blur', _hideLabelContextMenu);
        window.addEventListener('resize', _hideLabelContextMenu);
        // Also dismiss on transcript scroll — the click target moves out
        // from under the cursor, so the menu would be misleading.
        if (typeof transcriptContainer !== 'undefined' && transcriptContainer) {
            transcriptContainer.addEventListener('scroll', _hideLabelContextMenu, { passive: true });
        }
        _ctxDismissWired = true;
    }

    const colorVar = _COLOR_VAR[colorKey] || '--accent';
    const labelName = (typeof colorLabels !== 'undefined' && colorLabels[colorKey])
        ? colorLabels[colorKey]
        : (colorKey.charAt(0).toUpperCase() + colorKey.slice(1));
    _ctxMenu.querySelector('.transcript-context-menu-dot').style.background = `var(${colorVar})`;
    _ctxMenu.querySelector('.transcript-context-menu-label').textContent = `Remove ${labelName} highlight`;
    _ctxMenu.dataset.sectionId = String(sectionId);

    // Show off-screen first to measure, then clamp into viewport.
    _ctxMenu.style.display = '';
    _ctxMenu.style.left = '-9999px';
    _ctxMenu.style.top = '-9999px';
    const rect = _ctxMenu.getBoundingClientRect();
    const maxX = window.innerWidth - rect.width - 8;
    const maxY = window.innerHeight - rect.height - 8;
    _ctxMenu.style.left = Math.max(8, Math.min(x, maxX)) + 'px';
    _ctxMenu.style.top  = Math.max(8, Math.min(y, maxY)) + 'px';
}

function _hideLabelContextMenu() {
    if (_ctxMenu) _ctxMenu.style.display = 'none';
}


// ── Transcript search ──
//
// Live filter + match navigation + timecode jump for the transcript tab.
// Inserted as a sibling above #labelToolbar so it sits at the top of the
// right column, transcript-tab-only. Wired by transcriptSearchInit() which
// is idempotent — the host page may call it once on first render. The
// Pro Collections Transcript tab calls transcriptSearchReset() after each
// transcriptInit() interview swap so the prior interview's match state
// doesn't leak.
//
// Highlighting strategy: class-only on .tw spans (.search-match,
// .search-match-active). Never wraps text in <mark> — that would break
// allWords identity across re-init and complicate getWordFromEvent. Match
// granularity is the .tw, which is one word in the common path; substring
// matches inside a .tw still light the whole word, which aligns with how
// editors think.

let _searchEls = null;
let _searchMatches = [];
let _searchActiveIdx = -1;
let _searchMarkedTws = new Set();
let _searchHiddenParas = new Set();
let _searchDebounceTimer = null;
let _searchHandlersWired = false;

// Accepts MM:SS, HH:MM:SS, and full SMPTE HH:MM:SS:FF / HH:MM:SS;FF —
// editors paste source timecodes straight out of their NLE.
const _TC_RE = /^\d{1,2}:\d{2}(:\d{2})?([:;]\d{1,2})?$/;

function transcriptSearchInit(container) {
    if (_searchEls) return;  // Idempotent — DOM created once per page.
    if (!container) container = (typeof transcriptContainer !== 'undefined') ? transcriptContainer : null;
    if (!container) return;

    // Anchor: insert above #labelToolbar if present, else above the
    // container's tab-content parent.
    const toolbar = document.getElementById('labelToolbar');
    const anchor = toolbar || container.closest('.tab-content');
    if (!anchor || !anchor.parentNode) return;

    const wrap = document.createElement('div');
    wrap.className = 'transcript-search';
    wrap.id = 'transcriptSearch';
    wrap.innerHTML = `
        <div class="transcript-search-inner">
            <svg class="transcript-search-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
                <circle cx="7" cy="7" r="4.5"/>
                <line x1="10.5" y1="10.5" x2="14" y2="14" stroke-linecap="round"/>
            </svg>
            <input type="text" class="transcript-search-input" id="transcriptSearchInput"
                   placeholder="Search transcript or jump to timecode (e.g. 14:30 or 14:23:11:05)"
                   spellcheck="false" autocomplete="off">
            <button class="transcript-search-clear" id="transcriptSearchClear"
                    type="button" title="Clear (Esc)" aria-label="Clear search" style="display:none;">×</button>
        </div>
        <div class="transcript-search-meta">
            <span class="transcript-search-count" id="transcriptSearchCount"></span>
            <button class="transcript-search-nav" id="transcriptSearchPrev"
                    type="button" title="Previous match (Shift+Enter)" aria-label="Previous match">↑</button>
            <button class="transcript-search-nav" id="transcriptSearchNext"
                    type="button" title="Next match (Enter)" aria-label="Next match">↓</button>
        </div>
    `;
    anchor.parentNode.insertBefore(wrap, anchor);

    _searchEls = {
        wrap,
        input: wrap.querySelector('#transcriptSearchInput'),
        clear: wrap.querySelector('#transcriptSearchClear'),
        count: wrap.querySelector('#transcriptSearchCount'),
        prev: wrap.querySelector('#transcriptSearchPrev'),
        next: wrap.querySelector('#transcriptSearchNext'),
        container,
    };

    _searchEls.input.addEventListener('input', () => {
        _searchEls.clear.style.display = _searchEls.input.value ? '' : 'none';
        clearTimeout(_searchDebounceTimer);
        _searchDebounceTimer = setTimeout(() => {
            _searchRun(_searchEls.input.value);
        }, 150);
    });

    _searchEls.input.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            e.preventDefault();
            if (_searchEls.input.value) {
                _searchEls.input.value = '';
                _searchEls.clear.style.display = 'none';
                _searchRun('');
            }
            _searchEls.input.blur();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (_searchMatches.length === 0) return;
            _searchSetActive(_searchActiveIdx + (e.shiftKey ? -1 : 1), true);
        }
    });

    _searchEls.clear.addEventListener('click', () => {
        _searchEls.input.value = '';
        _searchEls.clear.style.display = 'none';
        _searchRun('');
        _searchEls.input.focus();
    });

    _searchEls.prev.addEventListener('click', () => {
        if (_searchMatches.length) _searchSetActive(_searchActiveIdx - 1, true);
    });
    _searchEls.next.addEventListener('click', () => {
        if (_searchMatches.length) _searchSetActive(_searchActiveIdx + 1, true);
    });

    if (!_searchHandlersWired) {
        window.addEventListener('keydown', (e) => {
            if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'f') return;
            if (e.shiftKey || e.altKey) return;  // Don't hijack Cmd+Shift+F etc.
            const tab = document.getElementById('tab-transcript');
            if (!tab || !tab.classList.contains('active')) return;
            if (!_searchEls || _searchEls.wrap.style.display === 'none') return;
            e.preventDefault();
            _searchEls.input.focus();
            _searchEls.input.select();
        });
        _searchHandlersWired = true;
    }

    _searchUpdateCounter();
}

function transcriptSearchReset() {
    if (!_searchEls) return;
    clearTimeout(_searchDebounceTimer);
    _searchClearVisualState();
    _searchMatches = [];
    _searchActiveIdx = -1;
    _searchEls.input.value = '';
    _searchEls.clear.style.display = 'none';
    _searchUpdateCounter();
}

function _searchClearVisualState() {
    for (const w of _searchMarkedTws) {
        w.classList.remove('search-match', 'search-match-active');
    }
    _searchMarkedTws.clear();
    for (const p of _searchHiddenParas) p.style.display = '';
    _searchHiddenParas.clear();
}

function _searchRun(query) {
    _searchClearVisualState();
    _searchMatches = [];
    _searchActiveIdx = -1;

    const q = (query || '').trim();
    if (!q) {
        _searchUpdateCounter();
        return;
    }

    if (_TC_RE.test(q)) {
        _searchTimecodeJump(q);
        _searchUpdateCounter();
        return;
    }

    const qLower = q.toLowerCase();
    const paras = _searchEls.container.querySelectorAll('.para-block');

    for (const para of paras) {
        const tws = para.querySelectorAll('.para-text .tw');
        if (!tws.length) continue;

        let flat = '';
        const offsets = [];
        for (const tw of tws) {
            const text = tw.textContent;
            offsets.push({ tw, start: flat.length, end: flat.length + text.length });
            flat += text;
        }
        const flatLower = flat.toLowerCase();

        const ranges = [];
        let pos = 0;
        while (true) {
            const idx = flatLower.indexOf(qLower, pos);
            if (idx === -1) break;
            ranges.push({ start: idx, end: idx + qLower.length });
            pos = idx + qLower.length;
        }

        if (ranges.length === 0) {
            para.style.display = 'none';
            _searchHiddenParas.add(para);
            continue;
        }

        for (const range of ranges) {
            const matchTws = [];
            for (const o of offsets) {
                if (o.end > range.start && o.start < range.end) {
                    matchTws.push(o.tw);
                    _searchMarkedTws.add(o.tw);
                }
            }
            if (matchTws.length) {
                _searchMatches.push({ paraBlock: para, twNodes: matchTws, firstTw: matchTws[0] });
            }
        }
    }

    for (const w of _searchMarkedTws) w.classList.add('search-match');

    if (_searchMatches.length > 0) _searchSetActive(0, true);
    _searchUpdateCounter();
}

function _searchSetActive(idx, scroll) {
    if (_searchMatches.length === 0) return;
    if (idx < 0) idx = _searchMatches.length - 1;
    if (idx >= _searchMatches.length) idx = 0;

    if (_searchActiveIdx >= 0 && _searchActiveIdx < _searchMatches.length) {
        for (const w of _searchMatches[_searchActiveIdx].twNodes) {
            w.classList.remove('search-match-active');
        }
    }
    _searchActiveIdx = idx;
    const m = _searchMatches[idx];
    for (const w of m.twNodes) w.classList.add('search-match-active');

    if (scroll && m.firstTw && typeof m.firstTw.scrollIntoView === 'function') {
        m.firstTw.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    _searchUpdateCounter();
}

function _searchUpdateCounter() {
    if (!_searchEls) return;
    const total = _searchMatches.length;
    const q = (_searchEls.input.value || '').trim();
    if (!q) {
        _searchEls.count.textContent = '';
    } else if (_TC_RE.test(q)) {
        // Counter text was set by _searchTimecodeJump; leave it.
    } else if (total === 0) {
        _searchEls.count.textContent = 'No matches';
    } else {
        _searchEls.count.textContent = `${_searchActiveIdx + 1} of ${total} matches`;
    }
    const disabled = total === 0;
    _searchEls.prev.disabled = disabled;
    _searchEls.next.disabled = disabled;
}

function _searchTimecodeJump(tcStr) {
    // Strip an SMPTE frames field (HH:MM:SS:FF / ;FF) — second-level
    // precision is plenty for a transcript jump.
    let cleaned = tcStr;
    const smpte = cleaned.match(/^(\d{1,2}:\d{2}:\d{2})[:;]\d{1,2}$/);
    if (smpte) cleaned = smpte[1];
    let target = _segTcToSec(cleaned);
    // When source-TC display is on AND the typed value lands at/past the
    // project's embedded start TC, the user is pasting SOURCE timecode —
    // subtract the offset to get media-relative seconds. Anything below
    // the offset (e.g. "14:30" meaning 14m30s into the media) stays
    // relative. (Multi-project views key off the primary project's
    // offset — the only one the search field can sensibly mean.)
    if (typeof tcDisplayMode !== 'undefined' && tcDisplayMode &&
            typeof TC_BY_PROJECT !== 'undefined') {
        const tc = TC_BY_PROJECT[typeof PROJECT_ID !== 'undefined' ? PROJECT_ID : ''];
        if (tc && tc.fps) {
            const offset = tc.frames / tc.fps;
            if (target >= offset) target = target - offset;
        }
    }
    const paras = Array.from(_searchEls.container.querySelectorAll('.para-block'));
    let best = null;
    let bestStart = -Infinity;
    for (const p of paras) {
        const start = parseFloat(p.dataset.start);
        if (Number.isNaN(start)) continue;
        if (start <= target && start > bestStart) {
            best = p;
            bestStart = start;
        }
    }
    if (!best && paras.length) best = paras[0];
    if (best) {
        best.scrollIntoView({ behavior: 'smooth', block: 'center' });
        if (typeof jumpTo === 'function') {
            jumpTo(target, best.dataset.project);
        }
        _searchEls.count.textContent = `Jump to ${tcStr}`;
    } else {
        _searchEls.count.textContent = `No paragraph at ${tcStr}`;
    }
}
