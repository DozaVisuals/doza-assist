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

    // Create new section
    const id = ++sectionIdCounter;
    labelSections.push({ id, start: startTime, end: endTime, color, text: text.substring(0, 200) });

    renderAllHighlights();
    updateSelectCount();
    saveLabels();
}

function findSectionAt(time) {
    return labelSections.find(s => time >= s.start && time <= s.end);
}

function removeSection(id) {
    labelSections = labelSections.filter(s => s.id !== id);
    renderAllHighlights();
    updateSelectCount();
    saveLabels();
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
}

function clearAllLabels() {
    if (!confirm('Remove all color labels?')) return;
    labelSections = [];
    renderAllHighlights();
    updateSelectCount();
    saveLabels();
    if (typeof showToast === 'function') showToast('Labels cleared');
}

function saveLabels() {
    clearTimeout(saveLabelTimeout);
    saveLabelTimeout = setTimeout(async () => {
        const sections = labelSections.map(s => ({
            start: s.start, end: s.end, color: s.color, text: s.text
        }));
        try {
            await fetch(`/project/${PROJECT_ID}/labels`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ color_labels: colorLabels, labeled_sections: sections }),
            });
        } catch (err) {
            console.error('Failed to save labels:', err);
        }
    }, 500);
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

    // Reset and reload labeled sections.
    labelSections = [];
    sectionIdCounter = 0;
    const saved = opts.labeledSections || [];
    saved.forEach(sec => {
        const id = ++sectionIdCounter;
        labelSections.push({
            id,
            start: sec.start,
            end: sec.end,
            color: sec.color,
            text: sec.text || '',
        });
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
}
