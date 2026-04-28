"""TF-IDF paragraph retrieval index for chat.

Pre-computes a sparse TF-IDF vector per transcript paragraph and serves
top-K cosine-similarity matches against arbitrary user queries. Built as a
zero-dependency baseline so retrieval works without numpy or an embedding
model. When Ollama exposes ``nomic-embed-text``, callers can swap in dense
embeddings without touching the chat-path integration.

The cosine score is sparse-dot-product on L2-normalized vectors, which is
mathematically identical to cosine similarity but avoids materializing dense
arrays. Per-paragraph vectors and the global IDF table are JSON-serializable
so the index persists alongside ``segment_vectors.json`` in each project's
data directory.
"""

import json
import math
import os
import re
from collections import Counter

# Stopwords kept narrow on purpose: aggressive removal hurts retrieval more
# than it helps because TF-IDF already down-weights common terms via IDF.
# This is a tiny "boilerplate-only" set — actual ranking quality comes from
# the IDF math, not the stoplist.
_STOP = frozenset({
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'have', 'in', 'is', 'it', 'of', 'on', 'or', 'that', 'the',
    'to', 'was', 'were', 'will', 'with',
})

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_INDEX_VERSION = 1


def _tokenize(text):
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class TfidfIndex:
    """Sparse TF-IDF retrieval over a fixed paragraph corpus.

    Build once per transcript via :func:`build_paragraph_index`, query many
    times. ``query_paragraphs`` returns the original paragraph dicts in
    descending cosine-similarity order.
    """

    def __init__(self, paragraphs, idf, vectors):
        self.paragraphs = paragraphs
        self.idf = idf
        self.vectors = vectors

    @classmethod
    def build(cls, paragraphs):
        # Two-pass build: first collect document frequencies, then weight
        # each paragraph's term counts by IDF and L2-normalize so cosine
        # similarity reduces to a sparse dot product.
        n_docs = len(paragraphs)
        df = Counter()
        token_lists = []
        for p in paragraphs:
            tokens = _tokenize(p.get('text', ''))
            token_lists.append(tokens)
            for tok in set(tokens):
                df[tok] += 1
        # Smoothed IDF (the +1 keeps a token that appears in every doc from
        # going to log(1) = 0 and dropping out entirely).
        idf = {tok: math.log((n_docs + 1) / (cnt + 1)) + 1 for tok, cnt in df.items()}

        vectors = []
        for tokens in token_lists:
            tf = Counter(tokens)
            total = max(1, len(tokens))
            vec = {tok: (cnt / total) * idf.get(tok, 0) for tok, cnt in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            vec = {tok: v / norm for tok, v in vec.items()}
            vectors.append(vec)
        return cls(list(paragraphs), idf, vectors)

    def _query_vector(self, query_text):
        tokens = _tokenize(query_text)
        if not tokens:
            return None
        tf = Counter(tokens)
        total = max(1, len(tokens))
        vec = {tok: (cnt / total) * self.idf.get(tok, 0) for tok, cnt in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {tok: v / norm for tok, v in vec.items() if v > 0}

    def query_paragraphs(self, query_text, k=12, min_score=0.05):
        """Return up to ``k`` paragraphs ranked by query relevance.

        ``min_score`` filters out paragraphs whose cosine similarity falls
        below the threshold — without it a TF-IDF query against a long
        transcript returns *every* paragraph that contains *any* query
        token, including incidental hits on common nouns. The threshold
        is intentionally low (0.05) because TF-IDF on short queries
        produces lower magnitudes than long ones.
        """
        q_vec = self._query_vector(query_text)
        if not q_vec:
            return []
        scored = []
        for i, vec in enumerate(self.vectors):
            score = 0.0
            # Iterate the smaller vector — the query is almost always shorter
            # than the paragraph so this is the cheap direction.
            small, large = (q_vec, vec) if len(q_vec) <= len(vec) else (vec, q_vec)
            for tok, w in small.items():
                if tok in large:
                    score += w * large[tok]
            if score >= min_score:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.paragraphs[i] for _score, i in scored[:k]]

    def to_dict(self):
        return {
            'version': _INDEX_VERSION,
            'paragraphs': self.paragraphs,
            'idf': self.idf,
            'vectors': self.vectors,
        }

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or data.get('version') != _INDEX_VERSION:
            return None
        return cls(
            paragraphs=data.get('paragraphs') or [],
            idf=data.get('idf') or {},
            vectors=data.get('vectors') or [],
        )


def build_paragraph_index(transcript, max_paragraph_seconds=60):
    """Build a TF-IDF index over the transcript's same-speaker paragraphs.

    Reuses :func:`ai_analysis._build_paragraphs` so the index granularity
    matches the chat path's grounding unit — every retrieved hit corresponds
    to a paragraph the chat prompt formatter knows how to render.
    """
    from ai_analysis import _build_paragraphs
    paragraphs = _build_paragraphs(transcript, max_paragraph_seconds=max_paragraph_seconds)
    return TfidfIndex.build(paragraphs)


def save_index(index, path):
    """Persist ``index`` as JSON. Caller owns directory creation."""
    if not isinstance(index, TfidfIndex):
        return False
    with open(path, 'w') as f:
        json.dump(index.to_dict(), f)
    return True


def load_index(path):
    """Load a previously saved index. Returns ``None`` on missing/invalid file
    so callers can degrade gracefully to keyword/vector-only retrieval.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return TfidfIndex.from_dict(json.load(f))
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None
