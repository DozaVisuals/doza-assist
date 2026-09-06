"""Decide the engine for an Auto-detect transcription (1.1).

Only ``language == 'en'`` takes the fast Parakeet path; Auto-detect used to
go straight to Whisper, so an editor who picked Auto-detect once (the picker
was sticky) pushed every English interview through the slow engine.

The probe: transcribe the first PROBE_SECONDS of the extracted audio with
Parakeet and score how English the words are (share of tokens found in a
list of common English words). English speech scores well above the
threshold; another language pushed through an English-only model comes out
as near-gibberish and scores far below it. English routes the whole file to
Parakeet; anything else, or any failure, keeps Auto-detect and Whisper
exactly as before. transcribe.py is not touched: the job in app.py calls
this before it picks the engine.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import Callable

PROBE_SECONDS = 45
MIN_TOKENS = 12
ENGLISH_THRESHOLD = 0.5

# About 550 of the most common spoken English words (function words, common
# verbs, everyday nouns). Real conversational English lands 60 to 80 percent
# of its tokens here; other languages decoded by an English-only model
# land far lower.
COMMON_ENGLISH_WORDS = frozenset("""
the be to of and a in that have i it for not on with he as you do at this but
his by from they we say her she or an will my one all would there their what so
up out if about who get which go me when make can like time no just him know
take people into year your good some could them see other than then now look
only come its over think also back after use two how our work first well way
even new want because any these give day most us is are was were been has had
did does doing am being very really thing things something anything nothing
much many more little big great long little own same right left still also
never always often sometimes again already yet ever every each both few lot
lots kind sort part place case point week month hour minute morning night
today tomorrow yesterday early late here where why yeah yes okay ok oh well
maybe actually basically probably definitely exactly obviously honestly
question answer problem issue idea story example reason person man woman
child family friend mother father company business job money school home
house city country world life hand eye head heart face side end start
started starting stop stopped keep kept let put said says tell told talk
talked talking ask asked need needed help helped try tried find found call
called feel felt seem seemed leave left mean meant might must should shall
show showed hear heard play played run ran move moved live lived believe
believed hold held bring brought happen happened write wrote provide provided
sit sat stand stood lose lost pay paid meet met include included continue
set learn learned change changed lead led understand understood watch follow
followed create created speak spoke read allow allowed add added spend spent
grow grew open opened walk walked win won offer offered remember love
consider appear buy bought wait waited serve die send sent expect build built
stay fall cut reach kill remain suggest raise pass sell require report decide
pull return explain hope develop carry break receive agree support hit
produce eat cover catch draw choose cause listen realize wonder finish
everything everyone everybody anyone anybody someone somebody nobody nothing
another others whatever whenever wherever whether while during before after
through between among against without within along across around behind
under above below near far down off away together apart instead rather
quite pretty almost enough too either neither nor although though unless
until since because whereas however therefore anyway besides meanwhile
different important able available possible interesting difficult easy hard
happy sure clear real true simple small large young old high low next last
best better bad worse worst free full whole general public local national
we've we're we'll i'm i've i'll i'd you're you've you'll they're they've
he's she's it's that's there's what's who's here's let's don't doesn't didn't
isn't aren't wasn't weren't can't couldn't won't wouldn't shouldn't haven't
hasn't hadn't
""".split())

_TOKEN = re.compile(r"[a-z']+")


def english_score(text: str) -> tuple[float, int]:
    """(share of tokens that are common English words, token count)."""
    tokens = [t.strip("'") for t in _TOKEN.findall((text or '').lower())]
    tokens = [t for t in tokens if t]
    if not tokens:
        return 0.0, 0
    hits = sum(1 for t in tokens if t in COMMON_ENGLISH_WORDS)
    return hits / len(tokens), len(tokens)


def looks_english(text: str, threshold: float = ENGLISH_THRESHOLD,
                  min_tokens: int = MIN_TOKENS) -> bool:
    """True when the text reads as English with enough words to judge."""
    score, n = english_score(text)
    return n >= min_tokens and score >= threshold


def transcript_text(result) -> str:
    """Join the segment texts of a transcribe result."""
    if not isinstance(result, dict):
        return ''
    segs = result.get('segments') or []
    return ' '.join((s.get('text') or '').strip() for s in segs if isinstance(s, dict)).strip()


def trim_head(audio_path: str, ffmpeg: str, seconds: int = PROBE_SECONDS) -> str:
    """The first ``seconds`` of a WAV as a new temp file (caller removes)."""
    fd, out = tempfile.mkstemp(prefix='doza_langprobe_', suffix='.wav')
    os.close(fd)
    subprocess.run(
        [ffmpeg, '-y', '-v', 'error', '-i', audio_path, '-t', str(seconds),
         '-ac', '1', '-ar', '16000', out],
        check=True, capture_output=True, timeout=120)
    return out


def probe_language(audio_path: str, ffmpeg: str | None,
                   transcribe_head: Callable[[str], object],
                   seconds: int = PROBE_SECONDS) -> dict:
    """Run the probe. Returns {'language': 'en' | None, 'score', 'tokens',
    'error'}; ``language`` is None whenever the probe cannot decide, which
    the caller treats as "keep Auto-detect, use Whisper"."""
    out: dict = {'language': None, 'score': 0.0, 'tokens': 0, 'error': None}
    if not audio_path or not os.path.exists(audio_path):
        out['error'] = 'no audio to probe'
        return out
    tmp = None
    try:
        if ffmpeg:
            tmp = trim_head(audio_path, ffmpeg, seconds)
            head = tmp
        else:
            head = audio_path
        result = transcribe_head(head)
        text = transcript_text(result)
        score, n = english_score(text)
        out['score'] = round(score, 3)
        out['tokens'] = n
        if n >= MIN_TOKENS and score >= ENGLISH_THRESHOLD:
            out['language'] = 'en'
    except Exception as exc:  # any failure keeps Auto-detect
        out['error'] = str(exc) or exc.__class__.__name__
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass
    return out
