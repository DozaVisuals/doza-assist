"""
Transcript analyzer for My Style (Editorial DNA Level 1).
Extracts narrative style metrics from a transcript with word-level timestamps.
Pure Python — no AI calls. Classification handled separately in classifier.py.
"""

import re
import statistics


# A "beat" is a continuous run of speech before a >1.5s gap or speaker change
BEAT_GAP_THRESHOLD = 1.5


def analyze_transcript(transcript):
    """
    Analyze a transcript dict (same format as transcribe_file() output) and
    return all computable style metrics.

    Returns a dict with keys: speech_pacing, structural_rhythm, soundbite_craft,
    content_patterns, plus raw data for classifier.py to use.
    """
    segments = transcript.get('segments', [])
    duration = transcript.get('duration', 0)

    if not segments or duration <= 0:
        return _empty_metrics(duration)

    beats = _extract_beats(segments)
    beat_lengths = [b['duration'] for b in beats if b['duration'] > 0]

    total_words = sum(b['word_count'] for b in beats)
    total_speech_time = sum(b['duration'] for b in beats)

    # --- Speech Pacing ---
    wpm = (total_words / total_speech_time * 60) if total_speech_time > 0 else 0
    avg_beat = statistics.mean(beat_lengths) if beat_lengths else 0
    median_beat = statistics.median(beat_lengths) if beat_lengths else 0
    stddev_beat = statistics.stdev(beat_lengths) if len(beat_lengths) > 1 else 0
    longest_beat = max(beat_lengths) if beat_lengths else 0
    shortest_beat = min(beat_lengths) if beat_lengths else 0

    speech_pacing = {
        'words_per_minute': round(wpm, 1),
        'avg_beat_length': round(avg_beat, 2),
        'median_beat_length': round(median_beat, 2),
        'stddev_beat_length': round(stddev_beat, 2),
        'longest_beat': round(longest_beat, 2),
        'shortest_beat': round(shortest_beat, 2),
        # rhythm_descriptor filled by classifier
        'rhythm_descriptor': '',
    }

    # --- Structural Rhythm ---
    speech_ratio = (total_speech_time / duration) if duration > 0 else 0

    # Speaker switching: count changes in speaker across beats
    speaker_changes = 0
    for i in range(1, len(beats)):
        if beats[i]['speaker'] != beats[i - 1]['speaker']:
            speaker_changes += 1
    switches_per_min = (speaker_changes / duration * 60) if duration > 0 else 0

    # Position of longest beat
    longest_beat_pos = _classify_position(beats, duration)

    # Pacing per third
    third = duration / 3
    first_third_wpm = _wpm_in_range(beats, 0, third)
    middle_third_wpm = _wpm_in_range(beats, third, third * 2)
    last_third_wpm = _wpm_in_range(beats, third * 2, duration)

    structural_rhythm = {
        'total_duration_seconds': round(duration, 2),
        'speech_to_silence_ratio': round(speech_ratio, 3),
        'speaker_switches_per_minute': round(switches_per_min, 2),
        'longest_beat_position': longest_beat_pos,
        'pacing_first_third_wpm': round(first_third_wpm, 1),
        'pacing_middle_third_wpm': round(middle_third_wpm, 1),
        'pacing_last_third_wpm': round(last_third_wpm, 1),
        # energy_arc filled by classifier
        'energy_arc': '',
    }

    # --- Soundbite Craft ---
    sentence_end_re = re.compile(r'[.!?]["\')\]]*\s*$')
    clean_cuts = 0
    total_cuts = 0
    soundbite_lengths = []
    gaps_before_cut = []

    for i, beat in enumerate(beats):
        total_cuts += 1
        text = beat['text'].strip()
        if sentence_end_re.search(text):
            clean_cuts += 1
            soundbite_lengths.append(beat['duration'])

        # Gap before next beat
        if i < len(beats) - 1:
            gap = beats[i + 1]['start'] - beat['end']
            if gap > 0:
                gaps_before_cut.append(gap)

    clean_ratio = (clean_cuts / total_cuts) if total_cuts > 0 else 0
    avg_soundbite = statistics.mean(soundbite_lengths) if soundbite_lengths else avg_beat
    avg_gap = statistics.mean(gaps_before_cut) if gaps_before_cut else 0

    soundbite_craft = {
        'avg_soundbite_length': round(avg_soundbite, 2),
        'clean_cut_ratio': round(clean_ratio, 3),
        'avg_gap_before_cut': round(avg_gap, 3),
    }

    # --- Content Patterns ---
    all_text = ' '.join(b['text'] for b in beats)
    sentences = re.split(r'(?<=[.!?])\s+', all_text)
    questions = sum(1 for s in sentences if s.strip().endswith('?'))
    statements = sum(1 for s in sentences if s.strip() and not s.strip().endswith('?'))
    q_ratio = (questions / statements) if statements > 0 else 0

    content_patterns = {
        'question_to_statement_ratio': round(q_ratio, 3),
        # topic_count filled by classifier (needs AI)
        'topic_count': 0,
    }

    # --- Raw data for classifier ---
    first_15s_text = _text_in_range(segments, 0, 15)
    last_15s_text = _text_in_range(segments, max(0, duration - 15), duration)
    opening_third_text = _text_in_range(segments, 0, third)
    closing_third_text = _text_in_range(segments, third * 2, duration)

    return {
        'speech_pacing': speech_pacing,
        'structural_rhythm': structural_rhythm,
        'soundbite_craft': soundbite_craft,
        'content_patterns': content_patterns,
        '_raw': {
            'first_15s_text': first_15s_text,
            'last_15s_text': last_15s_text,
            'opening_third_text': opening_third_text,
            'closing_third_text': closing_third_text,
            'beat_count': len(beats),
            'total_words': total_words,
        }
    }


def merge_metrics(existing_profile, new_metrics, new_duration):
    """
    Merge new file metrics into an existing profile using duration-weighted averaging.
    Returns updated metric sections (not a full profile — caller assembles that).
    """
    if not existing_profile or 'speech_pacing' not in existing_profile:
        return new_metrics

    old_dur = existing_profile.get('structural_rhythm', {}).get('total_duration_seconds', 0)
    new_dur = new_duration
    total_dur = old_dur + new_dur

    if total_dur <= 0:
        return new_metrics

    def wavg(old_val, new_val):
        """Duration-weighted average of two values."""
        return (old_val * old_dur + new_val * new_dur) / total_dur

    merged = {}
    for section_key in ['speech_pacing', 'structural_rhythm', 'soundbite_craft', 'content_patterns']:
        old_section = existing_profile.get(section_key, {})
        new_section = new_metrics.get(section_key, {})
        merged_section = {}

        for k in new_section:
            old_v = old_section.get(k, new_section[k])
            new_v = new_section[k]
            if isinstance(new_v, (int, float)) and isinstance(old_v, (int, float)):
                merged_section[k] = round(wavg(old_v, new_v), 3)
            else:
                # Non-numeric (strings like rhythm_descriptor) — keep new
                merged_section[k] = new_v
        merged[section_key] = merged_section

    # Update total duration to reflect combined
    merged['structural_rhythm']['total_duration_seconds'] = round(total_dur, 2)

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_beats(segments):
    """
    Group transcript segments into beats. A beat is a continuous run of speech
    from one speaker, broken by a >1.5s gap or speaker change.
    """
    beats = []
    current_beat = None

    for seg in segments:
        start = seg.get('start', 0)
        end = seg.get('end', start)
        text = seg.get('text', '').strip()
        speaker = seg.get('speaker', 'Speaker')

        if not text:
            continue

        if current_beat is None:
            current_beat = _new_beat(start, end, text, speaker)
            continue

        gap = start - current_beat['end']
        same_speaker = (speaker == current_beat['speaker'])

        if gap > BEAT_GAP_THRESHOLD or not same_speaker:
            # Close current beat, start new one
            beats.append(_finalize_beat(current_beat))
            current_beat = _new_beat(start, end, text, speaker)
        else:
            # Extend current beat
            current_beat['end'] = end
            current_beat['text'] += ' ' + text
            current_beat['word_count'] += len(text.split())

    if current_beat:
        beats.append(_finalize_beat(current_beat))

    return beats


def _new_beat(start, end, text, speaker):
    return {
        'start': start,
        'end': end,
        'text': text,
        'speaker': speaker,
        'word_count': len(text.split()),
    }


def _finalize_beat(beat):
    beat['duration'] = beat['end'] - beat['start']
    return beat


def _classify_position(beats, duration):
    """Classify where the longest beat falls in the timeline."""
    if not beats:
        return 'middle'

    longest = max(beats, key=lambda b: b['duration'])
    midpoint = (longest['start'] + longest['end']) / 2
    ratio = midpoint / duration if duration > 0 else 0.5

    if ratio < 0.1:
        return 'opening'
    elif ratio < 0.33:
        return 'first_third'
    elif ratio < 0.67:
        return 'middle'
    elif ratio < 0.9:
        return 'last_third'
    else:
        return 'closing'


def _wpm_in_range(beats, start_time, end_time):
    """Compute words per minute for beats that overlap a time range."""
    words = 0
    speech_seconds = 0

    for beat in beats:
        # Check overlap
        overlap_start = max(beat['start'], start_time)
        overlap_end = min(beat['end'], end_time)
        if overlap_end <= overlap_start:
            continue

        overlap_ratio = (overlap_end - overlap_start) / beat['duration'] if beat['duration'] > 0 else 0
        words += beat['word_count'] * overlap_ratio
        speech_seconds += (overlap_end - overlap_start)

    range_duration = end_time - start_time
    if range_duration <= 0 or words <= 0:
        return 0
    return words / range_duration * 60


def _text_in_range(segments, start_time, end_time):
    """Extract all transcript text that falls within a time range."""
    texts = []
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', seg_start)
        if seg_end > start_time and seg_start < end_time:
            texts.append(seg.get('text', ''))
    return ' '.join(texts).strip()


def _empty_metrics(duration):
    """Return zeroed-out metrics for an empty or invalid transcript."""
    return {
        'speech_pacing': {
            'words_per_minute': 0, 'avg_beat_length': 0, 'median_beat_length': 0,
            'stddev_beat_length': 0, 'longest_beat': 0, 'shortest_beat': 0,
            'rhythm_descriptor': '',
        },
        'structural_rhythm': {
            'total_duration_seconds': duration, 'speech_to_silence_ratio': 0,
            'speaker_switches_per_minute': 0, 'longest_beat_position': 'middle',
            'pacing_first_third_wpm': 0, 'pacing_middle_third_wpm': 0,
            'pacing_last_third_wpm': 0, 'energy_arc': '',
        },
        'soundbite_craft': {
            'avg_soundbite_length': 0, 'clean_cut_ratio': 0, 'avg_gap_before_cut': 0,
        },
        'content_patterns': {
            'question_to_statement_ratio': 0, 'topic_count': 0,
        },
        '_raw': {
            'first_15s_text': '', 'last_15s_text': '',
            'opening_third_text': '', 'closing_third_text': '',
            'beat_count': 0, 'total_words': 0,
        }
    }
