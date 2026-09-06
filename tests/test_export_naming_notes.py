"""export_naming (timeline/file names, counters) and export_notes (verbatim
clip notes) helpers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import export_naming as en  # noqa: E402
import export_notes as notes  # noqa: E402

P = {'name': 'PAC 2026 Interview', 'export_counts': {'selects': 2, 'story:the grind': 1}}


def test_timeline_names_use_en_dash_and_counter():
    assert en.timeline_name(P, 'selects', 3) == 'PAC 2026 Interview – Selects 3'
    assert en.timeline_name(P, 'markers', 1) == 'PAC 2026 Interview – Markers 1'
    assert en.timeline_name({'name': ''}, 'selects', 1) == 'Interview – Selects 1'
    assert en.timeline_name({'name': ' Two   spaces '}, 'selects', 1) == 'Two spaces – Selects 1'


def test_story_names_carry_n_only_from_the_second_export():
    assert en.timeline_name(P, 'story', 1, story_title='The Grind') == 'PAC 2026 Interview – Story: The Grind'
    assert en.timeline_name(P, 'story', 2, story_title='The Grind') == 'PAC 2026 Interview – Story: The Grind 2'
    assert en.timeline_name(P, en.story_kind('The Grind'), None, story_title='The Grind') == 'PAC 2026 Interview – Story: The Grind'
    assert en.story_kind(' The  Grind ') == 'story:the grind'


def test_override_wins_and_is_counter_free():
    assert en.timeline_name(P, 'selects', 7, override='  Client   cut ') == 'Client cut'
    assert en.timeline_name(P, 'selects', 7, override='   ') == 'PAC 2026 Interview – Selects 7'


def test_counters_and_events():
    assert en.export_count(P, 'selects') == 2 and en.next_count(P, 'selects') == 3
    assert en.export_count(P, 'markers') == 0 and en.next_count({}, 'selects') == 1
    assert en.export_count({'export_counts': {'selects': 'x'}}, 'selects') == 0
    assert en.event_name_for(P) == 'PAC 2026 Interview'
    assert en.event_name_for({'name': ''}) == 'Doza Assist' and en.project_base({}) == 'Interview'


def test_filename_matches_timeline_and_is_safe():
    assert en.filename_for('PAC 2026 – Selects 1') == 'PAC 2026 – Selects 1.fcpxml'
    assert en.filename_for('24/7: The Grind', 'xml') == '24-7- The Grind.xml'
    assert en.safe_filename('a\x00b\tc') == 'a b c' and en.safe_filename('///', 'Export') == '---'
    assert en.safe_filename('', 'Fallback') == 'Fallback'
    assert 'Doza' not in en.timeline_name(P, 'selects', 1)
    assert 'Doza' not in en.timeline_name(P, 'story', 3, story_title='Opening')


SEGS = [
    {'start': 0.0, 'end': 4.0, 'speaker': 'SPEAKER_00', 'text': 'We wanted to build the new stadium.'},
    {'start': 4.0, 'end': 8.0, 'speaker': 'SPEAKER_00', 'text': 'Downtown, near the river.'},
    {'start': 8.0, 'end': 12.0, 'speaker': 'SPEAKER_01', 'text': 'And the fans loved it?'},
    {'start': 12.0, 'end': 16.0, 'speaker': 'SPEAKER_00',
     'words': [{'word': 'They', 'start': 12.0}, {'word': 'did.', 'start': 12.5}, {'word': 'Every', 'start': 14.0}, {'word': 'night.', 'start': 15.0}]},
]
NAMES = {'SPEAKER_00': 'Sarah', 'SPEAKER_01': 'Mike'}


def test_verbatim_folds_turns_and_labels_speakers():
    v = notes.verbatim_for_range(SEGS, 0, 16, NAMES)
    assert v == ('Sarah: We wanted to build the new stadium. Downtown, near the river.\n'
                 'Mike: And the fans loved it?\n'
                 'Sarah: They did. Every night.')
    # word timings honor the range; whole segments are used when there are none
    assert notes.verbatim_for_range(SEGS, 13.9, 16, NAMES) == 'Sarah: Every night.'
    assert notes.verbatim_for_range(SEGS, 9, 10, NAMES) == 'Mike: And the fans loved it?'
    assert notes.verbatim_for_range(SEGS, 20, 30, NAMES) == ''
    assert notes.verbatim_for_range(SEGS, 5, 5, NAMES) == ''


def test_verbatim_without_speakers_has_no_prefix():
    plain = [{'start': 0, 'end': 3, 'text': 'Hello there.'}, {'start': 3, 'end': 6, 'text': 'General Kenobi.'}]
    assert notes.verbatim_for_range(plain, 0, 6) == 'Hello there. General Kenobi.'
    # raw pyannote labels stay when no names map them
    assert notes.verbatim_for_range(SEGS[:1], 0, 4) == 'SPEAKER_00: We wanted to build the new stadium.'


def test_cap_cuts_at_a_sentence_then_a_word():
    text = ('Alpha beta gamma. ' * 100).strip()      # 1799 chars
    capped = notes.cap_text(text, 1000)
    assert len(capped) <= 1000 and capped.endswith('gamma.…')
    words = 'word ' * 300
    capped = notes.cap_text(words, 1000)
    assert len(capped) <= 1000 and capped.endswith('word…') and '  ' not in capped
    assert notes.cap_text('short', 1000) == 'short'
    long_verbatim = notes.verbatim_for_range(
        [{'start': 0, 'end': 5, 'text': 'x' * 3000}], 0, 5)
    assert len(long_verbatim) == 1000 and long_verbatim.endswith('…')


def test_compose_clip_note():
    assert notes.compose_clip_note('Hero moment', 'Sarah', 'Sarah: We did it.') == 'Hero moment — Sarah\n\nSarah: We did it.'
    assert notes.compose_clip_note('', 'Sarah', 'text') == 'Sarah\n\ntext'
    assert notes.compose_clip_note('Hero moment', '', '') == 'Hero moment'
    assert notes.compose_clip_note('', '', 'only words') == 'only words'
    assert notes.compose_clip_note(None, None, None) == ''
