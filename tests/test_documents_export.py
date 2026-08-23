"""Documents export: full transcript / selects as Word, PDF, Excel.

Pins the builders in exporters/documents.py and the /export/document
route. These are producer/client deliverables, separate from the NLE
timeline exports (fcpxml_export / exporters.*_xml), which this feature
must not touch — test_export_matrix.py keeps guarding those.
"""

import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app as app_module
from exporters import documents as docs


def _project(**over):
    p = {
        'id': 'p1', 'name': 'Interview: Müller / Nowak', 'status': 'transcribed',
        'filename': 'interview.mov', 'source_path': '/nonexistent/interview.mov',
        'created_at': '2026-08-21T12:00:00',
        'speaker_names': {'SPEAKER_00': 'Anna Müller', 'SPEAKER_01': 'Piotr Nowak'},
        'color_labels': {'blue': 'Keeper', 'green': 'B-roll idea'},
        'labeled_sections': [
            {'start': 1.0, 'end': 4.5, 'color': 'blue', 'text': 'strong open'},
            {'start': 20.0, 'end': 25.0, 'color': 'green', 'text': ''},
        ],
        'analysis': {
            'social_clips': [{'title': 'Why we started', 'start': 4.0, 'end': 9.0, 'platform': 'TikTok'}],
            'story_beats': [{'label': 'Turning point', 'start': 10.0, 'end': 15.0,
                             'description': 'the pivot'}],
            'strongest_soundbites': [{'text': 'Zażółć gęślą jaźń — это работает', 'start': 2.0,
                                      'end': 3.0, 'why': 'punchy'}],
        },
        'transcript': {'language': 'pl', 'duration': 30.0, 'segments': [
            {'start': 0.0, 'end': 5.0, 'text': 'Zażółć gęślą jaźń.', 'speaker': 'SPEAKER_00',
             'words': [{'start': 0.2, 'end': 1.0, 'word': 'Zażółć'},
                       {'start': 1.2, 'end': 2.0, 'word': 'gęślą'},
                       {'start': 2.2, 'end': 3.0, 'word': 'jaźń.'}]},
            {'start': 5.0, 'end': 10.0, 'text': 'Это работает очень хорошо.', 'speaker': 'SPEAKER_01',
             'words': []},
            {'start': 10.0, 'end': 16.0, 'text': 'And then everything changed.', 'speaker': 'SPEAKER_01',
             'words': []},
        ]},
    }
    p.update(over)
    return p


# ── collectors ────────────────────────────────────────────────────────

def test_transcript_rows_resolve_speaker_names():
    rows = docs.transcript_rows(_project())
    assert [r['speaker'] for r in rows] == ['Anna Müller', 'Piotr Nowak', 'Piotr Nowak']
    assert rows[0]['text'].startswith('Zażółć')


def test_paragraphs_merge_consecutive_same_speaker():
    paras = docs.transcript_paragraphs(docs.transcript_rows(_project()))
    assert [p['speaker'] for p in paras] == ['Anna Müller', 'Piotr Nowak']
    assert paras[1]['text'] == 'Это работает очень хорошо. And then everything changed.'
    assert paras[1]['end'] == 16.0


def test_selects_rows_follow_nle_category_semantics():
    rows = docs.selects_rows(_project(), ['labels', 'social', 'story', 'soundbites'])
    cats = sorted({r['category'] for r in rows})
    assert cats == ['B-roll idea', 'Keeper', 'Social Clip', 'Soundbite', 'Story Beat']
    keeper = next(r for r in rows if r['category'] == 'Keeper')
    assert keeper['speaker'] == 'Anna Müller'
    assert keeper['quote'] == 'gęślą jaźń.'          # word-level slice of [1.0, 4.5)
    assert keeper['note'] == 'strong open'
    assert rows == sorted(rows, key=lambda r: r['start'])


def test_selects_rows_respect_category_filter():
    rows = docs.selects_rows(_project(), ['story'])
    assert [r['category'] for r in rows] == ['Story Beat']
    assert docs.selects_rows(_project(), ['transcript']) == []


def test_source_tc_formatting():
    tc = {'frames': 90000, 'fps': 25.0, 'drop': False}     # 01:00:00:00 @25
    assert docs.fmt_source_tc(61.0, tc) == '01:01:01:00'
    assert docs.fmt_source_tc(61.0, {'frames': 0, 'fps': 29.97, 'drop': True}) == ''
    assert docs.fmt_source_tc(61.0, None) == ''
    assert docs.fmt_clock(3661) == '01:01:01'
    assert docs.fmt_duration(83) == '1:23'


# ── builders ──────────────────────────────────────────────────────────

def _docx_text(path):
    with zipfile.ZipFile(path) as z:
        return z.read('word/document.xml').decode('utf-8')


def test_transcript_docx(tmp_path):
    out = str(tmp_path / 't.docx')
    docs.build_transcript_docx(_project(), docs.transcript_rows(_project()), out)
    xml = _docx_text(out)
    assert 'Anna Müller' in xml and 'Piotr Nowak' in xml
    assert 'Zażółć gęślą jaźń.' in xml and 'Это работает' in xml
    assert 'SPEAKER_00' not in xml


def test_transcript_pdf_uses_unicode_font(tmp_path):
    out = str(tmp_path / 't.pdf')
    docs.build_transcript_pdf(_project(), docs.transcript_rows(_project()), out)
    data = Path(out).read_bytes()
    assert data.startswith(b'%PDF')
    regular, _bold = docs.register_unicode_fonts()
    if regular == 'DozaSans':
        # The Unicode TrueType family is embedded as subset fonts (/F2+0 …)
        # with FontFile2 streams. reportlab still emits its default Helvetica
        # /F1 object on every canvas, so its presence proves nothing; the
        # embedded TrueType program does.
        assert b'FontFile2' in data
        assert b'/F2+0' in data


def test_transcript_xlsx_is_valid_workbook(tmp_path):
    out = str(tmp_path / 't.xlsx')
    docs.build_transcript_xlsx(_project(), docs.transcript_rows(_project()), out)
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {'[Content_Types].xml', '_rels/.rels', 'xl/workbook.xml',
                'xl/_rels/workbook.xml.rels', 'xl/styles.xml',
                'xl/worksheets/sheet1.xml'} <= names
        sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
        assert z.testzip() is None
    assert '<t xml:space="preserve">Speaker</t>' in sheet      # header
    assert 'Anna Müller' in sheet and 'Zażółć gęślą jaźń.' in sheet
    assert 'state="frozen"' in sheet
    assert '<row r="4">' in sheet and '<row r="5">' not in sheet  # 3 segments + header


def test_selects_docx_pdf_xlsx(tmp_path):
    p = _project()
    rows = docs.selects_rows(p, ['labels', 'social', 'story'])
    docx_path = docs.build_selects_docx(p, rows, str(tmp_path / 's.docx'))
    assert 'Turning point' in _docx_text(docx_path) and 'Keeper' in _docx_text(docx_path)
    pdf_path = docs.build_selects_pdf(p, rows, str(tmp_path / 's.pdf'))
    assert Path(pdf_path).read_bytes().startswith(b'%PDF')
    xlsx_path = docs.build_selects_xlsx(p, rows, str(tmp_path / 's.xlsx'))
    with zipfile.ZipFile(xlsx_path) as z:
        sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
    assert 'Social Clip' in sheet and 'TikTok' in sheet and 'Story Beat' in sheet


def test_xlsx_writer_escapes_and_types(tmp_path):
    out = docs.write_xlsx(str(tmp_path / 'w.xlsx'),
                          [('S', ['A', 'B'], [['<x>&"', 42], ['\x00ctrl', 1.5]])])
    with zipfile.ZipFile(out) as z:
        sheet = z.read('xl/worksheets/sheet1.xml').decode('utf-8')
    assert '&lt;x&gt;&amp;"' in sheet
    assert '<c r="B2"><v>42</v></c>' in sheet
    assert '\x00' not in sheet and 'ctrl' in sheet


def test_allocate_path_uniquifies(tmp_path):
    a = docs.allocate_path(str(tmp_path), 'My / Project: one', 'transcript', 'docx')
    Path(a).write_bytes(b'x')
    b = docs.allocate_path(str(tmp_path), 'My / Project: one', 'transcript', 'docx')
    assert os.path.basename(a) == 'My Project one – Transcript.docx'
    assert os.path.basename(b) == 'My Project one – Transcript (2).docx'
    assert os.path.dirname(a).endswith('documents')


# ── route ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    app_module.app.config['PROJECTS_DIR'] = str(tmp_path / 'projects')
    app_module.app.config['EXPORTS_DIR'] = str(tmp_path / 'exports')
    Path(app_module.app.config['PROJECTS_DIR']).mkdir(parents=True, exist_ok=True)
    app_module.app.config['TESTING'] = True
    monkeypatch.setattr(app_module, '_reveal_in_finder', lambda path: None)
    return app_module.app.test_client()


def _seed(pid='p1', **over):
    pdir = Path(app_module.app.config['PROJECTS_DIR']) / pid
    pdir.mkdir(parents=True)
    json.dump(_project(id=pid, **over), open(pdir / 'meta.json', 'w'))


@pytest.mark.parametrize('what', ['transcript', 'selects'])
@pytest.mark.parametrize('fmt', ['docx', 'pdf', 'xlsx'])
def test_route_builds_every_document(client, what, fmt):
    _seed()
    resp = client.post('/project/p1/export/document', json={
        'what': what, 'format': fmt, 'categories': ['labels', 'social', 'story']})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    d = resp.get_json()
    assert d['delivery'] == 'file' and d['format'] == fmt and d['what'] == what
    assert os.path.isfile(d['file'])
    assert d['file'].endswith('.' + fmt)
    assert d['count'] > 0
    assert os.sep + 'documents' + os.sep in d['file']


def test_route_rejects_bad_format_and_empty_selects(client):
    _seed()
    r = client.post('/project/p1/export/document', json={'what': 'transcript', 'format': 'odt'})
    assert r.status_code == 400
    r = client.post('/project/p1/export/document', json={'what': 'selects', 'format': 'docx',
                                                          'categories': []})
    assert r.status_code == 400
    assert 'No selects' in r.get_json()['error']


def test_route_404_and_transcript_not_ready(client):
    assert client.post('/project/nope/export/document', json={}).status_code == 404
    _seed('p2', transcript={'segments': []})
    r = client.post('/project/p2/export/document', json={'what': 'transcript', 'format': 'pdf'})
    assert r.status_code == 400
    assert 'not ready' in r.get_json()['error']
