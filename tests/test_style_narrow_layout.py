"""In a window under 1100 px the player column must flow above the content
instead of staying pinned: the media-query rule has to outrank the sticky
`.player-col` rule declared after it (a bare `.player-col` lost to it, and
the column painted over the text below in a compressed window)."""
import os
import re

STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'style.css')


def test_narrow_window_rule_outranks_the_sticky_rule():
    css = open(STYLE).read()
    block = re.search(r'@media \(max-width: 1100px\) \{(.*?)\n\}', css, re.S).group(1)
    assert '.page-two-col > .player-col {' in block
    assert 'position: static' in block
    # The picture is capped so a wide, short window keeps the controls in view.
    assert '.video-player-box' in block and 'max-height: 48vh' in block
