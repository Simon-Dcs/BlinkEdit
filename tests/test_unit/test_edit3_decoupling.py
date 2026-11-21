import numpy as np
from embodied_gen.harness.asset_edit import grounding as g
from embodied_gen.harness.asset_edit import local_program as code


def test_selector_and_abstention():
    ids = ['3', '8']
    assert g.selected('{"id":8}', ids) == 8
    for text in ('{"id":true}', '{"id":88}', '{"id":null}', 'not JSON'):
        assert g.selected(text, ids) is None


def test_distinct_hues_use_surface_area_not_triangle_counts():
    tri = np.array(
        [
            [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]],
        ]
    )
    c = np.array([[1, 1, 0], [0, 0, 1]])
    assert (
        g.hue_support(tri, c, 60) > 0.99 and g.hue_support(tri, c, 240) < 0.01
    )


def test_patch_preserves_external_lines_and_declines_escapes():
    s = 'import math\nwidth = 8\nheight = 4\nprint(width*height)\n'
    assert code.apply_reply(
        s, [(2, 3)], '{"edits":[{"start":3,"end":3,"code":"height = 9"}]}'
    ) == s.replace('height = 4', 'height = 9')
    for x in (
        '{"edits":[]}',
        '{"edits":[{"start":1,"end":3,"code":"pass"}]}',
        '{"edits":[{"start":2,"end":2,"code":"if !"}]}',
    ):
        assert code.apply_reply(s, [(2, 3)], x) is None


def test_same_statement_can_only_be_replaced_once():
    msg = '{"edits":[{"start":1,"end":1,"code":"x=2"},{"start":1,"end":1,"code":"x=4"}]}'
    assert code.apply_reply('x=1', [(1, 1)], msg) is None
