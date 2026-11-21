from embodied_gen.harness.asset_edit.live_program import select


def trace(names):
    return {
        'objects': [
            dict(
                name=name,
                bounds=[[0, 0, 0], [1, 2, 3]],
                origin={
                    'stack': [
                        dict(fn='box', at=2, params={'depth': 4}),
                        dict(fn='body', at=6, params={}),
                    ]
                },
            )
            for name in names
        ]
    }


SRC = 'def box():\n    return 1\n\ndef body():\n    for i in range(4):\n        box()\nbody()\n'


def test_a_live_origin_not_a_dead_name_guides_the_patch():
    spans, e = select(
        SRC,
        'Reduce the four metal divider panels to half height.',
        '',
        trace([f'bay_wall_{i}' for i in range(4)]),
    )
    assert e['sites'] == [6] and all(a > 2 for a, b in spans)
    assert sum(a <= 6 <= b for a, b in spans) == 1


def test_multiple_or_untraced_subjects_decline():
    for names in [['door_1', 'door_2'], ['unrelated']]:
        assert select(SRC, 'Open the door.', '', trace(names))[0] is None
    t = trace(['door'])
    t['objects'][0]['origin'] = None
    assert select(SRC, 'Open the door.', '', t)[0] is None
