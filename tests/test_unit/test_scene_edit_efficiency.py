"""Deterministic state projection and opt-in consumer contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from embodied_gen.harness.scene_edit.planning import (
    DeterministicContext,
    compact_state,
)


def payload(ids):
    facts = [
        dict(
            id=f'current_{i}',
            asset='mug',
            room='dining_floor',
            support=['dining_table'],
            stable_id=identity,
            xyz=[0, 0, 1],
            bounds=[[0, 0, 0], [1, 1, 1]],
            collision=True,
        )
        for i, identity in enumerate(ids)
    ]
    return dict(
        objects=[f['id'] for f in facts],
        rooms=['office_floor', 'dining_floor'],
        assets=['mug'],
        facts=facts,
        history=[
            dict(
                status='execute',
                inserted=[
                    dict(
                        asset='mug',
                        instance_key='old_mug',
                        stable_id='id_1',
                        in_room='office_floor',
                        on_instance='office_table',
                    )
                ],
                deleted=[],
            )
        ],
    )


@pytest.mark.parametrize(
    'ids,resolution,current',
    [
        (['id_1'], 'resolved', 'current_0'),
        (['replacement'], 'absent', None),
        (['id_1', 'id_1'], 'ambiguous', None),
    ],
)
def test_history_identity_is_resolved_before_language_interpretation(
    ids, resolution, current
):
    state = compact_state(payload(ids))
    item = state['history'][0]['inserted'][0]
    assert item['resolution'] == resolution and item['current_id'] == current
    assert item['old_room'] == 'office_floor'
    if current:
        assert state['objects'][current]['room'] == 'dining_floor'
        assert state['objects'][current]['on'] == ['dining_table']
    assert 'stable_id' not in json.dumps(state)
    assert 'bounds' not in json.dumps(state)


def test_missing_identity_is_unknown_not_deleted():
    value = payload(['id_1'])
    value['history'][0]['inserted'][0]['stable_id'] = None
    assert (
        compact_state(value)['history'][0]['inserted'][0]['resolution']
        == 'unverified'
    )


def test_state_refresh_does_not_reuse_previous_objects():
    worker = DeterministicContext()
    assert worker.call('prepare_request', payload(['id_1']))['context'][
        'objects'
    ]
    refreshed = worker.call('prepare_request', payload([]))['context']
    assert refreshed['objects'] == {}
    assert refreshed['history'][0]['inserted'][0]['resolution'] == 'absent'


def test_fixed_execution_preserves_the_parsed_plan():
    worker = DeterministicContext()
    plan = dict(
        status='execute', delete=['lamp_a'], insert=[], message='delete'
    )
    assert worker.call('decide', {'plan': plan}) is plan
    assert (
        worker.call(
            'process_tool_result', {'plan': plan, 'error': 'search_exhausted'}
        )['retry']
        is False
    )


@pytest.mark.parametrize('published', [True, False])
def test_new_cli_passes_explicit_engine_flags_without_changing_legacy_defaults(
    tmp_path, monkeypatch, capsys, published
):
    from embodied_gen.harness.scene_edit import cli, runtime

    catalog = tmp_path / 'catalog.json'
    catalog.write_text('{"mug":"/asset/mug.obj"}')
    called = {}

    class Editor:
        def __init__(self, assets, policy, **options):
            called.update(assets=assets, policy=policy, **options)

        def edit(self, instruction, source, output, **options):
            called.update(
                instruction=instruction,
                source=source,
                output=output,
                **options,
            )
            return dict(
                status='execute' if published else 'failed',
                published=published,
            )

    monkeypatch.setattr(runtime, 'SceneEditor', Editor)
    monkeypatch.setattr(
        'sys.argv',
        [
            'scene-edit-cli',
            '--scene',
            'input.urdf',
            '--instruction',
            '放一个杯子',
            '--catalog',
            str(catalog),
            '--output',
            str(tmp_path / 'out'),
            '--engine',
            'deterministic',
            '--lean-parser',
            '--efficient-geometry',
            '--gpu',
            '4',
        ],
    )
    if published:
        cli.main()
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
    assert called['policy'] is None
    assert called['planning_mode'] == 'deterministic'
    assert (
        called['lean_parser'] is True and called['efficient_geometry'] is True
    )
    assert called['gpu_index'] == 4
    assert (
        json.loads(capsys.readouterr().out.strip().splitlines()[-1])[
            'published'
        ]
        == published
    )


def test_task_instruction_override_preserves_native_transport_and_tool_disabling(
    tmp_path, monkeypatch
):
    from embodied_gen.harness.scene_edit.model import ModelGateway

    instructions = tmp_path / 'instructions.txt'
    instructions.write_text('Parse scene edits.')
    gateway = ModelGateway(
        tmp_path / 'requests', instructions_file=instructions
    )
    seen = {}

    def run(command, **kwargs):
        seen['command'] = command
        Path(command[command.index('--output-last-message') + 1]).write_text(
            '{}'
        )
        return SimpleNamespace(
            returncode=0,
            stderr='',
            stdout=json.dumps(
                {
                    'type': 'turn.completed',
                    'usage': {'input_tokens': 17, 'output_tokens': 2},
                }
            ),
        )

    monkeypatch.setattr(gateway.gc, 'subprocess', SimpleNamespace(run=run))
    assert gateway.query('Return JSON.') == '{}'
    command = seen['command']
    assert any(x.startswith('model_instructions_file=') for x in command)
    assert '--ignore-user-config' in command and '--ignore-rules' in command
    assert (
        'read-only' in command
        and 'shell_tool' in command
        and 'plugins' in command
    )
    assert gateway.records[0]['usage']['input_tokens'] == 17
