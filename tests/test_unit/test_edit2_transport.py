"""Interleaved model roles keep their own capture directory and CLI flags."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


def test_simultaneous_model_gateways_do_not_stack_flags(tmp_path, monkeypatch):
    from embodied_gen.harness.scene_edit.model import ModelGateway

    gate = threading.Barrier(2)
    clients = [ModelGateway(tmp_path / role) for role in ('editor', 'judge')]

    def fake_exec(command, **kw):
        gate.wait(timeout=5)
        assert command.count('--json') == 1
        role = 'editor' if 'edit_this' in kw['input'] else 'judge'
        Path(command[command.index('--output-last-message') + 1]).write_text(
            role
        )
        return SimpleNamespace(
            returncode=0,
            stderr='',
            stdout=json.dumps(
                {
                    'type': 'turn.completed',
                    'usage': {'input_tokens': 3, 'output_tokens': 1},
                }
            ),
        )

    monkeypatch.setattr(clients[0].gc.subprocess, 'run', fake_exec)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(clients[0].query, 'edit_this')
        b = pool.submit(clients[1].query, 'judge_this')
        assert (a.result(), b.result()) == ('editor', 'judge')
    assert [
        (c.directory / 'request_00/response.txt').read_text() for c in clients
    ] == ['editor', 'judge']
