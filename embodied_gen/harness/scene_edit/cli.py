"""Explicit opt-in entry point for natural-language scene editing."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    start = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', help='Input scene (first session turn only)')
    p.add_argument(
        '--session',
        help='Resume or initialize a versioned static URDF session',
    )
    p.add_argument('--instruction', required=True)
    p.add_argument(
        '--catalog', required=True, help='JSON catalog_key -> OBJ path'
    )
    p.add_argument(
        '--policy', help='Frozen policy directory (required for legacy engine)'
    )
    p.add_argument(
        '--engine', choices=['legacy', 'deterministic'], default='legacy'
    )
    p.add_argument(
        '--lean-parser',
        action='store_true',
        help='Use task-specific Codex instructions',
    )
    p.add_argument(
        '--efficient-geometry',
        action='store_true',
        help='Avoid texture decoding and duplicate geometry reads',
    )
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int, default=4)
    p.add_argument(
        '--history', help='Previous result.json; its history is reused'
    )
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not (a.session or a.scene):
        p.error('--scene or --session is required')
    if a.session and (
        a.history or a.engine != 'deterministic' or not a.efficient_geometry
    ):
        p.error(
            '--session requires --engine deterministic --efficient-geometry; history is restored automatically'
        )
    if a.engine == 'legacy' and not a.policy:
        p.error('--policy is required for the legacy engine')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    os.environ.setdefault('GPT_PROVIDER', 'codex')
    os.environ.setdefault('MODEL_NAME', 'gpt-6-astra')
    from .runtime import SceneEditor

    history = (
        json.loads(Path(a.history).read_text())['history'] if a.history else []
    )
    editor = SceneEditor(
        json.loads(Path(a.catalog).read_text()),
        a.policy,
        gpu_index=a.gpu,
        planning_mode=a.engine,
        lean_parser=a.lean_parser,
        efficient_geometry=a.efficient_geometry,
    )
    if a.session:
        from .native_session import session_edit

        result = session_edit(
            editor,
            a.session,
            a.scene,
            a.instruction,
            a.output,
            seed=a.seed,
            start_mono=start,
        )
    else:
        result = editor.edit(
            a.instruction,
            a.scene,
            a.output,
            history=history,
            seed=a.seed,
            start_mono=start,
        )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in [
                    'status',
                    'output',
                    'published',
                    'verified_s',
                    'known_tokens',
                    'error',
                ]
            },
            ensure_ascii=False,
        )
    )
    if not result['published']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
