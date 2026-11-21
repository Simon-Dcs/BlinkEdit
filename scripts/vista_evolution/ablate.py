"""Test the explanatory memory on development scenes without tuning the winner."""

import json
import shutil
from pathlib import Path

from run import evaluate
from embodied_gen.harness.scene_edit.package import Versions
from embodied_gen.harness.scene_edit.records import write_json


def main():
    root = Path(
        Path('outputs/vista_evolution/latest_run.txt').read_text().strip()
    )
    selected = json.loads((root / 'selected.json').read_text())
    source = root / 'ablations' / 'without_memory'
    if not source.exists():
        shutil.copytree(selected['path'], source)
        memory = source / 'memory' / 'MEMORY.md'
        memory.chmod(0o644)
        memory.write_text('')
    commit, frozen = Versions(root / 'harness_repo').freeze(
        source,
        'Diagnostic: remove explanatory memory only',
        selected['commit'],
    )
    write_json(
        root / 'ablations' / 'request.json',
        {
            'intervention': 'Empty memory/MEMORY.md; policy, skill and guidance unchanged',
            'parent': selected['commit'],
            'commit': commit,
            'cells': ['D1_v0', 'D2_v0'],
            'use': 'Development diagnostic only; no change to final recommendation',
        },
        once=True,
    )
    result = evaluate(
        root,
        'H1_no_memory',
        frozen,
        commit,
        purpose='ablation',
        cells=['D1_v0', 'D2_v0'],
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    'successes',
                    'denominator',
                    'turns',
                    'tokens',
                    'verified_s',
                ]
            }
        )
    )


if __name__ == '__main__':
    main()
