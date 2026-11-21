"""One measured real edit, using the shared physical and semantic checks."""

import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['GPT_PROVIDER'] = 'codex'
os.environ['MODEL_NAME'] = 'gpt-6-astra'
os.environ['GPT_TIMEOUT'] = '180'
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / 'vista_evolution')
)
from scenarios import evaluate
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.runtime import SceneEditor

job = json.loads(Path(sys.argv[1]).read_text())
loaded = time.perf_counter()
editor = SceneEditor(
    job['catalog'], job.get('policy'), gpu_index=4, **job['options']
)
result = editor.edit(
    job['instruction'],
    job['input'],
    job['directory'],
    history=job['history'],
    seed=job['seed'],
    start_mono=job['start_mono'],
    max_model_requests=1,
    verify_callback=lambda a, b, status, r: evaluate(
        job['cell'], job['turn'], a, b, status, r
    ),
)
result.update(identity=job['identity'], startup_s=loaded - job['start_mono'])
write_json(Path(job['directory']) / 'result.json', result)
