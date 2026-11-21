"""One timed production edit. The evaluator is a separate, trusted observer."""

import json
import os
import sys
import time
from pathlib import Path

START = time.perf_counter()
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['GPT_PROVIDER'] = 'codex'
os.environ['MODEL_NAME'] = 'gpt-6-astra'
os.environ['GPT_TIMEOUT'] = '180'
from scenarios import evaluate
from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.runtime import SceneEditor

job = json.loads(Path(sys.argv[1]).read_text())
before = time.perf_counter()
editor = SceneEditor(job['catalog'], job['policy'], gpu_index=4)
rec = editor.edit(
    job['instruction'],
    job['input'],
    job['directory'],
    history=job.get('history', []),
    seed=job['seed'],
    start_mono=job['start_mono'],
    video=job.get('video', False),
    max_model_requests=job.get('max_model_requests', 2),
    verify_callback=lambda a, b, status, r: evaluate(
        job['cell'], job['turn'], a, b, status, r
    ),
)
rec.update(identity=job['identity'], startup_s=before - job['start_mono'])
write_json(Path(job['directory']) / 'result.json', rec)
