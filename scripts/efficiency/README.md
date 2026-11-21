# EmbodiedGen editing measurements

From the repository root:

```bash
bash scripts/efficiency/run_all.sh
```

This creates a new timestamped directory in `outputs/efficiency/`. It uses the
existing README-configured `embodiedgen` Conda environment, the current ChatGPT
login through the project's **native Codex backend**, and physical **GPU 7** for
SAPIEN rendering. No API key is read or exported. Check `codex login status`
before running. The model identifier is `gpt-6-astra`, reasoning effort `medium`.
All model calls consume subscription allowance.

The driver downloads 25 official demo asset files, constructs an **input fixture**
with two rooms, and invokes the original spatial-computing CLI/API for every edit.
It does not implement a mesh editor or placement solver. It does not run image-to-3D,
texture regeneration, Blender room generation, or the Claude Code front end.

## Workloads

- S1: semantic deletion of the office lamp.
- S2: semantic placement of a vase on the office table.
- H1: four sequential semantic placements on that table.
- A1/A2: the same four assets and exact target IDs, sequential CLI vs native batch.
- N1: an oversized sofa requested on the table, fixed orientation, 2,000 attempts.
- E1: **separately labelled outer-planner experiment**. A full Chinese instruction
  is sent through `GPT_CLIENT`, returning a constrained JSON plan. The harness
  validates identifiers, then calls the original deletion and batch-insert API.
  It removes the office floor lamp and inserts lamp/vase/book/mug on the table.

Each workload runs three times. Base seeds are 42–44; sequential insert j uses
base seed + j (j=0..3), while batch mode uses one stream. Python/NumPy are seeded; current
`trimesh==5.1.1` uses its own entropy-seeded sampler when no explicit seed is passed
by the library. Therefore saved artifacts, not bitwise-identical placements, are
the exact experimental record. No core randomness code is changed for measurement.

## What is measured

`trace_cli.py` wraps existing methods with monotonic inclusive/exclusive timers.
It adds only `--json` to native Codex subprocess arguments, preserving prompts,
responses, raw usage events, failures, and timeouts. Metrics do not log credentials
or the subprocess environment. `run_benchmark.py` also times process startup and
shutdown. The measured path ends after output URDF and native floorplan creation;
independent quality checks and SAPIEN rendering are timed separately.

In `metrics/`:

- `summary.json`: aggregate and per-case values.
- `per_case.csv`: wall time, exclusive stage times, tokens, and acceptance checks.
- `requests.csv`: one row per actual model request.
- Each `runs/*/call_*/request_*/` retains prompt, response, JSONL events and stderr.
- Each `runs/*/call_*/trace.json` retains the nested timing tree.
- Each `runs/*/scene/` retains original and edited URDF and mesh dependencies.

Input tokens include cached tokens; total is input + output. Reasoning details
are retained separately; missing usage is unknown, not zero. Subscription usage
is not converted into API billing. The current supervising chat's tokens are not
exposed by the execution environment and are excluded.

## Validation

`validate_render.py` loads the actual output URDF into SAPIEN on GPU 7, checks
inventory and unchanged poses, referenced meshes and conservative XY geometry,
and saves two views. On repetition 1 it also releases inserted assets as dynamic
actors for a two-second support diagnostic, using the original collision meshes.
That diagnostic is not an edit: its single-link wrappers live in
`validation_adapters/`, and the original fixed-joint scene stays unchanged.
Friction is .6/.5, contact offset 1 mm, density the native loader default.
It is a bounded support check, not robot-task or physical-parameter validation.

The environment pins GPU visibility; the renderer device is additionally supplied
at construction because the project's `SapienSceneManager` does not forward its
device argument to `RenderSystem`. Actual PID-to-GPU UUID records verify placement.

Run only one driver at a time: `latest_run.txt` selects its output root.
Existing case directories are skipped rather than overwritten. After an interrupted
case, inspect its trace and preserve it under a different run name before retrying.
Do not count a skipped incomplete case as completed.
