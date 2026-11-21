# Deterministic scene editing: bounded efficiency experiment

The baseline is the adopted H1 package, not the weaker H0. The candidate uses
fixed application code to resolve historical identities and project current
scene state, while the same model interprets each new instruction once.

Three implementation changes are tested together:

- A compact relation table replaces duplicated inventories, geometry numbers and
  explanatory memory. Geometry remains available to the trusted checks.
- Codex's documented `model_instructions_file` supplies task-specific instructions
  for this request only, replacing the general coding instructions. Authentication,
  model, reasoning effort, disabled tools and read-only sandbox are unchanged.
- Geometry checks skip texture decoding and reuse an already checked candidate's
  geometry within the same edit. URDFs, meshes, collisions, textured rendering and
  the two-second release check remain unchanged. There is no cross-request scene
  cache that could hide external edits.

## Use the opt-in implementation

```bash
source scripts/vista_evolution/env.sh
scene-edit-cli --engine deterministic --lean-parser --efficient-geometry \
  --scene /path/to/scene.urdf --instruction '在办公室桌上放一个花瓶。' \
  --catalog /path/to/catalog.json --output outputs/my_edit --gpu 4
```

The compact language context exposes object type, room, supports and confirmed
identities. For language targets requiring coordinates or distances, use the
legacy full-facts path. Geometry checking in both paths retains full precision.

The existing `--policy harnesses/scene_edit_vista` invocation stays on the legacy
path by default. See `scripts/vista_evolution/README.md` for result files, confirmed
history and atomic publication. Generated candidates are not loaded by the new
fixed implementation.

## Reproduce or resume the recorded comparison

The run is identified by `outputs/edit_efficiency/latest_run.txt`. Its manifest,
input references and source snapshots preserve the experiment. Existing results
are reused rather than rerun; do not change the version configuration under an
existing request ID.

```bash
source scripts/vista_evolution/env.sh
python scripts/edit_efficiency/benchmark.py pilot
python scripts/edit_efficiency/benchmark.py formal
python scripts/edit_efficiency/analyze.py
```

The fixed regression benchmark contains 24 whole tasks / 32 instructions per
version. H1 and E1 are interleaved by whole task; this is not a newly unseen set.
Both use cold processes, one planner request, the original placement budgets,
paired views and the same physical thresholds. All attempts enter cost totals;
usage missing after a provider error is unknown, not zero. Raw requests retain
actual prompts, commands, events, response and usage. `metrics/summary.json` uses
all-attempt costs; episode quality uses the completed execution. Admission stops
on GPU 4 conflicts, 80 model requests, two million known tokens, two experimental
hours or a repeated infrastructure failure.

The selected configuration requires no correctness regression, 24/24 candidate
whole tasks, and either at least 10% fewer tokens or 15% lower verified time, with
no more than a 5% increase in the other cost. A complete token comparison requires
known usage for both versions. Transport diagnostics are saved separately from
model tool calls.

Official configuration reference:
https://developers.openai.com/codex/config-reference/#model_instructions_file
