# Evidence-driven scene editing

This directory adapts VISTA's inspect, explore, freeze, evaluate and retain loop
to EmbodiedGen's existing scene-editing API. Models stay fixed. Production code
lives in `embodied_gen/harness/scene_edit/`; policies are small source packages
under `harnesses/`. Existing editing and generation entry points remain usable.

## Edit a scene

Use the existing `embodiedgen` environment, an authenticated Codex CLI, and
Bubblewrap. The policy runs in a private filesystem/network namespace. The
trusted host calls the project's original Codex adapter and native geometry API.

```bash
source scripts/vista_evolution/env.sh
python -m embodied_gen.harness.scene_edit.cli \
  --scene /path/to/scene.urdf \
  --instruction '在办公室桌上放一个花瓶，其他东西不要移动。' \
  --catalog /path/to/catalog.json \
  --policy harnesses/scene_edit_vista \
  --output outputs/my_edit --gpu 4
```

After installing the updated project, `scene-edit-cli` provides the same entry
point. The catalog maps asset names to OBJ paths, for example
`{"vase":"/path/to/vase.obj","mug":"/path/to/mug.obj"}`. Relative asset paths
are interpreted against the current working directory. Associated collision,
material and texture files must be present.

`result.json` reports status, output path, stage times and model usage.
`published.json` points to a verified scene; it is absent after failed execution.
Original input files are not overwritten. Non-edit responses keep the input
scene. The native batch API itself still retains its original partial-success
contract; transactions belong to this opt-in editor.
After a failed release diagnostic, preview images show the unpublished candidate;
`result.json.output` remains the authoritative unchanged input scene.

For another instruction, use the previous result's `output` as `--scene`, provide
its `result.json` through `--history`, and choose a new output directory. New
objects carry a stable identity in their URDF link attributes. External editors
should preserve `scene_edit_id` when renaming or moving an existing object.

## Run a bounded evolution experiment

The 2026-11-16 experiment uses GPU 4, existing official demo assets, the fixed
`gpt-6-astra / medium` model and the current ChatGPT subscription. The prior asset
bundle is `outputs/efficiency/20261112_214800/`. No robot policy checkpoints are
needed.

```bash
source scripts/vista_evolution/env.sh
python scripts/vista_evolution/run.py setup --root RUN_DIRECTORY
python scripts/vista_evolution/run.py evaluate --root RUN_DIRECTORY --version H0
python scripts/vista_evolution/evolve.py campaign --root RUN_DIRECTORY
```

`setup` creates the manifest, directories, fixed scene inputs and initial policy
version. The current run is recorded in `outputs/vista_evolution/latest_run.txt`;
specify `--root` when inspecting or resuming a particular run. `setup` must only be
run once in a new run directory. Evaluations resume completed cells without new
model calls, while immutable request IDs reject conflicting settings.

Each candidate is generated from actual requests, native outcomes, geometry and
two live observations. It runs autonomously after freezing in a private Git
repository. Internal development evidence, selection results and final unseen
layout results are separate. Final H0/candidate tasks are interleaved by episode.
The report identifies the adopted package; a submitted candidate is not itself
an adoption result.

## Cost and verification

The common host checks declared object changes, unchanged objects, support bounds,
conservative XY overlap, SAPIEN loading, paired 720x1080 previews, and a two-second
release diagnostic using original collision meshes. Translation and rotation
limits are 2 cm and 5 degrees. Hidden task answers belong only to the evaluator.

`verified_s` includes interpreter startup, context preparation, model round trip,
native edits and verification. Video encoding and final process cleanup are
recorded separately. `input_tokens` already includes cached input; reasoning is
part of output. Total tokens add input and output only.
The analysis includes every attempt in comparison costs, including transport
failures and their retries. Quality uses the completed execution. Unknown usage
is listed separately. Raw evaluation summaries retain completed-attempt costs;
`metrics/summary.json` supplies the all-attempt comparison used in the report.

The shared budget covers engineering probes, Meta calls, development tests,
selection and final verification: at most 240 observable model requests, five
million known tokens and four experimental hours. Infrastructure attempts retain
their costs. GPU conflicts stop admission rather than switching devices.
