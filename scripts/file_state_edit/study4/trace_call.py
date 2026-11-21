import json, shutil, tempfile, time, subprocess
from pathlib import Path
from sandbox import isolated
from embodied_gen.harness.asset_edit import live_program


def locate(root, out, src, instruction, feedback):
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        (w / 'live.py').write_text(src)
        try:
            ret = isolated(
                [
                    '/blender/blender',
                    '-b',
                    '--threads',
                    '4',
                    '--python-exit-code',
                    '1',
                    '--python',
                    '/probe.py',
                ],
                w,
                readonly=[
                    (
                        root.parent / 'tools/blender-5.1.0-linux-x64',
                        '/blender',
                    ),
                    (root.parent / 'tools/blender_deps', '/deps'),
                    (
                        Path(__file__).with_name('trace_blender.py'),
                        '/probe.py',
                    ),
                ],
                timeout=150,
                log=out / 'trace.log',
            )
        except subprocess.TimeoutExpired:
            ret = 124
        if ret == 0 and (w / 'objects.json').is_file():
            shutil.copyfile(w / 'objects.json', out / 'live-objects.json')
            trace = json.loads((w / 'objects.json').read_text())
            spans, e = live_program.select(src, instruction, feedback, trace)
            e.update(
                objects=len(trace['objects']),
                mapped=sum(o['origin'] is not None for o in trace['objects']),
            )
        else:
            spans, e = (
                None,
                dict(reason='trace_execution_failed', exit_code=ret),
            )
    e.update(seconds=time.perf_counter() - t, spans=spans)
    (out / 'binding.json').write_text(
        json.dumps(e, ensure_ascii=False, indent=2)
    )
    return spans, e
