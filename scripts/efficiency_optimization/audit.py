"""Audit retained evidence without re-running models, edits, or simulation."""
import argparse
import json
import math
from pathlib import Path
from common import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root')
    a = p.parse_args()
    root = Path(a.root or Path('outputs/efficiency_optimization/latest_run.txt').read_text().strip())
    records = [json.loads(l) for l in (root/'metrics/tasks.jsonl').read_text().splitlines()]
    requests = [json.loads(l) for l in (root/'metrics/requests.jsonl').read_text().splitlines()]
    errors = []

    def check(ok, text):
        if not ok:
            errors.append(text)
    check(len({r['task_id'] for r in records})==len(records),'Duplicate task IDs')
    check(len(requests)==sum(len(r['requests']) for r in records),'Request inventory does not reconcile')
    formal = [r for r in records if r['phase']=='formal']
    check(len(formal)==60,f'Expected 60 formal rounds, observed {len(formal)}')
    check(len(records)<=78,'Round limit exceeded')
    check(len(requests)<=100,'Request limit exceeded')
    check(sum(r['known_tokens'] for r in records)<=2000000,'Known token threshold exceeded')
    for r in records:
        d = Path(r['directory'])
        prefix = r['task_id']+': '
        check(r['verified_s']>=r['edit_s']>=0,prefix+'Invalid endpoint ordering')
        check(abs(r['verified_s']-r['process_overhead_s']-r['worker_task_s'])<1e-6,prefix+'Clock reconciliation')
        exclusive = sum(e['exclusive_s'] for e in r['events'])
        check(all(e['exclusive_s']>=-1e-6 for e in r['events']),prefix+'Negative exclusive phase time')
        check(exclusive<=r['worker_task_s']+.001,prefix+'Double-counted phase time')
        for q in r['requests']:
            rd = Path(q['directory'])
            raw = [json.loads(l) for l in (rd/'events.jsonl').read_text().splitlines() if l.startswith('{')]
            usage = [e['usage'] for e in raw if e.get('type')=='turn.completed']
            check(usage==q.get('raw_usage'),prefix+'Raw usage mismatch')
            check(q['model']=='gpt-6-astra',prefix+'Model changed')
            check(not q.get('tool_items'),prefix+'Planner invoked an additional tool; review needed')
            check((rd/'prompt.txt').is_file() and (rd/'response.txt').is_file(),prefix+'Missing model input/output')
        if r['phase']=='formal':
            check(r.get('version')=='optimization-v2',prefix+'Unfrozen candidate')
            for name in ['before_office.png','after_office.png','before_overview.png','after_overview.png']:
                check((d/name).exists(),prefix+'Missing preview '+name)
            gpu = json.loads((d/'gpu.json').read_text())
            check(gpu['visible']=='7' and all('GPU-63a185b7-8463-446a-808b-10425a0e9800' in row for row in gpu['rows']),prefix+'Wrong GPU')
            check((d/'published.json').exists()==r['accepted'],prefix+'Atomic publication mismatch')
            check(Path(r['output']).exists(),prefix+'Missing effective scene')
            check(r['unknown_usage_requests']==0,prefix+'Unknown usage')
    # Check that compressing context never altered M2's baseline prompt.
    prompt_comparisons = []
    for r in formal:
        if r['condition']=='M2' and r['task'] in ['T3','T4']:
            b = next(x for x in formal if x['condition']=='M0' and x['task']==r['task'] and x['seed']==r['seed'])
            same = (Path(r['directory'])/'planner_input.txt').read_text()==(Path(b['directory'])/'planner_input.txt').read_text()
            check(same,r['task_id']+': M2 changed the baseline prompt')
            prompt_comparisons.append(dict(task=r['task'],seed=r['seed'],identical=same))
    repeated_prompt_usage = []
    groups = {}
    for r in formal:
        if r['session_id'] is None:
            key = (r['condition'],r['task'])
            groups.setdefault(key,[]).append(r)
    for key, group in groups.items():
        prompts = [(Path(r['directory'])/'planner_input.txt').read_text() for r in group]
        inputs = [r['requests'][0]['usage']['input_tokens'] for r in group]
        repeated_prompt_usage.append(dict(condition=key[0],task=key[1],identical_submitted_prompts=len(set(prompts))==1,
                                         input_tokens=inputs,prompt_characters=[len(p) for p in prompts]))
    result = dict(passed=not errors,errors=errors,formal_rounds=len(formal),all_rounds=len(records),requests=len(requests),
                  paired_M0_M2_prompts=prompt_comparisons,repeated_prompt_usage=repeated_prompt_usage)
    save(root/'metrics/audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors:
        raise SystemExit(1)


if __name__=='__main__':
    main()
