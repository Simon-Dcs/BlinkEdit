import argparse
import csv
import json
import random
import subprocess
from pathlib import Path

from embodied_gen.harness.scene_edit.records import write_json


def freeze(r, ce, eh):
    chains = json.loads((r / 'setup/data/chains.json').read_text())
    byid = {x['chain']: x for x in chains}
    dev = [
        'R/50c96950_s0',
        'G/a4bef399_s0',
    ]  # first contiguous short, CC BY/CC0 chains in the two archive streams
    hosts = {byid[k]['host_uid'] for k in dev}
    formal = []
    for fam in 'APRMGX':
        candidates = sorted(
            (
                c
                for c in chains
                if c['chain'].startswith(fam + '/')
                and c['host_uid'] not in hosts
                and 3 <= c['n_turns'] <= 8
                and c['license_class'] == 'CC BY / CC0'
            ),
            key=lambda c: (c['n_turns'], c['chain']),
        )
        c = candidates[0]
        formal.append(c['chain'])
        hosts.add(c['host_uid'])
    tasks = list(
        map(json.loads, (ce / 'data/tasks.jsonl').read_text().splitlines())
    )
    attr = {
        a['source_id']: a
        for a in csv.DictReader((ce / 'data/ATTRIBUTION.csv').open())
    }
    allowed = set(
        json.loads((ce / 'data/analysis_subset.json').read_text())['case_ids']
    )
    pilot = [
        '3dcodebench_TVStand_seed0_L1_Q0',
        '3dcodebench_DoorCasing_seed0_L4_Q0',
    ]
    formal_ce = []
    rng = random.Random(20261118)
    used = set()
    for k in ['G1', 'G2', 'G3', 'G4', 'S', 'C']:
        pool = sorted(
            (
                t
                for t in tasks
                if t['edit_type'] == k
                and t['case_id'] in allowed
                and 'NC' not in attr[t['source_id']]['license']
            ),
            key=lambda t: t['case_id'],
        )
        rng.shuffle(pool)
        for _ in range(2):
            t = next(t for t in pool if t['source_id'] not in used)
            formal_ce.append(t['case_id'])
            used.add(t['source_id'])
    for task in tasks:
        if task['case_id'] in pilot + formal_ce:
            write_json(
                r / 'setup/public_code' / f"{task['case_id']}.json",
                {k: task[k] for k in ('source_id', 'instruction')},
                once=True,
            )
    write_json(
        r / 'manifest.json',
        dict(
            dev=dict(edithero=dev, code=pilot),
            formal=dict(edithero=formal, code=formal_ce),
            chain_info={k: byid[k] for k in dev + formal},
            attribution={
                t['source_id']: attr[t['source_id']]
                for t in tasks
                if t['case_id'] in pilot + formal_ce
            },
            roots=dict(code=str(ce), edithero=str(eh)),
            repo_revisions={
                str(p): subprocess.check_output(
                    ['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True
                ).strip()
                for p in [ce, eh]
            },
            dataset_revision=json.loads(
                (r / 'setup/hf-mirror.com.json').read_text()
            )['sha'],
            gpu=[6, 7],
            model='gpt-6-astra/medium',
            arms=['direct', 'F0', 'F1'],
            pilot_budget=dict(requests=80, tokens=1500000, seconds=7200),
            pilot_max_revisions=1,
            reviewer='joint clause-wise visual report per round; no hidden checklist',
            verifier='independent GPT per official item; six matched pairs',
            protocol='separate offline scoring; failed valid edits retained, no GT resets',
        ),
        once=True,
    )


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    a = p.parse_args()
    freeze(
        a.run, Path('../3DCodeEdit').resolve(), Path('../EditHero').resolve()
    )
