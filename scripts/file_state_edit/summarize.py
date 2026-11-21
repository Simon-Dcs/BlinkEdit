"""Join the raw usage ledger to the artifact, feedback and independent-scoring endpoints."""

import argparse
import csv
import statistics
from pathlib import Path

from embodied_gen.harness.scene_edit.records import write_json
from embodied_gen.harness.scene_edit.state_store import read


def avg(items):
    vals = [v for v in items if v is not None]
    return statistics.mean(vals) if vals else None


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def summarize(root, stage):
    m = read(root / 'manifest.json')
    rows = []
    requests = []
    pairs = []
    by_arm = {}
    for kind, label in [('edithero', 'edithero'), ('3dcodeedit', 'code')]:
        for key in m[stage][label]:
            for arm in m['arms']:
                d = root / kind / stage / key / arm
                chain = []
                total_n = (
                    m['chain_info'][key]['n_turns']
                    if kind == 'edithero'
                    else None
                )
                for p in sorted(d.glob('turn*/outcome.json')):
                    row = read(p)
                    t = p.parent
                    record_list = []
                    for log in sorted(t.rglob('record.json')):
                        rec = read(log)
                        u = rec.get('usage')
                        stage_name = rec['stage']
                        record = dict(
                            kind=kind,
                            case=key,
                            arm=arm,
                            turn=row['turn'],
                            role=stage_name,
                            status=rec['status'],
                            time_s=rec['latency_s'],
                            tools=len(rec.get('tool_items', [])),
                            input=u['input_tokens'] if u else None,
                            output=u['output_tokens'] if u else None,
                            cached=(u or {}).get('cached_input_tokens'),
                            known_tokens=(
                                u['input_tokens'] + u['output_tokens']
                            )
                            if u
                            else None,
                            path=str(log.relative_to(root)),
                        )
                        requests.append(record)
                        record_list.append(record)

                    def cost(stages, k):
                        return sum(
                            v[k] or 0
                            for v in record_list
                            if v['role'] in stages
                        )

                    model = {'editor', 'reviewer'}
                    score = (
                        read(t / 'score.json')
                        if (t / 'score.json').exists()
                        else {}
                    )
                    v = score.get('views', [])
                    # First dev candidate predates the explicit artifact_s timer; stat its original write,
                    # not a file copied later during report generation.
                    artifact_s = row.get('artifact_s')
                    if artifact_s is None and row['completed']:
                        artifact_s = (
                            Path(row['artifact']).stat().st_mtime
                            - row['start']
                        )
                    observed = score.get('score_s')
                    # Resumed item-by-item scoring still accounts for earlier requests.
                    offline = (
                        max(observed or 0, cost({'verifier'}, 'time_s'))
                        if score
                        else (cost({'verifier'}, 'time_s') or None)
                    )
                    turn = dict(
                        kind=kind,
                        case=key,
                        arm=arm,
                        turn=row['turn'],
                        delivered=row['completed'],
                        error=row.get('error'),
                        cached=row.get('cached'),
                        recomputed=row.get('nodes_recomputed'),
                        first_write_s=artifact_s,
                        checked_s=row.get('edit_s'),
                        system_s=row['elapsed_s'],
                        model_s=cost(model, 'time_s'),
                        local_s=row['elapsed_s'] - cost(model, 'time_s'),
                        offline_s=offline,
                        system_tokens=cost(model, 'known_tokens'),
                        verifier_tokens=cost({'verifier'}, 'known_tokens'),
                        unknown=sum(v['input'] is None for v in record_list),
                        editor_tokens=cost({'editor'}, 'known_tokens'),
                        reviewer_tokens=cost({'reviewer'}, 'known_tokens'),
                        reviewed=row.get('accept'),
                        all_pass=score.get('all_pass'),
                        IF=score.get('IF'),
                        CC_prev=score.get('CC_prev'),
                        CC_initial=score.get('CC_initial'),
                        heldout_MAE=avg(s['edit_MAE'] for s in v if s['view']),
                        scored=bool(score),
                    )
                    rows.append(turn)
                    chain.append(turn)
                if not chain:
                    continue
                final = chain[-1]
                agg = dict(
                    kind=kind,
                    case=key,
                    arm=arm,
                    n=len(chain),
                    delivered=sum(t['delivered'] for t in chain),
                    expected=total_n,
                    scored=sum(t['scored'] for t in chain),
                    final_pass=final['all_pass'],
                    first_pass=chain[0]['all_pass'],
                    accept=final['reviewed'],
                    final_exec=final['delivered'],
                    IF_n=sum(t['IF'] is not None for t in chain),
                    IF=avg(t['IF'] for t in chain),
                    CC_prev=avg(t['CC_prev'] for t in chain),
                    CC_initial=avg(t['CC_initial'] for t in chain),
                    heldout_MAE=avg(t['heldout_MAE'] for t in chain),
                    cold_s=chain[0]['system_s'],
                    later_s=avg(t['system_s'] for t in chain[1:]),
                )
                for k in [
                    'system_s',
                    'model_s',
                    'local_s',
                    'system_tokens',
                    'editor_tokens',
                    'reviewer_tokens',
                    'verifier_tokens',
                    'unknown',
                ]:
                    agg[k] = sum(t[k] for t in chain)
                agg['offline_s'] = sum(t['offline_s'] or 0 for t in chain)
                pairs.append(agg)
        for arm in m['arms']:
            group = [p for p in pairs if p['kind'] == kind and p['arm'] == arm]
            turns = [p for p in rows if p['kind'] == kind and p['arm'] == arm]
            n = len(m[stage][label])
            key = f'{kind}/{arm}'
            by_arm[key] = dict(
                assigned=n,
                started=len(group),
                turns=len(turns),
                delivered=sum(p['delivered'] for p in turns),
                scored=sum(p['scored'] for p in turns),
                final_pass=sum(p['final_pass'] is True for p in group),
                final_judged=sum(p['final_pass'] is not None for p in group),
                first_pass=sum(p['first_pass'] is True for p in group),
                exec_cases=sum(p['final_exec'] for p in group),
                accepted=sum(p['accept'] is True for p in group),
                IF_n=sum(t['IF'] is not None for t in turns),
                IF=avg(t['IF'] for t in turns),
                CC_prev=avg(t['CC_prev'] for t in turns),
                CC_initial=avg(t['CC_initial'] for t in turns),
                heldout_MAE=avg(t['heldout_MAE'] for t in turns),
            )
            for k in [
                'system_s',
                'model_s',
                'local_s',
                'system_tokens',
                'editor_tokens',
                'reviewer_tokens',
                'verifier_tokens',
                'offline_s',
                'unknown',
            ]:
                by_arm[key][k] = sum(p[k] for p in group)
    out = root / 'metrics'
    save_csv(out / f'{stage}-requests.csv', requests)
    save_csv(out / f'{stage}-turns.csv', rows)
    save_csv(out / f'{stage}-cases.csv', pairs)
    write_json(out / f'{stage}-summary.json', by_arm)
    return by_arm


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('stage')
    a = p.parse_args()
    summarize(a.root.resolve(), a.stage)
