"""Derive all numerical tables from retained measurements, never estimates."""
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from datetime import datetime

root = Path(Path('outputs/efficiency/latest_run.txt').read_text().strip())
rows, request_rows = [], []
for p in sorted(root.glob('runs/*/case.json')):
    case = json.loads(p.read_text())
    row = dict(run=p.parent.name,case=case['case'], repeat=case['repeat'],
               wall_s=case['seconds'], cli_success=case['cli_success'],
               llm_calls=0, input_tokens=0, cached_input_tokens=0, output_tokens=0,
               reasoning_output_tokens=0, unknown_usage=0)
    stages = defaultdict(float)
    for tracefile in sorted(p.parent.glob('call_*/trace.json')):
        trace = json.loads(tracefile.read_text())
        exclusive = sum(e['exclusive_s'] for e in trace['events'])
        assert exclusive <= trace['seconds']+.001, tracefile
        for e in trace['events']:
            stages[e['stage']] += e['exclusive_s']
        for j,q in enumerate(trace['requests']):
            row['llm_calls'] += 1
            if q['usage'] is None:
                row['unknown_usage'] += 1
            else:
                for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']:
                    row[k] += q['usage'].get(k,0)
            request_rows.append(dict(run=p.parent.name, call=tracefile.parent.name, request=j,
                                     seconds=q['seconds'], status=q['status'], model=q['model'],
                                     **(q['usage'] or {})))
    row['total_tokens'] = row['input_tokens']+row['output_tokens']
    row['uncached_input_tokens'] = row['input_tokens']-row['cached_input_tokens']
    stages['other_process_overhead'] = row['wall_s']-sum(stages.values())
    assert stages['other_process_overhead'] >= -.001
    row.update({k+'_s':v for k,v in stages.items()})
    val = p.parent/'validation.json'
    if val.exists():
        v=json.loads(val.read_text())
        expected_added = 0 if row['case'].startswith(('S1','N1')) else 1 if row['case'].startswith('S2') else 4
        expected_removed = 1 if row['case'].startswith(('S1','E1')) else 0
        row['inventory_ok'] = len(v['added'])==expected_added and len(v['removed'])==expected_removed and v['unchanged_poses']
        if expected_removed:
            row['inventory_ok'] = row['inventory_ok'] and v['removed']==['lamp_office_001']
        # Older validation records can derive type from our canonical new_* keys.
        types=sorted(x.get('asset_type',k.removeprefix('new_')) for k,x in v['assets'].items())
        if expected_added == 4:
            row['inventory_ok'] = row['inventory_ok'] and types==['book','lamp','mug','vase']
        row['geometry_screen_ok'] = all(x['xy_within_table_hull'] and x['collision_present'] and abs(x['support_gap_to_table_bbox_m'])<.01 for x in v['assets'].values()) and all(x['conservative_xy_overlap_m2']<1e-6 for x in v['pairwise'])
        if row['case'].startswith('N1'):
            row['expected_rejection_ok'] = not row['cli_success'] and not (p.parent/'scene/scene_updated.urdf').exists()
    rows.append(row)


def write_csv(path, rows):
    fields = sorted(set(k for r in rows for k in r))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


write_csv(root/'metrics/per_case.csv',rows)
write_csv(root/'metrics/requests.csv',request_rows)
summary={}
for key in sorted(set(r['case'] for r in rows)):
    group=[r for r in rows if r['case']==key]
    numeric=set(k for r in group for k,v in r.items() if isinstance(v,(int,float)) and not isinstance(v,bool))
    summary[key]=dict(n=len(group),successes=sum(r['cli_success'] for r in group),
                      mean={k:statistics.mean(r.get(k,0) for r in group) for k in numeric},
                      wall_min=min(r['wall_s'] for r in group), wall_max=max(r['wall_s'] for r in group),
                      wall_median=statistics.median(r['wall_s'] for r in group))
total={k:sum(r[k] for r in rows) for k in ['wall_s','llm_calls','input_tokens','cached_input_tokens','uncached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens','unknown_usage']}
result=dict(generated_at=datetime.now().astimezone().isoformat(),total=total,cases=summary,rows=rows)
(root/'metrics/summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(total=total,cases={k:{'mean_s':v['mean']['wall_s'],'min':v['wall_min'],'max':v['wall_max'],'mean_tokens':v['mean']['total_tokens'],'successes':v['successes']} for k,v in summary.items()}),indent=2))

# Data plot, not a generated illustration. Every height is a measured mean.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
keys=list(summary)
fig,axes=plt.subplots(1,2,figsize=(13,4.6))
colors=['#51657a','#137c8b','#cc7135','#9565ab','#40936e','#d5a027','#8b5a52']
for ax,field,ylabel in [(axes[0],'wall_s','Mean editing wall time (s)'),(axes[1],'total_tokens','Mean reported tokens (input + output)')]:
    ax.bar([k.split('_')[0] for k in keys],[summary[k]['mean'][field] for k in keys],color=colors[:len(keys)])
    ax.set_ylabel(ylabel);ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.suptitle('EmbodiedGen V2.1.0 | native spatial editing | 3 repeats / case',fontweight='bold')
fig.tight_layout()
fig.savefig(root/'comparison/efficiency.png',dpi=180)
plt.close(fig)

from PIL import Image, ImageDraw, ImageFont
panels=[('Input',root/'comparison/baseline/office.png'),
        ('S1: delete lamp',root/'runs/S1_delete_semantic_r1/office.png'),
        ('S2: insert vase',root/'runs/S2_insert_semantic_r1/office.png'),
        ('H1: four objects',root/'runs/H1_four_semantic_r1/office.png'),
        ('A2: native batch',root/'runs/A2_four_exact_batch_r1/office.png'),
        ('E1: plan + native edits',root/'runs/E1_full_request_r1/office.png')]
canvas=Image.new('RGB',(1440,3*530),'white')
draw=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
for i,(label,path) in enumerate(panels):
    if not path.exists():continue
    x=(i%2)*720;y=(i//2)*530
    draw.text((x+16,y+8),label,fill='#182c3b',font=font)
    canvas.paste(Image.open(path).resize((720,480)),(x,y+50))
canvas.save(root/'comparison/edits.png')
