"""Rebuild tables and figures from measured task/request records, including failures."""
import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import save

COLORS = dict(M0='#54657a', M1='#e59a3b', M2='#298c87', M3='#6c63aa')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dist(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=min(values), maximum=max(values), all=values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root')
    a = p.parse_args()
    root = Path(a.root or Path('outputs/efficiency_optimization/latest_run.txt').read_text().strip())
    rows = [json.loads(l) for l in (root/'metrics/tasks.jsonl').read_text().splitlines()]
    requests = [json.loads(l) for l in (root/'metrics/requests.jsonl').read_text().splitlines()]
    formal = [r for r in rows if r['phase']=='formal']
    data = []
    for r in rows:
        llm = sum(e['exclusive_s'] for e in r['events'] if e['stage']=='llm_rpc')
        def usage(key):
            return sum((q.get('usage') or {}).get(key,0) for q in r['requests'])
        data.append(dict(task_id=r['task_id'],condition=r['condition'],phase=r['phase'],task=r['task'],
            session_id=r['session_id'],turn_id=r['turn_id'],seed=r['seed'],status=r['status'],accepted=r['accepted'],
            verified_s=r['verified_s'],edit_s=r['edit_s'],llm_rpc_s=llm,non_llm_s=r['verified_s']-llm,
            process_overhead_s=r['process_overhead_s'],controller_wall_s=r['controller_wall_s'],
            input_tokens=usage('input_tokens'),cached_input_tokens=usage('cached_input_tokens'),
            uncached_input_tokens=usage('input_tokens')-usage('cached_input_tokens'),output_tokens=usage('output_tokens'),
            reasoning_output_tokens=usage('reasoning_output_tokens'),total_tokens=r['known_tokens'],
            requests=len(r['requests']),unknown_usage_requests=r['unknown_usage_requests'],
            video_encoding_s=r.get('video_encoding_s',0),directory=r['directory']))
    write_csv(root/'metrics/tasks.csv',data)
    write_csv(root/'metrics/requests.csv',[{k:v for k,v in q.items() if not isinstance(v,(list,dict))} | (q.get('usage') or {}) for q in requests])
    fdata = [r for r in data if r['phase']=='formal']
    sessions = []
    for sid in sorted({r['session_id'] for r in fdata if r['session_id']}):
        rs = [r for r in fdata if r['session_id']==sid]
        raw = sorted((r for r in formal if r['session_id']==sid),key=lambda r:r['turn_id'])
        sessions.append(dict(session_id=sid,condition=rs[0]['condition'],seed=rs[0]['seed'],turns=len(rs),
            accepted=sum(r['accepted'] for r in rs),verified_s=sum(r['verified_s'] for r in rs),
            session_wall_s=raw[-1]['verified_mono']-raw[0]['start_mono'],
            tokens=sum(r['total_tokens'] for r in rs),input_tokens=sum(r['input_tokens'] for r in rs),
            output_tokens=sum(r['output_tokens'] for r in rs),requests=sum(r['requests'] for r in rs)))
    write_csv(root/'metrics/sessions.csv',sessions)
    comparisons = []
    for row in fdata:
        if row['condition']=='M0':
            continue
        base = next((b for b in fdata if b['condition']=='M0' and b['task']==row['task'] and b['seed']==row['seed']),None)
        if base:
            comparisons.append(dict(task=row['task'],seed=row['seed'],condition=row['condition'],
                baseline_s=base['verified_s'],candidate_s=row['verified_s'],saved_s=base['verified_s']-row['verified_s'],
                reduction_fraction=1-row['verified_s']/base['verified_s'],
                baseline_tokens=base['total_tokens'],candidate_tokens=row['total_tokens'],
                token_reduction_fraction=1-row['total_tokens']/base['total_tokens'],
                baseline_ok=base['accepted'],candidate_ok=row['accepted']))
    write_csv(root/'metrics/comparisons.csv',comparisons)
    summary = dict(all_timed_rounds=len(rows),formal_rounds=len(formal),requests=len(requests),
        known_tokens=sum(r['total_tokens'] for r in data),unknown_usage_requests=sum(r['unknown_usage_requests'] for r in data),
        model_request_failures=sum(q['status']!='ok' for q in requests),
        input_tokens=sum(r['input_tokens'] for r in data),cached_input_tokens=sum(r['cached_input_tokens'] for r in data),
        output_tokens=sum(r['output_tokens'] for r in data),reasoning_output_tokens=sum(r['reasoning_output_tokens'] for r in data),
        pilot_rounds=sum(r['phase'].startswith('pilot') for r in rows),groups={},sessions=sessions)
    for task in sorted({r['task'] for r in fdata}):
        for c in ['M0','M1','M2','M3']:
            group = [r for r in fdata if r['task']==task and r['condition']==c]
            if not group:
                continue
            summary['groups'][f'{c}_{task}'] = dict(verified_s=dist([r['verified_s'] for r in group]),
                tokens=dist([r['total_tokens'] for r in group]),accepted=sum(r['accepted'] for r in group),
                status=[r['status'] for r in group],
                llm_s=dist([r['llm_rpc_s'] for r in group]),local_s=dist([r['non_llm_s'] for r in group]),
                total_consumption_per_accepted_tokens=sum(r['total_tokens'] for r in group)/sum(r['accepted'] for r in group)
                  if any(r['accepted'] for r in group) else None)
    if len(sessions)==6:
        s0 = [s for s in sessions if s['condition']=='M0']
        s1 = [s for s in sessions if s['condition']=='M1']
        reduction = 1-sum(s['tokens'] for s in s1)/sum(s['tokens'] for s in s0)
        summary['P1'] = dict(session_token_reduction=reduction,passed=reduction>=.10 and all(s['accepted']==4 for s in sessions),
            M0_tokens=sum(s['tokens'] for s in s0),M1_tokens=sum(s['tokens'] for s in s1),
            M0_session_s=dist([s['verified_s'] for s in s0]),M1_session_s=dist([s['verified_s'] for s in s1]))
        summary['P1']['M0_session_wall_s'] = dist([s['session_wall_s'] for s in s0])
        summary['P1']['M1_session_wall_s'] = dist([s['session_wall_s'] for s in s1])
    m0 = [r for r in fdata if r['condition']=='M0' and r['task'] in ['T3','T4']]
    m2 = [r for r in fdata if r['condition']=='M2' and r['task'] in ['T3','T4']]
    if len(m2)==6 and len(m0)==6:
        reduction = 1-sum(r['verified_s'] for r in m2)/sum(r['verified_s'] for r in m0)
        summary['P2'] = dict(verified_reduction=reduction,passed=reduction>=.15 and all(r['accepted'] for r in m2+m0),
            M0=dist([r['verified_s'] for r in m0]),M2=dist([r['verified_s'] for r in m2]),
            M2_first_s=m2[0]['verified_s'],M2_hot=dist([r['verified_s'] for r in m2[1:]]),
            M0_local=dist([r['non_llm_s'] for r in m0]),M2_local=dist([r['non_llm_s'] for r in m2]),
            M0_llm=dist([r['llm_rpc_s'] for r in m0]),M2_llm=dist([r['llm_rpc_s'] for r in m2]),
            amortization_basis='Sum of measured own request intervals incl first init and every reset; excludes interleaved waiting.')
    lifetime = root/'formal_resident_lifetime.json'
    if lifetime.exists():
        life = json.loads(lifetime.read_text())
        resident = [r for r in formal if r['condition']=='M2']
        active = sum(r['controller_wall_s'] for r in resident)
        summary['resident_lifetime'] = dict(**life, tasks=len(resident),active_controller_wall_s=active,
                                          interleaved_wait_and_shutdown_s=life['wall_s']-active)
    save(root/'metrics/summary.json',summary)
    if not fdata:
        print(json.dumps(summary,indent=2))
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    for j,c in enumerate(['M0','M1','M2']):
        for i,task in enumerate(['T1','T2','T3','T4']):
            rs = [r for r in fdata if r['task']==task and r['condition']==c]
            if not rs:
                continue
            x = i+(j-1)*.23
            vals = [r['verified_s'] for r in rs]
            axes[0].bar(x,statistics.mean(vals),width=.21,color=COLORS[c],label=c if task=='T3' else None,alpha=.85)
            axes[0].scatter([x]*len(vals),vals,c='black',s=10,zorder=3)
    axes[0].set(xticks=range(4),xticklabels=['T1 delete','T2 insert','T3 four','T4 mixed'],ylabel='Instruction to verified output (s)',title='All main-task repeats; startup included')
    axes[0].legend(frameon=False)
    for i,c in enumerate(['M0','M1']):
        ss = [s for s in sessions if s['condition']==c]
        if ss:
            vals = [s['tokens']/1000 for s in ss]
            axes[1].bar(i,statistics.mean(vals),width=.6,color=COLORS[c],alpha=.85)
            axes[1].scatter([i]*len(vals),vals,c='black',s=14)
    axes[1].set(xticks=[0,1],xticklabels=['M0 full history','M1 compact state'],ylabel='Total tokens / four-turn session (thousands)',title='Includes first turn and deleted-object turn')
    for i,c in enumerate(['M0','M1','M2']):
        rs = [r for r in fdata if r['condition']==c and r['task'] in ['T3','T4']]
        if not rs:
            continue
        rpc = statistics.mean(r['llm_rpc_s'] for r in rs)
        local = statistics.mean(r['non_llm_s'] for r in rs)
        axes[2].bar(i,rpc,color='#bac4cf',label='CLI + network + remote model' if i==0 else None)
        axes[2].bar(i,local,bottom=rpc,color='#43657b',label='Local runtime + verification' if i==0 else None)
        axes[2].text(i,rpc+local+.8,f'{rpc+local:.1f}s',ha='center')
    axes[2].set(xticks=[0,1,2],xticklabels=['M0 cold','M1 cold','M2 resident'],ylabel='Mean T3/T4 (s)',title='Remote and local time, not pure inference')
    axes[2].legend(frameon=False,fontsize=8,loc='upper left')
    axes[2].set_ylim(0,axes[2].get_ylim()[1]*1.2)
    fig.suptitle('EmbodiedGen | GPU 7 | gpt-6-astra / medium',fontsize=15,weight='bold')
    fig.savefig(root/'comparisons/performance.png',dpi=180)
    plt.close(fig)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    # Every after-render is retained in labelled contact sheets for visual acceptance.
    for page in range((len(formal)+11)//12):
        sheet = Image.new('RGB',(4*432,3*320),'#f4f5f7')
        draw = ImageDraw.Draw(sheet)
        for j,r in enumerate(formal[page*12:(page+1)*12]):
            path = Path(r['directory'])/'after_office.png'
            x,y = j%4*432,j//4*320
            if path.exists():
                im = Image.open(path).convert('RGB').resize((432,288))
                sheet.paste(im,(x,y+30))
            draw.text((x+8,y+4),r['task_id'].replace('formal_','')+(' OK' if r['accepted'] else ' FAIL'),font=font,fill='#1d2939')
        sheet.save(root/f'comparisons/contact_{page+1:02d}.jpg',quality=92)
    candidates = [next((r for r in formal if r['condition']==c and r['task']=='T4' and r['seed']==42),None) for c in ['M0','M1','M2']]
    if all(candidates):
        sheet = Image.new('RGB',(1080*2,760*2),'white')
        draw = ImageDraw.Draw(sheet)
        views = [(candidates[0],'before_office.png','Input: office + dining room'),
                 (candidates[0],'after_office.png','M0: full context / cold process'),
                 (candidates[1],'after_office.png','M1: compact context / cold process'),
                 (candidates[2],'after_office.png','M2: full context / resident process')]
        for j,(r,file,label) in enumerate(views):
            x,y = j%2*1080,j//2*760
            sheet.paste(Image.open(Path(r['directory'])/file).convert('RGB'),(x,y+40))
            draw.text((x+20,y+8),label,font=font,fill='#172331')
        sheet.save(root/'comparisons/scene_comparison.jpg',quality=94)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['groups','sessions']},indent=2))


if __name__ == '__main__':
    main()
