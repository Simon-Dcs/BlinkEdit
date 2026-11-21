"""Independent geometry diagnostics for the local program primitives (not blind checklist grades)."""

import sys, shutil, tempfile
from pathlib import Path
from prepare import R, read

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
from sandbox import isolated

CHECK = """import bpy,runpy,json,numpy as np
def observe(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    runpy.run_path(path,run_name='__main__')
    d=bpy.context.evaluated_depsgraph_get()
    ans={}
    for o in bpy.context.scene.objects:
        if o.type!='MESH' or o.hide_render:continue
        e=o.evaluated_get(d);mesh=e.to_mesh()
        v=np.empty(len(mesh.vertices)*3, dtype=np.float32)
        mesh.vertices.foreach_get('co',v);v=v.reshape(-1,3)
        M=np.asarray(e.matrix_world);v=v@M[:3,:3].T+M[:3,3]
        ans[o.name]=(v,len(mesh.polygons));e.to_mesh_clear()
    return ans
a=observe('/work/source.py');b=observe('/work/edited.py')
out={}
for name in sorted(set(a)|set(b)):
    if name not in a or name not in b:out[name]={'unmatched':True};continue
    v,f=a[name];w,g=b[name]; dv=np.ptp(v,axis=0);dw=np.ptp(w,axis=0)
    out[name]=dict(count_original=len(v),count_edited=len(w),faces=[f,g],unchanged=(v.shape==w.shape and bool(np.allclose(v,w,rtol=0,atol=1e-6)) and f==g),source_bounds=[v.min(0).tolist(),v.max(0).tolist()],edited_bounds=[w.min(0).tolist(),w.max(0).tolist()],axis_ratios=np.divide(dw,dv,out=np.zeros(3),where=dv>1e-8).tolist())
json.dump(out,open('/work/report.json','w'))
"""


def main():
    m = read(R / 'manifest.json')
    for p in sorted((R / '3dcodeedit/formal').glob('*/*/turn01/outcome.json')):
        r = read(p)
        if (
            r.get('route') != 'local'
            or not r['completed']
            or (p.parent / 'geometry.json').exists()
        ):
            continue
        pub = read(R / 'setup/public_code' / f"{r['case']}.json")
        src = (
            Path(m['roots']['code'])
            / 'data/sources'
            / f"{pub['source_id']}.py"
        )
        with tempfile.TemporaryDirectory() as temp:
            w = Path(temp)
            shutil.copyfile(src, w / 'source.py')
            shutil.copyfile(r['artifact'], w / 'edited.py')
            (w / 'check.py').write_text(CHECK)
            rc = isolated(
                [
                    '/blender/blender',
                    '-b',
                    '--threads',
                    '2',
                    '--python-exit-code',
                    '1',
                    '--python',
                    '/work/check.py',
                ],
                w,
                readonly=[
                    (R.parent / 'tools/blender-5.1.0-linux-x64', '/blender')
                ],
                timeout=180,
                log=p.parent / 'geometry.log',
            )
            if rc:
                raise RuntimeError(p)
            shutil.copyfile(w / 'report.json', p.parent / 'geometry.json')


if __name__ == '__main__':
    main()
