"""Stdlib-only child, launched in an empty mount and network namespace."""

import importlib.util
import json
import resource
import sys
from pathlib import Path

resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
sys.path.insert(0, "/policy")
spec = importlib.util.spec_from_file_location(
    "candidate_policy", "/policy/policy.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
policy = module.Policy()
state = {}
manifest = json.loads(Path('/policy/HARNESS.json').read_text())
calls = []
payload = {}


class SkillRuntime:
    def __init__(self, name, requires):
        self.name, self.requires = name, requires

    def call(self, alias, args):
        capability = self.requires[alias]
        calls.append(dict(skill=self.name, capability=capability, args=args))
        facts = payload.get('facts', [])
        if capability == 'scene.inspect@1':
            return facts
        if capability == 'scene.query@1':
            return next((f for f in facts if f['id'] == args['id']), None)
        raise ValueError('Capability not available in preparation hook')


skills = {}
for item in manifest['skills']:
    filename = '/policy/skills/' + item['name'] + '/tool.py'
    skill_spec = importlib.util.spec_from_file_location(
        'skill_' + item['name'], filename
    )
    skill_module = importlib.util.module_from_spec(skill_spec)
    skill_spec.loader.exec_module(skill_module)
    binding = skill_module.build(SkillRuntime(item['name'], item['requires']))
    if set(binding) != {
        'name',
        'description',
        'input_schema',
        'handler',
    } or not callable(binding['handler']):
        raise ValueError('Invalid skill binding')
    skills[item['name']] = binding['handler']
state['skills'] = skills
state['memory_text'] = (Path('/policy') / manifest['memory']).read_text()
for line in sys.stdin:
    try:
        request = json.loads(line)
        payload = request['payload']
        calls.clear()
        hook = getattr(policy, request["hook"], None)
        output = (
            hook(request["payload"], state) if hook else request["payload"]
        )
        print(
            json.dumps(
                {"ok": True, "output": output, "calls": calls},
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception as exc:
        print(json.dumps({"ok": False, "error": repr(exc)}), flush=True)
