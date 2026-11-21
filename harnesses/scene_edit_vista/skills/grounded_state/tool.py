"""Read current public facts through the declared observation capability."""

from copy import deepcopy


def build(runtime):
    def handler(args):
        try:
            result = runtime.call('inspect', {})
        except Exception:
            return {'ok': False, 'reason': 'inspection_failed'}
        if isinstance(result, dict):
            facts = result.get('facts')
        else:
            facts = result
        if not isinstance(facts, list):
            return {'ok': False, 'reason': 'invalid_inspection_response'}
        if any(not isinstance(f, dict) or not isinstance(f.get('id'), str)
               for f in facts):
            return {'ok': False, 'reason': 'invalid_fact_records'}
        return {'ok': True, 'facts': deepcopy(facts)}

    return {
        'name': 'grounded_state',
        'description': 'Observe current public scene facts before interpreting an edit.',
        'input_schema': {
            'type': 'object', 'properties': {}, 'additionalProperties': False
        },
        'handler': handler
    }
