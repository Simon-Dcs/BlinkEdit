"""Fixed scene-state projection for one language interpretation per instruction."""

from collections import defaultdict


def compact_state(payload):
    """Keep semantic relations; resolve confirmed identities before interpretation."""
    facts = payload['facts']
    by_identity = defaultdict(list)
    for fact in facts:
        if fact.get('stable_id'):
            by_identity[fact['stable_id']].append(fact['id'])
    objects = {
        f['id']: {'kind': f['asset'], 'room': f['room'], 'on': f['support']}
        for f in facts
        if f['id'] in payload['objects']
    }
    history = []
    for event in payload['history']:
        entry = {
            'status': event['status'],
            'deleted': event.get('deleted', []),
            'inserted': [],
        }
        if event['status'] == 'execute':
            for item in event.get('inserted', []):
                identity = item.get('stable_id')
                matches = by_identity.get(identity, []) if identity else []
                resolution = (
                    'unverified'
                    if not identity
                    else 'resolved'
                    if len(matches) == 1
                    else 'ambiguous'
                    if matches
                    else 'unverified'
                    if payload.get('unidentified_objects')
                    else 'absent'
                )
                entry['inserted'].append(
                    {
                        'kind': item['asset'],
                        'old_id': item['instance_key'],
                        'old_room': item.get('in_room'),
                        'old_on': item.get('on_instance'),
                        'current_id': matches[0]
                        if resolution == 'resolved'
                        else None,
                        'resolution': resolution,
                    }
                )
        history.append(entry)
    return {
        'objects': objects,
        'rooms': payload['rooms'],
        'assets': payload['assets'],
        'history': history,
    }


class DeterministicContext:
    """Fixed adapter to the existing executor; no generated policy is loaded."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def call(self, hook, payload):
        if hook == 'prepare_request':
            return {
                'context': compact_state(payload),
                'placement_attempts': 1,
                'guidance': (
                    'State objects are keyed by exact current ID; kind is asset type, '
                    'on lists actual supports (room floors or furniture). '
                    'History contains confirmed edits. Use resolved current_id for '
                    'historical references and its current room/support from objects. '
                    'An absent identity stays absent; never substitute a similar object. '
                    'Unverified or ambiguous identity is unknown, not absent. '
                    'Historical room/support describe past placement only.'
                ),
            }
        if hook == 'decide':
            return payload['plan']
        if hook == 'process_tool_result':
            return {
                'retry': False,
                'plan': payload['plan'],
                'message': payload['error'],
            }
        raise ValueError('Unknown fixed planning step: ' + hook)
