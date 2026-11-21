"""Ground requests using observations; never rewrite parsed edit semantics."""

from copy import deepcopy


def _facts(value):
    if not isinstance(value, list):
        return None
    if any(not isinstance(f, dict) or not isinstance(f.get('id'), str)
           for f in value):
        return None
    return deepcopy(value)


def _identity_context(history, facts):
    by_stable = {}
    for fact in facts:
        stable = fact.get('stable_id')
        if isinstance(stable, str) and stable:
            by_stable.setdefault(stable, []).append(fact)

    bindings = []
    for index, event in enumerate(history):
        if not isinstance(event, dict) or event.get('status') != 'execute':
            continue
        inserted = event.get('inserted', [])
        if not isinstance(inserted, list):
            continue
        for record in inserted:
            if not isinstance(record, dict):
                continue
            stable = record.get('stable_id')
            binding = {
                'history_index': index,
                'historical_instance_key': record.get('instance_key'),
                'stable_id': stable,
                'historical_asset': record.get('asset'),
                'historical_support': record.get('on_instance'),
                'historical_room': record.get('in_room'),
                'current': None
            }
            if not isinstance(stable, str) or not stable:
                binding['resolution'] = 'unverified_identity'
            else:
                matches = by_stable.get(stable, [])
                if len(matches) == 1:
                    binding['resolution'] = 'resolved'
                    binding['current'] = deepcopy(matches[0])
                elif matches:
                    binding['resolution'] = 'ambiguous_identity'
                else:
                    binding['resolution'] = 'absent_identity'
            bindings.append(binding)
    return bindings


class Policy:
    def prepare_request(self, payload, state):
        # Observe anew even when the previous request used the same state object.
        observation = None
        try:
            observation = state['skills']['grounded_state']({})
        except Exception:
            pass
        facts = None
        source = 'unavailable'
        if isinstance(observation, dict) and observation.get('ok') is True:
            facts = _facts(observation.get('facts'))
            if facts is not None:
                source = 'live_inspection'
        if facts is None:
            facts = _facts(payload.get('facts'))
            if facts is not None:
                source = 'current_request_facts'
        available = facts is not None
        if facts is None:
            facts = []

        history = deepcopy(payload.get('history', []))
        if not isinstance(history, list):
            history = []
        context = {
            'objects': ([f['id'] for f in facts] if available
                        else deepcopy(payload.get('objects', []))),
            'rooms': deepcopy(payload.get('rooms', [])),
            'assets': deepcopy(payload.get('assets', [])),
            'history': history,
            'facts': facts,
            'observation': {'source': source, 'available': available},
            'identity_bindings': (_identity_context(history, facts)
                                  if available else []),
            'experience': state.get('memory_text', '')
        }
        return {
            'context': context,
            'guidance': (
                'Parse this new instruction normally using the factual context. '
                'Current facts take precedence over names and historical locations. '
                'Use explicit support IDs to distinguish floor-supported objects '
                'from objects supported by furniture; use current room and world '
                'bounds when evaluating physical conditions. Coordinates and bounds '
                'are in metres. Do not infer support from name or centre height alone. '
                'For references to confirmed prior insertions, identity_bindings '
                'provides stable-ID matches and the current object fact. Use the '
                'current ID and evaluate conditions against its current room and '
                'support. Historical placement is historical context only. '
                'An absent stable identity is not a replacement with a similar name, '
                'asset or position. Do not bind unresolved historical references to '
                'other objects. Missing current facts mean unknown, not absent. '
                'Repeated deletion of a confirmed absent identity is a no-op. '
                'Clarify genuinely ambiguous references. Preserve unrelated objects '
                'and all requested assets, scales, rooms and targets. '
                'Only confirmed history establishes prior committed edits; attempted '
                'or failed plans do not establish objects or conversational referents.'
            ),
            'placement_attempts': 1
        }

    def decide(self, payload, state):
        # No evidence supports changing native insertion order or parsed semantics.
        return payload['plan']

    def process_tool_result(self, payload, state):
        # A full-budget attempt has already been selected. Never manufacture
        # confirmation, reinterpret a target, or request an unbudgeted retry.
        error = payload.get('error')
        return {
            'retry': False,
            'plan': payload['plan'],
            'message': '' if error is None else str(error)
        }
