"""Unlearned baseline: one language plan followed by native batch execution."""


class Policy:
    def prepare_request(self, payload, state):
        return {'context': {k: payload[k] for k in ['objects','rooms','assets','history']},
                'guidance': '', 'placement_attempts': 1}

    def decide(self, payload, state):
        return payload['plan']

    def process_tool_result(self, payload, state):
        return {'retry': False, 'plan': payload['plan'], 'message': payload['error']}
