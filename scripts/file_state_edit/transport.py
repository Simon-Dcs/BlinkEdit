"""Global, budget-bound admission for a rate-limited native subscription transport."""

from contextlib import contextmanager
from pathlib import Path
import fcntl, json, time


@contextmanager
def admission(root, dest):
    path = root / 'metrics/transport-policy.json'
    policy = json.loads(path.read_text()) if path.exists() else {}
    if not policy.get('serialize'):
        yield
        return
    state = root / 'metrics/admission.json'
    begin = time.monotonic()
    with (root / 'metrics/admission.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        gate = json.loads(state.read_text()) if state.exists() else {}
        delay = max(0, gate.get('next', 0) - time.time())
        budget = root / 'metrics/formal-budget.json'
        start = (
            json.loads(budget.read_text())['start']
            if budget.exists()
            else time.time()
        )
        deadline = (
            start
            + json.loads((root / 'metrics/formal-limits.json').read_text())[
                'seconds'
            ]
        )
        if time.time() + delay >= deadline:
            raise RuntimeError(
                'Study time budget reached in the admission queue'
            )
        if delay:
            time.sleep(delay)
        admit = time.time()
        wait = time.monotonic() - begin
        try:
            yield
        finally:
            # The CLI uses its own backoff internally. Any observed 429 extends
            # the cooldown for all jobs, including editing, review and grading.
            rate = any(
                '429 Too Many Requests' in p.read_text(errors='replace')
                for p in dest.rglob('events.jsonl')
            )
            end = time.time()
            state.write_text(
                json.dumps(
                    dict(
                        next=max(
                            admit + policy['start_gap_s'],
                            end + (policy['backoff_s'] if rate else 0),
                        ),
                        last=admit,
                        rate_limited=rate,
                    )
                )
            )
            if dest.exists():
                (dest / 'admission.json').write_text(
                    json.dumps(
                        dict(
                            queue_s=wait,
                            admitted=admit,
                            finished=end,
                            rate_limited=rate,
                        )
                    )
                )
