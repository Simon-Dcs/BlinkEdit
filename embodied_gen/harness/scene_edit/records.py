"""Small, durable records; explicit IDs rather than content-derived identities.

The write-once publication pattern follows VISTA meta_harness/storage.py,
commit 62eab4e6c4d6b413a7a8d653fe6e9ba1f5c50486 (Apache-2.0).
Copyright 2026 The RPent Authors. Adapted for EmbodiedGen.
"""

import json
import os
import tempfile
from pathlib import Path


def write_json(path, value, *, once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if once and path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                f"Record ID already has different content: {path}"
            )
        return
    fd, tmp = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if once:
            os.link(tmp, path)
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_json(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
