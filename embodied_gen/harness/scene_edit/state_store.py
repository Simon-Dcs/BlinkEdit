"""Single-writer revision store for static editing sessions. Only current.json publishes."""

import fcntl
import json
import uuid
from contextlib import contextmanager
from pathlib import Path

from .records import write_json


def read(path):
    return json.loads(Path(path).read_text())


def stamps(paths):
    out = {}
    for p in sorted(set(map(Path, paths))):
        try:
            st = p.stat()
            value = [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns]
        except FileNotFoundError:
            value = None
        out[str(p.absolute())] = value
    return out


class RevisionStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.pointer = self.root / 'current.json'

    @contextmanager
    def writer(self):
        with (self.root / '.writer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield self

    def current(self):
        return (
            read(self.pointer)['revision'] if self.pointer.exists() else None
        )

    def directory(self, revision):
        return self.root / 'revisions' / revision

    def begin(self):
        path = self.directory(uuid.uuid4().hex)
        (path / 'artifact').mkdir(parents=True)
        return path

    def load(self):
        revision = self.current()
        if revision is None:
            return None
        directory = self.directory(revision)
        manifest = read(directory / 'manifest.json')
        dependencies = stamps(manifest['dependencies'])
        hit = dependencies == manifest['dependencies']
        try:
            hit = hit and stamps(manifest['derived']) == manifest['derived']
            state = read(directory / 'state.json') if hit else None
        except (FileNotFoundError, ValueError):
            state = None
        return dict(
            revision=revision,
            artifact=Path(manifest['artifact']),
            state=state,
            history=manifest['history'],
            dependencies=dependencies,
        )

    def commit(
        self,
        directory,
        artifact,
        state,
        *,
        parent,
        expected,
        dependencies,
        delta,
        history,
        derived=(),
    ):
        directory, artifact = Path(directory), Path(artifact).resolve()
        if not artifact.is_file():
            raise ValueError('Missing revision artifact')
        if not Path(directory).resolve().parent == self.root / 'revisions':
            raise ValueError('Revision is outside this session')
        current_deps = stamps(dependencies)
        if None in current_deps.values():
            raise ValueError('Incomplete artifact dependencies')
        write_json(directory / 'state.json', state, once=True)
        write_json(directory / 'delta.json', delta, once=True)
        write_json(
            directory / 'manifest.json',
            dict(
                schema=1,
                parent=parent,
                artifact=str(artifact),
                history=history,
                dependencies=current_deps,
                derived=stamps([directory / 'state.json', *derived]),
            ),
            once=True,
        )
        if parent != self.current() or stamps(expected) != expected:
            raise RuntimeError(
                'Parent revision or its dependencies changed during edit'
            )
        write_json(self.pointer, dict(revision=directory.name))
