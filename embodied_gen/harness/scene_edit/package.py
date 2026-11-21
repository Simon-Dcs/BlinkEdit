"""Versioned policy packages, isolated from assets, evaluators and credentials."""

import json
import os
import select
import shutil
import subprocess
import time
from pathlib import Path


def package_files(root):
    root = Path(root).resolve()
    result = []
    for p in root.rglob("*"):
        if ".git" in p.relative_to(root).parts or "__pycache__" in p.parts:
            continue
        if p.is_symlink():
            raise ValueError("Policy packages cannot contain symlinks")
        if p.is_file():
            if p.stat().st_size > 128 * 1024:
                raise ValueError(
                    "Large files belong outside the policy package"
                )
            p.read_text(encoding="utf-8")
            result.append(str(p.relative_to(root)))
    if not {"HARNESS.json", "policy.py"} <= set(result):
        raise ValueError("Expected HARNESS.json and policy.py")
    manifest = json.loads((root / "HARNESS.json").read_text())
    if (
        set(manifest)
        != {
            'schema_version',
            'id',
            'capabilities',
            'policy',
            'skills',
            'memory',
        }
        or manifest.get("schema_version") != 1
        or manifest['policy'] != 'policy.py:Policy'
    ):
        raise ValueError("Unsupported policy package schema")
    capabilities = {
        'scene.inspect@1',
        'scene.query@1',
        'scene.batch_edit@1',
        'scene.preview@1',
    }
    if not set(manifest['capabilities']) <= capabilities:
        raise ValueError('Unknown capability declaration')
    for skill in manifest['skills']:
        name = skill['name']
        if not name or any(
            c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name
        ):
            raise ValueError('Invalid skill name')
        if not set(skill['requires'].values()) <= {
            'scene.inspect@1',
            'scene.query@1',
        }:
            raise ValueError(
                'Preparation skills can only inspect public scene facts'
            )
        if not {f'skills/{name}/tool.py', f'skills/{name}/SKILL.md'} <= set(
            result
        ):
            raise ValueError(
                'Skill declaration has no executable and documentation'
            )
    return sorted(result)


class IsolatedPolicy:
    """Only JSON is exchanged; the worker cannot mount the host filesystem."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        package_files(self.root)
        executable = shutil.which("bwrap")
        if not executable:
            raise RuntimeError(
                "bubblewrap is required for executable policies"
            )
        cmd = [
            executable,
            "--die-with-parent",
            "--unshare-all",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/scratch",
            "--ro-bind",
            str(self.root),
            "/policy",
            "--ro-bind",
            str(Path(__file__).with_name("policy_worker.py")),
            "/runner.py",
            "--chdir",
            "/policy",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--",
            "/usr/bin/python3",
            "-u",
            "-B",
            "/runner.py",
        ]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PATH": "/usr/bin:/bin"},
        )
        self.calls = []

    def call(self, hook, payload):
        self.process.stdin.write(
            json.dumps({"hook": hook, "payload": payload}) + "\n"
        )
        self.process.stdin.flush()
        data = b''
        deadline = time.monotonic() + 10
        while b'\n' not in data:
            remaining = deadline - time.monotonic()
            if (
                remaining <= 0
                or not select.select([self.process.stdout], [], [], remaining)[
                    0
                ]
            ):
                raise RuntimeError("Policy hook exceeded its 10-second limit")
            chunk = os.read(
                self.process.stdout.fileno(), min(8192, 256 * 1024 - len(data))
            )
            if not chunk:
                raise RuntimeError(
                    "Policy process exited before returning JSON"
                )
            data += chunk
            if len(data) >= 256 * 1024:
                raise RuntimeError("Policy output exceeded 256 KiB")
        line = data.split(b'\n', 1)[0].decode('utf-8')
        result = json.loads(line)
        if not result["ok"]:
            raise RuntimeError(result["error"])
        self.calls.extend(result.get('calls', []))
        return result["output"]

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Versions:
    """Private Git DAG; never stages, resets or commits the user's repository."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.repo = self.root / "repository"
        self.repo.mkdir(parents=True, exist_ok=True)
        if not (self.repo / ".git").exists():
            self.git("init", "-q")

    def git(self, *args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "user.name=Scene Edit",
                "-c",
                "user.email=scene-edit@localhost",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            cwd=self.repo,
            text=True,
        ).strip()

    def freeze(self, source, message, parent=None):
        files = package_files(source)
        # Each submitted package is a complete small source snapshot.
        for p in self.repo.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        for name in files:
            dest = self.repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(source) / name, dest)
        self.git("add", "-A")
        tree = self.git("write-tree")
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        commit = self.git(*args)
        self.git("update-ref", "refs/retained/" + commit, commit)
        frozen = self.root / "frozen" / commit
        if not frozen.exists():
            frozen.mkdir(parents=True)
            import io
            import tarfile

            archive = subprocess.check_output(
                ["git", "archive", commit], cwd=self.repo
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(frozen)
            for p in frozen.rglob("*"):
                p.chmod(0o555 if p.is_dir() else 0o444)
            frozen.chmod(0o555)
        return commit, frozen
