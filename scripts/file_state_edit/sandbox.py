"""Private Blender worker: executable + standard libraries + case inputs only."""

import subprocess
from pathlib import Path


def isolated(command, cwd, *, readonly=(), timeout=480, log):
    args = [
        'bwrap',
        '--unshare-all',
        '--new-session',
        '--die-with-parent',
        '--cap-drop',
        'ALL',
        '--clearenv',
        '--ro-bind',
        '/usr',
        '/usr',
        '--symlink',
        'usr/bin',
        '/bin',
        '--symlink',
        'usr/lib',
        '/lib',
        '--symlink',
        'usr/lib64',
        '/lib64',
        '--dev',
        '/dev',
        '--proc',
        '/proc',
        '--tmpfs',
        '/tmp',
        '--dir',
        '/home',
        '--setenv',
        'HOME',
        '/tmp',
        '--setenv',
        'PATH',
        '/usr/bin:/bin',
        '--setenv',
        'OMP_NUM_THREADS',
        '1',
        '--setenv',
        'OPENBLAS_NUM_THREADS',
        '1',
    ]
    for p in [
        '/etc/ld.so.cache',
        '/etc/fonts',
        '/usr/share/fonts',
        '/sys/devices/system/cpu',
    ]:
        if Path(p).exists():
            args += ['--ro-bind', p, p]
    args += ['--bind', str(Path(cwd).resolve()), '/work', '--chdir', '/work']
    for src, dst in readonly:
        args += ['--ro-bind', str(Path(src).resolve()), dst]
    with Path(log).open('wb') as stream:
        return subprocess.run(
            args + list(command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env={'PATH': '/usr/bin:/bin'},
        ).returncode


PROFILE = 'permissions.study.filesystem={":minimal"="read",":workspace_roots"="write"}'


def probe(directory):
    import tempfile

    with tempfile.TemporaryDirectory() as w:
        test = 'test ! -r /data2/dcs/3DCodeEdit/data/tasks.jsonl && test ! -r /data2/dcs/EmbodiedGen/README.md && test ! -r "$HOME/.codex/config.toml" && touch private.tmp'
        if isolated(
            ['/bin/sh', '-ec', test],
            w,
            log=directory / 'blender-isolation.log',
        ):
            raise RuntimeError('worker isolation')
        subprocess.run(
            [
                'codex',
                'sandbox',
                '-P',
                'study',
                '-c',
                PROFILE,
                '-C',
                w,
                '--',
                '/bin/sh',
                '-ec',
                test,
            ],
            check=True,
        )
