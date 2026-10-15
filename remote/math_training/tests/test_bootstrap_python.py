import os
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def choose(tmp_path, env):
    command=f'set -e; source "{ROOT}/bootstrap_python.sh"; select_bootstrap_python "{tmp_path}" "$(( $(date +%s) + 600 ))"; printf "CHOICE:%s:%s:%s\n" "$TASK_PYTHON" "$TASK_MANAGED" "$TASK_UV"'
    return subprocess.run(['bash','-c',command],env={**os.environ,**env},text=True,capture_output=True)


def test_detects_explicit_supported_python(tmp_path):
    fake=tmp_path/'python3.11'
    fake.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then echo "Python 3.11.9"; fi\nexit 0\n')
    fake.chmod(0o755)
    result=choose(tmp_path,{'GENIAC_BOOTSTRAP_PYTHON':str(fake)})
    assert result.returncode==0 and f'CHOICE:{fake}:0:' in result.stdout


def test_reports_explicit_unsupported_python(tmp_path):
    fake=tmp_path/'python3.14'
    fake.write_text('#!/bin/sh\nexit 1\n');fake.chmod(0o755)
    result=choose(tmp_path,{'GENIAC_BOOTSTRAP_PYTHON':str(fake)})
    assert result.returncode==2 and 'not Python 3.10' in result.stderr


def test_falls_back_to_local_uv_without_download(tmp_path):
    fake=tmp_path/'uv'
    fake.write_text('#!/bin/sh\nexit 0\n');fake.chmod(0o755)
    # The provided Python 3.14 script forces the search path to reject all system Python.
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    python=bin_dir/'python3'
    python.write_text('#!/bin/sh\nexit 1\n');python.chmod(0o755)
    result=choose(tmp_path,{'GENIAC_BOOTSTRAP_PYTHON':'', 'GENIAC_UV':str(fake),
                            'PATH':f'{bin_dir}:/usr/bin:/bin'})
    assert result.returncode==0 and f'CHOICE::1:{fake}' in result.stdout
