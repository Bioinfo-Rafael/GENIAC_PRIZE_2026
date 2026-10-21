"""Frozen count schedules for the four-condition curriculum scaling suite."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
CURRICULUM = HERE.parent / 'math_curriculum'
sys.path.insert(0, str(CURRICULUM))
from common import *
HERE = Path(__file__).resolve().parent
from continuation import counts, extend_plan, link_readonly
from c_utils import validate_plan

NAMES = ('R8_6k', 'R8_10k', 'R32_6k', 'R32_10k')


def count_plan(rows, boundaries):
    groups, milestones, start = [], {}, 0
    for name, end in boundaries:
        if not start < end <= len(rows): raise ValueError('Invalid count boundary')
        groups.extend(list(range(i, min(i+16, end))) for i in range(start, end, 16))
        milestones[name] = len(groups)
        start = end
    if start != len(rows): raise ValueError('Unscheduled examples')
    return make_plan(rows, groups, SEED, milestones)


def lr_factor(step, total, switch=None):
    """Same warmup/cosine rule as worker; rank32 lowers cap after update 65."""
    import math
    start, end, cap = (0, total, 1.) if switch is None else (
        (0, switch, 1.) if step < switch else (switch, total, .1))
    local = step-start
    warmup = max(1, round((end-start)*.05))
    if local < warmup: return cap*local/warmup
    return cap*max(0., .5*(1+math.cos(math.pi*(local-warmup)/max(1,end-start-warmup))))


def file_receipt(root, paths):
    files = {}
    for path in paths:
        if not path.exists(): raise ValueError('Missing artifact: '+str(path))
        for p in sorted(path.rglob('*')) if path.is_dir() else [path]:
            if p.is_file(): files[str(p.relative_to(root))] = sha(p)
    return files


def verify_files(root, files):
    for name, value in files.items():
        if not (root/name).is_file() or sha(root/name) != value:
            raise ValueError('Artifact changed: '+name)


def jsonl(path):
    """New suites have no timing/log files before their first successful update."""
    import json
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()] if Path(path).exists() else []
