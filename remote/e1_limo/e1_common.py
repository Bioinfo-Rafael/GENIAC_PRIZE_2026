"""E1's CPU contracts; existing math helpers remain authoritative."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent/'math_training'
sys.path.insert(0, str(LEGACY))
from core import read, write, sha, digest, jsonl, question_id
from suite_utils import immutable_json
from published_data import response_tokens
from c_utils import make_plan

INDEPENDENT_SHA = '7eafe2c0c46678af1483ea724948ccd394914771eb8362d4990ef4e91cf87d70'
POLICY = dict(dev='highest correct count; ties prefer E1-200k',
              adoption='dev strictly above A AND independent paired bootstrap 95% lower bound > 0',
              independent='A and frozen dev winner only; no checkpoint reselection',
              bootstrap_seed=42, bootstrap_draws=2000)
WEIGHTS = {'prm800k': .6, 'gsm8k': .2, 'limo': .2}


def plan_for(rows, targets=(200_000, 600_000), batch=16):
    """Whole responses, token-weighted 60/20/20, at most three exposures."""
    from collections import Counter
    if set(r['dataset'] for r in rows) != set(WEIGHTS):
        raise ValueError('All three sources are required')
    if any(response_tokens(r) <= 0 for r in rows):
        raise ValueError('Empty teacher response')
    pools = {s: [i for epoch in range(3) for i in sorted(
        (i for i,r in enumerate(rows) if r['dataset']==s),
        key=lambda i: digest([42,epoch,rows[i]['id']]))] for s in WEIGHTS}
    offsets, used = Counter(), Counter()
    groups, group, milestones = [], [], {}
    for target in targets:
        while sum(used.values()) < target:
            source = min(WEIGHTS, key=lambda s:(used[s]/WEIGHTS[s],s))
            if offsets[source] == len(pools[source]):
                raise ValueError('Insufficient whole examples within three exposures: '+source)
            i = pools[source][offsets[source]]; offsets[source] += 1
            used[source] += response_tokens(rows[i]); group.append(i)
            if len(group)==batch:
                groups.append(group); group=[]
        if group: groups.append(group); group=[]
        milestones[f'E1-{target//1000}k'] = len(groups)
    return make_plan(rows, groups, 42, milestones)


def choose(trials):
    names = ('E1-200k','E1-600k')
    if any(trials.get(n,{}).get('status')!='complete' for n in names):
        raise ValueError('Both dev evaluations must finish before selection')
    return max(names, key=lambda n:trials[n]['metrics']['correct'])


def adopt(baseline, trial, independent):
    return (trial['metrics']['correct'] > baseline['correct'] and
            independent['comparison']['paired_bootstrap_95'][0] > 0)


def validate_independent(rows):
    from collections import Counter
    if digest(rows) != INDEPENDENT_SHA:
        raise ValueError('Use the already frozen independent400, not a new split')
    if len({r['id'] for r in rows}) != 400 or Counter(r['dataset'] for r in rows) != {'prm800k':300,'gsm8k':100}:
        raise ValueError('Independent split must be 300 PRM + 100 GSM')
