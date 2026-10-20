"""Accuracy-first policy, deterministic matched budgets, and online diagnostics."""
import math
import random
from collections import Counter
from core import digest
from c_utils import make_plan
from published_data import response_tokens


POLICY = {'min_dev_delta': .02, 'confirmation_lower_bound': 0.,
          'primary': 'single_generation_accuracy',
          'repetition': 'reported only; no half-repetition gate'}


def select_accuracy(base, trials, min_delta=.02):
    eligible = [(name, r) for name, r in trials.items()
                if r.get('status') == 'complete'
                and r['metrics']['n'] == base['n']
                and r['metrics']['accuracy'] >= base['accuracy'] + min_delta - 1e-12]
    return (min(eligible, key=lambda x: (-x[1]['metrics']['correct'],
                                        x[1]['metrics']['unknown'], x[0]))[0]
            if eligible else None)


def matched_plan(rows, target, seed=42):
    """Whole examples, at most three passes; same token target for both sizes."""
    order = [i for epoch in range(3)
             for i in sorted(range(len(rows)), key=lambda i: digest([seed, epoch, rows[i]['id']]))]
    selected, total = [], 0
    for i in order:
        selected.append(i)
        total += response_tokens(rows[i])
        if total >= target:
            break
    if total < target:
        raise ValueError('Token budget exceeds three-pass capacity')
    groups = [selected[i:i+16] for i in range(0, len(selected), 16)]
    return make_plan(rows, groups, seed)


def balanced_subset(rows, count, seed=42):
    if count % 4:
        raise ValueError('Subset size must be divisible by four')
    pools = {s: sorted([r for r in rows if r['dataset'] == s],
                       key=lambda r: digest([seed, r['id']])) for s in ('prm800k', 'gsm8k')}
    if len(pools['prm800k']) < count*3//4 or len(pools['gsm8k']) < count//4:
        raise ValueError('Insufficient source examples for the requested 3:1 subset')
    return [r for i in range(count//4)
            for r in pools['prm800k'][3*i:3*i+3] + pools['gsm8k'][i:i+1]]


def coverage(outputs, rows, k=8):
    if set(outputs) != {r['id'] for r in rows} or any(len(v) != k for v in outputs.values()):
        raise ValueError('Incomplete coverage diagnostic')
    counts = [sum(bool(o['grade']['correct']) for o in outputs[r['id']]) for r in rows]
    return {'n': len(rows), 'k': k,
            'sampled_pass_at_1': sum(counts)/(len(rows)*k),
            'observed_pass_at_k': sum(c > 0 for c in counts)/len(rows),
            'all_wrong': counts.count(0), 'mixed': sum(0 < c < k for c in counts),
            'all_correct': counts.count(k),
            'interpretation': 'diagnostic only; not an oracle used at submission or for choosing training questions'}


def advantages(rewards):
    if len(rewards) < 2 or any(not math.isfinite(r) for r in rewards):
        raise ValueError('Invalid reward group')
    mean = sum(rewards)/len(rewards)
    std = (sum((r-mean)**2 for r in rewards)/len(rewards))**.5
    return [(r-mean)/(std+1e-4) for r in rewards]


def rollout_tokens(prefix, generated, eos, context):
    """Use sampled IDs exactly; never re-render a chat answer or invent EOS."""
    generated = list(generated)
    if eos in generated:
        generated = generated[:generated.index(eos)+1]
    if not prefix or not generated or len(prefix)+len(generated) > context:
        raise ValueError('Invalid rollout length')
    return {'input_ids': list(prefix)+generated,
            'labels': [-100]*len(prefix)+generated}


def online_groups(rows, rounds, prompts, seed):
    if not rows or rounds < 1 or prompts < 1 or prompts > len(rows):
        raise ValueError('Invalid online schedule')
    # No correctness-dependent resampling or hand-picked questions.
    result, epoch = [], 0
    while len(result) < rounds:
        indices = list(range(len(rows)))
        random.Random(seed+epoch).shuffle(indices)
        result.extend(indices[i:i+prompts] for i in range(0, len(indices), prompts))
        epoch += 1
    return result[:rounds]
