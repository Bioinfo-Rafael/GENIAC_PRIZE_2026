"""CPU-only schedules, diagnostics and prospective selection for the C suite."""
import bisect
import random
from collections import Counter
from core import digest, comparison
from published_data import response_tokens, schedule

POLICY = {'accuracy_loss_max':.03,'gsm_lost_max':8,'repetition_ratio_max':.5,
          'non_stop_ratio_max':.5,'correct_stop_gain_min':.05,'unknown_gain_max':.03}


def make_plan(rows, groups, seed=42, milestones=None):
    if not rows or not groups or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty schedule or duplicate question IDs')
    flat = [i for g in groups for i in g]
    if any(not g or len(g)>16 for g in groups) or any(type(i)!=int or i<0 or i>=len(rows) for i in flat):
        raise ValueError('Invalid update group')
    counts = Counter(flat)
    if max(counts.values())>3:
        raise ValueError('Exposure cap exceeded')
    by_source, unique = Counter(), Counter()
    for i,n in counts.items():
        by_source[rows[i]['dataset']] += n*response_tokens(rows[i])
        unique[rows[i]['dataset']] += 1
    cumulative, total = [],0
    for group in groups:
        total += sum(response_tokens(rows[i]) for i in group)
        cumulative.append(total)
    if milestones is None:
        milestones = {f'tokens_{t}':bisect.bisect_left(cumulative,t)+1
                      for t in (50_000,100_000,250_000,500_000,1_000_000) if t<=total}
        milestones['full'] = len(groups)
    if not milestones or any(type(s)!=int or not 1<=s<=len(groups) for s in milestones.values()):
        raise ValueError('Invalid milestone')
    return {'kind':'c_plan_v1','pairs_digest':digest(rows),'groups':groups,'seed':seed,
            'milestone_steps':milestones,'cumulative_tokens':cumulative,'actual_tokens':total,
            'tokens_by_source':dict(by_source),'unique_by_source':dict(unique),
            'exposures':{str(n):count for n,count in Counter(counts.values()).items()},'examples':len(flat)}


def validate_plan(rows, plan):
    expected = make_plan(rows,plan['groups'],plan['seed'],plan['milestone_steps'])
    # JSON object keys are strings after persistence.
    if digest(plan)!=digest(expected):
        raise ValueError('C training schedule or tokenized dataset changed')


def budget_plan(rows,target,seed=42):
    return make_plan(rows,schedule(rows,target,seed,mixed=True)['groups'],seed)


def reshuffle_plan(rows,plan,seed):
    """Same selected multiset and update sizes; change only example order."""
    flat = [i for g in plan['groups'] for i in g]
    random.Random(seed).shuffle(flat)
    groups,offset = [],0
    for group in plan['groups']:
        groups.append(flat[offset:offset+len(group)])
        offset += len(group)
    return make_plan(rows,groups,seed,plan['milestone_steps'])


def category(base, result):
    if result['n']!=base['n']:
        raise ValueError('Selection requires the same sample size')
    if (result['repetition_rate']>base['repetition_rate']*.5+1e-9 or
        result['non_stop_rate']>base['non_stop_rate']*.5+1e-9):
        return None
    if result['accuracy']>=base['accuracy']-1e-9:
        return 'accuracy_preserving'
    n = base['n']
    if (result['accuracy']>=base['accuracy']-.03-1e-9
        and result['by_dataset']['gsm8k']['correct']>=base['by_dataset']['gsm8k']['correct']-8
        and result['correct_and_stop']>=base['correct_and_stop']+.05*n-1e-9
        and result['unknown']<=base['unknown']+.03*n+1e-9):
        return 'termination_tradeoff'
    return None


def rank_key(base,item):
    name,result = item
    m = result['metrics']
    kind = category(base,m)
    if kind=='accuracy_preserving':
        return (0,-m['correct'],-m['correct_and_stop'],m['repeated'],m['mean_output_tokens'],name)
    return (1,-m['correct_and_stop'],-m['correct'],m['repeated'],m['mean_output_tokens'],name)


def select(base,trials):
    eligible = [(k,v) for k,v in trials.items() if v.get('status')=='complete'
                and v.get('selectable',True) and category(base,v['metrics'])]
    return min(eligible,key=lambda item:rank_key(base,item))[0] if eligible else None


def outcome_intervals(base,candidate):
    """Question bootstrap, not execution variance or multiplicity-adjusted inference."""
    result = {}
    for metric in ('correct','correct_and_stop','repeated','non_stop'):
        def transform(outputs):
            transformed = {}
            for key,values in outputs.items():
                v = values[0]
                if metric=='correct': value = v['grade']['correct']
                elif metric=='correct_and_stop': value = v['grade']['correct'] and v['finish_reason']=='stop'
                elif metric=='non_stop': value = v['finish_reason']!='stop'
                else:
                    words=v['text'].split()
                    value=any(n>=4 for n in Counter(tuple(words[i:i+10])
                              for i in range(max(0,len(words)-9))).values())
                transformed[key]={'grade':{'correct':bool(value)}}
            return transformed
        result[metric]=comparison(transform(base),transform(candidate))
    return result


def grade_diagnostics(outputs,rows):
    result = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        counts=Counter()
        for r in rows:
            if r['dataset']!=dataset: continue
            v=outputs[r['id']][0]
            flags=[k for k in ('extraction_failed','gold_unparseable','grader_error') if v['grade'][k]]
            counts['n']+=1
            counts.update(flags)
            counts['overlapping_unknown_flags']+=len(flags)>1
            counts['answer_method_'+v['grade'].get('answer_method','unrecorded')]+=1
            counts['both_boxed_and_hash_marker']+=('\\boxed' in v['text'] and '####' in v['text'])
        result[dataset]=dict(counts)
    return result
