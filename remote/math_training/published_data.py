"""Pinned public solutions; mechanical processing only. Never print dataset text."""
import argparse
import bisect
from collections import Counter
from pathlib import Path

from core import digest, jsonl, question_id, read, sha, write
from suite_utils import immutable_json

HERE = Path(__file__).resolve().parent


def canonical_tokens(tok, problem, solution, max_length):
    prefix = tok.apply_chat_template([{'role':'user', 'content':problem}],
                                    tokenize=True, add_generation_prompt=True)
    response = tok.encode(solution, add_special_tokens=False)
    # Reject embedded control tokens structurally; do not rewrite published content.
    if not response or set(response) & set(tok.all_special_ids):
        return None, 'embedded_special_or_empty'
    ids = prefix + response + [tok.eos_token_id]
    if len(ids) > max_length:
        return None, 'overlength'
    labels = [-100]*len(prefix) + response + [tok.eos_token_id]
    assert labels.count(tok.eos_token_id) == 1 and labels[-1] == tok.eos_token_id
    return {'input_ids':ids, 'labels':labels}, None


def response_tokens(row):
    return sum(t != -100 for t in row['chosen_tokens']['labels'][1:])


def schedule(rows, target, seed=42, batch=16, max_exposures=3, mixed=False):
    """Stop at a whole-example boundary; balance cumulative supervised tokens."""
    if not rows or target <= 0 or batch <= 0:
        raise ValueError('Empty data or invalid training budget')
    pools = {}
    for source in sorted({r['dataset'] for r in rows}):
        indices = [i for i,r in enumerate(rows) if r['dataset']==source]
        pools[source] = [i for epoch in range(max_exposures) for i in
                        sorted(indices, key=lambda i:digest([seed,epoch,rows[i]['id']]))]
    if mixed and set(pools) != {'gsm8k','limo'}:
        raise ValueError('Mixed schedule requires both published sources')
    used, offsets, order = Counter(), Counter(), []
    while sum(used.values()) < target:
        available = [s for s,indices in pools.items() if offsets[s] < len(indices)]
        if not available:
            raise ValueError('Requested tokens exceed the exposure cap')
        source = min(available, key=lambda s:(used[s],s))
        i = pools[source][offsets[source]]
        offsets[source] += 1
        order.append(i)
        used[source] += response_tokens(rows[i])
    groups = [order[i:i+batch] for i in range(0,len(order),batch)]
    cumulative, total = [], 0
    for group in groups:
        total += sum(response_tokens(rows[i]) for i in group)
        cumulative.append(total)
    milestones = {label:min(len(groups),bisect.bisect_left(cumulative,target*fraction)+1)
                  for label,fraction in [('quarter',.25),('half',.5),('full',1.)]}
    return {'groups':groups, 'target_tokens':target, 'actual_tokens':total,
            'tokens_by_source':dict(used), 'milestone_steps':milestones,
            'max_exposures':max_exposures, 'seed':seed, 'batch':batch,
            'examples':len(order), 'cumulative_tokens':cumulative}


def partition_limo(rows, excluded, seed=42):
    """Hash order, then fixed 5%/5%/90%, before any token-length filtering."""
    unique = {}
    duplicates = overlap = 0
    for row in rows:
        key = question_id(row['question'])
        if key in excluded:
            overlap += 1
        elif key in unique:
            duplicates += 1
        else:
            unique[key] = dict(row,id=key)
    ordered = sorted(unique.values(),key=lambda r:digest([seed,r['id']]))
    n = len(ordered)//20
    return {'dev':ordered[:n], 'test':ordered[n:2*n], 'train':ordered[2*n:]}, {
        'duplicates':duplicates, 'known_overlap':overlap, 'unique':len(ordered)}


def fetch_public(destination):
    from huggingface_hub import hf_hub_download
    manifest = read(HERE/'published_sources.json')
    paths = {}
    for name, spec in manifest.items():
        path = Path(hf_hub_download(repo_id=spec['repo_id'],repo_type='dataset',
                    revision=spec['revision'],filename=spec['filename']))
        if sha(path) != spec['sha256']:
            raise ValueError('Published source SHA mismatch: '+name)
        paths[name] = path
    immutable_json(destination/'sources.json',manifest)
    return paths


def prepare(context):
    import pyarrow.parquet as pq
    from worker import tokenizer_for
    ctx = read(context)
    root, inputs, cfg = Path(ctx['run']), Path(ctx['inputs']), ctx['config']
    paths = fetch_public(root)
    gsm = pq.read_table(paths['gsm8k']).to_pylist()
    limo = jsonl(paths['limo'])
    # Validate field types without logging field values or individual examples.
    for name, rows, fields in [('gsm8k',gsm,('question','answer')),
                               ('limo',limo,('question','solution','answer'))]:
        if not rows or any(not isinstance(r.get(f),str) for r in rows for f in fields):
            raise ValueError('Unexpected published dataset schema: '+name)
    split = read(inputs/'split.json')
    if digest(split['split']) != split['split_sha256']:
        raise ValueError('Evaluation split checksum mismatch')
    heldout = {question_id(r['problem']) for kind in ('dev','final') for r in split['split'][kind]}
    old_train = {question_id(r['problem']) for r in jsonl(inputs/'train.jsonl')}
    gsm_ids = {question_id(r['question']) for r in gsm}
    partitions, partition_stats = partition_limo(limo,heldout|old_train|gsm_ids,cfg['seed'])
    tok = tokenizer_for(inputs/'stage2')
    stats = {'control':'GSM8K only; PRM file mapping unresolved',
             'sources':{'gsm8k':len(gsm),'limo':len(limo)},'limo_partition':partition_stats,
             'limo_splits':{k:len(v) for k,v in partitions.items()},
             'lengths':{}, 'filters':{}, 'terminal_policy':'raw published response + one EOS',
             'known_overlap_scope':'run03 6000, existing dev/final800, entire GSM train; not base/Stage1 pretraining'}
    all_limo_counts = Counter()
    for row in limo:
        n = len(tok.apply_chat_template([{'role':'user','content':row['question']}],
                tokenize=True,add_generation_prompt=True)) + len(tok.encode(row['solution'],add_special_tokens=False)) + 1
        for limit in (2048,4096,8192,16384):
            all_limo_counts[str(limit)] += n <= limit
    stats['lengths']['limo_full_sequence_leq'] = dict(all_limo_counts)
    prepared = {}
    for name, rows, answer_field in [('gsm8k',gsm,'answer'),('limo',partitions['train'],'solution')]:
        accepted, seen, counts = [], set(), Counter()
        for row in rows:
            key = question_id(row['question'])
            if key in heldout or key in seen:
                counts['duplicate_or_evaluation_overlap'] += 1
                continue
            seen.add(key)
            tokens, reason = canonical_tokens(tok,row['question'],row[answer_field],cfg['context'])
            if reason:
                counts[reason] += 1
                continue
            accepted.append({'id':key,'dataset':name,'chosen_tokens':tokens})
        accepted.sort(key=lambda r:digest([cfg['seed'],r['id']]))
        prepared[name] = accepted
        stats['filters'][name] = dict(counts)
        stats[name] = {'eligible':len(accepted),'response_tokens':sum(map(response_tokens,accepted)),
                       'already_in_run03':sum(r['id'] in old_train for r in accepted)}
        immutable_json(root/f'{name}_pairs.json',accepted)
    # Auxiliary evaluation gold comes from the published answer field, never a generated correction.
    for kind in ('dev','test'):
        rows, too_long = [], 0
        for r in partitions[kind]:
            plen = len(tok.apply_chat_template([{'role':'user','content':r['question']}],
                        tokenize=True,add_generation_prompt=True))
            if plen + cfg['max_tokens'] > cfg['context']:
                too_long += 1
                continue
            rows.append({'id':r['id'],'problem':r['question'],'answer':r['answer'],'dataset':'limo'})
        immutable_json(root/f'limo_{kind}.json',rows)
        stats['limo_splits'][kind+'_prompt_overlength'] = too_long
    stats['mixed_available'] = len(prepared['limo']) >= 100
    stats['insufficient_limo_reason'] = None if stats['mixed_available'] else 'fewer than 100 whole eligible training examples'
    immutable_json(root/'data_report.json',stats)
    print('PUBLIC_DATA_AGGREGATES',stats,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context',required=True,type=Path)
    prepare(parser.parse_args().context)
