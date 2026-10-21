"""Fetch pinned run03 training cache/Stage1 only; never fetch competition inputs."""
import argparse
from pathlib import Path
from collections import Counter
from common import *


def fetch(destination, cache):
    import shutil
    import gdown
    for item in read(LEGACY/'input_manifest.json')['files']:
        if not (item['path'].startswith(('stage1/','stage2/')) or item['path'] in ('train.jsonl','split.json')):
            continue
        target = destination/item['path']
        if target.exists() and sha(target) == item['sha256']: continue
        if target.exists(): raise ValueError('Unexpected input already exists: '+str(target))
        target.parent.mkdir(parents=True,exist_ok=True)
        source = cache/item['path']
        if source.exists() and sha(source) == item['sha256']:
            shutil.copy2(source,target)
        else:
            tmp = target.with_suffix('.download')
            if not gdown.download(id=item['drive_id'],output=str(tmp),quiet=True) or sha(tmp) != item['sha256']:
                raise ValueError('Input download hash mismatch')
            tmp.rename(target)
    validate_inputs(destination)


def tokenize(root):
    from worker import tokenizer_for
    inputs = root/'inputs'
    validate_inputs(inputs,include_stage2=False)
    raw = jsonl(inputs/'train.jsonl')
    if Counter(r['dataset'] for r in raw) != {'prm800k':4500,'gsm8k':1500}:
        raise ValueError('Not the run03 training set')
    ids = [question_id(r['problem']) for r in raw]
    if ids != sorted(ids,key=lambda k:original_hash(['shuffle',42,k])) or len(set(ids)) != 6000:
        raise ValueError('Original cache order/identity changed')
    tok = tokenizer_for(inputs/'stage1')
    split = read(inputs/'split.json')
    if digest(split['split']) != split['split_sha256']: raise ValueError('Split checksum mismatch')
    heldout = {question_id(r['problem']) for part in ('dev','final') for r in split['split'][part]}
    if set(ids) & heldout: raise ValueError('Training/evaluation overlap')
    rows = []
    for r,key in zip(raw,ids):
        tokens = tok.apply_chat_template([{'role':'user','content':r['problem']},
            {'role':'assistant','content':full_solution(r['solution'],r['answer'])}],tokenize=True)
        if tokens[-1] != tok.eos_token_id: tokens.append(tok.eos_token_id)
        if len(tokens) > 2048: raise ValueError('No truncation or example filtering permitted')
        rows.append({'id':key,'dataset':r['dataset'],'input_ids':tokens})
    stats = {'examples':len(rows),'tokens':sum(len(r['input_ids']) for r in rows),
             'maximum_length':max(len(r['input_ids']) for r in rows),
             'supervised_tokens':sum(len(r['input_ids'])-1 for r in rows)}
    ref = read(HERE/'run03_reference.json')['observed_data']
    if stats['tokens'] != ref['tokens'] or stats['maximum_length'] != ref['maximum_length']:
        raise ValueError('Tokenization does not reproduce recorded run03 counts')
    order = order_indices(len(rows))
    immutable_json(root/'data/tokens.json',rows)
    immutable_json(root/'data/order.json',order)
    immutable_json(root/'data/groups.json',groups_for(order))
    immutable_json(root/'data/dev.json',split['split']['dev'])
    immutable_json(root/'data/report.json',dict(stats,tokenized_sha256=digest(rows),order_sha256=digest(order),
        loss='sum CE over every shifted nonpadding token / targets in the fixed 16-example update',
        first_label='ignored (no preceding token); each packed document boundary also ignored',
        order_provenance=read(HERE/'run03_reference.json')['order']))
    print('DATA',stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path);a=p.parse_args()
    fetch(a.root/'inputs',a.cache);tokenize(a.root)
