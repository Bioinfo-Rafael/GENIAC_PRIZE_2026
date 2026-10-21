"""run04 public selection only. Never import submission datasets here."""
import argparse
from shared import *
from run04_recipe import fetch_allowed_records,build_recipe,question_hash


def prepare(root):
    from worker import tokenizer_for
    inputs=root/'inputs';validate_inputs(inputs)
    report=root/'data/report.json'
    if report.exists():
        r=read(report)
        for f,key in [('tokens.json','tokenized_sha256'),('order.json','order_sha256'),('dev.json','dev_sha256')]:
            if digest(read(root/'data'/f))!=r[key]:raise ValueError('Prepared data changed: '+f)
        if groups_for_12k(read(root/'data/order.json'))!=read(root/'data/groups.json'):raise ValueError('Groups changed')
        print('DATA_REUSED',r['examples'],flush=True);return
    reference=read(HERE/'run04_reference.json')
    cache=root/'data/public_records.json'
    if cache.exists():records=read(cache)
    else:
        records=fetch_allowed_records(reference['sources']);immutable_json(cache,records)
    immutable_json(root/'data/public_records_identity.json',{'sha256':sha(cache),'sources':reference['sources']})
    split=read(inputs/'split.json')
    if digest(split['split'])!=split['split_sha256']:raise ValueError('Split checksum mismatch')
    excluded=set(reference['heldout_hashes'])|{question_hash(r['problem']) for part in ('dev','final') for r in split['split'][part]}
    raw,tokens,stats=build_recipe(records,tokenizer_for(inputs/'stage1'),excluded,COUNTS,max_length=2048,seed=42)
    if stats['tokens']!=3235460 or stats['max_tokens']!=1974:
        raise ValueError('Tokenization differs from validated run04 selection')
    rows=[dict(id=r['problem_hash'],dataset=r['dataset'],input_ids=t['input_ids']) for r,t in zip(raw,tokens)]
    # Confirm the smaller run03/D set is nested in the larger recipe, not a different resampling.
    old={question_hash(r['problem']) for r in jsonl(inputs/'train.jsonl')}
    if not old.issubset({r['id'] for r in rows}):raise ValueError('12k selection does not contain the original 6k training set')
    order=order_indices(len(rows));groups=groups_for_12k(order)
    for file,value in [('tokens',rows),('order',order),('groups',groups),('dev',split['split']['dev'])]:
        immutable_json(root/f'data/{file}.json',value)
    immutable_json(report,dict(examples=len(rows),counts=COUNTS,tokens=stats['tokens'],
        maximum_length=stats['max_tokens'],supervised_tokens=sum(len(r['input_ids'])-1 for r in rows),
        tokenized_sha256=digest(rows),order_sha256=digest(order),dev_sha256=digest(split['split']['dev']),
        selection_stats=stats,heldout_hashes_sha256=digest(sorted(excluded)),run03_nested=True,
        sources=reference['sources'],recipe_sha256=sha(HERE/'run04_recipe.py')))
    print('DATA_READY examples=12000 updates=750 tokens='+str(stats['tokens']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);prepare(p.parse_args().root)
