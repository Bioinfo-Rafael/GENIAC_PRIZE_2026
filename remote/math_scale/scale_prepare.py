"""Restore exact used order from both completed schedules; never fetch training data."""
from shared import *
from continuation import selected_indices, verify_source


def completed_order(root):
    rows, plan = read(root/'stage2/pairs.json'), read(root/'stage2/schedule.json')
    validate_plan(rows, plan)
    complete = read(root/'stage2/training/rft/complete.json')
    identity = read(root/'stage2/training_contract.json')
    signature = digest([digest(identity), sha(root/'stage2/pairs.json'), None, digest(plan)])
    if not complete.get('complete') or complete['signature'] != signature or complete['updates'] != len(plan['groups']) or complete['supervised_tokens'] != plan['actual_tokens']:
        raise ValueError('Source completion does not match frozen training contract')
    order = [i for group in plan['groups'] for i in group]
    if len(order) != len(set(order)): raise ValueError('Repeated source exposures')
    return rows, plan, [rows[i] for i in order]


def prepare(root, source):
    from worker import check_model
    if (root/'data/complete.json').exists():
        verify_files(root, read(root/'data/complete.json'))
        contract = read(root/'source.json')
        verify_files(Path('/'), contract['absolute_files'])
        return
    continuation = read(source/'continuation.json')
    verify_source(continuation)
    original = Path(continuation['source'])
    for run in (source, original):
        verify_files(run, read(run/'data/complete.json'))
    pool, old_plan, first = completed_order(original)
    added_rows, added_plan, second = completed_order(source)
    selected_indices(pool, old_plan, len(old_plan['groups']))
    used_rows = first+second
    used_ids = [r['id'] for r in used_rows]
    if len(first)!=1040 or len(used_rows)!=3000 or len(set(used_ids))!=3000:
        raise ValueError('Expected exactly 1040 + 1960 unique source examples')
    if read(source/'data/used_stage2_ids.json') != [r['id'] for r in first] or read(source/'data/additional_stage2_ids.json') != [r['id'] for r in second]:
        raise ValueError('Source IDs/order differ from realized schedules')
    lookup = {r['id']:i for i,r in enumerate(pool)}
    if len(lookup)!=len(pool) or any(r!=pool[lookup[r['id']]] for r in used_rows):
        raise ValueError('Tokenized rows changed between source runs')
    more, _, report = extend_plan(pool, [lookup[i] for i in used_ids], 10000)
    full = used_rows+more
    if len(full)<=6000: raise ValueError('Insufficient unique pool for distinct 6k and 10k stages')
    evaluation = read(original/'data/eval.json')
    if sha(original/'data/eval.json') != sha(source/'data/eval.json') or len(evaluation)!=400:
        raise ValueError('Fixed evaluation differs')
    if {r['id'] for r in full} & {r['id'] for r in evaluation}:
        raise ValueError('Training/evaluation overlap')
    tokcfg = read(original/'tokenizer/stage2/tokenizer_config.json')
    eos = tokcfg['eos_token']; eos = eos['content'] if isinstance(eos,dict) else eos
    eos_id = next(int(k) for k,v in tokcfg['added_tokens_decoder'].items() if v['content']==eos)
    for r in full:
        t=r['chosen_tokens']; labels=t['labels']; ids=t['input_ids']
        first_label=next(i for i,x in enumerate(labels) if x!=-100)
        if len(ids)>8192 or len(ids)!=len(labels) or first_label<1 or labels[first_label:]!=ids[first_label:] or labels[-1]!=eos_id:
            raise ValueError('Invalid complete response-only + EOS supervision')
    adapter=source/'stage2/training/rft/final'
    milestone=source/f"stage2/training/rft/milestones/step-{len(added_plan['groups']):05d}"
    for model in (adapter,milestone,original/'models/stage1',source/'models/cumulative3000'):
        if not check_model(model): raise ValueError('Missing/changed source model: '+str(model))
    if sha(adapter/'adapter_model.safetensors')!=sha(milestone/'adapter_model.safetensors'):
        raise ValueError('Source final/milestone weights differ')
    acfg=read(adapter/'adapter_config.json')
    if (acfg['r'],acfg['lora_alpha'])!=(8,16): raise ValueError('Source LoRA differs')
    lineage=read(source/'models/cumulative3000/lineage.json')
    parent=read(original/'models/stage1/lineage.json')
    if lineage['revision']!=REVISION or lineage['stage2_sha256']!=sha(adapter/'adapter_model.safetensors') or lineage['stage1_sha256']!=parent['stage1_sha256']:
        raise ValueError('Source lineage differs')
    required=[]
    for run in (source,original):
        required += [run/n for n in ('data/complete.json','stage2/pairs.json','stage2/schedule.json','stage2/training_contract.json','stage2/training/rft/complete.json')]
    required += [source/'continuation.json']
    for model in (adapter,milestone,original/'models/stage1',source/'models/cumulative3000'):
        required += [model/'artifacts.json']+[model/n for n in read(model/'artifacts.json')]
    required += [p for p in (original/'tokenizer').rglob('*') if p.is_file()]
    contract=dict(source=str(source),original=str(original),absolute_files={str(p.resolve()).lstrip('/'):sha(p) for p in required})
    immutable_json(root/'source.json',contract)
    immutable_json(root/'data/full_order.json',[r['id'] for r in full])
    immutable_json(root/'data/report.json',report)
    for rank, rows, bounds in [(8,more,[('R8_6k',3000),('R8_10k',len(more))]),
                               (32,full,[('warmup1040',1040),('history3000',3000),('R32_6k',6000),('R32_10k',len(full))])]:
        chain=root/f'chains/R{rank}'
        plan=count_plan(rows,bounds)
        immutable_json(chain/'stage2/pairs.json',rows)
        immutable_json(chain/'stage2/schedule.json',plan)
        immutable_json(chain/'identity.json',read(root/'identity.json'))
        for src,dst in [('models/stage1','models/stage1'),('stage1','stage1'),('tokenizer','tokenizer')]:
            link_readonly(original/src,chain/dst)
        recipe=dict(lora_rank=rank,lora_alpha=2*rank,learning_rate=5e-6 if rank==8 else 5e-5,
            lr_policy=dict(kind='cosine_5pct' if rank==8 else 'two_phase_cosine_5pct',switch_step=None if rank==8 else 65,
                optimizer='one fresh AdamW; no reset at 6000; restore optimizer/scheduler/scaler/RNG',total_steps=len(plan['groups'])))
        immutable_json(chain/'recipe.json',recipe)
        if rank==8:
            immutable_json(chain/'continuation.json',dict(initial_adapter=str(adapter),initial_adapter_sha256=sha(adapter/'adapter_model.safetensors')))
        for name,end in bounds:
            if name not in NAMES: continue
            exp=root/name
            link_readonly(chain/'models', exp/'models')
            link_readonly(original/'tokenizer',exp/'tokenizer')
            immutable_json(exp/'data/eval.json',evaluation)
            immutable_json(exp/'identity.json',read(root/'identity.json'))
            immutable_json(exp/'experiment.json',dict(name=name,rank=rank,alpha=rank*2,chain=str(chain),step=plan['milestone_steps'][name],
                realized=counts(full[:6000 if name.endswith('6k') else len(full)]),
                trained_this_chain=counts(rows[:end]),recipe=recipe,source=source.name,
                history_difference='R8 retains historical 1040/1960 separate cosine schedules and optimizer reset at 3000; R32 uses one optimizer, cosine reset at 1040 with low-LR tail to final. Not a rank-only causal comparison.'))
    paths=[root/'source.json',root/'data/full_order.json',root/'data/report.json']
    for rank in (8,32):
        paths += [root/f'chains/R{rank}'/n for n in ('stage2/pairs.json','stage2/schedule.json','recipe.json','identity.json')]
    paths += [root/n/f for n in NAMES for f in ('experiment.json','data/eval.json','identity.json')]
    paths += [root/'chains/R8/continuation.json']
    immutable_json(root/'data/complete.json',file_receipt(root,paths))
    print('DATA_READY',dict(questions=len(full),unused=report['unused_pool_by_source'],counts=counts(full),shortfall=report['shortfall']),flush=True)
