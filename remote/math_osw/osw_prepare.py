"""Reuse fixed source adapters/tokens and recover gold from pinned local sources."""
from collections import Counter
from helpers import *


def prepare(root, predecessor, maximum):
    from core import jsonl, question_id
    from accuracy_data import PRM
    from huggingface_hub import hf_hub_download
    from worker import check_model
    from scale_prepare import completed_order

    if (root/'data/complete.json').exists():
        verify_files(root, read(root/'data/complete.json'))
        verify_files(Path('/'), read(root/'source.json')['absolute_files'])
        return
    verify_files(predecessor, read(predecessor/'data/complete.json'))
    contract = read(predecessor/'source.json')
    verify_files(Path('/'), contract['absolute_files'])
    source, parent = Path(contract['source']), Path(contract['original'])
    full = read(predecessor/'chains/R32/stage2/pairs.json')
    evaluation = read(source/'data/eval.json')
    if Counter(r['dataset'] for r in evaluation) != {'prm800k':200, 'limo':200}:
        raise ValueError('Expected unchanged PRM200/LIMO200 evaluation')
    rows = validate_order(full, read(predecessor/'data/full_order.json'), evaluation, maximum)
    first = completed_order(parent)[2] + completed_order(source)[2]
    if first != full[:3000]:
        raise ValueError('3k ancestor differs from the frozen training prefix')
    adapter = source/'stage2/training/rft/final'
    if not check_model(adapter) or not check_model(parent/'models/stage1'):
        raise ValueError('Ancestor artifacts missing or modified')
    if (read(adapter/'adapter_config.json')['r'], read(adapter/'adapter_config.json')['lora_alpha']) != (8,16):
        raise ValueError('Expected rank8/alpha16 ancestor')

    # Never download training data during a one-hour run. These exact caches were
    # used to create the frozen curriculum. No official questions enter this path.
    prm = parent/'data/prm.jsonl'
    if sha(prm) != PRM['sha256']:
        raise ValueError('Pinned PRM cache changed')
    spec = read(HERE.parent/'math_training/published_sources.json')['limo']
    limo = Path(hf_hub_download(repo_id=spec['repo_id'], repo_type='dataset',
        revision=spec['revision'], filename=spec['filename'], local_files_only=True))
    if sha(limo) != spec['sha256']:
        raise ValueError('Pinned LIMO cache changed')
    needed = {(r['dataset'],r['id']) for r in rows}
    gold = {}
    for dataset, path, field in [('prm800k',prm,'problem'),('limo',limo,'question')]:
        for row in jsonl(path):
            key = (dataset, question_id(row[field]))
            if key in needed:
                record = dict(id=key[1], dataset=dataset, answer=str(row['answer']))
                if key[1] in gold and gold[key[1]] != record:
                    raise ValueError('Ambiguous source answer')
                gold[key[1]] = record
    if set(gold) != {r['id'] for r in rows}:
        raise ValueError('Missing source gold; never infer it from generated answers')
    chain = root/'chain'
    names = [(candidate_name(n),n) for n in range(16,maximum+1,16)]
    plan = count_plan(rows,names)
    recipe = dict(learning_rate=1e-6,lora_rank=8,lora_alpha=16,
        lr_policy=dict(kind='cosine_5pct',switch_step=None,total_steps=438))
    for name in ('models/stage1','stage1','tokenizer'):
        link_readonly(parent/name,chain/name)
    payloads = {
        'source.json':contract,
        'data/gold.json':gold,
        'data/ancestor_counts.json':counts(full[:3000]),
        'data/full_order.json':[r['id'] for r in full],
        'data/source_gold_hashes.json':{'prm':sha(prm),'limo':sha(limo)},
        'chain/stage2/pairs.json':rows,
        'chain/stage2/schedule.json':plan,
        'chain/recipe.json':recipe,
        'chain/identity.json':read(root/'identity.json'),
        'chain/continuation.json':dict(initial_adapter=str(adapter),
            initial_adapter_sha256=sha(adapter/'adapter_model.safetensors'))}
    for name,n in names:
        exp = root/name
        link_readonly(chain/'models',exp/'models')
        link_readonly(parent/'tokenizer',exp/'tokenizer')
        payloads.update({
            f'{name}/data/eval.json':evaluation,
            f'{name}/identity.json':read(root/'identity.json'),
            f'{name}/experiment.json':dict(name=name,rank=8,alpha=16,chain=str(chain),
                step=n//16,realized=counts(full[:3000+n]),trained_this_chain=counts(rows[:n]),
                requested_questions=3000+maximum,source=source.name,recipe=recipe,
                objective='OSW-weighted response+EOS token-mean NLL; two current-policy rollouts',
                history_difference='Small pilot prefix, original 438-batch cosine horizon; not a completed 10k run')})
    for path,value in payloads.items(): immutable_json(root/path,value)
    immutable_json(root/'data/complete.json',file_receipt(root,[root/p for p in payloads]))
    print('PREPARED',counts(rows),flush=True)
