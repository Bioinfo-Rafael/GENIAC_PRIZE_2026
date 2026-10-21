"""Thin adapters around the existing trainer, merger, official generator and grader."""
import argparse
import importlib.metadata
import os
import shutil
import time
from shared import *


def train(root, name):
    os.environ['UNSLOTH_RETURN_LOGITS']='0'
    import unsloth  # must precede transformers
    import transformers
    import torch
    import gpu
    exp=read(root/name/'experiment.json'); chain=Path(exp['chain']); recipe=read(chain/'recipe.json')
    previous=transformers.get_cosine_schedule_with_warmup
    switch=recipe['lr_policy']['switch_step']
    def scheduler(optimizer, warmup, total):
        if total!=recipe['lr_policy']['total_steps']: raise ValueError('Scheduler horizon changed')
        return torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,total,switch))
    transformers.get_cosine_schedule_with_warmup=scheduler
    try: gpu.train(chain,'stage2',exp['step'],recipe=recipe)
    finally: transformers.get_cosine_schedule_with_warmup=previous
    dest=chain/'stage2/training/rft'
    latest=read(dest/'latest.json')
    if latest['step']!=exp['step']: raise ValueError('Checkpoint does not match milestone')
    checkpoint=dest/latest['checkpoint']
    state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
    required={'optimizer','scheduler','scaler','torch_rng','cuda_rng','python_rng','step','signature'}
    if not required<=set(state) or state['step']!=exp['step']: raise ValueError('Incomplete resumable state')
    snapshot=root/name/'checkpoint'
    if not snapshot.exists():
        partial=snapshot.with_suffix('.partial')
        if partial.exists(): shutil.rmtree(partial)
        shutil.copytree(checkpoint,partial);partial.rename(snapshot)
    elif sha(snapshot/'state.pt')!=sha(checkpoint/'state.pt'):
        raise ValueError('Immutable milestone checkpoint differs')
    plan=read(chain/'stage2/schedule.json')
    immutable_json(root/name/'training_complete.json',dict(complete=True,step=exp['step'],
        signature=state['signature'],realized=exp['realized'],trained_this_chain=exp['trained_this_chain'],
        teacher_tokens_chain=plan['cumulative_tokens'][exp['step']-1],
        checkpoint_files=file_receipt(root/name,[snapshot]),recipe=recipe))


def merge(root,name):
    import gpu
    exp=read(root/name/'experiment.json')
    gpu.merge(Path(exp['chain']),'stage2',name)


def submit(root,name,phase):
    from submission import verify_assets, fetch, generate
    from worker import check_model
    sys.path.insert(0,str(HERE.parent/'rank64_12k'))
    import official_prompt
    verify_assets()
    exp=root/name; dest=exp/'submission';dest.mkdir(parents=True,exist_ok=True)
    if phase=='fetch':
        # Reuse the source's already downloaded official inputs only inside this
        # isolated submission phase; never expose them to training/evaluation.
        source=Path(read(root/'source.json')['source'])/'submission'
        for n in ('questions.json','question_identity.json'):
            if (source/n).exists() and not (dest/n).exists(): shutil.copyfile(source/n,dest/n)
        fetch(dest);return
    model=exp/'models'/name
    if not check_model(model): raise ValueError('Merged model changed')
    manifest=dict(candidate=name,config=CONFIG,experiment=read(exp/'experiment.json'),
        training=read(exp/'training_complete.json'),model_artifacts=read(model/'artifacts.json'),
        lineage=read(model/'lineage.json'),code=read(root/'identity.json'),
        source_sha256=sha(root/'source.json'),data_sha256=sha(root/'data/complete.json'),
        official_assets=read(LEGACY/'submission_assets/manifest.json'))
    immutable_json(dest/'manifest.json',manifest)
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt'];official_prompt._placeholder=official['prompt']['placeholder']
    from vllm import LLM
    original=LLM.chat
    timings=read(dest/'inference_times.json') if (dest/'inference_times.json').exists() else []
    def timed(self,*args,**kwargs):
        start=time.monotonic(); result=original(self,*args,**kwargs)
        timings.append(dict(seconds=time.monotonic()-start,questions=len(result)))
        write(dest/'inference_times.json',timings);return result
    LLM.chat=timed
    try:
        generate(dest,model_path=model,stage1_slug='gsm8k_2000_rank8',run_id=root.name+'_'+name,
            recipe_detail='Existing GSM2000 merged parent; PRM800K/LIMO response+EOS NLL; '+name+'; exact counts and fixed learning-rate history in provenance',
            build_messages=official_prompt._build_messages)
    finally: LLM.chat=original
    from submission_assets import check_submission as checker
    result=checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=result.errors,violations=result.violations,warnings=result.warnings))
    if result.errors or result.violations: raise ValueError('Official format check failed')
    saved=read(dest/'submission.json');ids=[p['question_id'] for p in saved['predictions']]
    if len(ids)!=232 or len(set(ids))!=232 or set(ids)!=checker.QUESTION_IDS: raise ValueError('Submission IDs/count differ')
    write(dest/'hashes.json',dict(files=file_receipt(dest,[dest/n for n in ('submission.json','format_check.json','manifest.json','generation_profile.json')]),
        code=read(root/'identity.json'),model_tokenizer_config=read(model/'artifacts.json')))


def evaluate_and_report(root,name):
    import evaluate
    exp=root/name
    evaluate.evaluate(exp,name)
    rows=read(exp/'data/eval.json'); outputs=read(exp/f'evaluations/{name}/generations/b0_eval.json')
    if set(outputs)!={r['id'] for r in rows} or any(len(v)!=1 for v in outputs.values()):
        raise ValueError('Fixed evaluation incomplete')
    contract=read(exp/f'evaluations/{name}/contract.json')
    source=Path(read(root/'source.json')['source'])
    candidates={b:source/f'evaluations/{b}' for b in ('300000','cumulative3000')}
    for prev in NAMES[:NAMES.index(name)]:
        candidates[prev]=root/prev/f'evaluations/{prev}'
    comparisons={}; changes={}; references={}
    for baseline,dest in candidates.items():
        old=read(dest/'contract.json')
        if {k:v for k,v in old.items() if k!='model_sha256'}!={k:v for k,v in contract.items() if k!='model_sha256'}:
            raise ValueError('Historical evaluation conditions differ: '+baseline)
        if baseline in ('300000','cumulative3000'):
            model=source/'models'/baseline
            if old['model_sha256']!=sha(model/'artifacts.json'): raise ValueError('Baseline model differs')
        before=read(dest/'generations/b0_eval.json')
        if set(before)!=set(outputs) or any(len(v)!=1 for v in before.values()): raise ValueError('Baseline IDs/count differ')
        if evaluate.metrics(before,rows)!=read(dest/'metrics.json'): raise ValueError('Cached metrics differ')
        comparisons[baseline]=comparison({k:v[0] for k,v in before.items()},{k:v[0] for k,v in outputs.items()})
        changes[baseline]=[dict(id=r['id'],dataset=r['dataset'],before=before[r['id']][0]['grade']['correct'],
            after=outputs[r['id']][0]['grade']['correct']) for r in rows]
        references[baseline]=dict(path=str(dest),generations_sha256=sha(dest/'generations/b0_eval.json'),
            metrics=read(dest/'metrics.json'),contract=old,reused=True)
    immutable_json(exp/'question_changes.json',changes)
    immutable_json(exp/'report.json',dict(name=name,metrics=read(exp/f'evaluations/{name}/metrics.json'),
        comparisons=comparisons,baselines=references,training=read(exp/'training_complete.json'),official_score=None,
        limitations='Public evaluation only. Paired bootstrap covers question sampling, not training variance; no multiplicity correction. No score-based recipe changes.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','merge','fetch','submit','evaluate'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--name',choices=NAMES);a=p.parse_args()
    if a.phase=='prepare':
        from scale_prepare import prepare
        prepare(a.root,a.source)
    elif a.phase=='train': train(a.root,a.name)
    elif a.phase=='merge': merge(a.root,a.name)
    elif a.phase in ('fetch','submit'): submit(a.root,a.name,a.phase)
    else: evaluate_and_report(a.root,a.name)
