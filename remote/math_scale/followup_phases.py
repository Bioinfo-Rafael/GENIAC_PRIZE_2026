"""Shared GPU/official submission phases, plus bounded-storage followup preparation."""
import argparse
import shutil
from shared import *
from followup_plan import schedule_for
import phases as original_phases


def prepare(root,predecessor):
    frozen=read(root/'plan.json')
    if (root/'data/complete.json').exists():
        verify_files(root,read(root/'data/complete.json'))
        verify_files(Path('/'),read(root/'source.json')['absolute_files'])
        return
    previous=read(predecessor/'suite_report.json')
    if previous['status']!='complete':raise ValueError('Preceding suite must complete successfully')
    if previous.get('experiments') and any(previous['experiments'][n]['status']!='complete' for n in NAMES):
        raise ValueError('Every original experiment must finish before followup')
    for receipt in (predecessor/'receipts').glob('*.json'):
        verify_files(predecessor,read(receipt)['files'])
    verify_files(predecessor,read(predecessor/'data/complete.json'))
    source_contract=read(predecessor/'source.json')
    verify_files(Path('/'),source_contract['absolute_files'])
    source=Path(source_contract['source']);parent=Path(source_contract['original'])
    full=read(predecessor/'chains/R32/stage2/pairs.json')
    if [r['id'] for r in full]!=read(predecessor/'data/full_order.json') or len({r['id'] for r in full})!=len(full):
        raise ValueError('Frozen training order changed')
    evaluation=read(source/'data/eval.json')
    from collections import Counter
    if Counter(r['dataset'] for r in evaluation)!={'prm800k':200,'limo':200}:raise ValueError('Expected fixed PRM200/LIMO200')
    if {r['id'] for r in full}&{r['id'] for r in evaluation}:raise ValueError('Evaluation overlap')
    immutable_json(root/'source.json',source_contract)
    immutable_json(root/'data/full_order.json',[r['id'] for r in full])
    immutable_json(root/'data/report.json',read(predecessor/'data/report.json'))
    paths=[root/'source.json',root/'data/full_order.json',root/'data/report.json',root/'plan.json']
    for chain_cfg in frozen['chains']:
        chain=root/'chains'/chain_cfg['key']
        rows,plan=schedule_for(full,chain_cfg)
        immutable_json(chain/'stage2/pairs.json',rows)
        immutable_json(chain/'stage2/schedule.json',plan)
        immutable_json(chain/'identity.json',read(root/'identity.json'))
        recipe=dict(lora_rank=chain_cfg['rank'],lora_alpha=chain_cfg['alpha'],learning_rate=chain_cfg['learning_rate'],
            lr_policy=dict(kind='cosine_5pct' if chain_cfg['switch_step'] is None else 'two_phase_cosine_5pct',
                switch_step=chain_cfg['switch_step'],total_steps=len(plan['groups']),
                optimizer='fresh AdamW at branch origin; restore optimizer/scheduler/scaler/RNG across every submission boundary'))
        immutable_json(chain/'recipe.json',recipe)
        for name in ('models/stage1','tokenizer','stage1'):
            link_readonly(parent/name,chain/name)
        if chain_cfg['start_questions']:
            adapter=(source if chain_cfg['start_questions']==3000 else parent)/'stage2/training/rft/final'
            immutable_json(chain/'continuation.json',dict(initial_adapter=str(adapter),initial_adapter_sha256=sha(adapter/'adapter_model.safetensors')))
            paths.append(chain/'continuation.json')
        paths += [chain/n for n in ('stage2/pairs.json','stage2/schedule.json','recipe.json','identity.json')]
        for entry in chain_cfg['experiments']:
            name=entry['name'];exp=root/name
            actual=min(entry['target'],len(full));end=actual-chain_cfg['start_questions']
            if name not in plan['milestone_steps']:raise ValueError('Not enough distinct questions for checkpoint '+name)
            link_readonly(chain/'models',exp/'models');link_readonly(parent/'tokenizer',exp/'tokenizer')
            immutable_json(exp/'data/eval.json',evaluation)
            immutable_json(exp/'identity.json',read(root/'identity.json'))
            immutable_json(exp/'experiment.json',dict(name=name,rank=chain_cfg['rank'],alpha=chain_cfg['alpha'],
                chain=str(chain),step=plan['milestone_steps'][name],realized=counts(full[:actual]),
                trained_this_chain=counts(rows[:end]),requested_questions=entry['target'],shortfall=entry['target']-actual,
                source=source.name,recipe=recipe,history_difference=frozen['history_difference']))
            paths += [exp/n for n in ('experiment.json','data/eval.json','identity.json')]
    immutable_json(root/'data/complete.json',file_receipt(root,paths))
    print('FOLLOWUP_DATA_READY',dict(questions=len(full),planned_submissions=len(frozen['execution_order'])),flush=True)


def merge(root,name):
    original_phases.merge(root,name)
    model=root/name/'models'/name
    archive=root/name/'model_manifest'
    for n in ('artifacts.json','lineage.json'):
        immutable_json(archive/n,read(model/n))
    immutable_json(archive/'rebuild.json',dict(parent=str((model.parent/'stage1').resolve()),
        chain=read(root/name/'experiment.json')['chain'],milestone=name,
        checkpoint=str(root/name/'checkpoint'),temporary_merged_path=str(model.resolve()),
        command='python followup_phases.py merge --root <followup root> --name '+name))


def evaluate(root,name):
    import evaluate as evaluator
    exp=root/name
    evaluator.evaluate(exp,name)
    rows=read(exp/'data/eval.json')
    outputs=read(exp/f'evaluations/{name}/generations/b0_eval.json')
    contract=read(exp/f'evaluations/{name}/contract.json')
    source=Path(read(root/'source.json')['source'])
    predecessor=Path(read(root/'request.json')['predecessor'])
    refs={'300000':source/'evaluations/300000','cumulative3000':source/'evaluations/cumulative3000',
          'R8_6k':predecessor/'R8_6k/evaluations/R8_6k'}
    comparisons={};changes={};baselines={}
    if set(outputs)!={r['id'] for r in rows} or any(len(v)!=1 for v in outputs.values()):raise ValueError('Incomplete fixed evaluation')
    for baseline,path in refs.items():
        old=read(path/'contract.json')
        if {k:v for k,v in old.items() if k!='model_sha256'}!={k:v for k,v in contract.items() if k!='model_sha256'}:
            raise ValueError('Baseline evaluation conditions differ')
        model=(source/'models'/baseline if baseline in ('300000','cumulative3000') else predecessor/'R8_6k/models/R8_6k')
        if old['model_sha256']!=sha(model/'artifacts.json'):raise ValueError('Baseline model differs')
        before=read(path/'generations/b0_eval.json')
        if set(before)!=set(outputs) or any(len(v)!=1 for v in before.values()):raise ValueError('Baseline IDs differ')
        if evaluator.metrics(before,rows)!=read(path/'metrics.json'):raise ValueError('Cached baseline metrics differ')
        comparisons[baseline]=comparison({k:v[0] for k,v in before.items()},{k:v[0] for k,v in outputs.items()})
        changes[baseline]=[dict(id=r['id'],dataset=r['dataset'],before=before[r['id']][0]['grade']['correct'],
            after=outputs[r['id']][0]['grade']['correct']) for r in rows]
        baselines[baseline]=dict(metrics=read(path/'metrics.json'),contract=old,generations_sha256=sha(path/'generations/b0_eval.json'),reused=True)
    immutable_json(exp/'question_changes.json',changes)
    immutable_json(exp/'report.json',dict(name=name,metrics=read(exp/f'evaluations/{name}/metrics.json'),
        comparisons=comparisons,baselines=baselines,training=read(exp/'training_complete.json'),official_score=None,
        limitations='Exploratory fixed-public-set comparisons, not independent confirmatory tests. Bootstrap covers question sampling, not run variance; unadjusted multiplicity. No score-driven changes.'))


def cleanup(root,name):
    """Delete ONLY this new suite's reconstructible temporary merged copy."""
    exp=root/name;model=exp/'models'/name
    for phase in ('train','submit','evaluate'):
        verify_files(root,read(root/'receipts'/f'{name}_{phase}.json')['files'])
    manifest=exp/'model_manifest/artifacts.json'
    if model.exists():
        expected=(root/'chains'/Path(read(exp/'experiment.json')['chain']).name/'models'/name)
        if model.is_symlink() or model.resolve()!=expected.resolve() or root.resolve() not in model.resolve().parents:
            raise ValueError('Cleanup path is outside the new followup workspace')
        verify_files(model,read(manifest))
        # Parent, all adapters/full states and original four models are untouched.
        shutil.rmtree(model)
    immutable_json(exp/'merged_released.json',dict(temporary_merged_weights_released=True,
        manifest_sha256=sha(manifest),checkpoint=str(exp/'checkpoint'),submission=str(exp/'submission/submission.json'),
        reconstructible=True))
    print('TEMPORARY_MERGE_RELEASED',name,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','merge','fetch','submit','evaluate','cleanup'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--name');a=p.parse_args()
    if a.phase!='prepare' and a.name not in read(a.root/'plan.json')['execution_order']:raise ValueError('Unknown frozen experiment')
    if a.phase=='prepare':prepare(a.root,a.source)
    elif a.phase=='train':original_phases.train(a.root,a.name)
    elif a.phase=='merge':merge(a.root,a.name)
    elif a.phase in ('fetch','submit'):original_phases.submit(a.root,a.name,a.phase)
    elif a.phase=='evaluate':evaluate(a.root,a.name)
    else:cleanup(a.root,a.name)
