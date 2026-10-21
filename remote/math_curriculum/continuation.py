"""Count-limited continuation using the original immutable token pool and split."""
import argparse
import importlib.metadata
from collections import Counter
from common import *
from c_utils import validate_plan

NAME = 'cumulative3000'


def selected_indices(rows, plan, completed_steps):
    validate_plan(rows, plan)
    if completed_steps != plan['milestone_steps']['300000'] or completed_steps != len(plan['groups']):
        raise ValueError('Source must be the completed 300k schedule')
    indices = [i for group in plan['groups'][:completed_steps] for i in group]
    if len(indices) != len(set(indices)):
        raise ValueError('Source schedule repeated questions; cannot infer a unique cumulative count')
    return indices


def counts(rows):
    n = Counter(r['dataset'] for r in rows)
    tokens = Counter()
    for r in rows:
        tokens[r['dataset']] += response_tokens(r)
    total = sum(tokens.values())
    return dict(questions=len(rows), questions_by_source=dict(n), teacher_tokens=total,
                teacher_tokens_by_source=dict(tokens),
                teacher_token_ratio={s:tokens[s]/total if total else 0 for s in ('prm800k','limo')})


def extend_plan(rows, used, target=3000):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate pool IDs')
    if len(used) != len(set(used)) or any(i < 0 or i >= len(rows) for i in used):
        raise ValueError('Invalid used indices')
    if len(used) >= target:
        raise ValueError('Source already meets/exceeds target; no continuation needed')
    if any(r['dataset'] not in ('prm800k','limo') or response_tokens(r)<=0 for r in rows):
        raise ValueError('Expected nonempty PRM/LIMO responses only')
    excluded = set(used)
    pools = {s:sorted((i for i,r in enumerate(rows) if r['dataset']==s and i not in excluded),
                      key=lambda i:digest([SEED,rows[i]['id']])) for s in ('prm800k','limo')}
    offsets = Counter()
    tokens = Counter(counts([rows[i] for i in used])['teacher_tokens_by_source'])
    selected=[]
    for _ in range(target-len(used)):
        options=[]
        for source,pool in pools.items():
            if offsets[source] < len(pool):
                i=pool[offsets[source]]; size=response_tokens(rows[i])
                ratio=(tokens['limo']+(size if source=='limo' else 0))/(sum(tokens.values())+size)
                options.append((abs(ratio-.2),source,i,size))
        if not options: break
        _,source,i,size=min(options)
        selected.append(i); offsets[source]+=1; tokens[source]+=size
    added=[rows[i] for i in selected]
    if not added: raise ValueError('No unused eligible questions')
    groups=[list(range(i,min(i+16,len(added)))) for i in range(0,len(added),16)]
    plan=make_plan(added,groups,SEED,{NAME:len(groups)})
    report=dict(target_cumulative_questions=target, existing=counts([rows[i] for i in used]),
        additional=counts(added), cumulative=counts([rows[i] for i in used]+added),
        shortfall=target-len(used)-len(added), unused_pool_by_source={s:len(p) for s,p in pools.items()},
        remaining_pool_by_source={s:len(p)-offsets[s] for s,p in pools.items()},
        final_update_examples=len(groups[-1]),
        mixture_policy='Choose the next seeded source head closest to cumulative 80:20 teacher tokens; never repeat; use PRM when LIMO exhausted',
        limo_exhausted=offsets['limo']==len(pools['limo']))
    return added,plan,report


def link_readonly(source, dest):
    source=source.resolve()
    if not source.exists(): raise ValueError('Missing source: '+str(source))
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_symlink():
        if dest.resolve()!=source: raise ValueError('Source link changed: '+str(dest))
    elif dest.exists(): raise ValueError('Refuse to replace existing destination: '+str(dest))
    else: dest.symlink_to(source,target_is_directory=source.is_dir())


def verify_source(contract):
    source=Path(contract['source'])
    for name,h in contract['files'].items():
        if sha(source/name)!=h: raise ValueError('Continuation source changed: '+name)


def prepare(root):
    from worker import check_model
    request=read(root/'continuation_request.json'); source=Path(request['source'])
    if root.resolve()==source.resolve() or source in root.resolve().parents:
        raise ValueError('Continuation must use a separate run directory')
    if (root/'continuation.json').exists(): verify_source(read(root/'continuation.json'))
    if (root/'data/complete.json').exists():
        for n,h in read(root/'data/complete.json').items():
            if sha(root/n)!=h: raise ValueError('Prepared continuation changed: '+n)
        return
    files={}
    def remember(path):
        files[str(path.relative_to(source))]=sha(path)
    for name,h in read(source/'data/complete.json').items():
        path=source/name
        if sha(path)!=h: raise ValueError('Source data changed: '+name)
        remember(path)
    required=['stage2/pairs.json','stage2/schedule.json','data/eval.json']
    if any(n not in files for n in required): raise ValueError('Source data lacks a frozen pool/split')
    rows=read(source/'stage2/pairs.json');plan=read(source/'stage2/schedule.json')
    complete=read(source/'stage2/training/rft/complete.json')
    used=selected_indices(rows,plan,complete['updates'])
    identity=read(source/'stage2/training_contract.json')
    expected=digest([digest(identity),sha(source/'stage2/pairs.json'),None,digest(plan)])
    if not complete.get('complete') or complete['signature']!=expected or complete['supervised_tokens']!=plan['actual_tokens']:
        raise ValueError('Source training completion/signature does not match schedule')
    step=complete['updates']; adapter=source/f'stage2/training/rft/milestones/step-{step:05d}'
    for model in (adapter,source/'stage2/training/rft/final',source/'models/stage1',source/'models/300000'):
        if not check_model(model): raise ValueError('Missing complete source model: '+str(model))
        remember(model/'artifacts.json')
        for name in read(model/'artifacts.json'): remember(model/name)
    if sha(adapter/'adapter_model.safetensors')!=sha(source/'stage2/training/rft/final/adapter_model.safetensors'):
        raise ValueError('Final adapter and 300k milestone differ')
    acfg=read(adapter/'adapter_config.json')
    if acfg['r']!=8 or acfg['lora_alpha']!=16: raise ValueError('Expected rank8/alpha16')
    lineage=read(source/'models/300000/lineage.json')
    if lineage['stage2_sha256']!=sha(adapter/'adapter_model.safetensors') or lineage['revision']!=REVISION:
        raise ValueError('300k merged model lineage differs')
    parent_lineage=read(source/'models/stage1/lineage.json')
    if parent_lineage['revision']!=REVISION or parent_lineage['stage1_sha256']!=lineage['stage1_sha256']:
        raise ValueError('Merged GSM parent lineage differs')
    if lineage['stage1_sha256']!=sha(source/'stage1/training/rft/final/adapter_model.safetensors'):
        raise ValueError('Stage 1 lineage differs')
    evaluation=read(source/'data/eval.json');eval_ids={r['id'] for r in evaluation}
    if len(eval_ids)!=len(evaluation) or eval_ids.intersection(r['id'] for r in rows):
        raise ValueError('Frozen split overlaps/duplicates')
    if any(len(r['chosen_tokens']['input_ids'])>8192 for r in rows): raise ValueError('Source exceeds context')
    added,new_plan,report=extend_plan(rows,used,request['target'])
    for name in ['data/complete.json','stage2/training/rft/complete.json','stage2/training_contract.json',
                 'stage1/training_contract.json','stage1/training/rft/final/adapter_model.safetensors',
                 'models/300000/lineage.json','models/stage1/lineage.json']:
        remember(source/name)
    for path in (source/'tokenizer').rglob('*'):
        if path.is_file(): remember(path)
    contract=dict(source=str(source),files=files,initial_adapter=str(adapter),
        initial_adapter_sha256=sha(adapter/'adapter_model.safetensors'),target=request['target'],
        optimizer_policy='new AdamW and cosine scheduler at 5e-6; own checkpoints resume own optimizer')
    immutable_json(root/'continuation.json',contract)
    # These references are only read by training/merge/submission. Never run Stage 1 phases here.
    for src,dst in [('models/stage1','models/stage1'),('models/300000','models/300000'),
                    ('tokenizer','tokenizer'),('stage1','stage1')]:
        link_readonly(source/src,root/dst)
    immutable_json(root/'data/eval.json',evaluation)
    immutable_json(root/'data/used_stage2_ids.json',[rows[i]['id'] for i in used])
    immutable_json(root/'data/additional_stage2_ids.json',[r['id'] for r in added])
    immutable_json(root/'stage2/pairs.json',added)
    immutable_json(root/'stage2/schedule.json',new_plan)
    report.update(source_run=source.name,source_schedule_sha256=sha(source/'stage2/schedule.json'),
        evaluation_sha256=sha(source/'data/eval.json'),evaluation_counts=dict(Counter(r['dataset'] for r in evaluation)),
        training_status='planned; verify stage2/training/rft/complete.json for completion')
    immutable_json(root/'data/report.json',report)
    names=['continuation.json','data/eval.json','data/used_stage2_ids.json','data/additional_stage2_ids.json',
           'stage2/pairs.json','stage2/schedule.json','data/report.json']
    immutable_json(root/'data/complete.json',{n:sha(root/n) for n in names})
    print('CONTINUATION_READY',report,flush=True)


def evaluate_baseline(root):
    # Preserve evaluate.py byte-for-byte: its hash is part of historical evaluation contracts.
    import evaluate
    source=Path(read(root/'continuation.json')['source']); old=source/'evaluations/300000'
    dest=root/'evaluations/300000'
    expected=dict(rows_sha256=sha(root/'data/eval.json'),model_sha256=sha(root/'models/300000/artifacts.json'),
        config=CONFIG,grader_sha256=sha(LEGACY/'grading.py'),extraction_sha256=sha(HERE/'evaluate.py'),
        packages={p:importlib.metadata.version(p) for p in ('vllm','transformers','math-verify')})
    if not (dest/'contract.json').exists() and (old/'contract.json').exists() and read(old/'contract.json')==expected:
        generation=old/'generations/b0_eval.json'
        if generation.exists() and (old/'metrics.json').exists():
            outputs=read(generation)
            if any(len(v)!=1 for v in outputs.values()): raise ValueError('Baseline must have one answer per question')
            if read(old/'metrics.json')!=evaluate.metrics(outputs,read(root/'data/eval.json')):
                raise ValueError('Cached baseline metrics differ')
            for name in ('generations/b0_eval.json','metrics.json','eval_rows.json','contract.json',
                         'timings/b0_eval.json','generation_environment.json'):
                if (old/name).exists():
                    immutable_json(dest/name,read(old/name))
            write(dest/'reuse.json',dict(source=str(old),contract=expected,
                  generations_sha256=sha(generation),reused=True))
            print('REUSE_BASELINE_EVALUATION',old,flush=True)
    evaluate.evaluate(root,'300000')


def report(root):
    import evaluate
    rows=read(root/'data/eval.json'); names=('300000',NAME)
    done=read(root/'stage2/training/rft/complete.json');plan=read(root/'stage2/schedule.json')
    if not done.get('complete') or done['updates']!=len(plan['groups']) or done['supervised_tokens']!=plan['actual_tokens']:
        raise ValueError('Continuation has not completed its planned questions')
    outputs={n:read(root/f'evaluations/{n}/generations/b0_eval.json') for n in names}
    contracts={n:read(root/f'evaluations/{n}/contract.json') for n in names}
    if any({k:v for k,v in contracts[n].items() if k!='model_sha256'} !=
           {k:v for k,v in contracts['300000'].items() if k!='model_sha256'} for n in names):
        raise ValueError('Evaluation conditions differ')
    if any(len(v)!=1 for o in outputs.values() for v in o.values()): raise ValueError('Expected one answer')
    scores={n:evaluate.metrics(o,rows) for n,o in outputs.items()}
    changes=[dict(id=r['id'],dataset=r['dataset'],
        correct_300k=outputs['300000'][r['id']][0]['grade']['correct'],
        correct_cumulative3000=outputs[NAME][r['id']][0]['grade']['correct']) for r in rows]
    result=dict(metrics=scores,comparison=comparison(*[{k:v[0] for k,v in outputs[n].items()} for n in names]),
        submission_candidate=NAME,data=read(root/'data/report.json'),
        training_complete=done,realized_counts=read(root/'data/report.json')['cumulative'],
        rule='Compare accuracy; submit continued model as requested, irrespective of score. No official inputs used.')
    immutable_json(root/'question_changes.json',changes)
    immutable_json(root/'selection.json',dict(candidate=NAME,rule=result['rule']))
    write(root/'report.json',result)
    print('ACCURACY_COMPARISON',scores,flush=True)



def write_summary(root):
    result=read(root/'report.json');runtime=read(root/'runtime_summary.json')
    data=result['data'];train=runtime['training_summary']['stage2']
    text=['# Stage 2 cumulative-question continuation', '',
          f"Source: `{data['source_run']}`; target: 3000; completed cumulative: {result['realized_counts']['questions']}.", '',
          f"Existing {data['existing']['questions']} + additional {data['additional']['questions']} unique questions; shortfall {data['shortfall']}.",
          f"Cumulative source counts: {result['realized_counts']['questions_by_source']}.",
          f"Cumulative teacher tokens: {result['realized_counts']['teacher_tokens_by_source']}.",
          f"Cumulative teacher-token ratios: {result['realized_counts']['teacher_token_ratio']}.", '',
          '| Model | Correct / n | Accuracy | PRM correct / n | LIMO correct / n |',
          '| --- | ---: | ---: | ---: | ---: |']
    for name,m in result['metrics'].items():
        prm=m['by_dataset']['prm800k'];limo=m['by_dataset']['limo']
        text.append(f"| {name} | {m['correct']} / {m['n']} | {m['accuracy']:.4%} | {prm.get('correct',0)} / {prm['n']} | {limo.get('correct',0)} / {limo['n']} |")
    text += ['',f"Additional training update-attempt seconds: {train['seconds']:.2f}; allocated peak GiB: {train['peak_allocated_vram_bytes']/2**30:.2f}.",
             f"All recorded phase-attempt seconds: {runtime['total_phase_seconds']:.2f}; device peak GiB: {runtime['peak_device_vram_bytes']/2**30:.2f}.",
             f"Evaluation phase seconds: {runtime['evaluation_seconds']:.2f}; submission generation seconds: {runtime['submission_generation_seconds']:.2f}.",
             'Cached baseline timings, if reused, are historical and excluded from new phase inference time.', '',
             'See [report.json](report.json), [question_changes.json](question_changes.json), [runtime_summary.json](runtime_summary.json).',
             'Unedited output: [submission/submission.json](submission/submission.json); [format_check.json](submission/format_check.json); [hashes.json](submission/hashes.json).',
             'Submit the continued checkpoint as requested, irrespective of evaluation ranking. No Omnicampus upload.', '']
    (root/'README.md').write_text('\n'.join(text))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','baseline','report'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    {'prepare':prepare,'baseline':evaluate_baseline,'report':report}[a.phase](a.root)
