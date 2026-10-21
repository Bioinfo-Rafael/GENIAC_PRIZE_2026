"""Read-only CPU audit of public dev, saved teacher tokens and update logs.

Never imports submission.py, fetches datasets, loads weights, trains or generates.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'math_training'))
from core import comparison, digest, read, sha, write
from suite_utils import immutable_json

CATEGORIES=('problem','explanation','final_box','after_final_box','eos','formatting','boundary')
STEPS=(125,250,375,500,750)


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def quantiles(values):
    values=sorted(values)
    if not values:return dict(n=0)
    return dict(n=len(values),mean=statistics.mean(values),min=values[0],max=values[-1],
        median=statistics.median(values),p90=values[math.ceil(.9*len(values))-1],
        p99=values[math.ceil(.99*len(values))-1])


def repeat(text):
    w=text.split()
    return any(v>=4 for v in Counter(tuple(w[i:i+10]) for i in range(len(w)-9)).values())


def single_outputs(path,ids):
    values=read(path)
    if set(values)!=set(ids) or any(len(v)!=1 for v in values.values()):
        raise ValueError('Expected exactly one saved output for every public dev ID')
    return {k:v[0] for k,v in values.items()}


def transitions(rows,before,after):
    records=[];groups=defaultdict(list)
    for row in rows:
        k=row['id'];a,b=before[k],after[k]
        ac,bc=bool(a['grade']['correct']),bool(b['grade']['correct'])
        change=('right_to_right' if bc else 'right_to_wrong') if ac else ('wrong_to_right' if bc else 'wrong_to_wrong')
        metadata={key:row[key] for key in ('type','subject','level','category')
                  if key in row and isinstance(row[key],(str,int,float))}
        r=dict(id=k,dataset=row['dataset'],metadata=json.dumps(metadata,ensure_ascii=False),change=change,
            D_correct=ac,R12_correct=bc,D_tokens=len(a['token_ids']),R12_tokens=len(b['token_ids']),
            D_stop=a['finish_reason']=='stop',R12_stop=b['finish_reason']=='stop',
            D_repeated=repeat(a['text']),R12_repeated=repeat(b['text']),
            D_unknown=any(a['grade'].get(x,False) for x in ('extraction_failed','grader_error','gold_unparseable')),
            R12_unknown=any(b['grade'].get(x,False) for x in ('extraction_failed','grader_error','gold_unparseable')))
        records.append(r)
        groups['all'].append(k);groups['dataset:'+row['dataset']].append(k)
        for key,value in metadata.items():groups[f'{key}:{value}'].append(k)
        length=len(row['problem'])
        groups['problem_characters:'+('<=250' if length<=250 else '251-750' if length<=750 else '>750')].append(k)
    summary={}
    for label,ids in groups.items():
        subset=[r for r in records if r['id'] in set(ids)]
        summary[label]=dict(comparison=comparison({k:before[k] for k in ids},{k:after[k] for k in ids}),
            transitions=dict(Counter(r['change'] for r in subset)),
            by_transition={change:dict(n=len(rs),D_tokens=statistics.mean(r['D_tokens'] for r in rs),
                R12_tokens=statistics.mean(r['R12_tokens'] for r in rs),
                D_repeated=sum(r['D_repeated'] for r in rs),R12_repeated=sum(r['R12_repeated'] for r in rs),
                D_non_stop=sum(not r['D_stop'] for r in rs),R12_non_stop=sum(not r['R12_stop'] for r in rs))
                for change in sorted({r['change'] for r in subset})
                if (rs:=[r for r in subset if r['change']==change])})
    return records,summary


def last_box_span(text,start,end):
    spans=[]
    for m in re.finditer(r'\\boxed\s*\{',text[start:end]):
        left=start+m.start();pos=start+m.end();depth=1
        while pos<end:
            if text[pos]=='{':depth+=1
            elif text[pos]=='}':depth-=1
            pos+=1
            if depth==0:spans.append((left,pos));break
    return spans[-1] if spans else None


def region_spans(text):
    user='<|im_start|>user\n';assistant='<|im_start|>assistant\n';end='<|im_end|>'
    if not text.startswith(user) or text.count(assistant)!=1:
        raise ValueError('Unexpected teacher chat layout')
    u0=len(user);u1=text.index(end,u0);a0=text.index(assistant,u1)+len(assistant)
    a1=text.index(end,a0)
    box=last_box_span(text,a0,a1)
    spans=[(u0,u1,'problem')]
    if box:spans.extend([(a0,box[0],'explanation'),(*box,'final_box'),(box[1],a1,'after_final_box')])
    else:spans.append((a0,a1,'explanation'))
    return spans,box


def split_tokens(tok,ids):
    text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ids:raise ValueError('Teacher tokens do not round-trip; refusing approximate token shares')
    spans,box=region_spans(text);counts=Counter()
    for token,(left,right) in zip(ids[1:],encoded['offset_mapping'][1:]):
        if token==tok.eos_token_id:category='eos'
        else:
            covering=[label for start,end,label in spans if start<=left and right<=end and right>left]
            crossing=any(max(start,left)<min(end,right) for start,end,_ in spans)
            category=covering[0] if covering else ('boundary' if crossing else 'formatting')
        counts[category]+=1
    if sum(counts.values())!=len(ids)-1:raise AssertionError('Target accounting mismatch')
    last_eos=next((i for i,x in enumerate(ids) if x==tok.eos_token_id),None)
    return counts,dict(has_final_box=box is not None,eos_count=ids.count(tok.eos_token_id),
        tokens_after_first_eos=0 if last_eos is None else len(ids)-last_eos-1)


def teacher_audit(root,name,tok,common_ids):
    rows=read(root/'data/tokens.json');groups=read(root/'data/groups.json')
    flattened=[i for group in groups for i in group]
    if sorted(flattened)!=list(range(len(rows))) or any(len(g)!=16 for g in groups):
        raise ValueError('Expected one epoch of fixed 16-example updates')
    per=[];stats=defaultdict(list);counts_by_group=defaultdict(Counter)
    for i,row in enumerate(rows):
        counts,flags=split_tokens(tok,row['input_ids']);subset='original_6000' if row['id'] in common_ids else 'added_6000'
        per.append(dict(id=row['id'],dataset=row['dataset'],subset=subset,tokens=len(row['input_ids']),
                        targets=len(row['input_ids'])-1,**{c:counts[c] for c in CATEGORIES},**flags))
        for key in ['all','dataset:'+row['dataset'],'subset:'+subset]:
            stats[key].append(per[-1]);counts_by_group[key].update(counts)
        if (i+1)%2000==0:print(f'TEACHER {name} {i+1}/{len(rows)}',flush=True)
    logs={int(p.stem):read(p) for p in (root/'conditions'/name/'updates').glob('*.json')}
    updates=[];weights=Counter()
    for step,indices in enumerate(groups,1):
        totals=Counter({c:sum(per[i][c] for i in indices) for c in CATEGORIES});n=sum(totals.values())
        for c in CATEGORIES:weights[c]+=totals[c]/n
        record=dict(step=step,targets=n,mean_sequence_tokens=statistics.mean(per[i]['tokens'] for i in indices),
                    max_sequence_tokens=max(per[i]['tokens'] for i in indices),
                    original_examples=sum(per[i]['subset']=='original_6000' for i in indices),
                    **{c+'_weight':totals[c]/n for c in CATEGORIES})
        log=logs.get(step,{})
        if log and (log['step']!=step or log['supervised_tokens']!=n):raise ValueError('Training log/token group mismatch')
        record.update({key:log.get(key) for key in ('loss','gradient_norm','learning_rate','seconds')})
        updates.append(record)
    summary={key:dict(length=quantiles([r['tokens'] for r in rs]),targets=sum(counts_by_group[key].values()),
        target_counts=dict(counts_by_group[key]),
        target_shares={c:counts_by_group[key][c]/sum(counts_by_group[key].values()) for c in CATEGORIES},
        examples_without_box=sum(not r['has_final_box'] for r in rs),
        eos_count_distribution=dict(Counter(r['eos_count'] for r in rs)),
        examples_with_tokens_after_eos=sum(bool(r['tokens_after_first_eos']) for r in rs)) for key,rs in stats.items()}
    summary['update_weight_mean']={c:weights[c]/len(groups) for c in CATEGORIES}
    windows=[]
    for start in range(0,len(groups),25):
        rs=updates[start:start+25];measured=[r for r in rs if r['loss'] is not None]
        windows.append(dict(start_step=start+1,end_step=start+len(rs),logged_updates=len(measured),
            mean_loss=statistics.mean(r['loss'] for r in measured) if measured else None,
            target_weighted_loss=sum(r['loss']*r['targets'] for r in measured)/sum(r['targets'] for r in measured) if measured else None,
            mean_sequence_tokens=statistics.mean(r['mean_sequence_tokens'] for r in rs),
            seconds=sum(r['seconds'] for r in measured) if measured else None))
    summary['logged_updates']=len(logs)
    summary['milestones']=[dict(step=n,examples=n*16,
        original_examples=sum(r['original_examples'] for r in updates[:n]),
        added_examples=n*16-sum(r['original_examples'] for r in updates[:n]),
        cumulative_targets=sum(r['targets'] for r in updates[:n]))
        for n in STEPS if n<=len(groups)]
    return summary,per,updates,windows


def checkpoint_inventory(root):
    folder=root/'conditions/R12'
    available=[]
    for p in sorted(folder.glob('checkpoint-*')):
        if not re.fullmatch(r'checkpoint-\d+',p.name):continue
        available.append(dict(step=int(p.name.split('-')[1]),path=str(p),
            adapter_present=(p/'adapter_model.safetensors').is_file(),manifest_present=(p/'artifacts.json').is_file()))
    steps={r['step'] for r in available if r['adapter_present'] and r['manifest_present']}
    return dict(available=available,requested=[dict(step=n,status=('reuse_saved_final_dev' if n==750 and (root/'evaluation/R12/generations/b0_eval.json').is_file() else
        'available_not_evaluated' if n in steps else 'missing_no_retraining')) for n in STEPS],
        retention='engine.save_checkpoint retains only the latest two checkpoint directories',
        limitation='Missing intermediate weights cannot be reconstructed from scalar loss logs. No training or inference was run.')


def audit(args):
    root=args.r12.resolve();source=Path(read(root/'source.json')['path']).resolve();out=args.output.resolve()
    if any(out==p or p in out.parents for p in (root,source)):raise ValueError('Use a separate diagnostic output directory')
    out.mkdir(parents=True,exist_ok=True)
    for model_root,name,steps in ((source,'D',375),(root,'R12',750)):
        done=read(model_root/f'conditions/{name}/complete.json')
        if not done.get('complete') or done.get('steps')!=steps:
            raise ValueError('Only completed D/R12 runs may be audited')
    paths=[root/'data/dev.json',root/'source.json',source/'conditions/D/complete.json',root/'conditions/R12/complete.json']
    for name in ('D','R12'):
        paths.extend(root/f'evaluation/{name}'/x for x in ('contract.json','generations/b0_eval.json'))
    for p,name in ((source,'D'),(root,'R12')):
        paths.extend(p/'data'/x for x in ('tokens.json','groups.json','report.json'))
        paths.extend(sorted((p/'conditions'/name/'updates').glob('*.json')))
    identity=dict(source_r12=str(root),source_D=str(source),files={str(p):sha(p) for p in paths},
        code_sha256=sha(Path(__file__)))
    immutable_json(out/'identity.json',identity)
    dev=read(root/'data/dev.json')
    if len(dev)!=400 or len({r['id'] for r in dev})!=400:raise ValueError('Expected fixed public dev400')
    contracts=[read(root/f'evaluation/{n}/contract.json') for n in ('D','R12')]
    for contract in contracts:
        if contract['dev_sha']!=digest(dev):raise ValueError('Saved evaluation does not match dev400')
    a,b=({k:v for k,v in c.items() if k!='model_sha'} for c in contracts)
    if a!=b:raise ValueError('D and R12 evaluation contracts differ; no automatic re-inference')
    outputs={n:single_outputs(root/f'evaluation/{n}/generations/b0_eval.json',[r['id'] for r in dev]) for n in ('D','R12')}
    records,changes=transitions(dev,outputs['D'],outputs['R12'])
    write(out/'dev_transitions.json',changes);csv_write(out/'dev_question_changes.csv',records)
    checkpoints=checkpoint_inventory(root);write(out/'checkpoints.json',checkpoints)
    from worker import tokenizer_for
    tok=tokenizer_for(source/'inputs/stage1')
    old={r['id']:r['input_ids'] for r in read(source/'data/tokens.json')}
    new={r['id']:r['input_ids'] for r in read(root/'data/tokens.json')}
    if len(old)!=6000 or len(new)!=12000 or not set(old).issubset(new):
        raise ValueError('Expected nested 6000/12000 public teacher examples')
    if any(old[k]!=new[k] for k in old):raise ValueError('Common teacher tokens changed')
    common=set(old)
    summaries={}
    for path,name in ((source,'D'),(root,'R12')):
        summaries[name],per,updates,windows=teacher_audit(path,name,tok,common)
        csv_write(out/f'{name}_teacher_tokens.csv',per);csv_write(out/f'{name}_updates.csv',updates);csv_write(out/f'{name}_loss_windows.csv',windows)
    write(out/'teacher_summary.json',summaries)
    result=dict(dev=changes['all'],checkpoints=checkpoints,teacher=summaries,
        limitations=['Token shares describe NLL target weights, not gradient norms or actual per-region loss.',
                    'Only aggregate online training loss was saved; historical region-specific NLL cannot be recovered.',
                    'last balanced boxed span is a final-answer proxy; other final answers are not identified.',
                    'Scalar loss averages compare different batches and different model states.',
                    'Public dev metadata categories only; missing subject labels are not inferred.',
                    'No official submission inputs or outputs used; no training, inference or dataset download.'])
    write(out/'report.json',result)
    lines=['# D/R12 saved-artifact diagnosis','',json.dumps(changes['all']['comparison'],ensure_ascii=False),'',
        '| Model | Category | Targets | Global token share | Mean per-update weight |','|---|---|---:|---:|---:|']
    for name,s in summaries.items():
        for c in CATEGORIES:
            lines.append(f"| {name} | {c} | {s['all']['target_counts'].get(c,0)} | {s['all']['target_shares'][c]:.4%} | {s['update_weight_mean'][c]:.4%} |")
    lines+=['','## Checkpoints','',*['- '+str(v['step'])+': '+v['status'] for v in checkpoints['requested']],
            '','## Limits','',*['- '+v for v in result['limitations']]]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('DEV',json.dumps(changes['all']['comparison']),flush=True)
    print('CHECKPOINTS',json.dumps(checkpoints['requested']),flush=True)
    import tarfile
    with tarfile.open(out.with_suffix('.tar.gz'),'w:gz') as archive:
        for file in sorted(out.iterdir()):
            if file.is_file() and file.suffix in ('.json','.csv','.md'):archive.add(file,arcname=file.name)
    print('COMPLETE',str(out),flush=True)
    print('PUBLIC_DIAGNOSTICS_ARCHIVE',str(out.with_suffix('.tar.gz')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--r12',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
