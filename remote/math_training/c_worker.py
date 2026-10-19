"""Isolated CPU preprocessing and adapter interpolation. Never print training text."""
import argparse
import shutil
from pathlib import Path
from core import read, write, jsonl, sha, question_id, digest, fingerprint, tokenize_answer
from published_data import fetch_public, canonical_tokens
from c_utils import make_plan
from suite_utils import immutable_json


def gsm_legacy_target(answer):
    import re
    from grading import last_boxed
    body,separator,value = answer.partition('####')
    if not separator or not body.strip() or not value.strip():
        raise ValueError('GSM structured final-answer field missing')
    body = re.sub(r'<<[^>]*>>','',body).strip()
    if last_boxed(body) is not None:
        return body
    return body+'\n\nTherefore, $\\boxed{'+(last_boxed(value) or value.strip())+'}$.'


def format_tokens(tok,problem,answer,form,limit):
    # Validate control tokens even for the legacy terminal variant.
    direct,reason = canonical_tokens(tok,problem,answer,limit)
    if reason: return None
    return tokenize_answer(tok,problem,answer,limit) if form.endswith('1') else direct


def prepare(ctx):
    import pyarrow.parquet as pq
    from worker import tokenizer_for
    root=Path(ctx['run']); source=Path(ctx['config']['c_source_suite'])
    public=Path(ctx['config']['c_public_data'])
    files=fetch_public(root)
    raw={}
    for dataset,rows in [('gsm8k',pq.read_table(files['gsm8k']).to_pylist()),('limo',jsonl(files['limo']))]:
        for row in rows:
            raw.setdefault(question_id(row['question']),(dataset,row))
    tok=tokenizer_for(Path(ctx['inputs'])/'stage2')
    original=read(public/'gsm8k_pairs.json')+read(public/'limo_pairs.json')
    forms={k:[] for k in ('F00','F10','F01','F11')}
    removed=[]
    for item in original:
        dataset,row=raw[item['id']]
        response=row['answer'] if dataset=='gsm8k' else row['solution']
        variants={}
        for form in forms:
            target=gsm_legacy_target(response) if dataset=='gsm8k' and form[1]=='1' else response
            variants[form]=format_tokens(tok,row['question'],target,form,ctx['config']['context'])
        if variants['F00']!=item['chosen_tokens']:
            raise ValueError('Pinned public source no longer reproduces original tokenization')
        if any(v is None for v in variants.values()):
            removed.append(item['id'])
            continue
        for form,tokens in variants.items():
            forms[form].append(dict(item,chosen_tokens=tokens))
    old=source/'training_trials/C_gsm_limo_nll'
    old_rows,old_plan=read(old/'pairs.json'),read(old/'schedule.json')
    index={r['id']:i for i,r in enumerate(forms['F00'])}
    groups=[]; old_to_new={}
    for step,group in enumerate(old_plan['groups'],1):
        kept=[index[old_rows[i]['id']] for i in group if old_rows[i]['id'] in index]
        if kept: groups.append(kept)
        old_to_new[step]=len(groups)
    milestones={label:old_to_new[old_plan['milestone_steps'][label]] for label in ('half','full')}
    if not groups or min(milestones.values())<1:
        raise ValueError('No matched examples for the fixed format comparison')
    for form,rows in forms.items():
        immutable_json(root/form/'pairs.json',rows)
        immutable_json(root/form/'schedule.json',make_plan(rows,groups,milestones=milestones))
    stats={'common_examples':len(index),'removed_for_common_length':len(removed),
           'removed_ids':removed,'old_updates':len(old_plan['groups']),'matched_updates':len(groups),
           'old_presentations':sum(map(len,old_plan['groups'])),
           'matched_presentations':sum(map(len,groups)),
           'comparison':'same IDs/order/update groups across formats; token counts may differ'}
    immutable_json(root/'format_report.json',stats)
    print('FORMAT_AGGREGATES', {k:v for k,v in stats.items() if k!='removed_ids'},flush=True)


def interpolate(left,right,destination,alpha):
    """Interpolate LoRA A/B parameters, not the products B@A or merged model weights."""
    import torch
    from safetensors.torch import load_file,save_file
    from worker import check_model,compatible_adapter
    if not 0<=alpha<=1: raise ValueError('Alpha must be within [0,1]')
    destination=Path(destination); left=Path(left); right=Path(right)
    manifest={'left_sha256':sha(left/'adapter_model.safetensors'),
              'right_sha256':sha(right/'adapter_model.safetensors'),'alpha':alpha,
              'left_config_sha256':sha(left/'adapter_config.json'),
              'right_config_sha256':sha(right/'adapter_config.json'),
              'method':'LoRA parameter interpolation; rank preserved'}
    if check_model(destination):
        if read(destination/'interpolation.json')!=manifest:
            raise ValueError('Interpolation source changed')
        return
    lc,rc=read(left/'adapter_config.json'),read(right/'adapter_config.json')
    for key in ('r','lora_alpha','target_modules','bias','fan_in_fan_out','modules_to_save',
                'use_rslora','use_dora','rank_pattern','alpha_pattern','lora_dropout'):
        l,r=lc.get(key),rc.get(key)
        if key=='target_modules': l,r=sorted(l),sorted(r)
        if l!=r: raise ValueError('Incompatible LoRA configuration: '+key)
    a,b=load_file(str(left/'adapter_model.safetensors')),load_file(str(right/'adapter_model.safetensors'))
    if a.keys()!=b.keys(): raise ValueError('Adapter parameter keys differ')
    result={}
    for key,x in a.items():
        y=b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype or not x.is_floating_point():
            raise ValueError('Adapter shape/dtype mismatch')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite source adapter')
        result[key]=(x.clone() if alpha==0 else y.clone() if alpha==1 else
                     torch.lerp(x.float(),y.float(),alpha).to(x.dtype)).contiguous()
        if alpha in (0,1) and not torch.equal(result[key],x if alpha==0 else y):
            raise ValueError('Interpolation endpoint is not exact')
    tmp=destination.with_name(destination.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    compatible_adapter(right,tmp)
    (tmp/'adapter_model.safetensors').unlink()  # only the temporary symlink
    save_file(result,str(tmp/'adapter_model.safetensors'))
    for name in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja'):
        if (right/name).exists(): shutil.copy2(right/name,tmp/name)
    write(tmp/'interpolation.json',manifest)
    write(tmp/'artifacts.json',fingerprint(tmp))
    tmp.rename(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','interpolate'])
    parser.add_argument('--context',type=Path)
    parser.add_argument('--left',type=Path)
    parser.add_argument('--right',type=Path)
    parser.add_argument('--destination',type=Path)
    parser.add_argument('--alpha',type=float)
    args=parser.parse_args()
    if args.phase=='prepare': prepare(read(args.context))
    else: interpolate(args.left,args.right,args.destination,args.alpha)
