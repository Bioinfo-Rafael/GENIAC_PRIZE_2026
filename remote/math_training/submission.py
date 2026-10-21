"""Submission helpers; legacy C-suite CLI and explicit-model generation API."""
import argparse
import datetime
import fcntl
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys

from core import read,write,sha,digest
from suite import link_dir
from suite_utils import immutable_json
from submission_assets import check_submission as checker

HERE=Path(__file__).resolve().parent


def verify_assets():
    for name,expected in read(HERE/'submission_assets/manifest.json').items():
        if sha(HERE/'submission_assets'/name)!=expected:
            raise ValueError('Distributed submission asset changed: '+name)


def validate_questions(rows):
    ids=[r['question_id'] for r in rows]
    if len(ids)!=len(checker.QUESTION_IDS) or len(set(ids))!=len(ids) or set(ids)!=checker.QUESTION_IDS:
        raise ValueError('Official question IDs/count differ')
    for r in rows:
        if (r['task'] not in checker.TASKS or not isinstance(r['subtask'],str)
            or not isinstance(r['turns'],list) or len(r['turns'])!=1
            or not isinstance(r['turns'][0],str) or not r['turns'][0]):
            raise ValueError('Unexpected official question schema')


def fetch(root):
    from submission_assets.evalset import load_eval_questions
    path=root/'questions.json'
    rows=read(path) if path.exists() else load_eval_questions()
    validate_questions(rows)
    immutable_json(path,rows)
    immutable_json(root/'question_identity.json',{'sha256':sha(path),'count':len(rows)})
    print('OFFICIAL_QUESTIONS_READY count='+str(len(rows)),flush=True)


def validate_partial(partial,signature,rows):
    if partial['signature']!=signature: raise ValueError('Submission resume signature differs')
    predictions=partial['predictions']; raw=partial['raw_outputs']
    if len(predictions)>len(rows) or len(raw)!=len(predictions):
        raise ValueError('Invalid partial output length')
    for q,p,r in zip(rows,predictions,raw):
        if any(p[k]!=q[k] for k in ('question_id','task','subtask')):
            raise ValueError('Partial question order changed')
        if (r['question_id']!=q['question_id'] or p['answer_text']!=r['text']
            or p['completion_tokens']!=len(r['token_ids'])
            or p['prompt_tokens']!=len(r['prompt_token_ids'])):
            raise ValueError('Partial answer differs from recorded model output')


def prediction(q,out):
    if len(out.outputs)!=1: raise ValueError('Expected exactly one model output')
    v=out.outputs[0]
    return ({'question_id':q['question_id'],'task':q['task'],'subtask':q['subtask'],
        'answer_text':v.text,'prompt_tokens':len(out.prompt_token_ids),'completion_tokens':len(v.token_ids)},
        {'question_id':q['question_id'],'text':v.text,'token_ids':list(v.token_ids),
         'prompt_token_ids':list(out.prompt_token_ids),'finish_reason':v.finish_reason,'stop_reason':v.stop_reason})


def generate(root, *, model_path=None, recipe_detail=None, stage1_slug="shared_stage1", build_messages=None, run_id=None):
    from vllm import LLM,SamplingParams
    import torch
    from worker import check_model
    manifest=read(root/'manifest.json'); cfg=manifest['config']
    model_path=Path(model_path) if model_path is not None else root/'models/rft'
    if recipe_detail is not None or build_messages is not None or run_id is not None:
        immutable_json(root/'generation_profile.json',dict(model_path=str(model_path.resolve()),
            model_artifacts_sha256=sha(model_path/'artifacts.json'),recipe_detail=recipe_detail,
            stage1_slug=stage1_slug,run_id=run_id,official_chat_builder=build_messages is not None,
            manifest_sha256=sha(root/'manifest.json')))
    if (root/'complete.json').exists():
        if sha(root/'submission.json')!=read(root/'complete.json')['sha256']:
            raise ValueError('Completed submission changed')
        print('SUBMISSION_READY '+str(root/'submission.json'),flush=True);return
    rows=read(root/'questions.json');validate_questions(rows)
    if sha(root/'questions.json')!=read(root/'question_identity.json')['sha256']:
        raise ValueError('Cached questions changed')
    if not check_model(model_path): raise ValueError('Merged model missing')
    packages={x:importlib.metadata.version(x) for x in ('vllm','torch','transformers')}
    signature=digest([manifest,sha(root/'questions.json'),packages,sha(model_path/'artifacts.json')])
    path=root/'partial.json'
    partial=read(path) if path.exists() else {'signature':signature,'predictions':[],'raw_outputs':[]}
    validate_partial(partial,signature,rows)
    template=(HERE/'chat_template.jinja').read_text()
    llm=LLM(model=str(model_path),dtype='half',max_model_len=cfg['context'],
        gpu_memory_utilization=cfg['vllm_memory'],max_num_seqs=cfg['eval_chunk'],
        seed=cfg['seed'],trust_remote_code=False,enable_prefix_caching=True)
    tok=llm.get_tokenizer()
    if tok.chat_template!=template: raise ValueError('Tokenizer template mismatch')
    prompts=[];conversations=[];framings=set()
    for q in rows:
        messages=build_messages(q['turns'][0]) if build_messages else [{'role':'user','content':q['turns'][0]}]
        conversations.append(messages)
        rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        text=q['turns'][0]
        if rendered.count(text)!=1: raise ValueError('Question not preserved exactly once in prompt')
        before,after=rendered.split(text)
        framings.add((before,after))
        ids=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        if len(ids)+cfg['max_tokens']>cfg['context']:
            raise ValueError('Question exceeds reserved context; no truncation performed')
        prompts.append(ids)
    if framings!={checker.REFERENCE_FRAMING}: raise ValueError('Unexpected prompt framing')
    for start in range(len(partial['predictions']),len(rows),cfg['eval_chunk']):
        end=min(len(rows),start+cfg['eval_chunk'])
        params=[SamplingParams(n=1,temperature=0.,seed=cfg['seed'],max_tokens=cfg['max_tokens']) for _ in rows[start:end]]
        if build_messages is None:
            outputs=llm.generate([{'prompt_token_ids':p} for p in prompts[start:end]],params,use_tqdm=False)
        else:
            outputs=llm.chat(conversations[start:end],params,use_tqdm=False)
        if len(outputs)!=end-start: raise ValueError('Incomplete generation batch')
        for q,out,expected in zip(rows[start:end],outputs,prompts[start:end]):
            if list(out.prompt_token_ids)!=expected: raise ValueError('Engine prompt differs')
            p,r=prediction(q,out);partial['predictions'].append(p);partial['raw_outputs'].append(r)
        write(path,partial)
        print(f'GENERATED {end}/{len(rows)}',flush=True)
    validate_partial(partial,signature,rows)
    prefix,suffix=next(iter(framings))
    meta={'path':'ii','base_model':cfg['base_model'],'recipe':'custom',
        'recipe_detail':recipe_detail if recipe_detail is not None else 'run03 Stage 2 continued with published GSM8K/LIMO NLL, F10; frozen C-suite selection',
        'run_id':run_id or root.name,'stage1_slug':stage1_slug,'stage2_slug':manifest['candidate'],
        'adapter_source':'local','generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'colab_env':'remote '+torch.cuda.get_device_name(0),'vllm_version':packages['vllm'],
        'chat_template':{'applied':True,'sha256':sha(HERE/'chat_template.jinja'),'is_default':True,
            'framing_variants':1,'framing_prefix':prefix,'framing_suffix':suffix},
        'inference_config':{'max_model_len':cfg['context'],'max_tokens_by_task':{t:cfg['max_tokens'] for t in checker.TASKS},
            'dtype':'float16','enforce_eager':False,'gpu_memory_utilization':cfg['vllm_memory'],
            'temperature':0.,'seed':cfg['seed'],'chunk_size':cfg['eval_chunk'],'n':1},
        'provenance':manifest,'packages':packages}
    submission={'submission_meta':meta,'predictions':partial['predictions']}
    draft=root/'submission.pending.json';write(draft,submission)
    result=checker.check(str(draft))
    write(root/'format_check.json',{'errors':result.errors,'violations':result.violations,'warnings':result.warnings})
    if result.errors or result.violations: raise ValueError('Submission format check failed; see format_check.json')
    final=root/'submission.json'
    if final.exists() and read(final)['predictions']!=submission['predictions']:
        raise ValueError('Refusing to replace different completed predictions')
    draft.replace(final)
    write(root/'complete.json',{'status':'submission_ready','count':len(rows),'sha256':sha(final)})
    print('SUBMISSION_READY '+str(final),flush=True)


def run(args):
    from worker import check_model
    source=args.source_suite.expanduser().resolve();workspace=args.workspace.expanduser().resolve()
    report=read(source/'suite_report.json');selection=read(source/'selection.json')
    candidate=selection['candidate']
    if not candidate or candidate!=report['selection']['candidate']:
        raise ValueError('No consistent frozen selected candidate')
    adapter=Path(report['recommended_adapter']).resolve()
    if adapter!=Path(report['trials'][candidate]['adapter']).resolve():
        raise ValueError('Recommended adapter is not the selected candidate')
    if source not in adapter.parents or not check_model(adapter): raise ValueError('Selected adapter missing or changed')
    stage1=(source/'shared/models/stage1').resolve()
    if not check_model(stage1): raise ValueError('Shared Stage 1 missing or changed')
    cfg=read(source/'suite_context.json')['config']
    inputs=workspace/'inputs'
    for item in read(HERE/'input_manifest.json')['files']:
        if item['path'].startswith(('stage1/','stage2/')) and sha(inputs/item['path'])!=item['sha256']:
            raise ValueError('Original input adapter/tokenizer changed')
    root=workspace/'submissions'/(source.name+'_'+candidate)
    root.mkdir(parents=True,exist_ok=True)
    code={str(p.relative_to(HERE)):sha(p) for p in HERE.rglob('*') if p.is_file()
          and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jinja','.sh','.txt')}
    manifest={'candidate':candidate,'source_suite':str(source),'adapter':str(adapter),
        'training_recipe':report['trials'][candidate].get('recipe'),
        'adapter_sha256':sha(adapter/'adapter_model.safetensors'),'adapter_artifacts_sha256':sha(adapter/'artifacts.json'),
        'stage1_artifacts_sha256':sha(stage1/'artifacts.json'),'selection_sha256':sha(source/'selection.json'),
        'config':{k:cfg[k] for k in ('base_model','base_revision','stage2_sha256','context','max_tokens','seed','eval_chunk','vllm_memory')},
        'code':code}
    immutable_json(root/'manifest.json',manifest)
    if (root/'complete.json').exists():
        if sha(root/'submission.json')!=read(root/'complete.json')['sha256']: raise ValueError('Completed submission changed')
        print('SUBMISSION_READY '+str(root/'submission.json'));return
    if not (root/'code').exists(): shutil.copytree(HERE,root/'code',ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for name,expected in code.items():
        if sha(root/'code'/name)!=expected: raise ValueError('Submission snapshot changed')
    link_dir(stage1,root/'models/stage1');link_dir(adapter,root/'training/rft/final')
    immutable_json(root/'context.json',{'run':str(root),'inputs':str(inputs),'config':cfg,'code':code,'signature':digest(manifest)})
    jobs=[('questions',workspace/'.venv-submission-data/bin/python',[str(root/'code/submission.py'),'fetch','--root',str(root)]),
          ('merge',workspace/'.venv-train/bin/python',[str(root/'code/worker.py'),'merge','--context',str(root/'context.json'),'--candidate','rft']),
          ('generate',workspace/'.venv-eval/bin/python',[str(root/'code/submission.py'),'generate','--root',str(root)])]
    for name,python,command in jobs:
        print('PHASE '+name,flush=True)
        subprocess.run([str(python),'-u',*command],check=True)
    print('SUBMISSION_READY '+str(root/'submission.json'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['run','fetch','generate','login'])
    p.add_argument('--workspace',type=Path,default=Path.home()/'geniac_math')
    p.add_argument('--source-suite',type=Path)
    p.add_argument('--root',type=Path)
    args=p.parse_args();verify_assets()
    if args.phase=='login':
        from huggingface_hub import get_token,login
        if not get_token():
            from getpass import getpass
            login(token=getpass('Hugging Face read token (hidden): '),add_to_git_credential=False)
        print('HF authentication available');return
    if args.phase=='fetch': fetch(args.root)
    elif args.phase=='generate': generate(args.root)
    else:
        if args.source_suite is None: p.error('--source-suite is required')
        args.workspace=args.workspace.expanduser().resolve()
        with (args.workspace/'.math_training.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: p.error('Another GPU experiment owns this workspace')
            run(args)


if __name__=='__main__': main()
