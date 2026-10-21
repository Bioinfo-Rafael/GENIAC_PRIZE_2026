"""Submit the completed E1-600k directly, without evaluating or selecting models."""
import argparse
import fcntl
import os
import signal
import subprocess
import time
from e1_common import *


def validate_training(source):
    from worker import check_model
    ctx=read(source/'suite_context.json');cfg=ctx['config']
    if cfg.get('suite_kind')!='e1_limo_v1':raise ValueError('Expected E1 source suite')
    for name,expected in ctx['code'].items():
        if sha(source/'code'/name)!=expected:raise ValueError('Source code changed: '+name)
    training=source/'training_trials/E1'
    plan=read(training/'schedule.json')
    complete=read(training/'training/rft/complete.json')
    if (complete.get('complete') is not True or complete['updates']!=len(plan['groups'])
        or complete['supervised_tokens']!=plan['actual_tokens'] or plan['actual_tokens']<600000):
        raise ValueError('E1-600k training is not complete; no training is restarted by this command')
    if not check_model(training/'training/rft/final'):raise ValueError('Missing final E1 adapter')
    if not check_model(training/'models/stage1'):raise ValueError('Missing Stage1 parent')
    if read(training/'context.json')['config']!=cfg:raise ValueError('Training configuration differs')
    return cfg,training,complete


def generate_direct(source,root):
    from submission import verify_assets,generate
    import sys
    sys.path.insert(0,str(HERE.parent/'rank64_12k'))
    import official_prompt
    cfg,training,complete=validate_training(source)
    verify_assets()
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt'];official_prompt._placeholder=official['prompt']['placeholder']
    model=training/'models/rft'
    code={str(p.relative_to(root/'code')):sha(p) for p in sorted((root/'code').rglob('*'))
          if p.is_file() and p.suffix in ('.py','.json','.jinja','.sh','.txt') and '__pycache__' not in p.parts}
    manifest=dict(candidate='E1-600k',source_suite=str(source),config=cfg,config_sha256=digest(cfg),
        selection='User explicitly requested final E1-600k directly; no dev/independent selection; improvement unverified',
        training_complete=complete,source_identity=read(source/'suite_context.json'),
        adapter_sha256=sha(training/'training/rft/final/adapter_model.safetensors'),
        model_artifacts_sha256=sha(model/'artifacts.json'),
        stage1_artifacts_sha256=sha(training/'models/stage1/artifacts.json'),
        data_identity=read(source/'prepared_identity.json'),submission_code=code,
        official_assets=read(LEGACY/'submission_assets/manifest.json'))
    immutable_json(root/'manifest.json',manifest)
    generate(root,model_path=model,stage1_slug='run01_stage1',run_id=source.name+'_direct600k',
        recipe_detail='run03 rank8 continued SFT; PRM/GSM/LIMO teacher-token mixture 60/20/20; response-only NLL; lr5e-6; BF16 NF4; maxlen8192; seed42; final 600k checkpoint; direct submission without evaluation-based selection',
        build_messages=official_prompt._build_messages)
    write(root/'hashes.json',{n:sha(root/n) for n in ('submission.json','manifest.json','format_check.json')})


def run(source,root):
    cfg,training,complete=validate_training(source)
    immutable_json(root/'request.json',dict(model='E1-600k',source=str(source),evaluation=False,
        training_complete_sha256=sha(training/'training/rft/complete.json')))
    workspace=source.parent.parent
    train=os.environ.get('GENIAC_TRAIN_PYTHON',str(workspace/'.venv-train/bin/python'))
    evaluate=os.environ.get('GENIAC_EVAL_PYTHON',str(workspace/'.venv-eval/bin/python'))
    data=os.environ.get('GENIAC_DATA_PYTHON',str(workspace/'.venv-submission-data/bin/python'))
    if (root/'complete.json').exists():
        if sha(root/'submission.json')!=read(root/'complete.json')['sha256']:raise ValueError('Submission changed')
        print('SUBMISSION_READY',root/'submission.json',flush=True);return
    jobs=[('merge',train,[str(source/'code/worker.py'),'merge','--context',str(training/'context.json'),'--candidate','rft']),
          ('fetch',data,[str(LEGACY/'submission.py'),'fetch','--root',str(root)]),
          ('generate',evaluate,[str(HERE/'direct_submit.py'),'generate','--source-suite',str(source)])]
    child=None
    def stop(*_):
        if child is not None:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            child.wait()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    report=dict(status='running',model='E1-600k',evaluations_skipped=True,improvement_verified=False,
                source_training=complete,phases=[])
    try:
        for name,python,command in jobs:
            print('PHASE direct_submission_'+name,flush=True)
            start=time.monotonic();log=root/(name+'.log')
            with log.open('a') as stream:
                child=subprocess.Popen([python,'-u',*command],stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                code=child.wait();child=None
            report['phases'].append(dict(phase=name,seconds=time.monotonic()-start,exit_code=code))
            write(root/'report.json',report)
            if code:raise RuntimeError(f'{name} failed ({code}); see {log}')
        report['status']='complete'
        print('SUBMISSION_READY',root/'submission.json',flush=True)
    except BaseException:
        report['status']='interrupted_or_failed';raise
    finally:write(root/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['run','generate'])
    p.add_argument('--source-suite',type=Path,required=True);a=p.parse_args()
    source=a.source_suite.expanduser().resolve();root=source/'direct_submission';root.mkdir(exist_ok=True)
    if a.phase=='generate':generate_direct(source,root);return
    workspace=source.parent.parent
    with (workspace/'.math_training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another GPU experiment is active; no process was stopped')
        run(source,root)

if __name__=='__main__':main()
