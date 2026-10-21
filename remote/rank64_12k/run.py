"""Single condition: prepare -> fresh SFT -> dev -> unedited submission JSON."""
import argparse
import os
import signal
import subprocess
import time
from shared import *


def link(source,target):
    source=source.resolve();target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink():
        if target.resolve()!=source:raise ValueError('Input link changed')
    elif target.exists():raise ValueError('Refusing to replace experiment directory')
    else:target.symlink_to(source,target_is_directory=True)


def validate_source(source):
    validate_inputs(source/'inputs')
    for name,expected in read(HERE/'source_contract.json')['source_rank_code'].items():
        if sha(source/'code/stage1_rank_ablation'/name)!=expected:
            raise ValueError('Unrecognized reference implementation: '+name)
    cfg=read(source/'selected_backend.json');contract=read(source/'conditions/D/contract.json')
    done=read(source/'conditions/D/complete.json')
    if contract['config']!=cfg or done['backend']!=cfg or cfg['weights']!='nf4':
        raise ValueError('D backend identity differs')
    if done['steps']!=375 or done['condition']!=CONDITIONS['D']:raise ValueError('Not the completed 6k rank64 D')
    if read(source/'data/report.json')['examples']!=6000:raise ValueError('D data size differs')
    if not verify_model(source/'parents/stage1') or not verify_model(source/'conditions/D/final'):
        raise ValueError('D artifacts missing')
    current=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    if sorted(current)!=sorted(read(source/'packages.json')):
        raise ValueError('Training environment differs from D; restore its packages, do not silently change backend')
    return cfg,current


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source-rank',type=Path,required=True);p.add_argument('--eval-python',required=True)
    p.add_argument('--data-python',required=True);a=p.parse_args()
    root=a.root.resolve();source=a.source_rank.resolve();root.mkdir(parents=True,exist_ok=True)
    cfg,packages=validate_source(source)
    if root==source or source in root.parents:raise ValueError('Use a separate output directory')
    immutable_json(root/'source.json',dict(path=str(source),contract_sha256=sha(source/'conditions/D/contract.json'),
        backend_sha256=sha(source/'selected_backend.json'),stage1_sha256=sha(source/'parents/stage1/artifacts.json')))
    immutable_json(root/'selected_backend.json',cfg);immutable_json(root/'packages.json',packages)
    files={str(f.relative_to(HERE.parent)):sha(f) for directory in [HERE,RANK,LEGACY]
           for f in directory.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.json','.txt','.sh','.jinja')}
    immutable_json(root/'identity.json',dict(code=files,recipe=RECIPE,source=read(root/'source.json')))
    link(source/'inputs',root/'inputs');link(source/'parents/stage1',root/'parents/stage1')
    link(source/'conditions/D',root/'conditions/D')
    child=None;stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def phase(label,script,args=(),python=None,private=False):
        nonlocal child
        if stopping:raise SystemExit(75)
        write(root/'status.json',dict(status='running',phase=label))
        print('PHASE',label,flush=True);started=time.monotonic()
        log=root/('private_logs' if private else 'logs')/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            cmd=[python or sys.executable,'-u',str(HERE/script),'--root',str(root),*map(str,args)]
            child=subprocess.Popen(cmd,stdout=f if private else subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if not private:
                for line in child.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            code=child.wait();child=None
        with (root/'phase_times.jsonl').open('a') as f:f.write(json.dumps(dict(phase=label,seconds=time.monotonic()-started,exit_code=code))+'\n')
        if code or stopping:raise RuntimeError(f'{label} stopped (exit={code}); '+('private remote log retained; do not paste its contents' if private else str(log)))
    try:
        phase('prepare_run04_12k','prepare.py')
        phase('reference_reuse_check','dev_eval.py',['--source',source,'--check-only'],a.eval_python)
        phase('train_rank64_12k','train.py',['train'])
        phase('merge_rank64_12k','train.py',['merge'])
        for name in read(root/'reference_eval_plan.json')['reinfer']:
            phase('reference_merge_'+name,'train.py',['reference-merge','--condition',name])
        phase('dev400','dev_eval.py',['--source',source],a.eval_python)
        phase('dev_report','report.py')
        phase('submission_fetch','submit.py',['fetch'],a.data_python,private=True)
        phase('submission_generate_and_check','submit.py',['generate'],a.eval_python,private=True)
        write(root/'status.json',dict(status='complete',submission=str(root/'submission/submission.json')))
        print('COMPLETE',root,'SUBMISSION',root/'submission/submission.json',flush=True)
    except BaseException as exc:
        write(root/'status.json',dict(status='stopped' if stopping else 'failed',error=str(exc)))
        raise

if __name__=='__main__':main()
