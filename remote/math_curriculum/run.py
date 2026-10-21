"""Resumable sequential phases; never train a comparison model."""
import argparse
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import time
from common import *


def verify_receipt(root,receipt):
    for name,value in receipt['files'].items():
        if not (root/name).is_file() or sha(root/name)!=value:
            raise ValueError('Completed phase artifact changed: '+name)


def phase_order():
    return ['prepare','stage1_train','stage1_merge','stage2_100000','merge_100000','eval_100000',
            'stage2_300000','merge_300000','eval_300000','select','submission_fetch','submission_generate']


def select_without_evaluation(root):
    """Keep any already frozen decision; otherwise use the final training milestone."""
    if (root/'selection.json').exists():
        return read(root/'selection.json')
    decision=dict(candidate='300000',rule='final training checkpoint; public evaluation skipped by user request',
                  evaluated=False,checkpoint_comparison_performed=False)
    immutable_json(root/'selection.json',decision)
    immutable_json(root/'report.json',dict(selected='300000',evaluation_status='skipped',
        checkpoint_comparison_performed=False,data=read(root/'data/report.json')))
    return decision


def run(a):
    root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    submission_first=getattr(a,'submission_first',True)
    if submission_first:
        immutable_json(root/'submission_first_policy.json',dict(
            mode='skip_public_evaluation_use_final_checkpoint',controller_sha256=sha(Path(__file__)),
            preserve_existing_selection=True))
    code={str(p.relative_to(HERE.parent)):sha(p) for folder in
          (HERE,LEGACY,HERE.parent/'public_eval_diagnostics',HERE.parent/'rank64_12k')
          for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
          and p.suffix in ('.py','.json','.jinja','.sh','.txt')}
    immutable_json(root/'identity.json',dict(code=code,config=CONFIG,recipe='gsm2000_prm80_limo20_v1'))
    for role,python in [('train',sys.executable),('eval',a.eval_python),('data',a.data_python)]:
        packages=subprocess.check_output([python,'-m','pip','freeze'],text=True).splitlines()
        immutable_json(root/f'packages_{role}.json',sorted(packages))
    for receipt in sorted((root/'receipts').glob('*.json')):
        verify_receipt(root,read(receipt))
    if (root/'status.json').exists() and read(root/'status.json')['status']=='complete':
        print('ALREADY_COMPLETE',root/'submission/submission.json',flush=True);return
    gpu_memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
    if not gpu_memory.strip() or int(gpu_memory.splitlines()[0])<45000:
        raise RuntimeError('A 48GB-class CUDA GPU is required')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active: raise RuntimeError('GPU has active compute processes; retry after they finish')
    if shutil.disk_usage(root).free<40*2**30: raise RuntimeError('Need at least 40 GiB free disk')
    child=None;stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    def phase(name,script,args=(),python=None,outputs=(),private=False):
        nonlocal child
        receipt=root/'receipts'/f'{phase_order().index(name):02d}_{name}.json'
        if receipt.exists(): verify_receipt(root,read(receipt));print('SKIP',name,flush=True);return
        if stopped: raise SystemExit(75)
        write(root/'status.json',dict(status='running',phase=name))
        print('PHASE',name,flush=True);start=time.monotonic();peak=0
        log=root/('private_logs' if private else 'logs')/(name+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[python or sys.executable,'-u',str(HERE/script),'--root',str(root),*map(str,args)]
        with log.open('a') as out:
            offset=log.stat().st_size;heartbeat=0
            child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                if not private:
                    with log.open() as stream:
                        stream.seek(offset);chunk=stream.read();offset=stream.tell()
                    if chunk: print(chunk,end='',flush=True)
                memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                    capture_output=True,text=True)
                if memory.returncode==0:
                    peak=max(peak,max((int(s.strip()) for s in memory.stdout.splitlines()),default=0)*2**20)
                elapsed=time.monotonic()-start
                if elapsed-heartbeat>=30:
                    print(f'PHASE_PROGRESS {name} elapsed={elapsed:.0f}s peak_device_VRAM={peak/2**30:.2f}GiB',flush=True)
                    heartbeat=elapsed
                time.sleep(1)
            if not private:
                with log.open() as stream: stream.seek(offset);print(stream.read(),end='',flush=True)
            code=child.returncode;child=None
        timing=dict(phase=name,seconds=time.monotonic()-start,peak_device_vram_bytes=peak,exit_code=code)
        with (root/'phase_times.jsonl').open('a') as stream: stream.write(json.dumps(timing)+'\n')
        if code or stopped: raise RuntimeError(f'{name} exited {code}; inspect its '+('private remote log' if private else 'log'))
        paths=[]
        for name in outputs:
            path=root/name
            if not path.exists(): raise ValueError('Missing phase artifact: '+name)
            paths.extend(sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path])
        write(receipt,dict(files={str(p.relative_to(root)):sha(p) for p in paths},timing=timing))
    try:
        phase('prepare','entry.py',outputs=['data/complete.json','data/eval.json','data/report.json',
              'stage1/pairs.json','stage1/schedule.json','stage2/pairs.json','stage2/schedule.json','tokenizer'])
        phase('stage1_train','gpu.py',['train','--stage','stage1'],outputs=['stage1/training/rft/final'])
        phase('stage1_merge','gpu.py',['merge','--stage','stage1'],outputs=['models/stage1'])
        for milestone in (('300000',) if submission_first else ('100000','300000')):
            step=read(root/'stage2/schedule.json')['milestone_steps'][milestone]
            phase('stage2_'+milestone,'gpu.py',['train','--stage','stage2','--until',step],
                  outputs=[f'stage2/training/rft/milestones/step-{step:05d}'])
            phase('merge_'+milestone,'gpu.py',['merge','--stage','stage2','--milestone',milestone],
                  outputs=['models/'+milestone])
            if not submission_first:
                phase('eval_'+milestone,'evaluate.py',['--milestone',milestone],a.eval_python,
                  outputs=[f'evaluations/{milestone}/metrics.json',f'evaluations/{milestone}/generations/b0_eval.json'])
        if submission_first:
            decision=select_without_evaluation(root)
            print('PUBLIC_EVALUATION_SKIPPED selected='+decision['candidate'],flush=True)
        else:
            phase('select','evaluate.py',python=a.eval_python,outputs=['selection.json','question_changes.json','report.json'])
        phase('submission_fetch','submit.py',['fetch'],a.data_python,outputs=['submission/question_identity.json'],private=True)
        phase('submission_generate','submit.py',['generate'],a.eval_python,
              outputs=['submission/submission.json','submission/format_check.json','submission/hashes.json',
                       'submission/manifest.json','submission/complete.json','submission/inference_times.json'],private=True)
        timings=jsonl(root/'phase_times.jsonl')
        training={s:jsonl(root/s/'training/rft/train.jsonl') for s in ('stage1','stage2')}
        training_summary={s:dict(seconds=sum(x.get('seconds',0) for x in updates),
            peak_allocated_vram_bytes=max((x.get('max_vram',0) for x in updates),default=0),
            supervised_tokens=read(root/s/'schedule.json').get('actual_tokens'),
            processed_token_attempts=sum(x.get('teacher_tokens_this_update',0) for x in updates))
            for s,updates in training.items()}
        write(root/'runtime_summary.json',dict(phases=timings,training=training,training_summary=training_summary,
            total_phase_seconds=sum(t['seconds'] for t in timings),
            evaluation_seconds=sum(t['seconds'] for t in timings if t['phase'].startswith('eval_')),
            submission_phase_seconds=sum(t['seconds'] for t in timings if t['phase']=='submission_generate'),
            submission_generation_seconds=sum(t['seconds'] for t in read(root/'submission/inference_times.json')),
            peak_device_vram_bytes=max(t['peak_device_vram_bytes'] for t in timings)))
        write(root/'status.json',dict(status='complete',submission=str(root/'submission/submission.json')))
        print('COMPLETE',root/'submission/submission.json',flush=True)
    except BaseException as exc:
        write(root/'status.json',dict(status='stopped' if stopped else 'failed',error=str(exc)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--eval-python',required=True);p.add_argument('--data-python',required=True)
    p.add_argument('--evaluate-checkpoints',dest='submission_first',action='store_false',
                   help='Optional original public-evaluation workflow; default skips it')
    run(p.parse_args())
