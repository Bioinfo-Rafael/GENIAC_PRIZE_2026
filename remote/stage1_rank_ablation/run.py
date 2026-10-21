"""B -> short C/D estimates -> C -> D, then reference A and final comparison."""
import argparse
import importlib.util
import subprocess
import signal
import time
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--eval-python',required=True)
    p.add_argument('--reference',type=Path);a=p.parse_args();root=a.root.resolve();root.mkdir(parents=True,exist_ok=True)
    files={str(x.relative_to(HERE)):sha(x) for x in HERE.rglob('*') if x.is_file() and x.suffix in ('.py','.json','.txt','.sh')}
    immutable_json(root/'identity.json',dict(code=files,helpers={x:sha(LEGACY/x) for x in ['core.py','worker.py','grading.py','suite_utils.py','chat_template.jinja','input_manifest.json']}))
    packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    immutable_json(root/'packages.json',packages.splitlines())
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
        else:raise SystemExit(75)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def phase(label,script,args=(),python=None,optional=False):
        nonlocal child
        log=root/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
        write(root/'status.json',dict(status='running',phase=label))
        print('PHASE',label,flush=True);started=time.monotonic()
        with log.open('a') as f:
            proc=subprocess.Popen([python or sys.executable,'-u',str(HERE/script),'--root',str(root),*map(str,args)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            child=proc
            for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            code=proc.wait();child=None
        with (root/'phase_times.jsonl').open('a') as f:f.write(json.dumps(dict(phase=label,seconds=time.monotonic()-started,exit_code=code))+'\n')
        if stopping:code=75
        if code and (not optional or stopping):
            write(root/'status.json',dict(status='stopped' if code==75 else 'failed',phase=label,exit_code=code))
            raise SystemExit(code)
        return code
    phase('prepare','prepare.py',['--cache',a.cache])
    phase('parents','engine.py',['parents'])
    selected=root/'selected_backend.json'
    if not selected.exists():
        # A tiny bounded search; every candidate must also fit rank64 before B starts.
        candidates=[]
        for backend,weights,attention in backend_candidates(importlib.util.find_spec('unsloth') is not None,importlib.util.find_spec('flash_attn') is not None):
            for mb in (2,4):
                candidates.append(dict(backend=backend,weights=weights,attention=attention,microbatch=mb,checkpointing=True,layout='padded'))
        # One no-checkpoint candidate, no large Cartesian sweep or full-weight LoRA.
        candidates.append(dict(backend='hf',weights='nf4',attention='sdpa',microbatch=2,checkpointing=False,layout='padded'))
        accepted=[]
        for cfg in candidates:
            name=candidate_id(cfg);config=root/'benchmarks'/f'{name}.config.json';write(config,cfg)
            output=root/'benchmarks'/f'{name}.json'
            if not output.exists():phase(name,'engine.py',['smoke','--condition','D','--config',config,'--output',output],optional=True)
            result=read(output) if output.exists() else {'status':'failed'}
            if result['status']=='ok':accepted.append(result)
        if not accepted:raise RuntimeError('No eligible 4bit backend; inspect benchmarks/logs')
        write(selected,min(accepted,key=lambda r:r['seconds_per_update'])['config'])
    def train_eval(name):
        phase(name+'_train','engine.py',['train','--condition',name,'--config',selected])
        phase(name+'_merge','engine.py',['merge','--condition',name])
        phase(name+'_eval','evaluate.py',['--condition',name],python=a.eval_python)
    try:
        train_eval('B')
        for name in ['C','D']:
            output=root/'benchmarks'/f'{name}_estimate.json'
            if not output.exists() or read(output).get('status')!='ok':phase(name+'_estimate','engine.py',['bench','--condition',name,'--config',selected,'--output',output])
            r=read(output)
            if r['status']!='ok':raise RuntimeError('Failed benchmark: '+name)
            print(f"ESTIMATE {name}: {r['estimated_training_seconds']/60:.1f} minutes for 375 updates (approximate; excludes setup/save/eval)",flush=True)
        train_eval('C');train_eval('D')
        reference_args=['--condition','A']+(['--reference',a.reference] if a.reference else [])
        if phase('A_reuse_check','evaluate.py',reference_args+['--reuse-only'],python=a.eval_python,optional=True):
            phase('A_merge_existing_only','engine.py',['merge','--condition','A'])
            phase('A_eval','evaluate.py',reference_args,python=a.eval_python)
    finally:
        before=read(root/'status.json')
        phase('report','report.py')
        write(root/'status.json',before)
    write(root/'status.json',dict(status='complete'))

if __name__=='__main__':main()
