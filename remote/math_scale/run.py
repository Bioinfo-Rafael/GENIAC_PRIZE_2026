"""Sequential submission-first suite, with a persisted end-to-end 600-minute deadline."""
import argparse
import datetime
import os
import signal
import shutil
import subprocess
import time
import json
from shared import *


class BudgetStop(RuntimeError): pass


def estimate(root,name):
    exp=read(root/name/'experiment.json');chain=Path(exp['chain'])
    plan=read(chain/'stage2/schedule.json')
    latest=chain/'stage2/training/rft/latest.json'
    done=read(latest)['step'] if latest.exists() else 0
    remaining=max(0,exp['step']-done)
    logs=jsonl(chain/'stage2/training/rft/train.jsonl')
    # Historical measured 123 updates / 367.8s. Rank32 initially gets a 2x allowance.
    speed=max([x['seconds'] for x in logs[-30:]] or [3.*(2 if exp['rank']==32 else 1)])
    phases=jsonl(root/'phase_times.jsonl')
    def measured(phase,default):
        return max([x['seconds'] for x in phases if x['phase']==phase and x['exit_code']==0] or [default])
    submission=measured('submit',255)*1.75+60
    evaluation=measured('evaluate',400)*1.75
    merge=measured('merge',40)*1.75
    training=remaining*speed*1.5+120
    required=training+merge+submission+evaluation+180
    return dict(remaining_updates=remaining,seconds_per_update=speed,training_seconds=training,
        merge_seconds=merge,submission_seconds=submission,evaluation_seconds=evaluation,required_seconds=required)


class Runner:
    def __init__(self,args):
        self.a=args;self.root=args.root.resolve();self.child=None;self.stopped=False
        self.root.mkdir(parents=True,exist_ok=True)
        budget=self.root/'budget.json'
        if not budget.exists():
            start=float(os.environ.get('MATH_SCALE_STARTED',time.time()))
            immutable_json(budget,dict(started_epoch=start,deadline_epoch=start+36000,max_seconds=36000,
                policy='Wall clock includes startup, preparation, all phases, failed attempts and downtime. Never reset on resume.'))
        self.budget=read(budget)
        self.deadline=self.budget['deadline_epoch']
        self.report=read(self.root/'suite_report.json') if (self.root/'suite_report.json').exists() else dict(
            status='running',experiments={n:dict(status='pending',path=str(self.root/n)) for n in NAMES})
        signal.signal(signal.SIGTERM,self.stop);signal.signal(signal.SIGINT,self.stop)

    def stop(self,*_):
        self.stopped=True
        if self.child is not None and self.child.poll() is None:
            os.killpg(self.child.pid,signal.SIGTERM)

    def save(self):
        now=time.time();phases=jsonl(self.root/'phase_times.jsonl')
        self.report.update(budget=self.budget,elapsed_seconds=now-self.budget['started_epoch'],remaining_seconds=max(0,self.deadline-now))
        for n,item in self.report['experiments'].items():
            item['phase_seconds']={p:sum(x['seconds'] for x in phases if x['name']==n and x['phase']==p) for p in ('train','merge','fetch','submit','evaluate')}
            if (self.root/n/'report.json').exists(): item['result']=read(self.root/n/'report.json')
        write(self.root/'suite_report.json',self.report)
        training={}
        for rank in (8,32):
            logs=jsonl(self.root/f'chains/R{rank}/stage2/training/rft/train.jsonl')
            training[f'R{rank}']=dict(update_attempt_seconds=sum(x['seconds'] for x in logs),
                peak_allocated_vram_bytes=max([x['max_vram'] for x in logs] or [0]))
        write(self.root/'runtime_summary.json',dict(budget=self.budget,elapsed_wall_seconds=now-self.budget['started_epoch'],
            total_phase_seconds=sum(x['seconds'] for x in phases),phases=phases,training=training,
            peak_device_vram_bytes=max([x['peak_device_vram_bytes'] for x in phases] or [0]),status=self.report['status']))

    def phase(self,name,phase,python,outputs):
        receipt=self.root/'receipts'/f'{name}_{phase}.json'
        if receipt.exists():
            verify_files(self.root,read(receipt)['files']);print('SKIP',name,phase,flush=True);return
        if self.stopped: raise RuntimeError('Stopped by signal; resume with same run ID')
        if time.time()>self.deadline-180: raise BudgetStop('Insufficient time before hard deadline')
        private=phase in ('fetch','submit')
        log=self.root/('private_logs' if private else 'logs')/f'{name}_{phase}.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        self.report['phase']=name+':'+phase;self.save()
        print('PHASE',name,phase,flush=True)
        start=time.monotonic();peak=0;offset=log.stat().st_size if log.exists() else 0;last_print=0;sent=False
        cmd=[python,'-u',str(getattr(self,'phase_script',HERE/'phases.py')),phase,'--root',str(self.root)]
        if phase=='prepare': cmd+=['--source',str(self.a.source.resolve())]
        else:cmd+=['--name',name]
        with log.open('a') as out:
            self.child=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            while self.child.poll() is None:
                now=time.time()
                if (self.stopped or now>=self.deadline-120) and not sent:
                    os.killpg(self.child.pid,signal.SIGTERM);sent=True
                if now>=self.deadline-15:
                    os.killpg(self.child.pid,signal.SIGKILL)
                memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                if memory.returncode==0: peak=max(peak,max((int(x) for x in memory.stdout.splitlines()),default=0)*2**20)
                if not private:
                    with log.open() as stream:
                        stream.seek(offset);chunk=stream.read();offset=stream.tell()
                    if chunk: print(chunk,end='',flush=True)
                elapsed=time.monotonic()-start
                if elapsed-last_print>=30:
                    print('PROGRESS',name,phase,round(elapsed),'seconds; remaining',round(self.deadline-now),flush=True)
                    last_print=elapsed;self.save()
                time.sleep(1)
            code=self.child.returncode
            # Some inference engines launch process descendants; clean this phase group only.
            try:os.killpg(self.child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            self.child=None
        if not private:
            with log.open() as stream:stream.seek(offset);print(stream.read(),end='',flush=True)
        timing=dict(name=name,phase=phase,seconds=time.monotonic()-start,exit_code=code,peak_device_vram_bytes=peak)
        with (self.root/'phase_times.jsonl').open('a') as stream:stream.write(json.dumps(timing)+'\n')
        self.save()
        if time.time()>=self.deadline-120:raise BudgetStop('Saved at update boundary or interrupted phase before 600-minute deadline')
        if code or self.stopped:raise RuntimeError(f'{name} {phase} exited {code}; see '+str(log))
        files=file_receipt(self.root,[self.root/p for p in outputs])
        write(receipt,dict(files=files,timing=timing))

    def run(self):
        root=self.root
        try:
            if root==self.a.source.resolve() or root in self.a.source.resolve().parents or self.a.source.resolve() in root.parents:
                raise ValueError('Suite and source must be separate directories')
            if self.report['status']=='complete':
                for receipt in (root/'receipts').glob('*.json'):verify_files(root,read(receipt)['files'])
                print('ALREADY_COMPLETE',root,flush=True);return
            if time.time()>self.deadline-180:raise BudgetStop('Original 600-minute budget exhausted')
            if shutil.disk_usage(root).free<60*2**30:raise RuntimeError('Need 60 GiB free for four merged models, preserved checkpoints and temporary writes')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True)
            if 'RTX 6000 Ada' not in gpu or int(gpu.split(',')[-1])<45000:raise ValueError('Expected RTX 6000 Ada 48GB')
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:raise RuntimeError('GPU already has compute processes')
            code={str(p.relative_to(HERE.parent)):sha(p) for folder in (HERE,CURRICULUM,LEGACY,HERE.parent/'rank64_12k')
                  for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jinja','.sh','.txt') and '__pycache__' not in p.parts}
            immutable_json(root/'identity.json',dict(recipe='math_scale_v1',code=code,config=CONFIG,source=str(self.a.source.resolve())))
            for role,python in [('train',sys.executable),('eval',self.a.eval_python),('data',self.a.data_python)]:
                packages=sorted(subprocess.check_output([python,'-m','pip','freeze'],text=True,timeout=30).splitlines())
                immutable_json(root/f'packages_{role}.json',packages)
            immutable_json(root/'gpu.json',dict(description=gpu))
            # Preparation is revalidated on resume, including immutable source model hashes.
            self.phase('suite','prepare',sys.executable,['data/complete.json'])
            from scale_prepare import prepare
            prepare(root,self.a.source.resolve())
            for name in NAMES:
                item=self.report['experiments'][name]
                exp=read(root/name/'experiment.json')
                if item['status']=='complete':
                    for receipt in (root/'receipts').glob(name+'_*.json'):verify_files(root,read(receipt)['files'])
                    continue
                prediction=estimate(root,name);prediction['remaining_budget_seconds']=self.deadline-time.time()
                item['estimate']=prediction;self.save()
                print('BUDGET_ESTIMATE',name,prediction,flush=True)
                trained=(root/'receipts'/f'{name}_train.json').exists()
                if not trained and prediction['required_seconds']>prediction['remaining_budget_seconds']:
                    raise BudgetStop('Not starting '+name+': cannot reserve training + submission + evaluation time')
                item['status']='running';self.save()
                self.phase(name,'train',sys.executable,[f'{name}/checkpoint',f'{name}/training_complete.json'])
                self.phase(name,'merge',sys.executable,[f'{name}/models/{name}'])
                self.phase(name,'fetch',self.a.data_python,[f'{name}/submission/question_identity.json'])
                self.phase(name,'submit',self.a.eval_python,[f'{name}/submission/{n}' for n in ('submission.json','format_check.json','hashes.json','manifest.json','complete.json','inference_times.json')])
                item.update(status='submission_ready',submission=str(root/name/'submission/submission.json'),submission_sha256=sha(root/name/'submission/submission.json'))
                self.save();print('SUBMISSION_READY',name,item['submission'],flush=True)
                self.phase(name,'evaluate',self.a.eval_python,[f'{name}/report.json',f'{name}/question_changes.json',f'{name}/evaluations/{name}'])
                item['status']='complete';self.save()
            self.report['status']='complete';self.report.pop('error',None)
        except BudgetStop as exc:
            self.report.update(status='budget_stopped',error=str(exc))
            for item in self.report['experiments'].values():
                if item['status']=='pending':item.update(status='skipped_budget',reason=str(exc))
            print('BUDGET_STOP',str(exc),flush=True)
        except BaseException as exc:
            self.report.update(status='stopped_resumable',error=str(exc))
            print('STOPPED_RESUMABLE',str(exc),flush=True)
            raise
        finally:self.save()
        print('SUITE_END',self.report['status'],root/'suite_report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--eval-python',required=True);p.add_argument('--data-python',required=True)
    Runner(p.parse_args()).run()
