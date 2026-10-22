"""Nonrenewable 60-minute coordinator; submission is mandatory, eval is optional."""
import argparse
import os
import signal
import shutil
import subprocess
import time
import zipfile
from helpers import *


class Runner:
    def __init__(self,args):
        self.a=args; self.root=args.root.resolve(); self.child=None; self.stopped=False
        self.budget=read(self.root/'budget.json')
        self.deadline=self.budget['deadline_epoch']
        self.report=dict(status='running',submission_ready=False)
        signal.signal(signal.SIGTERM,self.stop); signal.signal(signal.SIGINT,self.stop)

    def stop(self,*_):
        self.stopped=True
        if self.child and self.child.poll() is None:
            try: os.killpg(self.child.pid,signal.SIGTERM)
            except ProcessLookupError: pass

    def save(self):
        self.report.update(elapsed_seconds=time.time()-self.budget['started_epoch'],
            remaining_seconds=max(0,self.deadline-time.time()),budget=self.budget)
        write(self.root/'suite_report.json',self.report)

    def phase(self,phase,python,end):
        receipt=self.root/'receipts'/f'{phase}.json'
        if receipt.exists():
            verify_files(self.root,read(receipt)['files']); return
        end=min(end,self.deadline-15)
        if self.stopped or time.time() >= end: raise TimeoutError('No time for '+phase)
        log=self.root/('private_logs' if phase in ('fetch','submit') else 'logs')/(phase+'.log')
        log.parent.mkdir(parents=True,exist_ok=True)
        self.report['phase']=phase; self.save()
        started=time.time(); timed_out=False; last_progress=started
        with log.open('a') as stream:
            self.child=subprocess.Popen([python,'-u',str(HERE/'phases.py'),phase,'--root',str(self.root)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while self.child.poll() is None:
                    if self.stopped or time.time() >= end:
                        timed_out=True; os.killpg(self.child.pid,signal.SIGTERM)
                        try: self.child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(self.child.pid,signal.SIGKILL); self.child.wait()
                        break
                    if time.time()-last_progress >= 30:
                        self.save()
                        print('PROGRESS',phase,'elapsed_seconds=',round(time.time()-started),
                            'remaining_seconds=',round(self.deadline-time.time()),flush=True)
                        last_progress=time.time()
                    time.sleep(.5)
                code=self.child.returncode
            finally:
                try: os.killpg(self.child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                self.child=None
        self.report.setdefault('phases',{})[phase]=dict(seconds=time.time()-started,exit_code=code)
        self.save()
        if timed_out or code: raise RuntimeError(f'{phase} failed or exceeded its deadline; see {log}')
        name=read(self.root/'selected.json')['name'] if (self.root/'selected.json').exists() else None
        paths={'prepare':['data/complete.json'],'fetch':['official_inputs/question_identity.json'],
            'train':['selected.json',f'{name}/training_complete.json'],
            'merge':[f'{name}/models/{name}/artifacts.json'],
            'submit':[f'{name}/submission/submission.json',f'{name}/submission/format_check.json',f'{name}/submission/hashes.json'],
            'evaluate':[f'{name}/evaluations/{name}/metrics.json']}[phase]
        write(receipt,dict(files=file_receipt(self.root,[self.root/p for p in paths])))
        print('PHASE_DONE',phase,round(time.time()-started),flush=True)

    def package(self):
        name=read(self.root/'selected.json')['name']
        exp=self.root/name
        path=self.root/(name+'.zip')
        partial=path.with_suffix('.zip.partial')
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(exp/'submission/submission.json',name+'.json')
            for file in ('format_check.json','hashes.json','manifest.json'):
                archive.write(exp/'submission'/file,'provenance/'+file)
            archive.write(exp/'training_complete.json','provenance/training_complete.json')
            archive.write(self.root/'budget.json','provenance/budget.json')
        with zipfile.ZipFile(partial) as archive:
            if archive.testzip() is not None: raise ValueError('ZIP validation failed')
        partial.replace(path)
        self.report.update(submission_ready=True,name=name,zip=str(path),zip_sha256=sha(path),
            submission=str(exp/'submission/submission.json'),submission_sha256=sha(exp/'submission/submission.json'))
        self.save()

    def run(self):
        try:
            request=dict(predecessor=str(self.a.predecessor.resolve()),maximum_questions=128,
                algorithm='OSW pilot v1',horizon=438,learning_rate=1e-6,seed=42,
                rollout=dict(count=2,temperature=1.,top_p=1.,top_k=0,max_tokens=2048),
                policy='fixed unique prefix; last finished batch; immutable absolute deadline')
            immutable_json(self.root/'request.json',request)
            if time.time() >= self.budget['train_deadline_epoch'] and not (self.root/'selected.json').exists():
                raise TimeoutError('Original training window exhausted')
            if shutil.disk_usage(self.root).free < 14*2**30: raise RuntimeError('Need 14 GiB free')
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
            if len(gpu.strip().splitlines())!=1 or 'RTX 6000 Ada' not in gpu or int(gpu.split(',')[-1]) < 45000:
                raise ValueError('Expected one RTX6000 Ada 48GB')
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip()
            if active: raise RuntimeError('GPU busy; no jobs will be terminated')
            code={str(p.relative_to(HERE.parent)):sha(p) for p in HERE.parent.rglob('*')
                if p.is_file() and p.suffix in ('.py','.json','.jinja','.sh') and '__pycache__' not in p.parts}
            immutable_json(self.root/'identity.json',dict(recipe='osw_pilot_v1',code=code,config=CONFIG,request=request))
            for role,python in [('train',self.a.train_python),('eval',self.a.eval_python),('data',self.a.data_python)]:
                packages=subprocess.check_output([python,'-m','pip','freeze'],text=True,timeout=30).splitlines()
                immutable_json(self.root/f'packages_{role}.json',sorted(packages))
            self.phase('prepare',self.a.train_python,self.budget['started_epoch']+240)
            verify_files(self.root,read(self.root/'data/complete.json'))
            verify_files(Path('/'),read(self.root/'source.json')['absolute_files'])
            self.phase('fetch',self.a.data_python,self.budget['started_epoch']+300)
            self.phase('train',self.a.train_python,self.budget['train_deadline_epoch']+60)
            self.phase('merge',self.a.train_python,self.deadline-1050)
            self.phase('submit',self.a.eval_python,self.deadline-180)
            self.package()  # Deliverable exists before optional evaluation begins.
            if not self.stopped and self.deadline-time.time() > 660:
                try:
                    self.phase('evaluate',self.a.eval_python,self.deadline-90)
                    self.report['evaluation']='complete'
                except Exception as exc:
                    self.report['evaluation']='failed_or_budget_stopped'
                    self.report['evaluation_error']=str(exc)
            else: self.report['evaluation']='skipped_budget'
            self.report['status']='complete' if self.report['evaluation']=='complete' else 'submission_ready'
        except BaseException as exc:
            self.report.update(status='failed',error=str(exc))
            raise
        finally: self.save()
        print('SUBMISSION_READY',self.report['submission'],flush=True)
        print('ZIP_READY',self.report['zip'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--predecessor',type=Path,required=True)
    for role in ('train','eval','data'): p.add_argument('--'+role+'-python',required=True)
    Runner(p.parse_args()).run()
