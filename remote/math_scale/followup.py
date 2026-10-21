"""Wait for the original suite, then consume its remaining wall-clock budget."""
import argparse
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from shared import *
from followup_plan import build_plan,remaining_disk_bytes

# Unique module identity avoids collision with historical run.py modules.
spec=importlib.util.spec_from_file_location('scale_coordinator',HERE/'run.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def estimate(root,predecessor,name):
    exp=read(root/name/'experiment.json');chain=Path(exp['chain']);plan=read(chain/'stage2/schedule.json')
    latest=chain/'stage2/training/rft/latest.json';done=read(latest)['step'] if latest.exists() else 0
    target=exp['step'];remaining=max(0,target-done)
    tokens=max(0,plan['cumulative_tokens'][target-1]-(plan['cumulative_tokens'][done-1] if done else 0))
    logs=jsonl(chain/'stage2/training/rft/train.jsonl')
    if not logs:
        logs=jsonl(predecessor/f"chains/R{8 if exp['rank']==8 else 32}/stage2/training/rft/train.jsonl")
    throughput=sum(x['teacher_tokens_this_update'] for x in logs)/sum(x['seconds'] for x in logs) if logs else 1200
    history=jsonl(root/'phase_times.jsonl')+jsonl(predecessor/'phase_times.jsonl')
    def seconds(phase,default):
        vals=[x['seconds'] for x in history if x['phase']==phase and x['exit_code']==0]
        return max(vals[-6:] or [default])
    trained=(root/'receipts'/f'{name}_train.json').exists()
    training=0 if trained else tokens/max(1,throughput)*1.3+90+remaining*.15
    merging=0 if (root/'receipts'/f'{name}_merge.json').exists() else seconds('merge',40)*1.25
    submission=0 if (root/'receipts'/f'{name}_submit.json').exists() else seconds('submit',270)*1.25+15
    evaluation=0 if (root/'receipts'/f'{name}_evaluate.json').exists() else seconds('evaluate',400)*1.25
    return dict(remaining_updates=remaining,remaining_teacher_tokens=tokens,measured_teacher_tokens_per_second=throughput,
        training_seconds=training,merge_seconds=merging,submission_seconds=submission,evaluation_seconds=evaluation,
        required_seconds=training+merging+submission+evaluation+120,
        policy='Observed throughput + 30% training margin, 25% inference margin, startup/checkpoint/finalization allowance; no score input')


class Followup(base.Runner):
    def __init__(self,args):
        root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
        predecessor=args.source.resolve()
        if root==predecessor or root in predecessor.parents or predecessor in root.parents:raise ValueError('Use a separate sibling output')
        inherited=read(predecessor/'budget.json')
        immutable_json(root/'budget.json',inherited)
        immutable_json(root/'request.json',dict(predecessor=str(predecessor),same_deadline=True))
        immutable_json(root/'plan.json',build_plan())
        existed=(root/'suite_report.json').exists()
        super().__init__(args)
        self.phase_script=HERE/'followup_phases.py'
        self.plan=read(root/'plan.json');self.predecessor=predecessor;self.locks=[]
        if not existed:
            self.report=dict(status='waiting_predecessor',experiments={n:dict(status='pending',path=str(root/n)) for n in self.plan['execution_order']})
        self.save()

    def save(self):
        now=time.time();phases=jsonl(self.root/'phase_times.jsonl')
        stop=self.report.get('finished_epoch',now)
        elapsed=stop-self.budget['started_epoch']
        self.report.update(budget=self.budget,elapsed_wall_seconds=elapsed,remaining_seconds=max(0,self.deadline-now))
        summaries={}
        for name,item in self.report['experiments'].items():
            item['phase_seconds']={p:sum(x['seconds'] for x in phases if x['name']==name and x['phase']==p)
                for p in ('train','merge','fetch','submit','evaluate','cleanup')}
            result=self.root/name/'report.json'
            if result.exists():
                r=read(result);item.update(metrics=r['metrics'],comparisons=r['comparisons'],training=r['training'])
        for chain in getattr(self,'plan',{}).get('chains',[]):
            logs=jsonl(self.root/'chains'/chain['key']/'stage2/training/rft/train.jsonl')
            summaries[chain['key']]=dict(update_attempt_seconds=sum(x['seconds'] for x in logs),
                peak_allocated_vram_bytes=max([x['max_vram'] for x in logs] or [0]))
        write(self.root/'suite_report.json',self.report)
        write(self.root/'runtime_summary.json',dict(budget=self.budget,elapsed_wall_seconds=elapsed,
            followup_phase_seconds=sum(x['seconds'] for x in phases),phases=phases,training=summaries,
            peak_device_vram_bytes=max([x['peak_device_vram_bytes'] for x in phases] or [0]),status=self.report['status']))

    def wait_predecessor(self):
        workspace=self.predecessor.parent.parent
        paths=[workspace/'.launch.lock',workspace/'.math_training.lock']
        last_print=0
        while True:
            if self.stopped:raise RuntimeError('Queue stopped by signal')
            if time.time()>=self.deadline-180:raise base.BudgetStop('Original deadline reached while waiting')
            status=read(self.predecessor/'suite_report.json')['status']
            if status in ('stopped_resumable','budget_stopped'):
                raise RuntimeError('Preceding suite stopped; no subsequent training permitted: '+status)
            if status=='complete':
                files=[p.open('a') for p in paths]
                try:
                    for stream in files:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    for stream in files:stream.close()
                else:
                    self.locks=files
                    print('PREDECESSOR_COMPLETE_LOCKS_ACQUIRED',flush=True);return
            if time.time()-last_print>=60:
                print('FOLLOWUP_QUEUED waiting_for='+str(self.predecessor)+' remaining_seconds='+str(round(self.deadline-time.time())),flush=True)
                self.save();last_print=time.time()
            time.sleep(5)

    def execute_one(self,name):
        root=self.root;item=self.report['experiments'][name]
        item['status']='running';self.save()
        self.phase(name,'train',sys.executable,[f'{name}/checkpoint',f'{name}/training_complete.json'])
        self.phase(name,'merge',sys.executable,[f'{name}/model_manifest'])
        self.phase(name,'fetch',self.a.data_python,[f'{name}/submission/question_identity.json'])
        self.phase(name,'submit',self.a.eval_python,[f'{name}/submission/{n}' for n in ('submission.json','format_check.json','hashes.json','manifest.json','complete.json','inference_times.json')])
        item.update(status='submission_ready',submission=str(root/name/'submission/submission.json'),submission_sha256=sha(root/name/'submission/submission.json'))
        self.save();print('SUBMISSION_READY',name,item['submission'],flush=True)
        self.phase(name,'evaluate',self.a.eval_python,[f'{name}/report.json',f'{name}/question_changes.json',f'{name}/evaluations/{name}'])
        self.phase(name,'cleanup',sys.executable,[f'{name}/merged_released.json'])
        item['status']='complete';self.save()

    def run(self):
        root=self.root
        try:
            if self.report['status']=='complete':
                for receipt in (root/'receipts').glob('*.json'):verify_files(root,read(receipt)['files'])
                print('ALREADY_COMPLETE',root,flush=True);return
            self.wait_predecessor()
            self.report['status']='running';self.report.pop('error',None);self.save()
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            if active:raise RuntimeError('GPU has other compute processes')
            code={str(p.relative_to(HERE.parent)):sha(p) for folder in (HERE,CURRICULUM,LEGACY,HERE.parent/'rank64_12k')
                  for p in folder.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.jinja','.sh','.txt') and '__pycache__' not in p.parts}
            immutable_json(root/'identity.json',dict(recipe='math_scale_followup_v1',code=code,config=CONFIG,plan_sha256=sha(root/'plan.json')))
            for role,python in [('train',sys.executable),('eval',self.a.eval_python),('data',self.a.data_python)]:
                immutable_json(root/f'packages_{role}.json',sorted(subprocess.check_output([python,'-m','pip','freeze'],text=True,timeout=30).splitlines()))
            self.phase('suite','prepare',sys.executable,['data/complete.json'])
            # On resume verify frozen input hashes without mutating the predecessor.
            from followup_phases import prepare
            prepare(root,self.predecessor)
            for chain in self.plan['chains']:
                chain_skipped=False
                for entry in chain['experiments']:
                    name=entry['name'];item=self.report['experiments'][name]
                    if item['status']=='complete':
                        for receipt in (root/'receipts').glob(name+'_*.json'):verify_files(root,read(receipt)['files'])
                        continue
                    if chain_skipped or item['status'].startswith('skipped'):
                        item.update(status='skipped_dependency',reason='Previous checkpoint in this fixed chain did not fit the budget');chain_skipped=True;self.save();continue
                    prediction=estimate(root,self.predecessor,name)
                    prediction.update(remaining_budget_seconds=self.deadline-time.time(),required_disk_bytes=remaining_disk_bytes(root,chain),free_disk_bytes=shutil.disk_usage(root).free)
                    item['estimate']=prediction;self.save();print('BUDGET_ESTIMATE',name,prediction,flush=True)
                    if prediction['required_seconds']>prediction['remaining_budget_seconds']:
                        item.update(status='skipped_budget',reason='Cannot reserve training + complete submission + public evaluation');chain_skipped=True;self.save();continue
                    if prediction['free_disk_bytes']<prediction['required_disk_bytes']:
                        item.update(status='skipped_disk',reason='Preserve all old artifacts; insufficient room for this chain');chain_skipped=True;self.save();continue
                    self.execute_one(name)
            skipped=any(x['status'].startswith('skipped') for x in self.report['experiments'].values())
            self.report.update(status='budget_completed' if skipped else 'complete',finished_epoch=time.time())
        except base.BudgetStop as exc:
            self.report.update(status='budget_stopped',error=str(exc),finished_epoch=time.time())
            for item in self.report['experiments'].values():
                if item['status']=='pending':item.update(status='skipped_budget',reason=str(exc))
            print('BUDGET_STOP',str(exc),flush=True)
        except BaseException as exc:
            self.report.update(status='stopped_resumable',error=str(exc))
            print('STOPPED_RESUMABLE',str(exc),flush=True);raise
        finally:
            self.save()
            for stream in self.locks:stream.close()
        print('FOLLOWUP_END',self.report['status'],root/'suite_report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--eval-python',required=True);p.add_argument('--data-python',required=True)
    Followup(p.parse_args()).run()
