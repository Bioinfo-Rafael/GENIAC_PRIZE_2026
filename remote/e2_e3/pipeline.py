"""180 minute, resumable E2/E3 experiment; no E1 dependency."""
import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import time
from common import *
from suite import Suite,PhaseTimeout,PhaseFailure,link_dir
from suite_utils import summarize,candidate_report
from c_utils import make_plan

CAPS={'initialize':600,'synthesis':1500,'sft':900,'e2_eval':1200,'rl':2700,'e3_eval':1200,'final':2400}


class Experiments(Suite):
    def __init__(self,args):
        args.budget_minutes=180
        args.config_overrides=dict(suite_kind='e2_e3_v1',dtype='bfloat16',context=4096,generations=1,
            eval_chunk=16,microbatch=1,effective_batch=16,auto_oom_microbatch=True,
            published_schedule=True,sft_initialization='continue',learning_rate=5e-6,objective='rft',
            train_seed=42,seed=42,checkpoint_every=1,online_rounds=32,online_prompts=4,
            online_generations=8,online_lr=1e-6,dr_normalization=True,
            experiment_code={p.name:sha(p) for p in HERE.glob('*.py')},selection_policy=POLICY,
            independent_sha256=INDEPENDENT_SHA,
            dependency_code={str(p.relative_to(HERE.parent)):sha(p) for p in
                (HERE.parent/'public_eval_diagnostics/data.py',HERE.parent/'rank64_12k/official_prompt.py')})
        super().__init__(args)
        clock=self.root/'clock.json'
        immutable_json(clock,read(clock) if clock.exists() else dict(started_at=args.started_at,deadline=args.started_at+10800))
        self.deadline=read(clock)['deadline'];self.bucket='initialize'
        self.summary.update(inputs=str(self.inputs),selection_policy=POLICY)
        self.summary.setdefault('spent',{});self.summary.setdefault('evaluations',{})
        self.summary.setdefault('recipes',{});self.summary.setdefault('failures',[])
        self.data=self.root/'data'

    def limit(self):
        # Training/evaluation cannot consume the final 40 minutes of wall clock.
        reserve=0 if self.bucket=='final' else 2400
        return min(self.deadline-reserve,time.time()+max(0,CAPS[self.bucket]-self.summary['spent'].get(self.bucket,0)))

    def command(self,cmd,name):
        stop=self.limit()-30
        if time.time()>=stop:raise PhaseTimeout('Budget exhausted: '+self.bucket)
        path=self.root/'logs'/(name+'.log');path.parent.mkdir(exist_ok=True)
        self.summary['active_phase']=name;self.save();print('PHASE',name,flush=True)
        start=time.monotonic();peak=0;code=None;failure=None
        with path.open('a') as stream:
            self.active=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while self.active.poll() is None:
                    if time.time()>=stop:raise PhaseTimeout('Deadline reached: '+name)
                    try:
                        mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=2)
                        peak=max(peak,max(map(int,mem.splitlines())))
                    except (OSError,ValueError,subprocess.SubprocessError):pass
                    write(self.root/'status.json',dict(status='running',active_phase=name,bucket=self.bucket,
                        remaining_seconds=max(0,self.deadline-time.time()),
                        phase_remaining_seconds=max(0,stop-time.time()),peak_device_vram_mib=peak))
                    try:self.active.wait(timeout=min(2,max(.1,stop-time.time())))
                    except subprocess.TimeoutExpired:pass
                code=self.active.returncode
                if code:raise PhaseFailure(f'{name} exit={code}; see {path}')
            except BaseException as exc:
                failure=str(exc);self.terminate();raise
            finally:
                self.active=None
                duration=time.monotonic()-start
                self.summary['spent'][self.bucket]=self.summary['spent'].get(self.bucket,0)+duration
                record=dict(phase=name,bucket=self.bucket,seconds=duration,exit_code=code,error=failure,
                    peak_device_vram_mib=peak,remaining_seconds=max(0,self.deadline-time.time()))
                with (self.root/'phase_times.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                self.save()
        return duration

    def attempt(self,name,fn):
        try:return fn()
        except (PhaseTimeout,PhaseFailure) as exc:
            self.summary['failures'].append(dict(phase=name,reason=str(exc)))
            self.save();print('SKIP',name,str(exc),flush=True)
            return None

    def initialize(self):
        self.command([self.args.train_python,str(self.snapshot/'fetch_inputs.py'),'--destination',str(self.inputs)],'inputs')
        for item in read(self.snapshot/'input_manifest.json')['files']:
            if sha(self.inputs/item['path'])!=item['sha256']:raise ValueError('Input hash mismatch')
        if read(self.inputs/'stage2/adapter_config.json')['r']!=8:raise ValueError('Expected run03 rank8')
        split=read(self.inputs/'split.json')
        if digest(split['split'])!=split['split_sha256']:raise ValueError('Dev split changed')
        self.dev=split['split']['dev'];self.independent=read(self.args.independent)
        if len(self.dev)!=400 or digest(self.independent)!=INDEPENDENT_SHA:raise ValueError('Frozen evaluation split mismatch')
        immutable_json(self.root/'independent_rows.json',self.independent)
        from worker import check_model
        ctx=read(self.source/'context.json')
        for k in ('base_model','base_revision','stage2_sha256'):
            if ctx['config'][k]!=self.cfg[k]:raise ValueError('Parent mismatch: '+k)
        if sha(self.source/'code/input_manifest.json')!=sha(self.snapshot/'input_manifest.json'):raise ValueError('Parent inputs changed')
        # Check the established builder as well as file manifests.
        from suite_utils import LEGACY_CODE
        if sha(self.source/'code/worker.py') not in (LEGACY_CODE['worker.py'],self.code['worker.py']):raise ValueError('Unrecognized parent builder')
        for n in ('stage1','b0'):
            if not check_model(self.source/'models'/n):raise ValueError('Invalid parent '+n)
            link_dir(self.source/'models'/n,self.shared/'models'/n)
        immutable_json(self.root/'parent_identity.json',{n:sha(self.shared/'models'/n/'artifacts.json') for n in ('stage1','b0')})
        for label,python in [('train',self.args.train_python),('eval',self.args.eval_python),('data',self.args.data_python)]:
            packages=json.loads(subprocess.check_output([python,'-c','import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],text=True))
            immutable_json(self.root/f'packages_{label}.json',packages)
        if shutil.disk_usage(self.workspace).free < 45*2**30:
            raise ValueError('Need 45 GiB free for isolated candidate exports/checkpoints')
        self.phase(self.context(self.shared,self.cfg),'preflight')
        self.save()

    def secure_fallback(self):
        from submit import reuse,check
        dest=self.root/'fallback_submission'
        if (dest/'complete.json').exists():
            if sha(dest/'submission.json')!=read(dest/'complete.json')['sha256']:raise ValueError('Fallback changed')
            check(dest/'submission.json');return
        paths=[self.args.baseline_submission] if self.args.baseline_submission else []
        paths+=list((self.workspace/'submissions').glob('*'))
        for source in paths:
            if reuse(source,dest,self.shared/'models/b0',self.cfg['stage2_sha256']):return
        self.bucket='final'
        self.submit('A')

    def submit(self,name):
        dest=self.root/('fallback_submission' if name=='A' else 'candidate_submission')
        if (dest/'complete.json').exists():return
        self.command([self.args.data_python,str(HERE/'submit.py'),'fetch','--root',str(self.root),'--name',name],'submission_fetch_'+name)
        self.command([self.args.eval_python,str(HERE/'submit.py'),'generate','--root',str(self.root),'--name',name],'submission_generate_'+name)

    def evaluate(self,name,adapter,split,count):
        rows=self.dev if split=='dev' else self.independent
        root=self.root/'evaluations'/split/name;self.share_models(root)
        immutable_json(root/'eval_rows.json',rows)
        branch='b0' if name=='A' else 'rft'
        ctx=self.context(root,self.cfg)
        if name!='A':
            link_dir(adapter,root/'training/rft/final')
            self.phase(ctx,'merge',branch='rft',tag=split+'_'+name+'_merge')
        output=root/f'generations/{branch}_eval.json'
        saved=read(output) if output.exists() else {}
        pending=count-len(saved)
        if pending>0:
            if not saved:
                self.phase(ctx,'generate',branch=branch,count=8,tag=split+'_'+name+'_speed8')
            times=read(root/f'timings/{branch}_eval.json')
            per=max(t['seconds']/t['questions'] for t in times)
            remaining=max(0,count-len(read(output)))
            estimate=remaining*per*1.5+60
            write(root/'speed_estimate.json',dict(seconds_per_question=per,target=count,estimated_remaining=estimate))
            if time.time()+estimate>self.limit()-30:raise PhaseTimeout('Measured evaluation does not fit: '+name+f' {estimate:.0f}s')
            self.phase(ctx,'generate',branch=branch,count=count,tag=split+'_'+name+'_'+str(count))
        saved=read(output)
        outputs={r['id']:saved[r['id']] for r in rows[:count]}
        if any(v[0]['grade']['grader_error'] or v[0]['grade']['gold_unparseable'] for v in outputs.values()):
            raise PhaseFailure('Invalid grader/gold result; refusing candidate selection')
        metrics=summarize(outputs,rows[:count]);metrics['normal_finish_rate']=1-metrics['non_stop_rate']
        for v in metrics['by_dataset'].values():v['accuracy']=v['correct']/v['n']
        result=dict(status='complete',metrics=metrics,path=str(root),adapter=str(adapter) if adapter else None)
        if name!='A':
            baseline=read(self.root/f'evaluations/{split}/A/generations/b0_eval.json')
            baseline={r['id']:baseline[r['id']] for r in rows[:count]}
            result.update(candidate_report(baseline,outputs,rows[:count]))
            result['metrics']=metrics
            result['changes']=[dict(id=r['id'],dataset=r['dataset'],baseline_correct=baseline[r['id']][0]['grade']['correct'],
                candidate_correct=outputs[r['id']][0]['grade']['correct']) for r in rows[:count]]
        table=self.summary['evaluations'].setdefault(split if count==400 else split+str(count),{})
        table[name]=result;self.save();return result

    def screen(self,name,adapter):
        self.evaluate('A',None,'dev',128)
        quick=self.evaluate(name,adapter,'dev',128)
        if quick['comparison']['paired_bootstrap_95'][1]<0:
            self.summary.setdefault('excluded',{})[name]='quick128 CI upper bound < 0';self.save();return None
        self.evaluate('A',None,'dev',400)
        return self.evaluate(name,adapter,'dev',400)

    def e2(self):
        self.bucket='synthesis'
        ctx=self.context(self.data,self.cfg)
        if not (self.data/'sources.json').exists():
            self.command([self.args.train_python,str(HERE/'data.py'),'prepare','--context',str(ctx),
                '--independent',str(self.root/'independent_rows.json')],'prepare_public')
        synth=self.data/'synthesis_context.json'
        immutable_json(synth,dict(read(ctx),baseline_model=str(self.shared/'models/b0')))
        if not (self.data/'synthesis_complete.json').exists():
            self.command([self.args.eval_python,str(HERE/'data.py'),'synthesize','--context',str(synth),
                '--deadline',str(self.limit()-40)],'synthesis')
        pairs=read(self.data/'pairs.json')
        if len(pairs)<32:
            self.summary['e2_skipped']='Fewer than 32 mechanically verified examples';self.save();return
        self.bucket='sft';root=self.root/'training/E2';self.share_models(root)
        immutable_json(root/'pairs.json',pairs)
        order=sorted(range(len(pairs)),key=lambda i:digest([42,pairs[i]['id']]))
        immutable_json(root/'schedule.json',make_plan(pairs,[order[i:i+16] for i in range(0,len(order),16)]))
        ctx=self.context(root,self.cfg)
        from worker import check_model
        if not check_model(root/'training/rft/final'):
            bench=self.root/'benchmark';self.share_models(bench)
            for n in ('pairs.json','schedule.json'):link_dir(root/n,bench/n)
            bc=self.context(bench,dict(self.cfg,benchmark_steps=1,benchmark_mode='longest'))
            if not (bench/'timings/bench_rft.json').exists():self.phase(bc,'bench',branch='rft')
            profile=read(bench/'timings/bench_rft.json')
            latest=root/'training/rft/latest.json'
            completed=read(latest)['step'] if latest.exists() else 0
            estimate=profile['seconds_per_update']*(len(read(root/'schedule.json')['groups'])-completed)*1.5+60
            write(root/'estimate.json',dict(estimated_seconds=estimate,policy='one full epoch or skip; no silent truncation'))
            if time.time()+estimate>self.limit()-30:raise PhaseTimeout('E2 full epoch cannot fit measured 15-minute budget')
            self.phase(ctx,'train',branch='rft')
        self.summary['recipes']['E2']=dict(kind='format SFT',examples=len(pairs),epochs=1,lr=5e-6,rank=8,parent='run03')
        self.save();self.bucket='e2_eval';self.screen('E2',root/'training/rft/final')

    def e3(self):
        self.bucket='rl'
        # E3 also works when synthesis timed out after source preparation.
        if not (self.data/'online_rows.json').exists():
            self.command([self.args.train_python,str(HERE/'data.py'),'prepare','--context',str(self.context(self.data,self.cfg)),
                '--independent',str(self.root/'independent_rows.json')],'prepare_rl')
        root=self.root/'training/E3';self.share_models(root)
        immutable_json(root/'online_rows.json',read(self.data/'online_rows.json'))
        ctx=self.context(root,self.cfg) # No initial_adapter override: original run03 stage2.
        def train(n):
            self.bucket='rl'
            if (root/f'training/rft/milestones/step-{n:05d}/artifacts.json').exists():return
            self.command([self.args.train_python,str(self.snapshot/'online_worker.py'),'--context',str(ctx),
                '--until-round',str(n)],'E3_round_'+str(n))
        train(1)
        first=read(root/'training/rft/round_metrics/00001.json')
        completed=read(root/'training/rft/latest.json')['step']
        estimate=first['seconds']*max(0,16-completed)*1.5+60
        write(root/'speed_estimate.json',dict(first_round_seconds=first['seconds'],remaining16_seconds=estimate))
        if completed<16 and time.time()+estimate>self.limit()-30:raise PhaseTimeout('16 RL rounds do not fit measured 45-minute budget')
        train(16)
        self.summary['recipes']['E3-16']=dict(kind='mean-centered fixed-length REINFORCE',rounds=16,parent='run03',lr=1e-6)
        self.save();self.bucket='e3_eval'
        result=self.screen('E3-16',root/'training/rft/milestones/step-00016')
        base=self.summary['evaluations'].get('dev',{}).get('A')
        if result and base and result['metrics']['correct']>base['metrics']['correct']:
            self.bucket='rl'
            # Extra rounds are first to go; reserve full evaluation + final window.
            if time.time()+first['seconds']*16*1.5+1200>min(self.limit(),self.deadline-2400):
                self.summary['e3_extension']='skipped: insufficient measured time';self.save();return
            train(32)
            self.summary['recipes']['E3-32']=dict(kind='mean-centered fixed-length REINFORCE',rounds=32,parent='run03',lr=1e-6)
            self.save();self.bucket='e3_eval';self.screen('E3-32',root/'training/rft/milestones/step-00032')

    def finish(self):
        self.bucket='final'
        selection=self.root/'selection.json'
        if not selection.exists():
            self.attempt('baseline_dev400',lambda:self.evaluate('A',None,'dev',400))
            trials=self.summary['evaluations'].get('dev',{})
            name=choose(trials) if 'A' in trials else 'A'
            immutable_json(selection,dict(candidate=name,policy=POLICY,dev_sha256=digest(trials),selected_before_independent=True))
        name=read(selection)['candidate'];adopt='A';reason='baseline selected or independent improvement unconfirmed'
        if name!='A':
            trial=self.summary['evaluations']['dev'][name]
            base=self.attempt('independent_A',lambda:self.evaluate('A',None,'independent',400))
            candidate=self.attempt('independent_candidate',lambda:self.evaluate(name,Path(trial['adapter']),'independent',400)) if base else None
            if candidate and candidate['comparison']['paired_bootstrap_95'][0]>0:
                # Inference must finish too; otherwise the already secured run03 JSON wins.
                success=self.attempt('candidate_submission',lambda:(self.submit(name),True)[1])
                if success:adopt=name;reason='independent paired CI lower bound > 0'
        self.publish(adopt,name,reason)

    def publish(self,adopt,name,reason):
        decision=dict(model=adopt,selected_candidate=name,reason=reason)
        immutable_json(self.root/'adoption.json',decision)
        source=self.root/('fallback_submission' if adopt=='A' else 'candidate_submission')
        from submit import check
        if sha(source/'submission.json')!=read(source/'complete.json')['sha256']:raise ValueError('Submission changed')
        report=check(source/'submission.json')
        dest=self.root/'submission';dest.mkdir(exist_ok=True)
        for n in ('submission.json','complete.json','manifest.json','hashes.json'):shutil.copyfile(source/n,dest/n)
        write(dest/'format_check.json',report)
        with (self.root/'question_changes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=['split','model','id','dataset','baseline_correct','candidate_correct']);writer.writeheader()
            for split,results in self.summary['evaluations'].items():
                for model,result in results.items():
                    for row in result.get('changes',[]):writer.writerow(dict(split=split,model=model,**row))
        self.summary.update(status='complete' if not self.summary['failures'] else 'partial_with_submission',adoption=decision)
        self.save()

    def run(self):
        if (self.root/'submission/complete.json').exists():
            if sha(self.root/'submission/submission.json')!=read(self.root/'submission/complete.json')['sha256']:raise ValueError('Submission changed')
            print('ALREADY_COMPLETE',self.root,flush=True);return
        self.initialize();self.secure_fallback()
        if not (self.root/'selection.json').exists():
            self.attempt('E2',self.e2);self.attempt('E3',self.e3)
        self.finish()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--started-at',type=float)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--baseline-submission',type=Path);p.add_argument('--train-python',default=sys.executable)
    p.add_argument('--eval-python',required=True);p.add_argument('--data-python',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.started_at=a.started_at or time.time();a.workspace=a.workspace.expanduser().resolve()
    if not re.fullmatch(r'e2e3_[A-Za-z0-9_-]+',a.run_id):p.error('Run ID must start e2e3_')
    a.workspace.mkdir(parents=True,exist_ok=True)
    with (a.workspace/'.math_training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another experiment owns workspace; no job stopped')
        # Foreign GPU processes are checked before importing torch creates a CUDA context.
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if processes:p.error('GPU is busy; rerun after existing jobs finish')
        import torch
        if not torch.cuda.is_available() or '6000 Ada' not in torch.cuda.get_device_name():p.error('RTX 6000 Ada required')
        suite=Experiments(a)
        def stop(*_):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:suite.run()
        except KeyboardInterrupt:
            suite.terminate();suite.summary['status']='interrupted'
        except Exception as exc:
            suite.summary.update(status='failed',error=str(exc))
            # Preserve a valid fallback as the deliverable even on an unexpected phase error.
            if (suite.root/'fallback_submission/complete.json').exists():
                suite.summary['failures'].append(dict(phase='unexpected',reason=str(exc)))
                immutable_json(suite.root/'selection.json',read(suite.root/'selection.json') if (suite.root/'selection.json').exists() else dict(candidate='A',reason=str(exc)))
                suite.publish('A',read(suite.root/'selection.json')['candidate'],'unexpected failure; secured run03 fallback')
            else:raise
        finally:
            suite.summary['active_phase']=None;suite.save()
            write(suite.root/'status.json',dict(status=suite.summary['status'],remaining_seconds=max(0,suite.deadline-time.time())))
            print('STATUS',suite.summary['status'],'ARTIFACTS',suite.root,flush=True)

if __name__=='__main__':main()
