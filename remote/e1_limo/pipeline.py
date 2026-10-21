"""Exactly one E1 condition; freeze dev winner before opening independent scores."""
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
from e1_common import *
from suite import Suite, link_dir, PhaseFailure
from accuracy_suite import AccuracySuite
from suite_utils import summarize, candidate_report, LEGACY_CODE


class E1(AccuracySuite):
    def __init__(self,args):
        args.budget_minutes=0
        args.config_overrides=dict(suite_kind='e1_limo_v1',dtype='bfloat16',context=8192,
            generations=1,eval_chunk=8,microbatch=1,effective_batch=16,
            auto_oom_microbatch=True,published_schedule=True,sft_initialization='continue',
            learning_rate=5e-6,train_seed=42,seed=42,objective='rft',checkpoint_every=10**9,checkpoint_at_milestones=True,
            e1_code={p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.sh')},
            duplicate_code_sha=sha(HERE.parent/'public_eval_diagnostics/data.py'),
            official_prompt_sha=sha(HERE.parent/'rank64_12k/official_prompt.py'),
            independent_sha256=INDEPENDENT_SHA,selection_policy=POLICY)
        Suite.__init__(self,args)
        self.deadline=self.cutoff=float('inf')
        self.data=self.root/'data'; self.baselines={}
        immutable_json(self.root/'policy.json',POLICY)
        self.summary.update(selection_policy=POLICY)
        self.summary_path=self.root/'report.json'
        if self.summary_path.exists(): self.summary=read(self.summary_path)

    def command(self,cmd,name):
        """Record device-level memory too: vLLM workers allocate outside this process."""
        log=self.root/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        print('PHASE',name,flush=True);self.summary['active_phase']=name;self.save()
        started=time.monotonic();peak=0
        with log.open('a') as f:
            self.active=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while self.active.poll() is None:
                    try:
                        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used',
                            '--format=csv,noheader,nounits'],text=True,timeout=5)
                        peak=max(peak,max(int(v) for v in memory.splitlines()))
                    except (OSError,ValueError,subprocess.SubprocessError): pass
                    try: self.active.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
                code=self.active.returncode
            except KeyboardInterrupt:
                self.terminate();raise
            finally:
                code=self.active.returncode
                self.active=None
                record=dict(phase=name,seconds=time.monotonic()-started,exit_code=code,
                            peak_device_vram_mib=peak)
                with (self.root/'phase_times.jsonl').open('a') as timings:
                    timings.write(json.dumps(record)+'\n')
        if code==75: raise KeyboardInterrupt
        if code: raise PhaseFailure(f'{name} failed ({code}); inspect {log}')
        return record['seconds']

    def terminate(self):
        if self.active is None:return
        try:os.killpg(self.active.pid,signal.SIGTERM)
        except ProcessLookupError:return
        # A long 8192-token accumulation must finish before its state can be saved.
        # Do not inherit Suite's 25-second SIGKILL fallback.
        print('STOP_REQUESTED waiting for worker checkpoint boundary',flush=True)
        self.active.wait()

    def phase(self, context, phase, branch='b0', purpose='eval', count=128, tag=None):
        if phase != 'generate':
            return Suite.phase(self,context,phase,branch,purpose,count,tag)
        ctx=read(context);root=Path(ctx['run'])
        rows=read(root/('final_rows.json' if purpose=='final' else 'eval_rows.json'))[:count]
        output=root/f'generations/{branch}_{purpose}.json'
        identity=dict(config=ctx['config'],rows_sha256=digest(rows),
            model_sha256=sha(root/f'models/{branch}/artifacts.json'),
            code=self.code,packages=self.packages['eval'])
        immutable_json(root/'evaluation_identity.json',identity)
        if output.exists() and set(read(output))=={r['id'] for r in rows}:
            summarize(read(output),rows);return 0.
        # Full equality is deliberately strict; no prior scores enter checkpoint selection.
        for candidate in sorted((self.workspace/'suites').glob('*/**/evaluation_identity.json')):
            if candidate.parent==root or read(candidate)!=identity:continue
            for previous in sorted((candidate.parent/'generations').glob('*_eval.json')):
                values=read(previous)
                if set(values)=={r['id'] for r in rows} and all(len(v)==1 for v in values.values()):
                    summarize(values,rows)
                    write(output,values)
                    immutable_json(root/'evaluation_reuse.json',dict(source=str(previous),sha256=sha(previous)))
                    return 0.
        immutable_json(root/'evaluation_cache_miss.json',dict(reason='No complete evaluation with identical model, rows, config, code and packages'))
        return Suite.phase(self,context,phase,branch,purpose,count,tag)

    def initialize(self):
        self.command([self.args.train_python,str(self.snapshot/'fetch_inputs.py'),
                      '--destination',str(self.inputs)],'verify_inputs')
        for item in read(self.snapshot/'input_manifest.json')['files']:
            if sha(self.inputs/item['path'])!=item['sha256']:raise ValueError('Input SHA mismatch')
        self.packages={}
        for label,python in [('train',self.args.train_python),('eval',self.args.eval_python)]:
            packages=json.loads(subprocess.check_output([python,'-c',
                'import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],text=True))
            immutable_json(self.root/f'packages_{label}.json',packages);self.packages[label]=packages
        independent=read(self.args.independent);validate_independent(independent)
        immutable_json(self.root/'independent_rows.json',independent)
        self.final=independent
        split=read(self.inputs/'split.json')
        if digest(split['split'])!=split['split_sha256']:raise ValueError('Dev split changed')
        self.dev=split['split']['dev']
        if len(self.dev)!=400:raise ValueError('Expected dev400')
        # Reuse only the established run03 merged parents, not prior candidate weights.
        from worker import check_model
        if self.source:
            source_ctx=read(self.source/'context.json')
            for key in ('base_model','base_revision','stage2_sha256'):
                if source_ctx['config'][key]!=self.cfg[key]:raise ValueError('Parent identity mismatch')
            if sha(self.source/'code/input_manifest.json')!=sha(self.snapshot/'input_manifest.json'):
                raise ValueError('Parent adapters differ')
            if sha(self.source/'code/worker.py') not in (LEGACY_CODE['worker.py'],self.code['worker.py']):
                raise ValueError('Unrecognized parent builder')
            for name in ('stage1','b0'):
                if not check_model(self.source/'models'/name):raise ValueError('Missing parent '+name)
                link_dir(self.source/'models'/name,self.shared/'models'/name)
            immutable_json(self.root/'parent_identity.json',{n:sha(self.shared/'models'/n/'artifacts.json')
                                                           for n in ('stage1','b0')})
        ctx=self.context(self.shared,self.cfg)
        self.phase(ctx,'preflight');self.phase(ctx,'merge')
        datactx=self.context(self.data,self.cfg)
        if not (self.data/'data_report.json').exists():
            self.command([self.args.train_python,str(HERE/'prepare.py'),'--context',str(datactx),
                          '--independent',str(self.root/'independent_rows.json')],'prepare')
        self.plan=read(self.data/'schedule.json');self.summary['data']=read(self.data/'data_report.json')
        immutable_json(self.root/'prepared_identity.json',{n:sha(self.data/n)
            for n in ('pairs.json','schedule.json','data_report.json')})
        self.baseline('dev',self.dev)
        self.summary['baseline']=summarize(self.baselines['dev'],self.dev);self.save()

    def training(self):
        root=self.root/'training_trials/E1';self.share_models(root)
        # Links avoid repeated serialization of the large token cache.
        for n in ('pairs.json','schedule.json'):link_dir(self.data/n,root/n)
        ctx=self.context(root,self.cfg)
        contract=dict(config=self.cfg,pairs_sha256=sha(root/'pairs.json'),schedule_sha256=sha(root/'schedule.json'),
            worker_sha256=self.code['worker.py'],train_packages=self.packages['train'],
            parent_sha256=sha(self.shared/'models/stage1/artifacts.json'))
        immutable_json(root/'reuse_contract.json',contract)
        from worker import check_model
        audit=[]
        if not (root/'training').exists():
            for previous in sorted((self.workspace/'suites').glob('*/training_trials/*/schedule.json')):
                other=previous.parent
                if other==root:continue
                other_contract=other/'reuse_contract.json'
                match=other_contract.exists() and read(other_contract)==contract
                complete=match and (other/'training/rft/complete.json').exists() and check_model(other/'training/rft/final')
                if complete:
                    complete=all(check_model(other/f'training/rft/milestones/step-{s:05d}')
                                 for s in self.plan['milestone_steps'].values())
                audit.append(dict(path=str(other),reusable=bool(complete),reason=(
                    'Exact model/data/schedule/code/package contract matches' if complete else
                    'No matching E1 model/data/schedule/code/package contract; legacy GSM/LIMO 50:50 differs from PRM/GSM/LIMO 60:20:20')))
                if complete:
                    link_dir(other/'training',root/'training');break
            write(self.root/'reuse_audit.json',audit)
        if not check_model(root/'training/rft/final'):
            benchmark=self.root/'benchmark';self.share_models(benchmark)
            for n in ('pairs.json','schedule.json'):link_dir(self.data/n,benchmark/n)
            if not (self.root/'benchmark.json').exists():
                # One memory probe and one short timing probe; no backend or batch sweep.
                for mode,steps in [('longest',1),('representative',3)]:
                    bc=self.context(benchmark,dict(self.cfg,benchmark_mode=mode,benchmark_steps=steps),mode+'.json')
                    self.phase(bc,'bench',branch='rft',tag='benchmark_'+mode)
                profile=read(benchmark/'timings/bench_rft.json')
                profile['estimated_training_seconds']=profile['seconds_per_update']*len(self.plan['groups'])
                immutable_json(self.root/'benchmark.json',profile)
            print('ESTIMATED_TRAINING_MINUTES',round(read(self.root/'benchmark.json')['estimated_training_seconds']/60,1),flush=True)
            self.phase(ctx,'train',branch='rft')
        self.summary['training']=dict(path=str(root),status='complete',plan=self.plan)
        self.save()
        for name,step in self.plan['milestone_steps'].items():
            self.evaluate(name,root/f'training/rft/milestones/step-{step:05d}',
                dict(kind='E1_NLL',supervised_tokens=self.plan['cumulative_tokens'][step-1]),'dev')

    def reports(self):
        results=[('dev','A',self.summary['baseline'],None)]
        results += [('dev',n,v['metrics'],v['comparison']) for n,v in self.summary['trials'].items()]
        results += [('independent','A',self.summary['independent']['baseline'],None)]
        v=self.summary['independent']['candidate']
        results += [('independent',self.summary['selection']['candidate'],v['metrics'],v['comparison'])]
        fields=['split','model','n','correct','accuracy','gsm8k_accuracy','prm800k_accuracy',
                'mean_output_tokens','repetition_rate','normal_finish_rate','delta_accuracy','ci95_low','ci95_high']
        with (self.root/'comparison.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for split,name,m,c in results:
                row=dict(split=split,model=name,**{k:m[k] for k in ('n','correct','accuracy','mean_output_tokens','repetition_rate')},normal_finish_rate=1-m['non_stop_rate'])
                for ds in ('gsm8k','prm800k'):
                    d=m['by_dataset'][ds];row[ds+'_accuracy']=d['correct']/d['n']
                if c:row.update(delta_accuracy=c['delta_accuracy'],ci95_low=c['paired_bootstrap_95'][0],ci95_high=c['paired_bootstrap_95'][1])
                writer.writerow(row)
        with (self.root/'question_changes.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['split','model','id','dataset','baseline_correct','candidate_correct','delta'])
            for split,rows in [('dev',self.dev),('final',self.final)]:
                base=self.baseline(split,rows)
                table=self.summary['trials'] if split=='dev' else self.summary['final_evaluations']
                for name,result in table.items():
                    outputs=read(Path(result['path'])/'generations/rft_eval.json')
                    for r in rows:
                        a=int(base[r['id']][0]['grade']['correct']);b=int(outputs[r['id']][0]['grade']['correct'])
                        writer.writerow(['independent' if split=='final' else split,name,r['id'],r['dataset'],a,b,b-a])
        self.summary['phase_times']=jsonl(self.root/'phase_times.jsonl')
        self.summary['training_phase_seconds']=sum(r['seconds'] for r in self.summary['phase_times'] if r['phase']=='E1_train_rft_eval_128')
        self.summary['evaluation_phase_seconds']=sum(r['seconds'] for r in self.summary['phase_times'] if 'generate' in r['phase'] or r['phase'].startswith('baseline_'))
        self.summary['resource_note']='Phase durations include process/model startup; memory is sampled device usage, training logs also record allocator peak. Reused phases report original provenance.'
        self.summary['peak_device_vram_mib']=max(r['peak_device_vram_mib'] for r in self.summary['phase_times'])
        self.save()
        (self.root/'README.md').write_text('# E1 LIMO混合SFT\n\n'+
            '採用モデル: '+self.summary['adoption']['model']+'\n\n'+
            '選択規則はpolicy.json。独立400問はdevで固定した1候補とAのみ評価。\n'+
            'comparison.csv / question_changes.csv / report.json / submission/submission.json を参照。\n'+
            '教師応答を切断せず閾値を初めて超えた例で保存。実トークン数はdata/data_report.json。\n'+
            'paired bootstrapは問題単位2,000回、seed42。探索の多重性や実行ばらつきは含まない。\n')

    def run(self):
        if self.summary.get('status')=='complete':
            done=self.root/'submission/complete.json'
            if sha(self.root/'submission/submission.json')!=read(done)['sha256']:raise ValueError('Submission changed')
            print('ALREADY_COMPLETE',self.root,flush=True);return
        self.summary['status']='running';self.initialize()
        if not (self.root/'selection.json').exists():
            self.training()
            immutable_json(self.root/'selection.json',dict(candidate=choose(self.summary['trials']),policy=POLICY,
                selected_before_independent=True,dev_metrics_sha256=digest(self.summary['trials'])))
        self.summary['selection']=read(self.root/'selection.json')
        name=self.summary['selection']['candidate'];trial=self.summary['trials'][name]
        # Only now open independent evaluation; each generation resumes by question ID.
        base=self.baseline('final',self.final)
        final=self.evaluate(name,Path(trial['adapter']),trial['recipe'],'final')
        self.summary['independent']=dict(baseline=summarize(base,self.final),candidate=final)
        accepted=adopt(self.summary['baseline'],trial,final)
        decision=dict(model=name if accepted else 'A',adapter=trial['adapter'] if accepted else str(self.inputs/'stage2'),
                      policy=POLICY,independent_comparison=final['comparison'])
        immutable_json(self.root/'adoption.json',decision);self.summary['adoption']=decision;self.save()
        submission=self.root/'submission';submission.mkdir(exist_ok=True)
        if not (submission/'complete.json').exists():
            if accepted:
                self.phase(Path(trial['path'])/'context.json','merge',branch='rft',tag='submission_merge')
            self.command([self.args.data_python,str(HERE/'submit.py'),'fetch','--root',str(self.root)],'submission_fetch')
            self.command([self.args.eval_python,str(HERE/'submit.py'),'generate','--root',str(self.root)],'submission_generate')
        self.reports();self.summary['status']='complete';self.save()
        print('COMPLETE',self.root,'SUBMISSION',submission/'submission.json',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True);p.add_argument('--run-id',required=True)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--independent',type=Path,required=True)
    p.add_argument('--train-python',default=sys.executable);p.add_argument('--eval-python',required=True)
    p.add_argument('--data-python',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.started_at=time.time();a.workspace=a.workspace.expanduser().resolve()
    a.independent=a.independent.expanduser().resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_id):p.error('Unsafe run ID')
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():p.error('BF16 CUDA GPU required')
    if '6000 Ada' not in torch.cuda.get_device_name():p.error('Expected RTX 6000 Ada')
    a.workspace.mkdir(parents=True,exist_ok=True)
    with (a.workspace/'.math_training.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('Another experiment owns GPU workspace')
        suite=E1(a)
        def stop(*_):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        write(suite.root/'pid.json',dict(pid=os.getpid(),started_at=a.started_at))
        try:suite.run()
        except KeyboardInterrupt:
            suite.terminate();suite.summary['status']='interrupted'
        except Exception as exc:
            suite.summary.update(status='failed',error=str(exc));raise
        finally:
            suite.summary['active_phase']=None;suite.save()
            write(suite.root/'status.json',dict(status=suite.summary['status'],ended_at=time.time()))

if __name__=='__main__':main()
