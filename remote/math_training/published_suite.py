"""GSM8K / GSM8K+LIMO x NLL / DFT from run03, on one GPU, within eight hours."""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from core import digest, jsonl, prepare, read, sha, write
from published_data import response_tokens, schedule
from suite import Suite, PhaseFailure, PhaseTimeout, link_dir, HERE
from suite_utils import LEGACY_CODE, candidate_report, eligible, immutable_json, select_best, summarize

# A/C first if the measured budget cannot accommodate all conditions.
SPECS = [('A_gsm_nll','rft',False), ('C_gsm_limo_nll','rft',True),
         ('B_gsm_dft','dft',False), ('D_gsm_limo_dft','dft',True)]


class PublishedSuite(Suite):
    def __init__(self,args):
        args.config_overrides = dict(context=8192, generations=1, eval_chunk=8,
            dtype=args.dtype, published_terminal=True, suite_kind='published_gsm_limo_v1')
        super().__init__(args)
        self.data = self.root/'published_data'
        self.summary['experiment'] = 'published GSM8K solutions vs GSM8K+LIMO; no self-generated training'
        self.summary['uncertainty_scope'] = 'Repeated-use dev400 is exploratory. Bootstrap excludes execution variability.'

    def initialize(self):
        self.cutoff = self.deadline-3600
        self.command([self.args.train_python,str(self.snapshot/'fetch_inputs.py'),
                      '--destination',str(self.inputs)],'fetch')
        for item in read(self.snapshot/'input_manifest.json')['files']:
            if sha(self.inputs/item['path']) != item['sha256']:
                raise ValueError('Input SHA mismatch: '+item['path'])
        split = read(self.inputs/'split.json')
        prepare(jsonl(self.inputs/'train.jsonl'),split,6000,self.cfg['seed'])
        self.dev, self.final = split['split']['dev'], split['split']['final']
        for label, python in [('train',self.args.train_python),('eval',self.args.eval_python)]:
            packages = json.loads(subprocess.check_output([python,'-c',
                'import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],text=True))
            immutable_json(self.root/f'packages_{label}.json',packages)
        if self.source:
            # Reuse weights only. Old generation caches are deliberately NOT imported.
            ctx = read(self.source/'context.json')
            for key in ('base_model','base_revision','stage2_sha256'):
                if ctx['config'][key] != self.cfg[key]:
                    raise ValueError('Source model identity differs: '+key)
            for name, expected in LEGACY_CODE.items():
                if sha(self.source/'code'/name) != expected or ctx['code'][name] != expected:
                    raise ValueError('Unrecognized source model builder: '+name)
            if sha(self.source/'code/input_manifest.json') != sha(self.snapshot/'input_manifest.json'):
                raise ValueError('Source adapter manifest differs')
            from worker import check_model
            manifests = {}
            for name in ('stage1','b0'):
                if not check_model(self.source/'models'/name):
                    raise ValueError('Missing source model: '+name)
                manifests[name] = sha(self.source/'models'/name/'artifacts.json')
                link_dir(self.source/'models'/name,self.shared/'models'/name)
            immutable_json(self.root/'source_models.json',{'path':str(self.source),'manifests':manifests})
        required_gib = 16 if self.source else 60
        if shutil.disk_usage(self.workspace).free < required_gib*2**30:
            raise ValueError(f'Need at least {required_gib} GiB free disk space')
        ctx = self.context(self.shared,self.cfg)
        immutable_json(self.shared/'eval_rows.json',self.dev)
        immutable_json(self.shared/'final_rows.json',self.final)
        self.phase(ctx,'preflight')
        self.phase(ctx,'merge')
        data_ctx = self.context(self.data,self.cfg)
        self.command([self.args.train_python,str(self.snapshot/'published_data.py'),
                      '--context',str(data_ctx)],'prepare_published')
        self.summary['data'] = read(self.data/'data_report.json')
        self.gsm, self.limo = read(self.data/'gsm8k_pairs.json'), read(self.data/'limo_pairs.json')
        if not self.gsm:
            raise ValueError('No eligible GSM8K training examples')
        self.phase(ctx,'generate',count=400)
        self.base_outputs = read(self.shared/'generations/b0_eval.json')
        self.summary['baseline'] = summarize(self.base_outputs,self.dev)
        self.summary['limo_dev_baseline'] = self.auxiliary(self.shared,'b0','dev')
        audit = self.root/'audit'
        audit_ctx = self.context(audit,self.cfg)
        self.share_models(audit)
        immutable_json(audit/'pairs.json',self.gsm[:32])
        if not (audit/'integrity.json').exists():
            self.phase(audit_ctx,'audit')
            self.phase(audit_ctx,'merge',branch='rft')
            self.phase(audit_ctx,'audit_weights')
            self.remove_merged(audit,'rft')
        if not read(audit/'integrity.json')['identical']:
            raise ValueError('No-update merged weights/configs changed')
        self.summary['audit'] = read(audit/'audit.json')
        self.save()

    def auxiliary(self, source, branch, split):
        rows = read(self.data/f'limo_{split}.json')
        if not rows:
            return {'n':0,'status':'empty_auxiliary_split'}
        root = source/f'limo_{split}'
        ctx = self.context(root,self.cfg)
        immutable_json(root/'eval_rows.json',rows)
        link_dir(source/'models'/branch,root/'models'/branch)
        self.phase(ctx,'generate',branch=branch,count=len(rows),tag=f'{source.name}_limo_{split}')
        return summarize(read(root/f'generations/{branch}_eval.json'),rows)

    def dataset(self,mixed):
        return self.gsm+self.limo if mixed else self.gsm

    def benchmark(self,name,objective,mixed,target):
        root = self.root/'benchmarks'/name
        self.share_models(root)
        rows = self.dataset(mixed)
        immutable_json(root/'pairs.json',rows)
        immutable_json(root/'schedule.json',schedule(rows,target,mixed=mixed))
        cfg = dict(self.cfg,objective=objective,published_schedule=True,published_mixed=mixed)
        result_file = root/'benchmark_choice.json'
        if result_file.exists():
            return read(result_file)
        timings = []
        for microbatch in (1,2,4):
            ctx = self.context(root,dict(cfg,microbatch=microbatch),f'mb{microbatch}.json')
            try:
                self.phase(ctx,'bench',branch='rft',tag=name+f'_bench_mb{microbatch}')
                timings.append(read(root/'timings/bench_rft.json'))
            except PhaseFailure as exc:
                write(root/f'mb{microbatch}_failure.json',{'error':str(exc)})
                if 'out of memory' not in str(exc).lower():
                    raise
                break
        if not timings:
            raise PhaseFailure('No microbatch fits the longest published examples')
        result = min(timings,key=lambda x:x['seconds_per_update'])
        immutable_json(result_file,result)
        write(root/'benchmarks.json',timings)
        return result

    def prepare_budget(self):
        path = self.root/'budget_plan.json'
        if path.exists():
            return read(path)
        specs = [s for s in SPECS if not s[2] or self.summary['data']['mixed_available']]
        for name,objective,mixed in SPECS:
            if mixed and not self.summary['data']['mixed_available']:
                self.summary['trials'][name] = {'status':'skipped','reason':'fewer than 100 eligible LIMO examples'}
        capacity = 3*sum(map(response_tokens,self.gsm))
        if any(s[2] for s in specs):
            capacity = min(capacity,6*sum(map(response_tokens,self.limo)))
        target = min(1_000_000,capacity)
        timings = {}
        for name,objective,mixed in specs:
            timings[name] = self.benchmark(name,objective,mixed,target)
        # Longest-sequence benchmarks are a conservative projection, not a speed claim.
        available = min(150*60,max(0,self.cutoff-time.time()-135*60))
        estimated = sum(t['updates']*t['seconds_per_update']*1.5 for t in timings.values())
        scale = min(1.,available/max(1,estimated))
        target = int(target*scale)
        if target < 10_000:
            raise PhaseTimeout('Insufficient measured budget for a meaningful four-condition comparison')
        plan = {'target_tokens':target,'capacity_tokens':capacity,'timings':timings,
                'specs':specs,'estimated_training_minutes':estimated*scale/60,
                'initial_target_tokens':min(1_000_000,capacity),
                'max_exposures':3,'selection_uses':'all quarter/half/full dev checkpoints; final remains held out'}
        immutable_json(path,plan)
        self.summary['budget_plan'] = plan
        self.save()
        return plan

    def evaluate_checkpoint(self,name,training,step,cfg,metadata):
        root = self.root/'evaluations'/name
        result = self.summary['trials'].get(name)
        if result and result.get('status')=='complete':
            return result
        self.share_models(root)
        link_dir(training/f'training/rft/milestones/step-{step:05d}',root/'training/rft/final')
        ctx = self.context(root,cfg)
        immutable_json(root/'eval_rows.json',self.dev)
        immutable_json(root/'final_rows.json',self.final)
        result = dict(metadata,status='running',path=str(root),branch='rft',step=step)
        self.summary['trials'][name] = result
        self.save()
        try:
            self.phase(ctx,'merge',branch='rft')
            self.phase(ctx,'generate',branch='rft',count=400)
            outputs = read(root/'generations/rft_eval.json')
            result.update(candidate_report(self.base_outputs,outputs,self.dev),status='complete')
            result['limo_dev'] = self.auxiliary(root,'rft','dev')
        except (PhaseFailure,PhaseTimeout) as exc:
            result.update(status='time_limited' if isinstance(exc,PhaseTimeout) else 'failed',reason=str(exc))
        finally:
            self.remove_merged(root,'rft')
            self.save()
        print('RESULT',name,result['status'],result.get('metrics',{}).get('accuracy'),flush=True)
        return result

    def train_condition(self,spec,budget,seed=42,only_milestone=None):
        name,objective,mixed = spec
        label = name if seed==42 else name+f'_seed{seed}'
        root = self.root/'training_trials'/label
        self.share_models(root)
        rows = self.dataset(mixed)
        immutable_json(root/'pairs.json',rows)
        plan = schedule(rows,budget['target_tokens'],seed,mixed=mixed)
        immutable_json(root/'schedule.json',plan)
        cfg = dict(self.cfg,objective=objective,published_schedule=True,published_mixed=mixed,
                   train_seed=seed,microbatch=budget['timings'][name]['microbatch'])
        ctx = self.context(root,cfg)
        record = {'objective':objective,'mixed':mixed,'train_seed':seed,'condition':name,
                  'status':'running','path':str(root),'schedule_sha256':sha(root/'schedule.json'),
                  'target_tokens':plan['target_tokens'],'actual_tokens':plan['actual_tokens'],
                  'tokens_by_source':plan['tokens_by_source'],'updates':len(plan['groups'])}
        self.summary.setdefault('training',{})[label] = record
        self.save()
        try:
            self.phase(ctx,'train',branch='rft',tag=label+'_train')
            record['status'] = 'complete'
        except (PhaseFailure,PhaseTimeout) as exc:
            record.update(status='time_limited' if isinstance(exc,PhaseTimeout) else 'failed',reason=str(exc))
        self.save()
        # A valid earlier milestone may still be evaluated after a later training failure.
        results = {}
        for milestone,step in plan['milestone_steps'].items():
            if only_milestone and milestone != only_milestone:
                continue
            if not (root/f'training/rft/milestones/step-{step:05d}/artifacts.json').exists():
                continue
            meta = dict(objective=objective,mixed=mixed,condition=name,train_seed=seed,
                        milestone=milestone,supervised_tokens=plan['cumulative_tokens'][step-1])
            results[milestone] = self.evaluate_checkpoint(label+'_'+milestone,root,step,cfg,meta)
        return results

    def repeat_dev(self,original,branch,name):
        root = self.root/'repeats'/name
        ctx = self.context(root,self.cfg)
        immutable_json(root/'eval_rows.json',self.dev)
        link_dir(original/'models'/branch,root/'models'/branch)
        self.phase(ctx,'generate',branch=branch,count=400,tag='repeat_'+name)
        return read(root/f'generations/{branch}_eval.json')

    def run(self):
        self.summary['status'] = 'running'
        self.initialize()
        if not (self.root/'selection.json').exists():
            budget = self.prepare_budget()
            self.cutoff = self.deadline-2*3600
            for spec in budget['specs']:
                self.train_condition(spec,budget)
            best = select_best(self.summary['baseline'],{k:v for k,v in self.summary['trials'].items()
                                                        if v.get('train_seed')==42})
            selected = None
            self.cutoff = self.deadline-3600
            if best:
                candidate = self.summary['trials'][best]
                spec = next(s for s in budget['specs'] if s[0]==candidate['condition'])
                replication = self.train_condition(spec,budget,seed=43,only_milestone=candidate['milestone'])
                result = replication.get(candidate['milestone'],{})
                confirmed = result.get('status')=='complete' and eligible(self.summary['baseline'],result['metrics'])
                self.summary['replication'] = {'candidate':best,'seed43_dev_gate':confirmed}
                if confirmed:
                    original = Path(candidate['path'])
                    try:
                        self.phase(original/'context.json','merge',branch='rft',tag='repeat_merge')
                        baseline_repeat = self.repeat_dev(self.shared,'b0','b0')
                        candidate_repeat = self.repeat_dev(original,'rft',best)
                        report = candidate_report(baseline_repeat,candidate_repeat,self.dev)
                        repeat_passed = eligible(summarize(baseline_repeat,self.dev),report['metrics'])
                        self.summary['repeat_dev'] = dict(report,passed=repeat_passed,
                            baseline=summarize(baseline_repeat,self.dev))
                        if repeat_passed:
                            selected = best
                    except (PhaseTimeout,PhaseFailure) as exc:
                        self.summary['repeat_dev'] = {'passed':False,'reason':str(exc)}
                    finally:
                        self.remove_merged(original,'rft')
            immutable_json(self.root/'selection.json',{'candidate':selected,
                'reason':'dev gate, seed43 and repeated inference passed' if selected else 'no confirmed candidate',
                'selected_before_final':True})
        self.final_evaluation()
        # Auxiliary test is opened only after selection is frozen.
        self.summary['limo_test'] = {'baseline':self.auxiliary(self.shared,'b0','test')}
        selected = read(self.root/'selection.json')['candidate']
        if selected:
            original = Path(self.summary['trials'][selected]['path'])
            # final_evaluation leaves this merged candidate available.
            self.summary['limo_test']['candidate'] = self.auxiliary(original,'rft','test')
            self.remove_merged(original,'rft')
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--budget-minutes',type=int,default=480)
    parser.add_argument('--started-at',type=float)
    parser.add_argument('--train-python',default=sys.executable)
    parser.add_argument('--eval-python',required=True)
    parser.add_argument('--dtype',choices=['auto','float16','bfloat16'],default='auto')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) or not 180<=args.budget_minutes<=480:
        parser.error('Use a safe run ID and a budget of 180–480 minutes')
    import torch
    if not torch.cuda.is_available():
        parser.error('CUDA GPU required')
    supports_bf16 = torch.cuda.is_bf16_supported()
    if args.dtype=='auto':
        args.dtype = 'bfloat16' if supports_bf16 else 'float16'
    if args.dtype=='bfloat16' and not supports_bf16:
        parser.error('GPU does not support BF16')
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    with (workspace/'.math_training.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another experiment owns this workspace')
        suite = PublishedSuite(args)
        write(suite.root/'pid.json',{'pid':os.getpid(),'started_at':time.time()})
        def stop(*_):
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT,stop)
        signal.signal(signal.SIGTERM,stop)
        try:
            suite.run()
        except PhaseTimeout as exc:
            suite.summary.update(status='time_limited',error=str(exc))
        except KeyboardInterrupt:
            suite.terminate()
            suite.summary['status'] = 'interrupted'
        except Exception as exc:
            suite.summary.update(status='failed',error=str(exc))
            raise
        finally:
            suite.summary['active_phase'] = None
            suite.save()
            write(suite.root/'status.json',{'status':suite.summary['status'],'ended_at':time.time()})
            print('STATUS:',suite.summary['status'],'ARTIFACTS:',suite.root,flush=True)


if __name__ == '__main__':
    main()
