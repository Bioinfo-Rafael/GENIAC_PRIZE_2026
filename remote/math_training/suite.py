"""Sequential, budgeted RFT/DFT/DPO experiment suite for one 48GB GPU."""
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
from suite_utils import (candidate_report, eligible, immutable_json, select_best,
                         summarize, validate_source, validate_recovery)

HERE = Path(__file__).resolve().parent
EXPERIMENTS = [
    ('c1_rft_512', 'rft', 'mixed', 5e-6, 512),
    ('c2_rft_low_lr_512', 'rft', 'mixed', 1e-6, 512),
    ('e1_rft_mixed', 'rft', 'mixed', 5e-6, 2048),
    ('e2_rft_positive', 'rft', 'positive', 5e-6, 2048),
    ('e3_dft_positive', 'dft', 'positive', 5e-6, 2048),
    ('e4_dpo', 'dpo', 'mixed', 5e-6, 2048),
    ('e5_dpo_nll', 'dpo_nll', 'mixed', 5e-6, 2048),
]


class PhaseTimeout(Exception):
    pass


class PhaseFailure(Exception):
    pass


def link_dir(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source:
            raise ValueError('Unexpected shared artifact: '+str(target))
    elif target.exists():
        raise ValueError('Refusing to replace existing artifact: '+str(target))
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


class Suite:
    def __init__(self, args):
        self.args = args
        self.workspace = args.workspace.expanduser().resolve()
        self.root = self.workspace/'suites'/args.run_id
        self.inputs = self.workspace/'inputs'
        self.deadline = (args.started_at or time.time()) + args.budget_minutes*60
        self.cutoff = self.deadline-60*60
        self.active = None
        self.cfg = read(HERE/'configs/ada_dpo.json')
        self.cfg.update(epochs=1, save_milestones=True, suite_benchmark=True)
        self.cfg.update(getattr(args, 'config_overrides', {}))
        self.code = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts
            and p.suffix in ('.py','.json','.txt','.jinja','.sh')}
        self.source = args.source_run.expanduser().resolve() if args.source_run else None
        recovery = getattr(args,'source_suite',None)
        self.recovery = recovery.expanduser().resolve() if recovery else None
        self.signature = digest({'code':self.code, 'config':self.cfg,
                                 'source':str(self.source) if self.source else None,
                                 'source_suite':str(self.recovery) if self.recovery else None})
        if self.root.exists() and not args.resume:
            raise ValueError('Suite directory exists; use --resume or a new --run-id')
        self.root.mkdir(parents=True, exist_ok=True)
        immutable_json(self.root/'suite_context.json', {'signature':self.signature,
            'code':self.code, 'config':self.cfg, 'source':str(self.source) if self.source else None,
            'source_suite':str(self.recovery) if self.recovery else None})
        self.snapshot = self.root/'code'
        if not self.snapshot.exists():
            shutil.copytree(HERE,self.snapshot,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        for name, expected in self.code.items():
            if sha(self.snapshot/name) != expected:
                raise ValueError('Suite snapshot changed: '+name)
        self.summary_path = self.root/'suite_report.json'
        self.summary = read(self.summary_path) if self.summary_path.exists() else {
            'status':'initializing', 'trials':{}, 'official_score':None,
            'selection_policy':{'min_dev_delta':.02,'max_repetition_delta':.03,
                                'max_non_stop_delta':.03, 'gsm_max_lost':4}}
        self.shared = self.root/'shared'
        self.shared.mkdir(exist_ok=True)

    def save(self):
        self.summary['remaining_minutes'] = round(max(0,self.deadline-time.time())/60,2)
        write(self.summary_path,self.summary)

    def command(self, cmd, name):
        if time.time() >= min(self.cutoff,self.deadline)-45:
            raise PhaseTimeout('Reserved time reached before '+name)
        logs = self.root/'logs'
        logs.mkdir(exist_ok=True)
        logfile = logs/(name+'.log')
        print('PHASE',name,'remaining_minutes',round((self.deadline-time.time())/60,1),flush=True)
        self.summary['active_phase'] = name
        self.save()
        started = time.time()
        with logfile.open('a') as stream:
            self.active = subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = self.active.wait(timeout=max(1,min(self.cutoff,self.deadline)-time.time()-40))
            except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
                self.terminate()
                if isinstance(exc,KeyboardInterrupt):
                    raise
                raise PhaseTimeout('Time limit: '+name) from exc
            finally:
                self.active = None
        duration = time.time()-started
        with (self.root/'phase_times.jsonl').open('a') as f:
            f.write(json.dumps({'phase':name,'seconds':duration,'exit_code':code})+'\n')
        if code == 75:
            raise PhaseTimeout('Worker stopped: '+name)
        if code:
            raise PhaseFailure(f'{name} failed ({code}); {logfile}\n'+logfile.read_text()[-3000:])
        return duration

    def terminate(self):
        if self.active is None:
            return
        try:
            os.killpg(self.active.pid,signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.active.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(self.active.pid,signal.SIGKILL)
            self.active.wait()

    def context(self, root, cfg, filename='context.json'):
        root = Path(root)
        root.mkdir(parents=True,exist_ok=True)
        ctx = {'config':cfg,'code':self.code,'signature':digest([self.signature,cfg,str(root)]),
               'run':str(root),'inputs':str(self.inputs)}
        path = root/filename
        immutable_json(path,ctx)
        return path

    def phase(self, context, phase, branch='b0', purpose='eval', count=128, tag=None):
        python = self.args.eval_python if phase=='generate' else self.args.train_python
        return self.command([python,'-u',str(self.snapshot/'worker.py'),phase,
            '--context',str(context),'--candidate',branch,'--purpose',purpose,'--count',str(count)],
            tag or f'{Path(context).parent.name}_{phase}_{branch}_{purpose}_{count}')

    def share_models(self, root):
        for name in ('stage1','b0'):
            link_dir(self.shared/'models'/name,Path(root)/'models'/name)

    def pool(self, count):
        root = self.root/'pools'/str(count)
        cfg = dict(self.cfg,prompts=count)
        context = self.context(root,cfg)
        immutable_json(root/'train_rows.json',self.rows[:count])
        immutable_json(root/'eval_rows.json',self.dev)
        self.share_models(root)
        dest = root/'generations/b0_train.json'
        if not dest.exists():
            previous = [p for p in (self.root/'pools').iterdir()
                        if p.name.isdigit() and int(p.name)<count and (p/'generations/b0_train.json').exists()]
            source = (max(previous,key=lambda p:int(p.name)) if previous else self.shared)/'generations/b0_train.json'
            if source.exists():
                write(dest,read(source))
        self.phase(context,'generate',purpose='train',count=count)
        data = read(dest)
        if set(data) != {r['id'] for r in self.rows[:count]}:
            raise ValueError('Pool generation incomplete')
        return root

    def dataset(self, pool, mode):
        root = pool/mode
        context = self.context(root,dict(self.cfg,prompts=int(pool.name),data_mode=mode))
        immutable_json(root/'train_rows.json',read(pool/'train_rows.json'))
        link_dir(pool/'generations/b0_train.json',root/'generations/b0_train.json')
        self.phase(context,'pairs',tag=f'dataset_{pool.name}_{mode}')
        return root

    def initialize(self):
        self.cutoff = min(self.deadline-3600,time.time()+30*60)
        self.command([self.args.train_python,str(self.snapshot/'fetch_inputs.py'),
                      '--destination',str(self.inputs)],'fetch')
        for item in read(self.snapshot/'input_manifest.json')['files']:
            if sha(self.inputs/item['path']) != item['sha256']:
                raise ValueError('Input checksum mismatch: '+item['path'])
        split = read(self.inputs/'split.json')['split']
        self.rows = prepare(jsonl(self.inputs/'train.jsonl'),read(self.inputs/'split.json'),6000,self.cfg['seed'])
        quick = set(split['quick_ids'])
        self.dev = [r for r in split['dev'] if r['id'] in quick]+[r for r in split['dev'] if r['id'] not in quick]
        self.final = split['final']
        if len(self.dev)!=400 or len(self.final)!=400 or len(quick)!=128:
            raise ValueError('Unexpected heldout split')
        packages = {}
        for label, python in [('train',self.args.train_python),('eval',self.args.eval_python)]:
            packages[label] = json.loads(subprocess.check_output([python,'-c',
                'import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],text=True))
            immutable_json(self.root/f'packages_{label}.json',packages[label])
        if self.source:
            provenance = validate_source(self.source,self.cfg,self.code,packages,self.rows,self.dev)
            from worker import check_model
            for name in ('stage1','b0'):
                if not check_model(self.source/'models'/name):
                    raise ValueError('Missing source model: '+name)
                provenance[f'models/{name}/artifacts.json'] = sha(self.source/'models'/name/'artifacts.json')
            if shutil.disk_usage(self.workspace).free < 12*2**30:
                raise ValueError('At least 12 GiB free is needed for candidate exports')
            immutable_json(self.root/'source_manifest.json',{'path':str(self.source),'files':provenance})
            for name in ('stage1','b0'):
                link_dir(self.source/'models'/name,self.shared/'models'/name)
            for purpose in ('train','eval'):
                dest = self.shared/f'generations/b0_{purpose}.json'
                immutable_json(dest,read(self.source/f'generations/b0_{purpose}.json'))
        elif shutil.disk_usage(self.workspace).free < 60*2**30:
            raise ValueError('At least 60 GiB free is needed for initial merged models')
        context = self.context(self.shared,self.cfg)
        immutable_json(self.shared/'train_rows.json',self.rows[:512])
        immutable_json(self.shared/'eval_rows.json',self.dev)
        immutable_json(self.shared/'final_rows.json',self.final)
        self.phase(context,'preflight')
        self.phase(context,'merge')  # verifies manifests even when models were imported
        from worker import check_model
        if not all(check_model(self.shared/'models'/name) for name in ('stage1','b0')):
            raise ValueError('Missing or invalid shared model manifests')
        self.phase(context,'generate',count=400)
        self.base_outputs = read(self.shared/'generations/b0_eval.json')
        self.summary['baseline'] = summarize(self.base_outputs,self.dev)
        pool = self.pool(512)
        data = self.dataset(pool,'mixed')
        if self.source and read(data/'pairs.json') != read(self.source/'pairs.json'):
            raise ValueError('Control pairs differ from the completed pilot')
        # Require exact weights/configs; measure, rather than assume, generation reproducibility.
        audit_root = self.root/'audit'
        audit_ctx = self.context(audit_root,self.cfg)
        self.share_models(audit_root)
        immutable_json(audit_root/'pairs.json',read(data/'pairs.json'))
        immutable_json(audit_root/'eval_rows.json',self.dev)
        if not (audit_root/'comparison.json').exists():
            self.phase(audit_ctx,'audit')
            self.phase(audit_ctx,'merge',branch='rft')
            self.phase(audit_ctx,'audit_weights')
            self.phase(audit_ctx,'generate',branch='rft',count=32)
            generated = read(audit_root/'generations/rft_eval.json')
            differences = [r['id'] for r in self.dev[:32]
                if generated[r['id']][0]['token_ids'] != self.base_outputs[r['id']][0]['token_ids']]
            changed_grades = [r['id'] for r in self.dev[:32]
                if generated[r['id']][0]['grade']['correct'] != self.base_outputs[r['id']][0]['grade']['correct']]
            write(audit_root/'comparison.json',{'n':32,'changed_outputs':differences,
                'changed_correctness':changed_grades,'interpretation':
                'Identical weights/configs; differing outputs indicate execution variability, not training updates.'})
        if not read(audit_root/'integrity.json')['identical']:
            raise ValueError('No-update model integrity check failed')
        variability = read(audit_root/'comparison.json')
        self.summary['inference_variability'] = variability
        if variability['changed_outputs']:
            print('AUDIT: identical weights/configs; generation variability:',
                  len(variability['changed_outputs']),'/32; continuing with variability recorded.',flush=True)
            self.summary['uncertainty_scope'] = (
                'Paired bootstrap resamples questions, not inference executions; small gains are not conclusive.')
        self.remove_merged(audit_root,'rft')
        self.summary['audit'] = read(audit_root/'audit.json')
        if self.recovery:
            self.import_recovery(packages)
        self.save()

    def import_recovery(self, packages):
        from worker import check_model
        generation = validate_recovery(self.recovery,self.cfg,self.code,packages,
                                       self.rows,self.dev,self.base_outputs)
        pool = self.root/'pools/2048'
        cache = pool/'generations/b0_train.json'
        immutable_json(cache,generation)
        old = read(self.recovery/'suite_report.json')
        provenance = {'source':str(self.recovery),
            'pool_sha256':sha(self.recovery/'pools/2048/generations/b0_train.json'),
            'report_sha256':sha(self.recovery/'suite_report.json'),
            'completed':{},'failed_trial_policy':'restart from run03 using cached generation; no optimizer import'}
        for name,objective,mode,lr,count in EXPERIMENTS:
            item = old['trials'].get(name,{})
            if item.get('status')!='complete':
                continue
            previous = self.recovery/'trials'/name
            cfg = read(previous/'context.json')['config']
            expected_cfg = dict(self.cfg,objective=objective,data_mode=mode,learning_rate=lr,
                                prompts=count,train_seed=42,nll_weight=.1,microbatch=item['microbatch'])
            if cfg != expected_cfg:
                raise ValueError('Recovery completed trial config differs: '+name)
            branch = item['branch']
            adapter = previous/f'training/{branch}/final'
            if not check_model(adapter):
                raise ValueError('Recovery completed adapter missing: '+name)
            output = read(previous/f'generations/{branch}_eval.json')
            report = candidate_report(self.base_outputs,output,self.dev)
            if report['metrics']!=item['metrics']:
                raise ValueError('Recovery metrics disagree with raw outputs: '+name)
            provenance['completed'][name] = {
                'output_sha256':sha(previous/f'generations/{branch}_eval.json'),
                'adapter_manifest_sha256':sha(adapter/'artifacts.json')}
            if name in self.summary['trials']:
                continue
            root = self.root/'trials'/name
            self.context(root,cfg)
            self.share_models(root)
            immutable_json(root/'eval_rows.json',self.dev)
            immutable_json(root/'final_rows.json',self.final)
            immutable_json(root/f'generations/{branch}_eval.json',output)
            link_dir(adapter,root/f'training/{branch}/final')
            self.summary['trials'][name] = dict(item,**report,path=str(root),imported_from=str(previous))
        immutable_json(self.root/'recovery_manifest.json',provenance)
        self.summary['recovery'] = provenance
        print('RECOVERY: reused 2048-question generation and completed trials:',
              ', '.join(provenance['completed']),flush=True)

    def remove_merged(self, root, branch):
        path = Path(root)/'models'/branch
        # Only disposable candidate exports created under this suite are deleted.
        if path.exists() and not path.is_symlink() and self.root in path.resolve().parents:
            shutil.rmtree(path)

    def trial(self, spec, pool, train_seed=42):
        name, objective, mode, lr, _ = spec
        old = self.summary['trials'].get(name,{})
        if old.get('status') == 'complete':
            return old
        root = self.root/'trials'/name
        branch = 'dpo' if objective.startswith('dpo') else 'rft'
        cfg = dict(self.cfg,objective=objective,data_mode=mode,learning_rate=lr,
                   train_seed=train_seed,nll_weight=.1,prompts=int(pool.name))
        setup = self.context(root,cfg,'setup_context.json')
        self.share_models(root)
        immutable_json(root/'eval_rows.json',self.dev)
        immutable_json(root/'final_rows.json',self.final)
        result = {'status':'running','objective':objective,'data_mode':mode,
                  'learning_rate':lr,'train_seed':train_seed,'prompts':int(pool.name),
                  'branch':branch,'path':str(root)}
        self.summary['trials'][name] = result
        self.save()
        try:
            data = self.dataset(pool,mode)
            immutable_json(root/'pairs.json',read(data/'pairs.json'))
            result['examples'] = len(read(root/'pairs.json'))
            if result['examples'] < self.cfg['min_pairs']:
                result.update(status='skipped',reason='insufficient_examples')
                return result
            if branch == 'dpo':
                self.phase(setup,'reference')
            chosen_path = root/'microbatch.json'
            if chosen_path.exists():
                chosen = read(chosen_path)
            else:
                timings = []
                for microbatch in (2,4,8):
                    bench_cfg = dict(cfg,microbatch=microbatch)
                    bench = self.context(root,bench_cfg,f'bench_context_{microbatch}.json')
                    try:
                        self.phase(bench,'bench',branch=branch,tag=f'{name}_bench_mb{microbatch}')
                        timing = read(root/f'timings/bench_{branch}.json')
                        timings.append(timing)
                    except PhaseFailure as exc:
                        write(root/f'benchmark_failure_{microbatch}.json',{'error':str(exc)})
                        if 'out of memory' not in str(exc).lower():
                            raise
                        break
                if not timings:
                    raise PhaseFailure('No microbatch configuration fits')
                chosen = min(timings,key=lambda x:x['seconds_per_update'])
                write(chosen_path,chosen)
                write(root/'microbatch_trials.json',timings)
            cfg['microbatch'] = chosen['microbatch']
            context = self.context(root,cfg)
            # Use measured upper update time and leave ten minutes for merge/dev evaluation.
            latest = root/f'training/{branch}/latest.json'
            completed = read(latest)['step'] if latest.exists() else 0
            estimate = max(0,chosen['updates']-completed)*chosen['seconds_per_update']*1.5+600
            if time.time()+estimate > min(self.cutoff,self.deadline-3600):
                raise PhaseTimeout('Insufficient time for training and dev evaluation')
            self.phase(context,'train',branch=branch)
            if shutil.disk_usage(self.workspace).free < 12*2**30:
                raise PhaseFailure('Need 12 GiB free to export a candidate')
            self.phase(context,'merge',branch=branch)
            self.phase(context,'generate',branch=branch,count=400)
            output = read(root/f'generations/{branch}_eval.json')
            result.update(candidate_report(self.base_outputs,output,self.dev),status='complete',
                          microbatch=cfg['microbatch'])
            self.remove_merged(root,branch)
        except PhaseTimeout as exc:
            result.update(status='time_limited',reason=str(exc))
        except PhaseFailure as exc:
            result.update(status='failed',reason=str(exc))
        finally:
            self.remove_merged(root,branch)
            if result.get('status') == 'complete':
                print('RESULT',name,'accuracy',result['metrics']['accuracy'],
                      'delta',result['comparison']['delta_accuracy'],
                      'repeated',result['metrics']['repeated'],flush=True)
            else:
                print('RESULT',name,result.get('status'),result.get('reason',''),flush=True)
            self.save()
        return result

    def run(self):
        self.summary['status'] = 'running'
        self.initialize()
        if (self.root/'selection.json').exists():
            self.final_evaluation()
            return
        self.cutoff = min(self.deadline-3600,time.time()+45*60)
        for spec in EXPERIMENTS[:2]:
            self.trial(spec,self.root/'pools/512')
        self.cutoff = min(self.deadline-3600,time.time()+75*60)
        try:
            pool = self.pool(2048)
        except PhaseTimeout as exc:
            self.summary['pool_2048'] = {'status':'time_limited','reason':str(exc)}
            pool = None
        if pool:
            self.cutoff = min(self.deadline-3600,time.time()+120*60)
            for spec in EXPERIMENTS[2:]:
                self.trial(spec,pool)
        self.cutoff = min(self.deadline-3600,time.time()+90*60)
        best = select_best(self.summary['baseline'],self.summary['trials'])
        selected = None
        if best:
            item = self.summary['trials'][best]
            spec = (best+'_seed43',item['objective'],item['data_mode'],item['learning_rate'],item['prompts'])
            replicate = self.trial(spec,self.root/'pools'/str(item['prompts']),train_seed=43)
            confirmed = replicate.get('status')=='complete' and eligible(self.summary['baseline'],replicate['metrics'])
            self.summary['replication'] = {'candidate':best,'confirmed':confirmed,'replicate':spec[0]}
            if confirmed:
                selected = best
                # Data expansion is optional and is attempted only when measured throughput fits.
                times_path = self.root/'pools/2048/timings/b0_train.json'
                if pool and times_path.exists():
                    times = read(times_path)
                    seconds_per_question = sum(x['seconds'] for x in times)/max(1,sum(x['questions'] for x in times))
                    estimate = seconds_per_question*(6000-2048)*1.5+25*60
                    if time.time()+estimate < self.cutoff:
                        try:
                            expanded = self.pool(6000)
                            spec = (best+'_6000',item['objective'],item['data_mode'],item['learning_rate'],6000)
                            result = self.trial(spec,expanded)
                            if (result.get('status')=='complete' and eligible(self.summary['baseline'],result['metrics'])
                                and result['metrics']['correct']>item['metrics']['correct']):
                                selected = spec[0]
                        except PhaseTimeout as exc:
                            self.summary['expansion'] = {'status':'time_limited','reason':str(exc)}
                    else:
                        self.summary['expansion'] = {'status':'skipped','reason':'measured estimate exceeds reserved time',
                                                     'estimated_minutes':estimate/60}
        immutable_json(self.root/'selection.json', {'candidate':selected,
            'reason':'dev gate and seed replication passed' if selected else 'no reproducible candidate passed dev gate',
            'selected_before_final':True})
        self.final_evaluation()

    def final_evaluation(self):
        self.cutoff = min(self.deadline,time.time()+60*60)
        selection = read(self.root/'selection.json')
        self.summary['selection'] = selection
        self.phase(self.shared/'context.json','generate',purpose='final',count=400,tag='final_b0')
        baseline = read(self.shared/'generations/b0_final.json')
        self.summary['final'] = {'baseline':summarize(baseline,self.final)}
        self.summary['recommended_adapter'] = str(self.inputs/'stage2')
        self.summary['recommendation'] = 'keep_b0'
        selected = selection['candidate']
        if selected:
            trial = self.summary['trials'][selected]
            root, branch = Path(trial['path']),trial['branch']
            self.phase(root/'context.json','merge',branch=branch,tag='final_merge_'+selected)
            self.phase(root/'context.json','generate',branch=branch,purpose='final',count=400,tag='final_'+selected)
            outputs = read(root/f'generations/{branch}_final.json')
            report = candidate_report(baseline,outputs,self.final)
            report['candidate'] = selected
            report['improvement_supported'] = report['comparison']['paired_bootstrap_95'][0]>0
            self.summary['final']['candidate'] = report
            self.summary['candidate_adapter'] = str(root/f'training/{branch}/final')
            if report['improvement_supported'] and eligible(self.summary['final']['baseline'],report['metrics']):
                self.summary['recommendation'] = 'candidate_improvement_supported'
                self.summary['recommended_adapter'] = self.summary['candidate_adapter']
            else:
                self.summary['recommendation'] = 'keep_b0_improvement_not_confirmed'
        self.summary['incomplete_trials'] = {name:result['status']
            for name,result in self.summary['trials'].items() if result.get('status')!='complete'}
        self.summary['status'] = 'suite_complete' if selected else 'suite_complete_no_improvement'
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--source-suite',type=Path,help='Reuse the pre-AMP-fix suite pool and completed trials')
    parser.add_argument('--budget-minutes',type=int,default=480)
    parser.add_argument('--started-at',type=float,default=None)
    parser.add_argument('--train-python',default=sys.executable)
    parser.add_argument('--eval-python',required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id):
        parser.error('Invalid run ID')
    if not 120<=args.budget_minutes<=480:
        parser.error('Suite budget must be 120–480 minutes')
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True,exist_ok=True)
    with (workspace/'.math_training.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another experiment owns this workspace')
        suite = Suite(args)
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
