"""Finite work, no wall-clock cutoff: matched SFT and online accuracy experiments."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from core import read, write, sha
from suite import Suite, PhaseFailure, link_dir
from suite_utils import immutable_json, summarize, candidate_report
from accuracy_utils import POLICY, select_accuracy, coverage


class AccuracySuite(Suite):
    def __init__(self, args):
        args.budget_minutes = 0
        args.source_run = None
        args.config_overrides = dict(
            suite_kind='accuracy_v1', dtype=args.dtype, context=8192, eval_chunk=8,
            microbatch=1, auto_oom_microbatch=True, published_schedule=True,
            sft_tokens=args.sft_tokens, online_rounds=args.online_rounds,
            online_prompts=args.online_prompts, online_generations=8,
            online_lr=args.online_lr, online_eval_every=args.online_eval_every,
            sft_initialization='continue')
        super().__init__(args)
        self.deadline = self.cutoff = float('inf')
        self.data = self.root/'data'
        self.summary.update(selection_policy=POLICY, time_limit=None,
            objective='Improve single direct output accuracy; repetition is secondary')

    def save(self):
        self.summary.pop('remaining_minutes', None)
        self.summary['elapsed_minutes'] = round((time.time()-self.args.started_at)/60, 2)
        write(self.summary_path, self.summary)

    def command(self, cmd, name):
        logs = self.root/'logs'; logs.mkdir(exist_ok=True)
        path = logs/(name+'.log')
        print('PHASE', name, flush=True)
        self.summary['active_phase'] = name; self.save()
        start = time.time()
        with path.open('a') as stream:
            self.active = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            try:
                code = self.active.wait()  # Deliberately no wall-clock timeout.
            except KeyboardInterrupt:
                self.terminate()
                raise
            finally:
                self.active = None
        with (self.root/'phase_times.jsonl').open('a') as f:
            f.write(json.dumps({'phase': name, 'seconds': time.time()-start, 'exit_code': code})+'\n')
        if code == 75:
            raise KeyboardInterrupt
        if code:
            raise PhaseFailure(f'{name} failed ({code}); {path}\n'+path.read_text()[-2500:])
        return time.time()-start

    def initialize(self):
        self.command([self.args.train_python, str(self.snapshot/'fetch_inputs.py'),
                      '--destination', str(self.inputs)], 'fetch')
        for item in read(self.snapshot/'input_manifest.json')['files']:
            if sha(self.inputs/item['path']) != item['sha256']:
                raise ValueError('Input checksum mismatch: '+item['path'])
        for label, python in [('train', self.args.train_python), ('eval', self.args.eval_python)]:
            packages = json.loads(subprocess.check_output([python, '-c',
                'import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],
                text=True))
            immutable_json(self.root/f'packages_{label}.json', packages)
        ctx = self.context(self.shared, self.cfg)
        self.phase(ctx, 'preflight')
        self.phase(ctx, 'merge')
        data_ctx = self.context(self.data, self.cfg)
        self.command([self.args.train_python, str(self.snapshot/'accuracy_data.py'),
                      '--context', str(data_ctx)], 'prepare_public_data')
        self.dev = read(self.inputs/'split.json')['split']['dev']
        self.confirmation = read(self.data/'confirmation_rows.json')
        self.final = read(self.data/'final_rows.json')
        self.summary['data'] = read(self.data/'data_report.json')
        self.baselines = {}
        self.baseline('dev', self.dev)
        self.summary['baseline'] = summarize(self.baselines['dev'], self.dev)
        self.save()

    def baseline(self, split, rows):
        if split in self.baselines:
            return self.baselines[split]
        root = self.root/'baselines'/split
        self.share_models(root)
        ctx = self.context(root, self.cfg)
        immutable_json(root/'eval_rows.json', rows)
        self.phase(ctx, 'generate', count=len(rows), tag='baseline_'+split)
        self.baselines[split] = read(root/'generations/b0_eval.json')
        return self.baselines[split]

    def evaluate(self, name, adapter, recipe, split='dev'):
        table = self.summary.setdefault('trials' if split == 'dev' else split+'_evaluations', {})
        if table.get(name, {}).get('status') == 'complete':
            return table[name]
        rows = {'dev': self.dev, 'confirmation': self.confirmation, 'final': self.final}[split]
        base = self.baseline(split, rows)
        root = self.root/'evaluations'/split/name
        self.share_models(root)
        link_dir(adapter, root/'training/rft/final')
        immutable_json(root/'eval_rows.json', rows)
        ctx = self.context(root, self.cfg)
        result = dict(status='running', adapter=str(adapter), path=str(root), recipe=recipe)
        table[name] = result; self.save()
        try:
            self.phase(ctx, 'merge', branch='rft', tag=f'{split}_{name}_merge')
            self.phase(ctx, 'generate', branch='rft', count=len(rows), tag=f'{split}_{name}_generate')
            outputs = read(root/'generations/rft_eval.json')
            result.update(candidate_report(base, outputs, rows), status='complete')
        finally:
            merged = root/'models/rft'
            if merged.exists() and not merged.is_symlink():
                shutil.rmtree(merged)
            self.save()
        print('RESULT', split, name, result['metrics']['accuracy'], flush=True)
        return result

    def diagnostic(self):
        root = self.root/'coverage'
        self.share_models(root)
        cfg = dict(self.cfg, generations=8, temperature=1., top_p=1., eval_chunk=4)
        ctx = self.context(root, cfg)
        rows = self.dev[:128]
        immutable_json(root/'train_rows.json', rows)
        self.phase(ctx, 'generate', purpose='train', count=len(rows), tag='baseline_pass_at_8')
        self.summary['coverage'] = coverage(read(root/'generations/b0_train.json'), rows)
        self.summary['coverage']['greedy_accuracy_same_questions'] = summarize(
            {r['id']: self.baselines['dev'][r['id']] for r in rows}, rows)['accuracy']
        self.save()

    def sft(self):
        profile = self.profile()
        for initialization, n, form, lr in [
                ('continue', 6000, 'F11', 1e-5), ('fresh', 6000, 'F11', 1e-5),
                ('continue', 12000, 'F11', 1e-5), ('fresh', 12000, 'F11', 1e-5),
                ('continue', 6000, 'F11', 5e-6), ('continue', 6000, 'F10', 5e-6)]:
            name = f'{initialization}_{n}_{form}_lr{lr:g}'.replace('-', 'm')
            status = self.summary.setdefault('training', {}).setdefault(name, {})
            if status.get('status') == 'complete': continue
            root = self.root/'training_trials'/name
            self.share_models(root)
            suffix = str(n) if form == 'F11' else 'F10'
            rows = read(self.data/f'pairs_{suffix}.json')
            plan = read(self.data/f'plan_{suffix}.json')
            immutable_json(root/'pairs.json', rows)
            immutable_json(root/'schedule.json', plan)
            cfg = dict(self.cfg, sft_initialization=initialization,
                       learning_rate=lr,
                       objective='rft', train_seed=42, microbatch=profile['microbatch'])
            ctx = self.context(root, cfg)
            status.update(status='running', learning_rate=cfg['learning_rate'],
                          schedule_tokens=plan['actual_tokens'], updates=len(plan['groups']))
            self.save()
            try:
                for step in sorted(set(plan['milestone_steps'].values())):
                    self.command([self.args.train_python, '-u', str(self.snapshot/'worker.py'),
                        'train', '--context', str(ctx), '--candidate', 'rft', '--until-step', str(step)],
                        name+f'_train_{step}')
                    adapter = root/f'training/rft/milestones/step-{step:05d}'
                    self.evaluate(name+f'_step{step}', adapter,
                        dict(kind='sft', initialization=initialization, examples=n, form=form,
                             step=step, learning_rate=cfg['learning_rate'],
                             supervised_tokens=plan['cumulative_tokens'][step-1]))
                status['status'] = 'complete'
            except PhaseFailure as exc:
                status.update(status='failed', reason=str(exc))
            self.save()

    def profile(self):
        path = self.root/'sft_profile.json'
        if path.exists(): return read(path)
        root = self.root/'benchmark'
        self.share_models(root)
        immutable_json(root/'pairs.json', read(self.data/'pairs_12000.json'))
        immutable_json(root/'schedule.json', read(self.data/'plan_12000.json'))
        measurements = []
        for batch in (1, 2, 4):
            try:
                for mode, steps in [('longest', 1), ('representative', 4)]:
                    cfg = dict(self.cfg, objective='rft', train_seed=42,
                               microbatch=batch, benchmark_mode=mode, benchmark_steps=steps)
                    ctx = self.context(root, cfg, f'{mode}_{batch}.json')
                    self.phase(ctx, 'bench', branch='rft', tag=f'profile_{mode}_{batch}')
                measurements.append(read(root/'timings/bench_rft.json'))
            except PhaseFailure as exc:
                if 'out of memory' not in str(exc).lower(): raise
                break
        if not measurements: raise PhaseFailure('No training microbatch fits')
        best = min(measurements, key=lambda m: m['seconds_per_update'])
        immutable_json(path, dict(best, measurements=measurements))
        self.summary['sft_profile'] = read(path); self.save()
        return read(path)

    def online(self):
        freeze = self.root/'online_starts.json'
        if not freeze.exists():
            best = select_accuracy(self.summary['baseline'], self.summary['trials'], min_delta=0.)
            starts = {'run03': str(self.inputs/'stage2')}
            if best:
                starts['best_sft'] = self.summary['trials'][best]['adapter']
            immutable_json(freeze, {'starts': starts, 'sft_candidate': best,
                                   'selected_before_online': True})
        for name, adapter in read(freeze)['starts'].items():
            name = 'online_'+name
            status = self.summary.setdefault('training', {}).setdefault(name, {})
            if status.get('status') in ('complete', 'early_stopped'): continue
            root = self.root/'training_trials'/name
            self.share_models(root)
            immutable_json(root/'online_rows.json', read(self.data/'online_rows.json'))
            cfg = dict(self.cfg, initial_adapter=adapter,
                       initial_adapter_sha256=sha(Path(adapter)/'adapter_model.safetensors'))
            ctx = self.context(root, cfg)
            checkpoints = sorted(set(range(cfg['online_eval_every'], cfg['online_rounds']+1,
                                           cfg['online_eval_every'])) | {cfg['online_rounds']})
            status['status'] = 'running'; self.save()
            try:
                for step in checkpoints:
                    self.command([self.args.train_python, '-u', str(self.snapshot/'online_worker.py'),
                        '--context', str(ctx), '--until-round', str(step)], name+f'_round_{step}')
                    milestone = root/f'training/rft/milestones/step-{step:05d}'
                    result = self.evaluate(name+f'_round{step}', milestone,
                        dict(kind='online_group_reinforce', initial_adapter=adapter, round=step))
                    # Store streak only once per checkpoint on resume.
                    if step > status.get('evaluated_round', 0):
                        bad = result['metrics']['accuracy'] < self.summary['baseline']['accuracy']-.05
                        status['bad_streak'] = status.get('bad_streak', 0)+1 if bad else 0
                        status['evaluated_round'] = step
                    if status.get('bad_streak', 0) >= 2:
                        status.update(status='early_stopped', reason='two dev checkpoints below b0 by >5pp')
                        break
                    self.save()
                else:
                    status['status'] = 'complete'
            except PhaseFailure as exc:
                status.update(status='failed', reason=str(exc))
            self.save()

    def finish(self):
        selection = self.root/'selection.json'
        if not selection.exists():
            choice = self.root/'dev_choice.json'
            if not choice.exists():
                immutable_json(choice, {'candidate': select_accuracy(
                    self.summary['baseline'], self.summary['trials']), 'policy': POLICY})
            best = read(choice)['candidate']
            confirmed = None
            if best:
                trial = self.summary['trials'][best]
                result = self.evaluate(best, Path(trial['adapter']), trial['recipe'], 'confirmation')
                if result['comparison']['paired_bootstrap_95'][0] > 0:
                    confirmed = best
            immutable_json(selection, {'candidate': confirmed, 'dev_winner': best,
                                       'selected_before_final': True, 'policy': POLICY})
        self.summary['selection'] = read(selection)
        base = self.baseline('final', self.final)
        self.summary['final'] = {'baseline': summarize(base, self.final)}
        self.summary.update(recommended_adapter=str(self.inputs/'stage2'),
                            recommendation='keep_run03', status='complete_no_confirmed_gain')
        best = self.summary['selection']['candidate']
        if best:
            trial = self.summary['trials'][best]
            result = self.evaluate(best, Path(trial['adapter']), trial['recipe'], 'final')
            self.summary['final']['candidate'] = result
            if result['comparison']['paired_bootstrap_95'][0] > 0:
                self.summary.update(recommended_adapter=trial['adapter'],
                                    recommendation='accuracy_gain_supported',
                                    status='complete_accuracy_gain')
            else:
                self.summary['recommendation'] = 'keep_run03_final_gain_unconfirmed'
        self.summary['incomplete_training'] = {k: v for k, v in self.summary['training'].items()
                                               if v['status'] == 'failed'}
        if self.summary['incomplete_training']:
            self.summary['status'] = 'partial_'+self.summary['status']
        self.summary['evaluation_note'] = (
            'Old dev reused for exploration; fresh confirmation/final exclude known run03 training. '
            'Bootstrap is per question, not corrected for dev search. No official score measured.')
        self.save()

    def run(self):
        self.summary['status'] = 'running'; self.save()
        self.initialize()
        if not (self.root/'selection.json').exists():
            self.diagnostic()
            self.sft()
            self.online()
        self.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--train-python', default=sys.executable)
    p.add_argument('--eval-python', required=True)
    p.add_argument('--started-at', type=float, default=None)
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16'], default='auto')
    p.add_argument('--sft-tokens', type=int, default=4_000_000)
    p.add_argument('--online-rounds', type=int, default=128)
    p.add_argument('--online-prompts', type=int, default=4)
    p.add_argument('--online-eval-every', type=int, default=16)
    p.add_argument('--online-lr', type=float, default=1e-6)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        p.error('Use a safe run ID')
    if min(args.sft_tokens, args.online_rounds, args.online_prompts, args.online_eval_every) < 1:
        p.error('Work amounts must be positive')
    if not 0 < args.online_lr <= 1e-4 or args.online_prompts > 12000:
        p.error('Invalid online learning rate or prompt count')
    import torch
    if not torch.cuda.is_available(): p.error('CUDA GPU required')
    if args.dtype == 'auto':
        args.dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        p.error('BF16 unsupported')
    args.started_at = args.started_at or time.time()
    args.workspace = args.workspace.expanduser().resolve()
    args.workspace.mkdir(parents=True, exist_ok=True)
    with (args.workspace/'.math_training.lock').open('a') as lock:
        try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: p.error('Another experiment owns this workspace')
        suite = AccuracySuite(args)
        write(suite.root/'pid.json', {'pid': os.getpid(), 'started_at': args.started_at})
        def stop(*_): raise KeyboardInterrupt
        signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
        try:
            suite.run()
        except KeyboardInterrupt:
            suite.terminate(); suite.summary['status'] = 'interrupted'
        except Exception as exc:
            suite.summary.update(status='failed', error=str(exc))
            raise
        finally:
            suite.summary['active_phase'] = None; suite.save()
            write(suite.root/'status.json', {'status': suite.summary['status'], 'ended_at': time.time()})
            print('STATUS:', suite.summary['status'], 'ARTIFACTS:', suite.root, flush=True)


if __name__ == '__main__':
    main()
