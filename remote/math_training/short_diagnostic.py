"""Stop the original accuracy suite safely; finish only a 32-round SFT-start diagnostic."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from core import read, write, sha, digest
from suite import link_dir
from suite_utils import immutable_json, candidate_report


def stop_owner(source, lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    pid = read(source/'pid.json')['pid']
    path = Path(f'/proc/{pid}/cmdline')
    argv = path.read_bytes().decode().strip('\0').split('\0') if path.exists() else []
    if (not any(x.endswith('/accuracy_suite.py') for x in argv)
            or '--run-id' not in argv or argv[argv.index('--run-id')+1] != source.name
            or '--workspace' not in argv
            or Path(argv[argv.index('--workspace')+1]).resolve() != source.parent.parent):
        raise RuntimeError('Workspace locked by an unverified process; no signal sent')
    print('Stopping original suite PID', pid, flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(1)
    raise RuntimeError('Original process did not release the lock; no forced kill attempted')


def reward_summary(root, through):
    metrics = []
    totals = dict(all_correct=0, all_incorrect=0, mixed=0, unknown_outputs=0)
    missing = []
    for step in range(1, through+1):
        path = root/f'training/rft/round_metrics/{step:05d}.json'
        if path.exists(): metrics.append(read(path))
        else: missing.append(step)
        path = root/f'rollouts/round-{step:05d}.json'
        if not path.exists(): continue
        for outputs in read(path).values():
            n = sum(bool(o['grade']['correct']) for o in outputs)
            totals['all_correct' if n == len(outputs) else 'all_incorrect' if n == 0 else 'mixed'] += 1
            totals['unknown_outputs'] += sum(any(o['grade'].get(k, False) for k in
                ('extraction_failed', 'grader_error', 'gold_unparseable')) for o in outputs)
    groups = sum(totals[k] for k in ('all_correct', 'all_incorrect', 'mixed'))
    end = next((r for r in metrics if r['round'] == through), None)
    return dict(totals, through_round=through, optimizer_updates=end['optimizer_updates'] if end else None,
        recorded_rounds=len(metrics), missing_metric_rounds=missing,
        optimizer_updates_per_round=end['optimizer_updates']/through if end and through else None,
        mixed_group_fraction=totals['mixed']/groups if groups else None,
        mean_round_seconds=sum(r['seconds'] for r in metrics)/len(metrics) if metrics else None)


def sft_summary(report, rows):
    trials = {k:v for k,v in report['trials'].items()
              if v.get('status') == 'complete' and v.get('recipe', {}).get('kind') == 'sft'}
    table = []
    for name, v in trials.items():
        r = v['recipe']
        table.append(dict(name=name, **r, accuracy=v['metrics']['accuracy'],
                          correct=v['metrics']['correct']))
    comparisons = []
    for name, a in trials.items():
        r = a['recipe']
        if r['examples'] != 6000: continue
        for other, b in trials.items():
            q = b['recipe']
            if q['examples'] != 12000 or any(r[k] != q[k] for k in ('initialization','form','learning_rate')):
                continue
            if abs(q['supervised_tokens']/r['supervised_tokens']-1) > .05: continue
            left = read(Path(a['path'])/'generations/rft_eval.json')
            right = read(Path(b['path'])/'generations/rft_eval.json')
            comparisons.append(dict(small=name, large=other,
                tokens_6000=r['supervised_tokens'], tokens_12000=q['supervised_tokens'],
                **candidate_report(left, right, rows)['comparison']))
    return {'checkpoints':table, 'size_comparisons':comparisons,
            'best_50k_through_1m':max((r for r in table if r['supervised_tokens'] <= 1_050_000),
                                    key=lambda r:r['accuracy'], default=None),
            'best_overall':max(table,key=lambda r:r['accuracy'],default=None),
            'note':'Exploratory dev comparisons, not independent confirmation; positive delta favors 12000'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-suite', required=True, type=Path)
    args = p.parse_args()
    source = args.source_suite.expanduser().resolve()
    workspace = source.parent.parent
    frozen = read(source/'suite_context.json')
    if frozen['config'].get('suite_kind') != 'accuracy_v1': p.error('Expected accuracy_v1 suite')
    snapshot = source/'code'
    for name, expected in frozen['code'].items():
        if sha(snapshot/name) != expected: raise ValueError('Original snapshot changed: '+name)
    out = source/'short_diagnostic'; out.mkdir(exist_ok=True)
    train_python = workspace/'.venv-train/bin/python'
    eval_python = workspace/'.venv-eval/bin/python'
    with (workspace/'.launch.lock').open('a') as launch_lock:
        stop_owner(source, launch_lock)
        with (workspace/'.math_training.lock').open('a') as model_lock:
            fcntl.flock(model_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = read(source/'suite_report.json')
            rows = read(source/'baselines/dev/eval_rows.json')
            baseline = read(source/'baselines/dev/generations/b0_eval.json')
            plan_path = out/'plan.json'
            if not plan_path.exists():
                candidates = {k:v for k,v in report['trials'].items() if v.get('status') == 'complete'
                    and v.get('recipe',{}).get('kind') == 'sft'}
                best = min(candidates, key=lambda k:(-candidates[k]['metrics']['correct'],
                                                   candidates[k]['metrics']['unknown'], k))
                old_starts = source/'online_starts.json'
                if old_starts.exists() and read(old_starts).get('sft_candidate'):
                    best = read(old_starts)['sft_candidate']
                immutable_json(plan_path, {'sft_candidate':best, 'adapter':report['trials'][best]['adapter'],
                    'first_completed':[k for k,v in report['trials'].items()
                        if k.startswith('online_run03_round') and v.get('status') == 'complete'],
                    'second_rounds':[16,32], 'purpose':'diagnosis only; no final selection'})
            plan = read(plan_path)
            result = {'status':'running', 'plan':plan, 'sft':sft_summary(report, rows),
                      'coverage':report.get('coverage'), 'first_rl':[], 'second_rl':[]}
            write(out/'report.json', result)
            active = None
            def stop(*_): raise KeyboardInterrupt
            signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
            def command(cmd, tag):
                nonlocal active
                print('PHASE', tag, flush=True)
                with (out/(tag+'.log')).open('a') as log:
                    active = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = active.wait()
                    except BaseException:
                        os.killpg(active.pid, signal.SIGTERM)
                        try: active.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(active.pid, signal.SIGKILL); active.wait()
                        raise
                    finally: active = None
                if code: raise RuntimeError(f'{tag} failed ({code}); see {out/(tag+".log")}')
            try:
                for name in plan['first_completed']:
                    trial = report['trials'][name]
                    step = trial['recipe']['round']
                    result['first_rl'].append(dict(round=step, accuracy=trial['metrics']['accuracy'],
                        comparison_to_run03=trial['comparison'],
                        updates=reward_summary(source/'training_trials/online_run03', step)))
                write(out/'report.json', result)
                root = source/'training_trials/online_best_sft'
                for model in ('stage1', 'b0'):
                    link_dir(source/'shared/models'/model, root/'models'/model)
                immutable_json(root/'online_rows.json', read(source/'data/online_rows.json'))
                cfg = dict(frozen['config'], initial_adapter=plan['adapter'],
                           initial_adapter_sha256=sha(Path(plan['adapter'])/'adapter_model.safetensors'))
                # Preserve the original 128-round schedule and checkpoint identity; cap execution only.
                context = dict(config=cfg, code=frozen['code'],
                    signature=digest([frozen['signature'],cfg,str(root)]), run=str(root), inputs=str(workspace/'inputs'))
                immutable_json(root/'context.json', context)
                sft_outputs = read(Path(report['trials'][plan['sft_candidate']]['path'])/'generations/rft_eval.json')
                for step in plan['second_rounds']:
                    adapter = root/f'training/rft/milestones/step-{step:05d}'
                    if not (adapter/'artifacts.json').exists():
                        latest = root/'training/rft/latest.json'
                        if latest.exists() and read(latest)['step'] > step:
                            raise ValueError('Already passed requested round without a milestone; cannot rewind')
                        command([str(train_python),'-u',str(snapshot/'online_worker.py'),'--context',
                                 str(root/'context.json'),'--until-round',str(step)],f'second_train_{step}')
                    evaluation = out/f'eval_{step}'
                    for model in ('stage1','b0'):
                        link_dir(source/'shared/models'/model, evaluation/'models'/model)
                    link_dir(adapter, evaluation/'training/rft/final')
                    immutable_json(evaluation/'eval_rows.json', rows)
                    immutable_json(evaluation/'context.json', dict(context, run=str(evaluation),
                                   signature=digest([context['signature'],str(evaluation)])))
                    outputs_path = evaluation/'generations/rft_eval.json'
                    prior = report['trials'].get(f'online_best_sft_round{step}', {})
                    if (not outputs_path.exists() and prior.get('status') == 'complete'
                            and Path(prior['adapter']).resolve() == adapter.resolve()):
                        immutable_json(outputs_path, read(Path(prior['path'])/'generations/rft_eval.json'))
                    if not outputs_path.exists() or set(read(outputs_path)) != {r['id'] for r in rows}:
                        command([str(train_python),str(snapshot/'worker.py'),'merge','--context',
                            str(evaluation/'context.json'),'--candidate','rft'],f'second_merge_{step}')
                        command([str(eval_python),str(snapshot/'worker.py'),'generate','--context',
                            str(evaluation/'context.json'),'--candidate','rft','--count',str(len(rows))],f'second_eval_{step}')
                    outputs = read(outputs_path)
                    result['second_rl'].append(dict(round=step,
                        compared_to_sft=candidate_report(sft_outputs,outputs,rows),
                        compared_to_run03=candidate_report(baseline,outputs,rows), updates=reward_summary(root,step)))
                    write(out/'report.json', result)
                    import shutil
                    merged = evaluation/'models/rft'
                    if merged.exists() and not merged.is_symlink(): shutil.rmtree(merged)
                result['status'] = 'diagnostic_complete'
            except BaseException as exc:
                result.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed', error=str(exc))
                raise
            finally:
                write(out/'report.json', result)
                print('STATUS',result['status'],'REPORT',out/'report.json',flush=True)


if __name__ == '__main__': main()
