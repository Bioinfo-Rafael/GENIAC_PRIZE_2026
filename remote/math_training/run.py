"""Budgeted RFT/DPO pilot. All artifacts stay in --workspace, outside Git."""
import argparse
import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from core import read, write, sha, digest, jsonl, prepare

HERE = Path(__file__).resolve().parent


class BudgetExpired(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--run-id', default='ada_run07_rft_dpo')
    parser.add_argument('--config', type=Path, default=HERE / 'configs/ada_dpo.json')
    parser.add_argument('--budget-minutes', type=float, default=180)
    parser.add_argument('--started-at', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--train-python', default=sys.executable)
    parser.add_argument('--eval-python', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--validate-only', action='store_true', help='Validate existing inputs on CPU; no downloads or GPU')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id): parser.error('Invalid run ID')
    if not 1 <= args.budget_minutes <= 180: parser.error('Budget must be between 1 and 180 minutes')
    cfg = read(args.config)
    if cfg['prompts'] not in (128,256,512) or cfg['pilot_prompts'] != 128:
        parser.error('This pilot supports prompts=128/256/512 and pilot_prompts=128')
    if cfg['microbatch'] < 1 or cfg['effective_batch'] % cfg['microbatch']:
        parser.error('microbatch must divide effective_batch')
    if cfg['dtype'] not in ('float16', 'bfloat16'): parser.error('Unsupported dtype')
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    inputs, root = workspace / 'inputs', workspace / 'runs' / args.run_id
    lock = (workspace / '.math_training.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('A math-training experiment already owns this workspace')
    if root.exists() and not args.resume and not args.validate_only:
        parser.error('Run directory exists. Use --resume or a new --run-id')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    deadline = (args.started_at or time.time()) + args.budget_minutes * 60
    write(root / 'pid.json', {'pid':os.getpid(), 'started_at':time.time()})
    active = None

    def remaining(): return deadline - time.time()

    def stop(*_):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def command(cmd, name):
        nonlocal active
        if remaining() <= 60: raise BudgetExpired()
        print('PHASE', name, 'remaining_minutes', round(remaining()/60,1), flush=True)
        with (root / 'logs' / (name + '.log')).open('a') as log:
            active = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = active.wait(timeout=max(1, remaining()-45))
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(active.pid, signal.SIGTERM)
                try: active.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(active.pid, signal.SIGKILL)
                    active.wait()
                raise BudgetExpired()
            finally:
                active = None
        if code == 75: raise BudgetExpired()
        if code != 0:
            tail = (root / 'logs' / (name+'.log')).read_text()[-4000:]
            raise RuntimeError(f'{name} failed ({code}); log: {root / "logs" / (name+".log")}\n{tail}')

    status = 'initializing'
    ctx = None
    try:
        if not args.validate_only:
            command([args.train_python, str(HERE/'fetch_inputs.py'), '--destination', str(inputs)], 'fetch')
        for item in read(HERE/'input_manifest.json')['files']:
            if sha(inputs/item['path']) != item['sha256']:
                raise ValueError('Input checksum mismatch: '+item['path'])
        split = read(inputs/'split.json')
        train = prepare(jsonl(inputs/'train.jsonl'), split, cfg['prompts'], cfg['seed'])
        quick = set(split['split']['quick_ids'])
        dev = split['split']['dev']
        eval_rows = [r for r in dev if r['id'] in quick] + [r for r in dev if r['id'] not in quick]
        if len(quick) != 128 or len(eval_rows) != 400: raise ValueError('Unexpected fixed eval split')
        if sha(inputs/'stage2/adapter_model.safetensors') != cfg['stage2_sha256']:
            raise ValueError('Not the saved run03 adapter')
        code = {str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.txt','.jinja','.sh')}
        signature = digest({'config':cfg, 'code':code, 'inputs':read(HERE/'input_manifest.json')})
        ctx = {'config':cfg, 'code':code, 'signature':signature, 'run':str(root), 'inputs':str(inputs)}
        context = root/'context.json'
        if context.exists() and read(context)['signature'] != signature:
            raise ValueError('Code/config/inputs changed: use a new --run-id; refusing mixed resume')
        write(context, ctx)
        write(root/'train_rows.json', train)
        write(root/'eval_rows.json', eval_rows)
        if args.validate_only:
            status = 'inputs_validated'
            print('INPUTS_VALID: 6000 training / 800 heldout; run03 checksum verified')
            return
        if shutil.disk_usage(workspace).free < 60*2**30:
            raise RuntimeError('At least 60 GiB free disk space is required')
        # Stable worker snapshot: a later git pull cannot change an active worker.
        snapshot = root/'code'
        if not snapshot.exists():
            shutil.copytree(HERE, snapshot, ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
        worker = snapshot/'worker.py'
        for label, python in [('train',args.train_python),('eval',args.eval_python)]:
            environment = subprocess.check_output([python,'-c',
                'import importlib.metadata as m,json; print(json.dumps({x.metadata["Name"]:x.version for x in m.distributions()}))'],text=True)
            import json
            packages = json.loads(environment)
            path = root/f'packages_{label}.json'
            if path.exists() and read(path)!=packages:
                raise ValueError('Environment changed on resume: '+label)
            write(path,packages)

        def phase(name, candidate='b0', purpose='eval', count=128):
            python = args.eval_python if name=='generate' else args.train_python
            label = f'{name}_{candidate}' if name in ('train','bench','merge') else f'{name}_{candidate}_{purpose}_{count}'
            command([python, '-u', str(worker), name, '--context', str(context),
                     '--candidate',candidate,'--purpose',purpose,'--count',str(count)],
                    label)

        phase('preflight')
        phase('merge')
        phase('generate')
        phase('generate', purpose='train', count=cfg['pilot_prompts'])
        times = read(root/'timings/b0_train.json')
        qps = sum(x['questions'] for x in times)/max(1,sum(x['seconds'] for x in times))
        # Preserve at least 80 min for reference, training and assessment.
        selection = root/'selection.json'
        if selection.exists():
            selected_count = read(selection)['prompts']
        else:
            selected_count = cfg['pilot_prompts']
            for n in (256,512):
                if n <= cfg['prompts'] and (n-cfg['pilot_prompts'])/qps*1.3 + 80*60 < remaining():
                    selected_count = n
            write(selection, {'prompts':selected_count, 'pilot_questions_per_second':qps})
        phase('generate', purpose='train', count=selected_count)
        phase('pairs')
        summary = read(root/'pair_summary.json')
        if summary['pairs'] < cfg['min_pairs']:
            status = 'insufficient_pairs'
            write(root/'next_action.json', {'status':status,'pairs':summary['pairs'],
                'action':'No DPO training. Inspect pair_summary; SFT/DFT is a separate future experiment.'})
            return
        phase('reference')
        for branch in ('rft','dpo'):
            if not (root/f'timings/bench_{branch}.json').exists():
                phase('bench', candidate=branch)
        estimate = 0
        for branch in ('rft','dpo'):
            timing = read(root/f'timings/bench_{branch}.json')
            latest = root/f'training/{branch}/latest.json'
            completed = read(latest)['step'] if latest.exists() else 0
            estimate += timing['seconds_per_update'] * max(0,timing['updates']-completed)
        eval_time = sum(x['seconds'] for x in read(root/'timings/b0_eval.json'))
        if estimate*1.3 + eval_time*2.5 + 15*60 > remaining():
            status = 'insufficient_time_for_paired_training'
            return
        for branch in ('rft','dpo'): phase('train', candidate=branch)
        for branch in ('rft','dpo'):
            phase('merge', candidate=branch)
            phase('generate', candidate=branch)
        status = 'quick_comparison_complete'
        # Evaluate all three on dev400 only when the measured budget allows it.
        if eval_time * (272/128) * 3 * 1.4 + 5*60 < remaining():
            for branch in ('b0','rft','dpo'): phase('generate', candidate=branch, count=400)
            status = 'dev_comparison_complete'
    except BudgetExpired:
        status = 'budget_exhausted'
    except KeyboardInterrupt:
        status = 'interrupted'
    except Exception as exc:
        status = 'failed'
        write(root/'error.json', {'type':type(exc).__name__, 'message':str(exc)})
        raise
    finally:
        if ctx is not None:
            # CPU-only report remains available even after a budget stop.
            from worker import report
            report(ctx)
        write(root/'status.json', {'status':status,'ended_at':time.time(),
                                  'remaining_seconds':max(0,remaining())})
        print('STATUS:',status,'ARTIFACTS:',root,flush=True)


if __name__ == '__main__':
    main()
