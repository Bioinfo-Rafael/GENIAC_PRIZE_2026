"""Linux sidecar: shorten an already running OSW job without resetting its clock."""
import argparse
import json
import os
from pathlib import Path
import signal
import time


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def updated(root):
    work = root / 'chain/stage2/training/rft'
    latest = read(work / 'latest.json')
    if not latest or not (work / latest['checkpoint'] / 'state.pt').is_file():
        return False
    try:
        lines = (work / 'train.jsonl').read_text().splitlines()
    except FileNotFoundError:
        return False
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row['step'] <= latest['step'] and row['nonzero_lr_updates'] > 0:
            return True
    return False


def processes(root):
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            args = (entry / 'cmdline').read_bytes().decode().split('\0')
            if '--root' not in args or args[args.index('--root') + 1] != str(root):
                continue
            for kind in ('run', 'phases'):
                script = str(root / 'code/math_osw' / (kind + '.py'))
                if script in args:
                    phase = args[args.index(script) + 1] if kind == 'phases' else 'runner'
                    result.append((int(entry.name), phase))
        except (OSError, ValueError, IndexError, UnicodeError):
            continue
    return result


def send(pid, sig):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    budget = read(root / 'budget.json')
    start = budget['started_epoch']
    # Leave five seconds for shutdown before the original start + 20 minutes.
    end = min(budget['deadline_epoch'], start + 1200) - 10
    control = dict(original_started_epoch=start, requested_deadline_epoch=end + 10,
                   train_stop_epoch=min(start + 420, end),
                   policy='first saved nonzero-LR update or seven minutes; skip optional evaluation',
                   installed_epoch=time.time())
    (root / 'cap20_request.json').write_text(json.dumps(control, indent=2))
    print('CAP20', json.dumps(control), flush=True)
    sent = set()
    while True:
        jobs = processes(root)
        if not jobs:
            print('RUN_EXITED', json.dumps(read(root / 'suite_report.json')), flush=True)
            return
        now = time.time()
        if now >= end:
            groups = set()
            for pid, phase in jobs:
                if phase != 'runner':
                    try:
                        if os.getpgid(pid) == pid:
                            groups.add(pid)
                    except ProcessLookupError:
                        pass
                send(pid, signal.SIGTERM)
            time.sleep(5)
            for group in groups:
                try:
                    os.killpg(group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for pid, _ in processes(root):
                send(pid, signal.SIGKILL)
            print('CAP20_STOPPED; check suite_report.json and submission files', flush=True)
            return
        ready = read(root / 'suite_report.json').get('submission_ready', False)
        for pid, phase in jobs:
            if pid in sent:
                continue
            if (phase == 'train' and (now >= control['train_stop_epoch'] or updated(root))) or (ready and phase == 'runner'):
                send(pid, signal.SIGTERM)
                sent.add(pid)
                print('EARLY_FINISH', phase, pid, 'elapsed', round(now-start), flush=True)
        time.sleep(1)


if __name__ == '__main__':
    main()
