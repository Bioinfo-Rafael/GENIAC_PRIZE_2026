"""One-command run02 training on Linux/CUDA. No official evaluation or submission."""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'colab_t4' / 'experiments'))
from run02_launcher import completed_training, restore_bundle, run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path.home() / 'geniac_workspace')
    parser.add_argument('--bundle', type=Path, help='Separately transferred run02_colab_inputs.zip')
    parser.add_argument('--run-id', default='cuda_20261013_run02_prmfull')
    parser.add_argument('--install', action='store_true', help='Install training packages into the active venv')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        parser.error('--run-id must contain only letters, digits, underscores or hyphens')
    if platform.system() != 'Linux':
        parser.error('Run this on the Linux/CUDA server, not the Mac.')
    workspace = args.workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    # Prevent concurrent runs from merging/writing adapters in the same workspace.
    import fcntl
    with (workspace / '.run02.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This workspace already has an active run02 process.')
        if args.bundle:
            record = json.loads((HERE / 'bundle_reference.json').read_text())
            print('Inputs:', restore_bundle(args.bundle.expanduser(), workspace, record['sha256']), flush=True)
        if completed_training(workspace, args.run_id):
            print('RUN02_TRAINING_COMPLETE (already complete; weights verified)')
            return
        stage1 = workspace / 'adapters' / 't4_20261013_run01_stage1'
        if not (stage1 / 'adapter_model.safetensors').is_file():
            parser.error('Stage 1 adapter missing. Transfer the inputs ZIP and supply --bundle.')
        logs = workspace / 'experiments' / args.run_id / 'logs'
        if args.install:
            if sys.prefix == sys.base_prefix:
                parser.error('--install requires an activated Python venv; system Python will not be modified.')
            run_logged([sys.executable, '-m', 'pip', 'install', '-r', str(HERE / 'requirements-train.txt')], logs / 'install.log')
        run_logged([sys.executable, '-c', "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0)); print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)"], logs / 'gpu.log')
        logs.mkdir(parents=True, exist_ok=True)
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (logs / 'packages.txt').write_text(freeze)
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        for phase in ('merge', 'train'):
            run_logged([sys.executable, '-u', str(HERE / 'run02_worker.py'), phase,
                        '--workspace', str(workspace), '--run-id', args.run_id], logs / (phase + '.log'))
        if not completed_training(workspace, args.run_id):
            raise RuntimeError('Training finished without a verified final adapter.')
        print('RUN02_TRAINING_COMPLETE', flush=True)
        print('Summary:', workspace / 'experiments' / args.run_id / 'training_summary.json')


if __name__ == '__main__':
    main()
