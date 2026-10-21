"""Small shared I/O and immutable experiment contract; imports no model/dataset API."""
import json
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
# Snapshot layout preserves the sibling package, but only audited helper files are copied.
LEGACY = HERE.parent/'math_training'
sys.path.insert(0, str(LEGACY))
from core import read, write, sha, digest, jsonl, question_id, fingerprint
from suite_utils import immutable_json

CONDITIONS = {'A': {'stage1': True, 'rank': 8, 'alpha': 16},
              'B': {'stage1': False, 'rank': 8, 'alpha': 16},
              'C': {'stage1': True, 'rank': 32, 'alpha': 64},
              'D': {'stage1': True, 'rank': 64, 'alpha': 128}}
MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
BASE = 'Qwen/Qwen3-4B-Base'
REVISION = '906bfd4b4dc7f14ee4320094d8b41684abff8539'
SEED = 42
TOTAL_STEPS = 375


def original_hash(value):
    # run03 uses default JSON separators (not core.digest's compact encoding).
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def full_solution(solution, answer):
    # Same final-expression rule as the archived run03 worker.
    # run03 recognizes boxed, not fbox. Keep that distinction.
    from reference_recipe import boxed_answer
    solution = solution.strip()
    return solution if boxed_answer(solution) is not None else (
        solution+'\n\nTherefore, $\\boxed{'+(boxed_answer(answer) or answer.strip())+'}$.')


def order_indices(n, seed=SEED):
    import torch
    from accelerate.data_loader import SeedableRandomSampler
    sampler = SeedableRandomSampler(range(n), data_seed=seed)
    return list(iter(sampler))


def groups_for(order):
    if len(order) != 6000 or sorted(order) != list(range(6000)):
        raise ValueError('Expected an exact permutation of run03 6000 examples')
    return [order[i:i+16] for i in range(0,6000,16)]


def validate_inputs(inputs, include_stage2=True):
    manifest = read(LEGACY/'input_manifest.json')
    for entry in manifest['files']:
        if entry['path'].startswith('stage2/') and not include_stage2: continue
        if entry['path'].startswith(('stage1/','stage2/')) or entry['path'] in ('train.jsonl','split.json'):
            if sha(inputs/entry['path']) != entry['sha256']:
                raise ValueError('Pinned input changed: '+entry['path'])


def verify_model(path):
    marker = path/'artifacts.json'
    if not marker.exists(): return False
    if any(sha(path/name) != expected for name,expected in read(marker).items()):
        raise ValueError('Model artifact changed: '+str(path))
    return True


def save_model(model, tok, target):
    import shutil
    if target.exists():
        if verify_model(target): return
        raise ValueError('Refusing to replace an unverified model directory')
    tmp = target.with_name(target.name+'.partial')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(tmp,safe_serialization=True,max_shard_size='2GB')
    tok.save_pretrained(tmp)
    write(tmp/'artifacts.json',fingerprint(tmp))
    tmp.rename(target)


def backend_candidates(unsloth_available, flash_available):
    families = [('hf','nf4','sdpa')]
    if flash_available: families += [('hf','nf4','flash_attention_2')]
    if unsloth_available: families += [('unsloth','nf4','auto')]
    return families


def candidate_id(c):
    return f"{c['backend']}_{c['weights']}_{c['attention']}_mb{c['microbatch']}_gc{int(c['checkpointing'])}_{c['layout']}"
