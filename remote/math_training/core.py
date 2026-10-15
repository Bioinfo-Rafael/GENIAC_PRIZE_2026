"""Deterministic, CPU-only experiment preparation. Never reads competition answers."""
import hashlib
import json
import os
import random
import unicodedata
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def question_id(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def fingerprint(path):
    path = Path(path)
    return {p.name: sha(p) for p in sorted(path.iterdir())
            if p.is_file() and (p.suffix in ('.json', '.safetensors', '.jinja'))}


def prepare(train, split, count, seed):
    if count < 4 or count % 4:
        raise ValueError('prompts must be a positive multiple of 4')
    if digest(split['split']) != split['split_sha256']:
        raise ValueError('Fixed split hash mismatch')
    heldout = split['split']['dev'] + split['split']['final']
    heldout_ids = {question_id(x['problem']) for x in heldout}
    if len(heldout_ids) != 800:
        raise ValueError('Expected 800 distinct held-out questions')
    seen, pools = set(), {'prm800k': [], 'gsm8k': []}
    for row in train:
        key = question_id(row['problem'])
        if key in heldout_ids or key in seen:
            raise ValueError('Training duplication/heldout overlap: ' + key)
        if row.get('problem_hash', key) != key:
            raise ValueError('Training question hash mismatch')
        seen.add(key)
        pools[row['dataset']].append(dict(row, id=key))
    if len(seen) != 6000 or {k: len(v) for k, v in pools.items()} != {'prm800k': 4500, 'gsm8k': 1500}:
        raise ValueError('Expected the unchanged run03 6000-example cache')
    for pool in pools.values():
        pool.sort(key=lambda x: digest([seed, x['id']]))
    # Every prefix divisible by four preserves the inherited 3:1 ratio.
    selected = []
    for i in range(count // 4):
        selected.extend(pools['prm800k'][3*i:3*i+3])
        selected.append(pools['gsm8k'][i])
    return selected


def classify(output):
    if output['finish_reason'] != 'stop':
        return 'truncated'
    grade = output['grade']
    if any(grade[k] for k in ('extraction_failed', 'grader_error', 'gold_unparseable')):
        return 'unknown'
    return 'correct' if grade['correct'] else 'incorrect'


def pairs_from_outputs(rows, generations, seed):
    pairs, stats = [], Counter()
    for row in rows:
        outputs = generations[row['id']]
        by_status = {'correct': [], 'incorrect': []}
        seen = set()
        for out in outputs:
            status = classify(out)
            stats['outputs_' + status] += 1
            if status in by_status and out['text'] not in seen:
                by_status[status].append(out)
                seen.add(out['text'])
        good, bad = by_status['correct'], by_status['incorrect']
        kind = 'mixed' if good and bad else ('only_correct' if good else ('only_incorrect' if bad else 'unusable'))
        stats['questions_' + kind] += 1
        if good and bad:
            chosen = min(good, key=lambda x: digest([seed, row['id'], x['index']]))
            rejected = min(bad, key=lambda x: digest([seed, row['id'], x['index']]))
            pairs.append({'id': row['id'], 'problem': row['problem'], 'dataset': row['dataset'],
                          'chosen': chosen['text'], 'rejected': rejected['text']})
    return pairs, dict(stats)


def tokenize_answer(tokenizer, problem, answer, max_length):
    prefix = tokenizer.apply_chat_template([{'role': 'user', 'content': problem}],
                                           tokenize=True, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': problem},
                                        {'role': 'assistant', 'content': answer}], tokenize=True)
    if ids[:len(prefix)] != prefix:
        raise ValueError('Prompt/completion token boundary changed')
    if tokenizer.eos_token_id is None:
        raise ValueError('Missing EOS')
    if ids[-1] != tokenizer.eos_token_id:
        ids.append(tokenizer.eos_token_id)
    if len(ids) > max_length:
        return None
    return {'input_ids': ids, 'labels': [-100] * len(prefix) + ids[len(prefix):]}


def update_groups(n, epochs, batch, seed):
    result = []
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        result.extend(order[i:i+batch] for i in range(0, n, batch))
    return result


def comparison(baseline, candidate, seed=42, draws=2000):
    if set(baseline) != set(candidate):
        raise ValueError('Cannot compare different evaluated question sets')
    ids = sorted(baseline)
    deltas = [int(candidate[k]['grade']['correct']) - int(baseline[k]['grade']['correct']) for k in ids]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(draws))
    return {'n': len(ids), 'delta_accuracy': sum(deltas)/len(deltas),
            'wrong_to_right': deltas.count(1), 'right_to_wrong': deltas.count(-1),
            'paired_bootstrap_95': [boot[int(draws*.025)], boot[int(draws*.975)]]}
