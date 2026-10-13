"""Mechanical preparation for run02. No model inference or semantic selection."""
import hashlib
import json
import re
import urllib.request
from collections import Counter


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def question_hash(text):
    import unicodedata
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def boxed_answer(text):
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if i and text[i - 1] == '\\':
            continue
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def full_solution(solution, answer):
    solution = solution.strip()
    if boxed_answer(solution) is not None:
        return solution
    return solution + '\n\nTherefore, $\\boxed{' + (boxed_answer(answer) or answer.strip()) + '}$.'


def fetch_allowed_records(sources):
    from datasets import load_dataset
    gsm = sources['gsm8k']
    assert gsm['id'] == 'openai/gsm8k' and gsm['config'] == 'main' and gsm['split'] == 'train'
    prm = sources['prm800k']
    url = ('https://media.githubusercontent.com/media/openai/prm800k/'
           + prm['revision'] + '/prm800k/math_splits/train.jsonl')
    assert prm['url'] == url
    with urllib.request.urlopen(url, timeout=120) as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != prm['file_sha256']:
        raise RuntimeError('PRM800Kの取得内容が固定版と異なります。')
    rows = []
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if all(isinstance(record.get(k), str) and record[k].strip()
               for k in ('problem', 'solution', 'answer')):
            rows.append({k: record[k].strip() for k in ('problem', 'solution', 'answer')}
                        | {'dataset': 'prm800k'})
    for row in load_dataset('openai/gsm8k', 'main', split='train', revision=gsm['revision']):
        solution, sep, answer = row['answer'].partition('####')
        if sep and row['question'].strip() and solution.strip() and answer.strip():
            rows.append({'problem': row['question'].strip(),
                         'solution': re.sub(r'<<[^>]*>>', '', solution).strip(),
                         'answer': answer.strip(), 'dataset': 'gsm8k'})
    return rows


def build_recipe(records, tokenizer, heldout_hashes, counts, max_length=2048, seed=42):
    """Fixed hash ordering, dedup, whole-example length filtering, no truncation."""
    blocked = set(heldout_hashes)
    seen, stats, pools = set(), Counter(), {name: [] for name in counts}
    if tokenizer.eos_token_id is None:
        raise RuntimeError('EOS token is required.')
    ordered = sorted(records, key=lambda r: (r['dataset'] != 'prm800k',
                      stable_hash([seed, question_hash(r['problem'])])))
    for row in ordered:
        key, name = question_hash(row['problem']), row['dataset']
        if key in blocked:
            stats[name + '_heldout_excluded'] += 1
            continue
        if key in seen:
            stats[name + '_duplicate_excluded'] += 1
            continue
        seen.add(key)
        target = full_solution(row['solution'], row['answer'])
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['problem']},
                                             {'role': 'assistant', 'content': target}], tokenize=True)
        if ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        if len(ids) > max_length:
            stats[name + '_over_length_excluded'] += 1
            continue
        pools[name].append((dict(row, problem_hash=key), ids))
    selected = []
    for name, n in counts.items():
        stats[name + '_eligible'] = len(pools[name])
        if len(pools[name]) < n:
            raise RuntimeError('全文が収まる教材が不足: ' + name)
        selected.extend(pools[name][:n])
    selected.sort(key=lambda pair: stable_hash(['shuffle', seed, pair[0]['problem_hash']]))
    raw = [r for r, ids in selected]
    tokenized = [{'input_ids': ids, 'attention_mask': [1] * len(ids)} for r, ids in selected]
    selected_hashes = [r['problem_hash'] for r in raw]
    assert len(selected_hashes) == len(set(selected_hashes)) and not set(selected_hashes) & blocked
    stats.update({'selected': len(raw), 'tokens': sum(len(x['input_ids']) for x in tokenized),
                  'max_tokens': max(len(x['input_ids']) for x in tokenized),
                  'all_end_with_eos': all(x['input_ids'][-1] == tokenizer.eos_token_id for x in tokenized)})
    return raw, tokenized, dict(stats)
