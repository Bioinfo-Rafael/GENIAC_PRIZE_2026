"""Development evaluation helpers. Embedded as readable cells in evaluate_dev.ipynb.
No data/model I/O occurs on import. Runtime defaults to Colab; the explicit local runner enables local execution.
"""
import contextlib
import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter
from fractions import Fraction
from pathlib import Path

EVAL_CODE_VERSION = "dev-v1"


def require_colab():
    # Historical function name retained for notebook compatibility.
    if os.environ.get("GENIAC_EVAL_RUNTIME") == "local":
        return
    import importlib.util
    try:
        is_colab = Path('/content').is_dir() and importlib.util.find_spec('google.colab') is not None
    except ModuleNotFoundError:
        is_colab = False
    if not is_colab:
        raise RuntimeError('データ処理・モデル実行は Google Colab に接続して行ってください。')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def problem_hash(text):
    # Mechanical equivalence only; paraphrases/semantic overlap are not detected.
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def training_inventory(specs):
    require_colab()
    if not specs:
        raise RuntimeError('学習済み問題のキャッシュを指定してください。')
    inventory, hashes, rows_by_name = [], set(), {}
    for spec in specs:
        path = Path(spec['path'])
        if not path.is_file():
            raise RuntimeError('学習キャッシュがありません: ' + str(path))
        rows = read_jsonl(path)
        if len(rows) != spec['expected_rows']:
            raise RuntimeError('学習キャッシュの件数不一致: ' + path.name)
        if any(not isinstance(r.get('problem'), str) or not r['problem'].strip() for r in rows):
            raise RuntimeError('学習キャッシュのproblem列が不正です。')
        hashes.update(problem_hash(r['problem']) for r in rows)
        inventory.append({'name': spec['name'], 'file_sha256': file_hash(path), 'rows': len(rows)})
        rows_by_name[spec['name']] = rows
    return inventory, hashes, rows_by_name


def select_split(records, excluded, seed=42):
    """Pure mechanical split. PRM wins exact cross-source ties; no answer filtering."""
    seen, pools, counts = set(excluded), {'prm800k': [], 'gsm8k': []}, Counter()
    for r in sorted(records, key=lambda x: (x['dataset'] != 'prm800k', x['source_id'])):
        key = problem_hash(r['problem'])
        if key in seen:
            counts['excluded_or_duplicate'] += 1
            continue
        seen.add(key)
        pools[r['dataset']].append(dict(r, id=key))
    split = {'dev': [], 'final': [], 'quick_ids': []}
    for name, ndev, nquick in [('prm800k', 300, 96), ('gsm8k', 100, 32)]:
        pool = sorted(pools[name], key=lambda r: digest([seed, r['id']]))
        if len(pool) < ndev * 2:
            raise RuntimeError('学習分除外後の候補が不足: ' + name)
        split['dev'].extend(pool[:ndev])
        split['final'].extend(pool[ndev:2 * ndev])
        split['quick_ids'].extend(r['id'] for r in pool[:nquick])
        counts[name + '_eligible'] = len(pool)
    split['counts'] = dict(counts)
    return split


def prepare_split(root, train_specs):
    require_colab()
    root = Path(root)
    inventory, excluded, training = training_inventory(train_specs)
    path = root / 'split_v1.json'
    if path.exists():
        saved = read_json(path)
        if digest(saved['split']) != saved['split_sha256']:
            raise RuntimeError('固定splitファイルが変更されています。')
        heldout = {r['id'] for s in ('dev', 'final') for r in saved['split'][s]}
        if heldout & excluded:
            raise RuntimeError('固定dev/finalと学習データが重複しています。学習側を修正してください。')
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        import urllib.request
        # ONLY approved sources, ONLY train partitions. No competition data access.
        revision = HfApi().dataset_info('openai/gsm8k').sha
        gsm = load_dataset('openai/gsm8k', 'main', split='train', revision=revision)
        records, invalid = [], Counter()
        for i, row in enumerate(gsm):
            body, separator, answer = row['answer'].partition('####')
            if not separator or not row['question'].strip() or not answer.strip():
                invalid['gsm8k'] += 1
                continue
            records.append({'dataset': 'gsm8k', 'source_id': str(i),
                            'problem': row['question'].strip(), 'answer': answer.strip(),
                            'subject': 'arithmetic'})
        # Resolve repository commit before downloading the permitted math_splits/train.jsonl.
        req = urllib.request.Request('https://api.github.com/repos/openai/prm800k/commits/main',
                                     headers={'User-Agent': 'geniac-development-evaluation'})
        with urllib.request.urlopen(req, timeout=60) as f:
            prm_revision = json.load(f)['sha']
        url = ('https://media.githubusercontent.com/media/openai/prm800k/' + prm_revision
               + '/prm800k/math_splits/train.jsonl')
        with urllib.request.urlopen(url, timeout=120) as f:
            payload = f.read()
        for i, line in enumerate(payload.decode().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ('problem', 'answer')):
                invalid['prm800k'] += 1
                continue
            records.append({'dataset': 'prm800k', 'source_id': str(row.get('unique_id', i)),
                            'problem': row['problem'].strip(), 'answer': row['answer'].strip(),
                            'subject': str(row.get('subject', 'unknown'))})
        split = select_split(records, excluded)
        saved = {'version': 1, 'seed': 42, 'split': split, 'split_sha256': digest(split),
                 'initial_training_inventory': inventory, 'invalid_schema_rows': dict(invalid),
                 'sources': {'gsm8k': {'id': 'openai/gsm8k', 'config': 'main', 'split': 'train',
                                      'revision': revision, 'license': 'MIT'},
                             'prm800k': {'url': url, 'revision': prm_revision, 'license': 'MIT',
                                         'file_sha256': hashlib.sha256(payload).hexdigest()}}}
        atomic_json(path, saved)
    # Future training must exclude both dev and final, including the quick subset.
    heldout = sorted(r['id'] for s in ('dev', 'final') for r in saved['split'][s])
    atomic_json(root / 'heldout_problem_hashes.json',
                {'normalization': 'NFKC + whitespace collapse + sha256(utf8)',
                 'split_sha256': saved['split_sha256'], 'hashes': heldout})
    return saved, inventory, training


def last_boxed(text):
    matches = list(re.finditer(r'\\(?:boxed|fbox)\s*\{', text))
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


def final_candidate(text):
    # Never search all intermediate numbers for a matching gold answer.
    if re.search(r'\\(?:boxed|fbox)\b', text):
        return last_boxed(text), 'boxed'
    matches = list(re.finditer(r'(?:^|\n)\s*(?:####|(?:final\s+)?answer\s*(?:is|:))\s*([^\n]+)',
                               text, re.IGNORECASE))
    if matches:
        return matches[-1].group(1).strip(), 'answer_marker'
    # A response consisting only of a number is also an unambiguous final answer.
    if numeric_answer(text) is not None:
        return text.strip(), 'numeric_only'
    return None, 'missing'


def numeric_answer(text):
    if text is None:
        return None
    s = text.strip().strip('$').strip().replace('−', '-')
    if s.endswith('.'):
        s = s[:-1]
    m = re.fullmatch(r'\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', s)
    if m:
        s = m[1] + '/' + m[2]
    # Remove only well-formed thousands separators; don't turn a list into a number.
    if re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', s):
        s = s.replace(',', '')
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?', s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def parse_math(answer):
    from math_verify import parse, LatexExtractionConfig
    # Input is ONLY the selected final expression, never the reasoning trace.
    answer = answer.strip()
    boxed = last_boxed(answer)
    if boxed is not None:
        answer = boxed
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer.strip('$')
    return parse('$' + answer + '$', extraction_config=[LatexExtractionConfig()],
                 extraction_mode='first_match', fallback_mode='no_fallback', parsing_timeout=3)


def prepare_gold(rows):
    gold, failures = {}, Counter()
    for row in rows:
        value = numeric_answer(row['answer']) if row['dataset'] == 'gsm8k' else parse_math(row['answer'])
        if value is None or value == []:
            failures[row['dataset']] += 1
        gold[row['id']] = value
    return gold, dict(failures)


def grade(row, text, gold):
    candidate, method = final_candidate(text)
    result = {'correct': False, 'extraction_failed': True, 'answer_method': method,
              'grader_error': False, 'gold_unparseable': gold is None or gold == []}
    if not candidate:
        return result
    try:
        if row['dataset'] == 'gsm8k':
            pred = numeric_answer(candidate)
            result['extraction_failed'] = pred is None
            result['correct'] = pred is not None and pred == gold
        else:
            from math_verify import verify
            pred = parse_math(candidate)
            result['extraction_failed'] = not bool(pred)
            result['correct'] = (not result['gold_unparseable'] and bool(pred)
                                 and bool(verify(gold, pred, strict=True, timeout_seconds=3, raise_on_error=True)))
    except Exception:
        result['grader_error'] = True
    return result


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.95996398454
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, center - radius), min(1, center + radius)]


def metrics(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    lengths = sorted(r['completion_tokens'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    return {'n': n, 'correct': correct, 'accuracy': correct / n,
            'wilson95': wilson(correct, n),
            'gold_unparseable_count': sum(r['gold_unparseable'] for r in rows),
            'gradable_n': sum(not r['gold_unparseable'] for r in rows),
            'gradable_accuracy': (correct / sum(not r['gold_unparseable'] for r in rows)
                                  if any(not r['gold_unparseable'] for r in rows) else None),
            'extraction_failure_rate': sum(r['extraction_failed'] for r in rows) / n,
            'length_stop_rate': sum(r['finish_reason'] == 'length' for r in rows) / n,
            'grader_error_count': sum(r['grader_error'] for r in rows),
            'completion_tokens_mean': statistics.mean(lengths),
            'completion_tokens_p95': lengths[math.ceil(.95 * n) - 1],
            'answer_methods': dict(Counter(r['answer_method'] for r in rows)),
            'finish_reasons': dict(Counter(r['finish_reason'] for r in rows))}


def resume_state(path, signature, ids):
    if not Path(path).exists():
        return {'signature': signature, 'rows': [], 'generation_seconds': 0, 'grading_seconds': 0}
    state = read_json(path)
    if state['signature'] != signature:
        raise RuntimeError('同じEVAL_RUN_IDに異なるモデル・設定・環境の結果があります。新しいIDにしてください。')
    actual = [r['id'] for r in state['rows']]
    if actual != ids[:len(actual)] or len(actual) > len(ids):
        raise RuntimeError('途中保存の問題ID/順序が固定splitと一致しません。')
    return state


def evaluate_rows(llm, rows, gold, signature, outdir, max_tokens, chunk_size, sampling_params_factory=None):
    require_colab()
    if sampling_params_factory is None:
        from vllm import SamplingParams
        sampling_params_factory = SamplingParams
    outdir = Path(outdir)
    state = resume_state(outdir / 'predictions_private.json', signature, [r['id'] for r in rows])
    params = sampling_params_factory(temperature=0, seed=42, max_tokens=max_tokens, n=1)
    for start in range(len(state['rows']), len(rows), chunk_size):
        batch = rows[start:start + chunk_size]
        conversations = [[{'role': 'user', 'content': r['problem']}] for r in batch]
        t0 = time.monotonic()
        outputs = llm.chat(conversations, params, use_tqdm=False)
        elapsed = time.monotonic() - t0
        if len(outputs) != len(batch):
            raise RuntimeError('生成結果数が要求数と一致しません。')
        t0 = time.monotonic()
        for row, output in zip(batch, outputs):
            if len(output.outputs) != 1:
                raise RuntimeError('単発生成以外の出力を検出しました。')
            answer = output.outputs[0]
            state['rows'].append({'id': row['id'], 'dataset': row['dataset'],
                                  'text': answer.text, 'completion_tokens': len(answer.token_ids),
                                  'prompt_tokens': len(output.prompt_token_ids),
                                  'finish_reason': answer.finish_reason,
                                  **grade(row, answer.text, gold[row['id']])})
        state['generation_seconds'] += elapsed
        state['grading_seconds'] += time.monotonic() - t0
        atomic_json(outdir / 'predictions_private.json', state)
        print(f"[eval] {len(state['rows'])}/{len(rows)} 保存済み / 今回の生成 {elapsed:.1f} 秒")
    by_dataset = {name: metrics([r for r in state['rows'] if r['dataset'] == name])
                  for name in ('prm800k', 'gsm8k')}
    summary = {'complete': len(state['rows']) == len(rows), 'signature': signature,
               'overall': metrics(state['rows']), 'by_dataset': by_dataset,
               'macro_accuracy': statistics.mean(v['accuracy'] for v in by_dataset.values()),
               'generation_seconds': state['generation_seconds'],
               'grading_seconds': state['grading_seconds'],
               'seconds_per_problem': state['generation_seconds'] / len(rows),
               'scope': 'development holdout; not an estimate of the official competition score'}
    atomic_json(outdir / 'summary.json', summary)
    return summary


def adapter_fingerprint(path):
    require_colab()
    path = Path(path)
    if not (path / 'adapter_config.json').is_file() or not (path / 'adapter_model.safetensors').is_file():
        raise RuntimeError('adapterの設定/重みがありません: ' + str(path))
    # Hash all relevant adapter/tokenizer artifacts, excluding model card prose.
    return {p.name: file_hash(p) for p in sorted(path.iterdir()) if p.is_file()
            and p.suffix in ('.json', '.safetensors', '.model', '.jinja', '.txt')}


def package_versions():
    from importlib.metadata import version, PackageNotFoundError
    out = {}
    for name in ('torch', 'transformers', 'peft', 'vllm', 'datasets', 'math-verify', 'sympy',
                 'latex2sympy2-extended', 'accelerate', 'tokenizers'):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def c1_target(solution, answer):
    # Same mechanical rule as the distributed math_sft/data.py (c1=800 chars).
    solution = solution.strip()
    answer = last_boxed(answer) or answer.strip()
    if len(solution) <= 800 and last_boxed(solution) is not None:
        return solution, solution
    head = solution[:800].rstrip()
    pivot = head.rfind('.')
    if pivot > 400:
        head = head[:pivot + 1]
    return head + '\n\nTherefore, $\\boxed{' + answer + '}$.', head


def training_diagnostics(tokenizer, rows):
    require_colab()
    counts = Counter()
    lengths = {'c1': [], 'full': []}
    for row in rows:
        solution, answer = row['solution'].strip(), row['answer'].strip()
        compressed, head = c1_target(solution, answer)
        counts['c1_removes_solution_suffix'] += head != solution
        full = solution if last_boxed(solution) is not None else (
            solution + '\n\nTherefore, $\\boxed{' + (last_boxed(answer) or answer) + '}$.')
        for mode, target in [('c1', compressed), ('full', full)]:
            ids = tokenizer.apply_chat_template([{'role': 'user', 'content': row['problem']},
                                                 {'role': 'assistant', 'content': target}], tokenize=True)
            lengths[mode].append(len(ids))
            counts[mode + '_over_2048'] += len(ids) > 2048
    n = len(rows)
    return {'n': n, 'c1_removes_solution_suffix_rate': counts['c1_removes_solution_suffix'] / n,
            'over_2048_rate': {k: counts[k + '_over_2048'] / n for k in lengths},
            'mean_formatted_tokens': {k: statistics.mean(v) for k, v in lengths.items()},
            'note': 'tokenizerによる再計算。実際のSFT labels/lossや過去のtruncationを測定したものではない。'}
