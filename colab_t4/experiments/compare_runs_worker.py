import sys


# [config] 最初は PHASE だけ変更。他は今回の run01 に合わせてある。
PHASE = sys.argv[1]                    # setup → 再起動 → merge → 再起動 → evaluate
EVAL_SIZE = sys.argv[3]                 # quick:128問 / dev:400問（finalはまだ評価しない）
CANDIDATE = sys.argv[2]
assert CANDIDATE in ("run02", "run03")
EVAL_RUN_ID = "compare_v1_" + CANDIDATE + "_" + EVAL_SIZE
WORKSPACE = "/content/drive/MyDrive/geniac_workspace"
STAGE1_ADAPTER_DIR = WORKSPACE + "/adapters/t4_20261013_run01_stage1"
TRAIN_RUN = {"run02":"t4_20261013_run02_prmfull", "run03":"t4_20261014_run03_prmfull6k"}[CANDIDATE]
STAGE2_ADAPTER_DIR = WORKSPACE + "/adapters/" + TRAIN_RUN + "_stage2"
TRAIN_CACHE_SPECS = [
    {"name": "run01_stage1", "path": WORKSPACE + "/data_cache/tulu3_persona_if_2000.jsonl", "expected_rows": 2000},
    {"name": "run01_stage2", "path": WORKSPACE + "/data_cache/gsm8k_2000.jsonl", "expected_rows": 2000},
]
DIAGNOSTIC_CACHE_NAME = "run01_stage2"
BASE_MODEL = "Qwen/Qwen3-4B-Base"
BASE_REVISION = '906bfd4b4dc7f14ee4320094d8b41684abff8539'               # merge時にcommitへ解決し記録する
MERGED_DIR = "/content/dev_compare_merged_" + CANDIDATE
DEV_ROOT = WORKSPACE + "/dev_eval"
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 2048
GPU_MEMORY_UTILIZATION = 0.85
ENFORCE_EAGER = False
CHUNK_SIZE = 8                      # chunk単位でDriveへ保存
TRAIN_CACHE_SPECS += [
    {"name":"run02_stage2", "path":WORKSPACE+"/data_cache/t4_20261013_run02_prmfull_2000.jsonl", "expected_rows":2000},
    {"name":"run03_stage2", "path":WORKSPACE+"/data_cache/t4_20261014_run03_prmfull6k_6000.jsonl", "expected_rows":6000},
]


# [colab-guard]
import importlib.util
from pathlib import Path
try:
    _is_colab = Path('/content').is_dir() and importlib.util.find_spec('google.colab') is not None
except ModuleNotFoundError:
    _is_colab = False
if not _is_colab:
    raise RuntimeError('Colab T4 に接続してから実行してください。ローカル実行は停止しました。')
if PHASE not in ('setup', 'merge', 'evaluate') or EVAL_SIZE not in ('quick', 'dev'):
    raise ValueError('PHASE / EVAL_SIZE を確認してください。')
if not __import__('re').fullmatch(r'[A-Za-z0-9_-]+', EVAL_RUN_ID):
    raise ValueError('EVAL_RUN_ID は英数字・ハイフン・アンダースコアのみです。')
if not (0 < MAX_NEW_TOKENS < MAX_MODEL_LEN and CHUNK_SIZE > 0):
    raise ValueError('生成長 / CHUNK_SIZE を確認してください。')

# [helpers]
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

HELPERS_SHA256 = 'f991f74623c6653ac624612db685efd76eec600f198dc39dd7ad3f5920dcc450'

# [inference-template] 配布 inference_config.json の値をそのまま同梱
CHAT_TEMPLATE = '{%- if tools %}\n    {{- \'<|im_start|>system\\n\' }}\n    {%- if messages[0].role == \'system\' %}\n        {{- messages[0].content + \'\\n\\n\' }}\n    {%- endif %}\n    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}\n    {%- for tool in tools %}\n        {{- "\\n" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}\n{%- else %}\n    {%- if messages[0].role == \'system\' %}\n        {{- \'<|im_start|>system\\n\' + messages[0].content + \'<|im_end|>\\n\' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = \'\' %}\n    {%- endif %}\n    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + content + \'<|im_end|>\' + \'\\n\' }}\n    {%- elif message.role == "assistant" %}\n        {{- \'<|im_start|>\' + message.role + \'\\n\' + content }}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- \'\\n\' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- \'<tool_call>\\n{"name": "\' }}\n                {{- tool_call.name }}\n                {{- \'", "arguments": \' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- \'}\\n</tool_call>\' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- \'<|im_end|>\\n\' }}\n    {%- elif message.role == "tool" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}\n            {{- \'<|im_start|>user\' }}\n        {%- endif %}\n        {{- \'\\n<tool_response>\\n\' }}\n        {{- content }}\n        {{- \'\\n</tool_response>\' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}\n            {{- \'<|im_end|>\\n\' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|im_start|>assistant\\n\' }}\n{%- endif %}'
TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()

# [workspace] setupではDriveを開かない。
if PHASE != 'setup':
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('ColabのランタイムをT4 GPUへ変更してください。')
    if 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('この評価は無料版T4を想定しています。T4に切り替えてください。')
    if not Path(WORKSPACE).is_dir():
        raise RuntimeError('親notebookでDriveを認証してください。')
    Path(DEV_ROOT).mkdir(parents=True, exist_ok=True)
    # Qwen/GSM8K/PRMは公開。既存HF認証があればそのまま利用。秘密情報を表示しない。
    MODEL_INPUT = {'base_model': BASE_MODEL, 'requested_revision': BASE_REVISION,
                   'stage1': adapter_fingerprint(STAGE1_ADAPTER_DIR),
                   'stage2': adapter_fingerprint(STAGE2_ADAPTER_DIR),
                   'template_sha256': TEMPLATE_SHA256, 'dtype': 'float16',
                   'order': ['base', 'stage1_merge', 'stage2_merge']}
    MERGE_MANIFEST = Path(MERGED_DIR) / 'dev_merge_manifest.json'
    report = read_json(Path(WORKSPACE)/'experiments'/TRAIN_RUN/'training_summary.json')
    if report.get('complete') is not True or file_hash(Path(STAGE2_ADAPTER_DIR)/'adapter_model.safetensors') != report['stage2_weights_sha256']:
        raise RuntimeError('学習完了記録とStage 2の重みが一致しません。')


# [merge-model]
if PHASE == 'merge':
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if Path(MERGED_DIR).exists():
        if not MERGE_MANIFEST.exists() or read_json(MERGE_MANIFEST)['input'] != MODEL_INPUT:
            raise RuntimeError('MERGED_DIR に未完了/別条件のモデルがあります。空の別パスを設定してください。')
        print('同条件のmerge完了記録があります。evaluateで整合性を再確認します。')
    else:
        t0 = time.monotonic()
        revision = HfApi().model_info(BASE_MODEL, revision=BASE_REVISION).sha
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, revision=revision, dtype=torch.float16,
            device_map='cuda', low_cpu_mem_usage=True, trust_remote_code=False)
        for label, adapter in [('Stage 1', STAGE1_ADAPTER_DIR), ('Stage 2', STAGE2_ADAPTER_DIR)]:
            print(label + ': adapterを読込・統合')
            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload(safe_merge=True)
            torch.cuda.empty_cache()
        cfg = read_json(Path(STAGE2_ADAPTER_DIR)/'tokenizer_config.json')
        from importlib.metadata import version
        kw = {}
        if int(version('transformers').split('.')[0]) < 5 and isinstance(cfg.get('extra_special_tokens'), list):
            kw = {'extra_special_tokens': {}, 'additional_special_tokens': cfg['extra_special_tokens']}
        tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR, trust_remote_code=False, **kw)
        for tid, token in cfg.get('added_tokens_decoder', {}).items():
            assert tokenizer.convert_tokens_to_ids(token['content']) == int(tid), 'Token ID changed'
        assert tokenizer.chat_template == CHAT_TEMPLATE, 'Adapter chat template mismatch'
        tokenizer.chat_template = CHAT_TEMPLATE
        model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size='2GB')
        tokenizer.save_pretrained(MERGED_DIR)
        artifacts = {p.name: file_hash(p) for p in sorted(Path(MERGED_DIR).iterdir()) if p.is_file()}
        manifest = {'input': MODEL_INPUT, 'base_commit': revision, 'files_sha256': artifacts,
                    'versions': package_versions(), 'merge_seconds': time.monotonic() - t0}
        atomic_json(MERGE_MANIFEST, manifest)
        atomic_json(Path(DEV_ROOT) / ('merge_' + digest(manifest)[:16] + '.json'), manifest)
        del model, tokenizer
    print('merge完了。子プロセスを終了してGPUメモリを解放します。')

# [prepare-dev]
if PHASE == 'evaluate':
    if not MERGE_MANIFEST.exists():
        raise RuntimeError('統合モデルがありません。PHASE="merge" を先に実行してください。')
    manifest = read_json(MERGE_MANIFEST)
    if manifest['input'] != MODEL_INPUT:
        raise RuntimeError('統合時とadapter/テンプレート等が異なります。別MERGED_DIRでmergeしてください。')
    for name, expected in manifest['files_sha256'].items():
        if not (Path(MERGED_DIR) / name).is_file() or file_hash(Path(MERGED_DIR) / name) != expected:
            raise RuntimeError('統合モデルのファイルが欠損/変更されています。')
    frozen_path = Path(DEV_ROOT)/'split_v1.json'
    if not frozen_path.is_file() or read_json(frozen_path)['split_sha256'] != '52d27952511b10494d31620fca969e40a1a3986bb1d3bea28ef5f09d9e96ac55':
        raise RuntimeError('固定splitがありません/異なります。評価問題を新規選択せず停止します。')
    split_manifest, current_training_inventory, training_rows = prepare_split(DEV_ROOT, TRAIN_CACHE_SPECS)
    dev_rows = split_manifest['split']['dev']
    if EVAL_SIZE == 'quick':
        quick_ids = set(split_manifest['split']['quick_ids'])
        dev_rows = [r for r in dev_rows if r['id'] in quick_ids]
    assert len(dev_rows) == (128 if EVAL_SIZE == 'quick' else 400)
    # Math-Verify warnings can include answer text. Keep those off notebook output.
    import logging
    for logger_name in ('math_verify', 'math_verify.parser', 'math_verify.grader', 'latex2sympy2_extended'):
        logging.getLogger(logger_name).disabled = True
    gold_answers, gold_failures = prepare_gold(dev_rows)
    print('評価件数:', dict(Counter(r['dataset'] for r in dev_rows)))
    print('正解側の解析失敗件数:', gold_failures)
    print('問題文・正解・回答プレビューは表示しません。')

# [llm-build] 新しいカーネルで実行。既存モデルがVRAMに残っていたら停止。
if PHASE == 'evaluate':
    if 'llm' in globals() or torch.cuda.memory_allocated() > 1024 ** 3:
        raise RuntimeError('セッションを再起動し、PHASE="evaluate"で再実行してください。')
    import ctypes, glob, importlib
    # 実行済み推論コピーと同じColab/CUDA互換対応。開発用のみに適用する。
    libdirs = sorted({str(Path(p).parent) for p in glob.glob(
        '/usr/local/lib/python3*/dist-packages/nvidia/**/lib*.so.12', recursive=True)})
    if libdirs:
        os.environ['LD_LIBRARY_PATH'] = ':'.join(libdirs + [os.environ.get('LD_LIBRARY_PATH', '')]).strip(':')
    cudart = glob.glob('/usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so.12')
    if cudart:
        ctypes.CDLL(cudart[0], mode=ctypes.RTLD_GLOBAL)
    os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'
    sysutils = importlib.import_module('vllm.utils.system_utils')
    if hasattr(sysutils, 'suppress_stdout'):
        sysutils.suppress_stdout = contextlib.nullcontext
    from vllm import LLM
    t0 = time.monotonic()
    llm = LLM(model=MERGED_DIR, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=GPU_MEMORY_UTILIZATION, dtype='float16',
              enforce_eager=ENFORCE_EAGER, trust_remote_code=False)
    model_load_seconds = time.monotonic() - t0
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template != CHAT_TEMPLATE:
        raise RuntimeError('vLLM tokenizerと配布テンプレートが一致しません。')
    # Require enough space for the entire configured generation budget; never silently truncate prompts.
    prompt_lengths = [len(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': r['problem']}], tokenize=True, add_generation_prompt=True)) for r in dev_rows]
    too_long = sum(n + MAX_NEW_TOKENS > MAX_MODEL_LEN for n in prompt_lengths)
    if too_long:
        raise RuntimeError(f'{too_long}件で入力＋出力上限が文脈長を超えます。共通条件を見直してください。問題は除外しません。')
    print('モデル読込秒:', round(model_load_seconds, 1), '/ 入力最大トークン:', max(prompt_lengths))

# [evaluate-and-save]
if PHASE == 'evaluate':
    signature = {'code_version': EVAL_CODE_VERSION, 'helpers_sha256': HELPERS_SHA256,
                 'model_sha256': digest({k: v for k, v in manifest.items() if k != 'merge_seconds'}),
                 'base_commit': manifest['base_commit'],
                 'split_sha256': split_manifest['split_sha256'], 'eval_size': EVAL_SIZE,
                 'selected_ids_sha256': digest([r['id'] for r in dev_rows]),
                 'training_inventory': current_training_inventory,
                 'versions': package_versions(), 'gpu': torch.cuda.get_device_name(0),
                 'config': {'max_model_len': MAX_MODEL_LEN, 'max_new_tokens': MAX_NEW_TOKENS,
                            'temperature': 0, 'seed': 42, 'n': 1, 'dtype': 'float16',
                            'gpu_memory_utilization': GPU_MEMORY_UTILIZATION,
                            'enforce_eager': ENFORCE_EAGER, 'chunk_size': CHUNK_SIZE,
                            'template_sha256': TEMPLATE_SHA256}}
    output_dir = Path(DEV_ROOT) / 'runs' / EVAL_RUN_ID
    summary = evaluate_rows(llm, dev_rows, gold_answers, signature, output_dir,
                            MAX_NEW_TOKENS, CHUNK_SIZE)

    summary.update({'run_id': EVAL_RUN_ID, 'gold_parse_failures': gold_failures,
                    'model_load_seconds_this_session': model_load_seconds})
    atomic_json(output_dir/'summary.json', summary)
    print(CANDIDATE, json.dumps(summary['by_dataset'], ensure_ascii=False))
    print('集計:', output_dir/'summary.json')

