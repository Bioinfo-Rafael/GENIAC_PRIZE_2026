"""Build a standalone, readable notebook from code-only sources. No data/model access."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE.parent / 'inference_config.json').read_text())
HELPERS = (HERE / 'dev_eval.py').read_text()
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip().splitlines(True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.strip().splitlines(True),
                  'execution_count': None, 'outputs': []})


markdown(r'''
# 開発用評価：現在の Stage 1 + Stage 2 モデルを測る

**学習し直す必要はありません。保存済みの2つのadapterを使います。**
許可教材の **PRM800K math_splits/train と GSM8K main/train** から、既存の学習問題を除いた固定問題で評価します。
大会の推論用データ・提出JSONは開きません。提出回数は消費しません。
開発正答率は公式の0.14095と異なる指標で、0.22到達を予測するものではありません。

### 実行方法（新しい Colab T4）

この1冊だけをアップロードしてください。補助コードと配布chat templateは通常のコードセルに同梱済みです。
学習時の Unsloth ランタイムは使い回しません。

1. 設定の `PHASE = "setup"` のまま **すべて実行** → 完了後「セッションを再起動」。
2. `PHASE = "merge"` に変えて **すべて実行** → 完了後、もう一度「セッションを再起動」。
3. `PHASE = "evaluate"` に変えて **すべて実行** → 最後の集計を確認。

**再起動は「切断してランタイムを削除」ではありません。** `/content/` の統合モデルを残します。
削除してしまった場合は setup → merge をやり直せば、Driveの同じ条件の途中保存から再開できます。
初回は `EVAL_SIZE = "quick"`（128問）。400問に進むときは `EVAL_SIZE="dev"`、`EVAL_RUN_ID`も別名へ。
同じモデル・条件の中断再開ではIDを変えません。400問は128問を含みますが、別runとして全件生成します。

Driveに run01 のadapter 2つと学習データのキャッシュが必要です。別の学習履歴がある場合は、
そのキャッシュも `TRAIN_CACHE_SPECS` に追加してください。この一覧にない過去の学習は自動検出できません。
モデル/データの処理はColabのみ。ローカルでは実行しないでください。
''')
code(r'''
# [config] 最初は PHASE だけ変更。他は今回の run01 に合わせてある。
PHASE = "setup"                    # setup → 再起動 → merge → 再起動 → evaluate
EVAL_SIZE = "quick"                 # quick:128問 / dev:400問（finalはまだ評価しない）
EVAL_RUN_ID = "run01_quick_v1"
WORKSPACE = "/content/drive/MyDrive/geniac_workspace"
STAGE1_ADAPTER_DIR = WORKSPACE + "/adapters/t4_20261013_run01_stage1"
STAGE2_ADAPTER_DIR = WORKSPACE + "/adapters/t4_20261013_run01_stage2"
TRAIN_CACHE_SPECS = [
    {"name": "run01_stage1", "path": WORKSPACE + "/data_cache/tulu3_persona_if_2000.jsonl", "expected_rows": 2000},
    {"name": "run01_stage2", "path": WORKSPACE + "/data_cache/gsm8k_2000.jsonl", "expected_rows": 2000},
]
DIAGNOSTIC_CACHE_NAME = "run01_stage2"
BASE_MODEL = "Qwen/Qwen3-4B-Base"
BASE_REVISION = "main"               # merge時にcommitへ解決し記録する
MERGED_DIR = "/content/dev_eval_merged"
DEV_ROOT = WORKSPACE + "/dev_eval"
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 2048
GPU_MEMORY_UTILIZATION = 0.85
ENFORCE_EAGER = False
CHUNK_SIZE = 8                      # chunk単位でDriveへ保存
''')
markdown('## 1. Colab確認と環境構築\nsetupだけが依存ライブラリを導入します。実行済み推論ノートブックと同じvLLM wheelを使用します。')
code(r'''
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
''')
code(r'''
# [setup-install] 導入エラーなら停止。完了後に手動でセッション再起動。
if PHASE == 'setup':
    import subprocess, sys
    with open('/content/dev_eval_install.log', 'w') as log:
        commands = [
            [sys.executable, '-m', 'pip', 'install', '-q', 'uv'],
            [sys.executable, '-m', 'uv', 'pip', 'install', '--system',
             'https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl',
             '--extra-index-url', 'https://download.pytorch.org/whl/cu129',
             'peft', 'accelerate', 'datasets', 'math-verify==0.9.0'],
            [sys.executable, '-m', 'uv', 'pip', 'uninstall', '--system', 'torchao', 'torchcodec'],
        ]
        for cmd in commands:
            if subprocess.run(cmd, stdout=log, stderr=log).returncode:
                raise RuntimeError('導入失敗。/content/dev_eval_install.log を確認してください。')
    print('setup完了 → セッションを再起動 → PHASE="merge" → すべて実行。')
''')
markdown('## 2. 評価処理の定義\nここは関数の定義だけです。圧縮payloadや別notebookの実行はありません。')
code('# [helpers]\n' + HELPERS + '\nHELPERS_SHA256 = ' + repr(hashlib.sha256(HELPERS.encode()).hexdigest()))
# Embed the supplied template literally, without fixing escape sequences or changing framing.
code('# [inference-template] 配布 inference_config.json の値をそのまま同梱\nCHAT_TEMPLATE = '
     + repr(CONFIG['chat_template']['jinja']) + '\nTEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()')
code(r'''
# [workspace] setupではDriveを開かない。
if PHASE != 'setup':
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('ColabのランタイムをT4 GPUへ変更してください。')
    if 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('この評価は無料版T4を想定しています。T4に切り替えてください。')
    from google.colab import drive
    drive.mount('/content/drive')
    Path(DEV_ROOT).mkdir(parents=True, exist_ok=True)
    # Qwen/GSM8K/PRMは公開。既存HF認証があればそのまま利用。秘密情報を表示しない。
    MODEL_INPUT = {'base_model': BASE_MODEL, 'requested_revision': BASE_REVISION,
                   'stage1': adapter_fingerprint(STAGE1_ADAPTER_DIR),
                   'stage2': adapter_fingerprint(STAGE2_ADAPTER_DIR),
                   'template_sha256': TEMPLATE_SHA256, 'dtype': 'float16',
                   'order': ['base', 'stage1_merge', 'stage2_merge']}
    MERGE_MANIFEST = Path(MERGED_DIR) / 'dev_merge_manifest.json'
''')
markdown('## 3. 保存済みadapterを統合（mergeのみ）\nBase → Stage 1 → Stage 2 の順。これは学習ではありません。約8GBの統合モデルをColab内に保存します。')
code(r'''
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
        tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR, trust_remote_code=False)
        tokenizer.chat_template = CHAT_TEMPLATE
        model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size='2GB')
        tokenizer.save_pretrained(MERGED_DIR)
        artifacts = {p.name: file_hash(p) for p in sorted(Path(MERGED_DIR).iterdir()) if p.is_file()}
        manifest = {'input': MODEL_INPUT, 'base_commit': revision, 'files_sha256': artifacts,
                    'versions': package_versions(), 'merge_seconds': time.monotonic() - t0}
        atomic_json(MERGE_MANIFEST, manifest)
        atomic_json(Path(DEV_ROOT) / ('merge_' + digest(manifest)[:16] + '.json'), manifest)
        del model, tokenizer
    print('merge完了 → セッションのみ再起動 → PHASE="evaluate" → すべて実行。')
''')
markdown(r'''
## 4. 固定問題の準備・学習分の除外（evaluateのみ）

初回のみPRM 300問＋GSM 100問をdevに、別の400問を最終確認用に確保します。
quickはdevのうち96＋32問。最終確認用はこのノートブックで採点しません。
NFKCと空白正規化後の問題ハッシュで重複を除き、問題内容・正誤で選び直しません。
言い換えなどの意味的重複を完全に検出する仕組みではありません。

`split_v1.json` はDriveに固定保存。以後の学習から `heldout_problem_hashes.json` の全800問を除外してください。
**既存の学習notebookには、この除外処理はまだ入っていません。** 次の学習実装時に追加します。
正解自体を採点器が読めない場合も問題は選び直さず、その件数と採点可能分の正答率を別に記録します。
''')
code(r'''
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
''')
markdown(r'''
## 5. vLLMで単発生成・機械採点

提出経路と同じFP16統合モデル・配布chat template・単一user turn・temperature=0・seed=42です。
開発用は全問最大2048トークン。大会のtask別上限（2048/4096）と完全に同一の評価ではありません。

答案は最後の `\boxed{}` / `\fbox{}`、行頭の `####` / `Answer:` / `Final answer is`、
または数値だけの回答から抽出します。途中に現れた数値で正答扱いにはしません。
GSM8Kは数値の厳密比較、PRMは[Math-Verify](https://github.com/huggingface/Math-Verify)の数式比較です。
自由文・証明の品質は採点できず、公式採点器の再現ではありません。
''')
code(r'''
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
''')
code(r'''
# [training-diagnostics] 現在のStage 2教材を集計。追加学習はしない。
if PHASE == 'evaluate':
    if DIAGNOSTIC_CACHE_NAME not in training_rows:
        raise RuntimeError('DIAGNOSTIC_CACHE_NAME を確認してください。')
    from transformers import AutoTokenizer
    saved_training_tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR, trust_remote_code=False)
    diagnostics = training_diagnostics(saved_training_tokenizer, training_rows[DIAGNOSTIC_CACHE_NAME])
    diagnostics['saved_adapter_template_matches_inference'] = (saved_training_tokenizer.chat_template == CHAT_TEMPLATE)
    diagnostics['saved_adapter_eos_matches_inference'] = (saved_training_tokenizer.eos_token_id == tokenizer.eos_token_id)
    del saved_training_tokenizer
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
''')
code(r'''
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
    summary.update({'run_id': EVAL_RUN_ID, 'training_diagnostics': diagnostics,
                    'gold_parse_failures': gold_failures, 'model_load_seconds_this_session': model_load_seconds,
                    'official_score_reference': 0.14095,
                    'official_score_note': 'ユーザー報告のrun01。今回の開発評価とは別のスコア。'})
    atomic_json(output_dir / 'summary.json', summary)
    lines = [f'run: {EVAL_RUN_ID} / {EVAL_SIZE}', '※公式スコアとは別の開発指標です。']
    for name, m in summary['by_dataset'].items():
        lo, hi = m['wilson95']
        lines.append(f"{name}: {m['correct']}/{m['n']} = {m['accuracy']:.3f} "
                     f"(95%区間 {lo:.3f}–{hi:.3f}), 抽出失敗 {m['extraction_failure_rate']:.1%}, "
                     f"長さ打切り {m['length_stop_rate']:.1%}, 正解側解析失敗 {m['gold_unparseable_count']}")
    lines.append(f"生成: {summary['generation_seconds'] / 60:.1f}分 / {summary['seconds_per_problem']:.1f}秒/問")
    lines.append(f"c1が解説を削る割合: {diagnostics['c1_removes_solution_suffix_rate']:.1%}")
    lines.append(f"全文で2048超: {diagnostics['over_2048_rate']['full']:.1%}")
    if gold_failures:
        lines.append('正解側解析失敗があるため全問正答率は下限。summary.jsonのgradable_accuracyも確認。')
    report = '\n'.join(lines)
    (output_dir / 'summary.txt').write_text(report + '\n')
    print(report)
    print('\n共有するファイル:', str(output_dir / 'summary.json'))
''')
markdown(r'''
## 6. 実行後、次にすること

`MyDrive/geniac_workspace/dev_eval/runs/<EVAL_RUN_ID>/summary.json`（または `summary.txt`）を共有してください。
問題・回答本文を含む `split_v1.json` と `predictions_private.json` は共有不要です。

- **抽出失敗が多い**：答案形式・EOS・採点器の対応範囲を先に確認。
- **長さによる打切りが多い**：出力長と繰返しの問題を切り分ける。長くすれば必ず改善するわけではありません。
- **GSMは解けるがPRMが弱い**：より難しい許可教材への変更を候補にする。
- **c1が多くの解説を削る**：全文解説での比較学習を候補にする。2048超の割合も併せて判断。

これは仮説を絞るための診断で、原因の証明ではありません。quickの差が小さければ固定dev400問で確認します。
正解側の解析失敗・採点器エラーがある場合は、モデル改善より先に測定を点検します。
ライブラリや採点規則を変更したら、旧モデルも同じ規則で再評価してください。

### 保存物

| Driveのdev_eval以下 | 用途 |
| --- | --- |
| `split_v1.json` | 固定dev/final、取得元commit、初回の学習履歴。内容はColabに留める |
| `heldout_problem_hashes.json` | 次回の学習から除外する800問のハッシュ |
| `merge_*.json` | 統合元adapterとBaseのcommit・ファイルのハッシュ |
| `runs/<ID>/predictions_private.json` | 途中再開用の生成回答。共有不要 |
| `runs/<ID>/summary.json` / `summary.txt` | 共有用の集計（問題・答案本文なし） |

環境構築・merge・実推論はこのファイルを作成した時点では未実行です。所要時間は実行結果で確認します。
''')
for i, cell in enumerate(cells):
    cell['id'] = f'dev-{i:02d}'
notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'},
                         'colab': {'name': 'evaluate_dev.ipynb'}, 'accelerator': 'GPU'}}
(HERE / 'evaluate_dev.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(f'evaluate_dev.ipynb: {len(cells)} cells')
