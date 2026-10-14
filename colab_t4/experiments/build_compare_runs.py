"""Build self-contained run02/run03 evaluation; no competition questions are accessed."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent
ref=json.loads((HERE/'run02_reference.json').read_text())
nb=json.loads((HERE/'evaluate_dev.ipynb').read_text())
parts=['import sys\n']
for c in nb['cells']:
 if c['cell_type']!='code':continue
 s=''.join(c['source']);label=s.splitlines()[0]
 if label.startswith('# [config]'):
  s=s.replace('PHASE = "setup"','PHASE = sys.argv[1]').replace('EVAL_SIZE = "quick"','EVAL_SIZE = sys.argv[3]')
  s=s.replace('EVAL_RUN_ID = "run01_quick_v1"','CANDIDATE = sys.argv[2]\nassert CANDIDATE in ("run02", "run03")\nEVAL_RUN_ID = "compare_v1_" + CANDIDATE + "_" + EVAL_SIZE')
  s=s.replace('STAGE2_ADAPTER_DIR = WORKSPACE + "/adapters/t4_20261013_run01_stage2"','TRAIN_RUN = {"run02":"t4_20261013_run02_prmfull", "run03":"t4_20261014_run03_prmfull6k"}[CANDIDATE]\nSTAGE2_ADAPTER_DIR = WORKSPACE + "/adapters/" + TRAIN_RUN + "_stage2"')
  s=s.replace('BASE_REVISION = "main"','BASE_REVISION = '+repr(ref['base']['commit']))
  s=s.replace('MERGED_DIR = "/content/dev_eval_merged"','MERGED_DIR = "/content/dev_compare_merged_" + CANDIDATE')
  s+='''
TRAIN_CACHE_SPECS += [
    {"name":"run02_stage2", "path":WORKSPACE+"/data_cache/t4_20261013_run02_prmfull_2000.jsonl", "expected_rows":2000},
    {"name":"run03_stage2", "path":WORKSPACE+"/data_cache/t4_20261014_run03_prmfull6k_6000.jsonl", "expected_rows":6000},
]
'''
 elif label.startswith('# [setup-install]'):continue
 elif label.startswith('# [workspace]'):
  s=s.replace("    from google.colab import drive\n    drive.mount('/content/drive')","    if not Path(WORKSPACE).is_dir():\n        raise RuntimeError('親notebookでDriveを認証してください。')")
  s+='''
    report = read_json(Path(WORKSPACE)/'experiments'/TRAIN_RUN/'training_summary.json')
    if report.get('complete') is not True or file_hash(Path(STAGE2_ADAPTER_DIR)/'adapter_model.safetensors') != report['stage2_weights_sha256']:
        raise RuntimeError('学習完了記録とStage 2の重みが一致しません。')
'''
 elif label.startswith('# [merge-model]'):
  s=s.replace("print('merge完了 → セッションのみ再起動 → PHASE=\"evaluate\" → すべて実行。')","print('merge完了。子プロセスを終了してGPUメモリを解放します。')")
  s=s.replace("tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR, trust_remote_code=False)","""cfg = read_json(Path(STAGE2_ADAPTER_DIR)/'tokenizer_config.json')
        from importlib.metadata import version
        kw = {}
        if int(version('transformers').split('.')[0]) < 5 and isinstance(cfg.get('extra_special_tokens'), list):
            kw = {'extra_special_tokens': {}, 'additional_special_tokens': cfg['extra_special_tokens']}
        tokenizer = AutoTokenizer.from_pretrained(STAGE2_ADAPTER_DIR, trust_remote_code=False, **kw)
        for tid, token in cfg.get('added_tokens_decoder', {}).items():
            assert tokenizer.convert_tokens_to_ids(token['content']) == int(tid), 'Token ID changed'
        assert tokenizer.chat_template == CHAT_TEMPLATE, 'Adapter chat template mismatch'""")
 elif label.startswith('# [prepare-dev]'):
  s=s.replace("    split_manifest, current_training_inventory, training_rows = prepare_split", "    frozen_path = Path(DEV_ROOT)/'split_v1.json'\n    if not frozen_path.is_file() or read_json(frozen_path)['split_sha256'] != "+repr(ref['split_sha256'])+":\n        raise RuntimeError('固定splitがありません/異なります。評価問題を新規選択せず停止します。')\n    split_manifest, current_training_inventory, training_rows = prepare_split")
 elif label.startswith('# [training-diagnostics]'):continue
 elif label.startswith('# [evaluate-and-save]'):
  s=s[:s.index("    summary.update(")]+'''
    summary.update({'run_id': EVAL_RUN_ID, 'gold_parse_failures': gold_failures,
                    'model_load_seconds_this_session': model_load_seconds})
    atomic_json(output_dir/'summary.json', summary)
    print(CANDIDATE, json.dumps(summary['by_dataset'], ensure_ascii=False))
    print('集計:', output_dir/'summary.json')
'''
 parts.append(s)
worker='\n\n'.join(parts).rstrip()+'\n'
compile(worker,'compare_worker','exec')
(HERE/'compare_runs_worker.py').write_text(worker)
helper=(HERE/'run02_launcher.py').read_text()
comparison=(HERE/'compare_results.py').read_text()
cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},source=s.splitlines(True),execution_count=None,outputs=[]))
md('''# run02 / run03 自動比較（固定128問）
新しいColab T4に接続し「すべて実行」を1回。run02/run03を保存したDriveを認証するだけです。
ZIP・HFトークン入力・手動再起動・設定変更は不要。学習はしません。
Base→Stage 1→各Stage 2をFP16で統合し、同じPRM96＋GSM32問をvLLMで評価します。
両モデルとも単一user turn、配布template、greedy、seed42、最大2048生成トークンです。
問題・正解・生成文は出力せず、集計だけ表示します。公式提出回数を消費しません。
モデルごとにmergeと評価を別プロセスで動かし、終了時にGPUメモリを解放します。
同じ条件で再実行すると保存済みの回答から再開します。別ノートブックとの同時実行は避けてください。
128問の小標本で、公式スコアの再現・0.22到達の保証ではありません。final400問は使いません。
''')
code('''# [config-and-drive]
from pathlib import Path
import sys, subprocess, json, os
EVAL_SIZE = 'quick'
if not Path('/content').is_dir():
    raise RuntimeError('新しいColab T4へ接続してください。')
WORKSPACE = Path('/content/drive/MyDrive/geniac_workspace')
from google.colab import drive
drive.mount('/content/drive')
for name in ('t4_20261013_run02_prmfull','t4_20261014_run03_prmfull6k'):
    assert (WORKSPACE/'experiments'/name/'training_summary.json').is_file(), '学習集計がありません: '+name
assert (WORKSPACE/'dev_eval/split_v1.json').is_file(), '固定dev splitがありません'
LOG_DIR = WORKSPACE/'dev_eval/comparisons/run02_run03_v1'/EVAL_SIZE/'logs'
''')
code(helper)
code('%%writefile /content/compare_runs_worker.py\n'+worker)
code('''# [install] 親kernelはtorch/vllmをimportしない。再実行時は同じ環境を維持。
marker = Path('/content/compare_runs_environment_v1.json')
if not marker.exists():
    commands = [
        [sys.executable, '-m', 'pip', 'install', '-q', 'uv'],
        [sys.executable, '-m', 'uv', 'pip', 'install', '--system',
         'https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl',
         '--extra-index-url', 'https://download.pytorch.org/whl/cu129',
         'peft', 'accelerate', 'datasets', 'math-verify==0.9.0'],
        [sys.executable, '-m', 'uv', 'pip', 'uninstall', '--system', 'torchao', 'torchcodec'],
    ]
    for command in commands:
        run_logged(command, LOG_DIR/'install.log')
    marker.write_text(json.dumps({'complete':True}))
print('評価環境の準備完了')
''')
code('''# [compare] Each child exits before the next model is loaded.
for candidate in ('run02', 'run03'):
    for phase in ('merge', 'evaluate'):
        print(candidate, phase, flush=True)
        run_logged([sys.executable, '-c', "import torch; assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0); free,total=torch.cuda.mem_get_info(); print('GPU free GiB:',free/2**30); assert free/total > 0.9, 'GPUメモリを別プロセスが使用中です。新しいT4サーバーへ接続してください。'"], LOG_DIR/'gpu_checks.log')
        run_logged([sys.executable, '-u', '/content/compare_runs_worker.py', phase, candidate, EVAL_SIZE], LOG_DIR/(candidate+'_'+phase+'.log'))
''')
code(comparison+'''
root = WORKSPACE/'dev_eval'
report = compare_results(root/'runs'/('compare_v1_run02_'+EVAL_SIZE), root/'runs'/('compare_v1_run03_'+EVAL_SIZE))
output = root/'comparisons/run02_run03_v1'/EVAL_SIZE/'comparison_summary.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
print('RUN02_RUN03_COMPARISON_COMPLETE')
print('共有する集計:', output)
''')
for i,c in enumerate(cells):
 c['id']=f'compare-{i:02d}'
 if c['cell_type']=='code':
  s=''.join(c['source']);s=s.split('\n',1)[1] if s.startswith('%%writefile') else s
  compile(s,c['id'],'exec')
nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'accelerator':'GPU'}}
(HERE/'compare_run02_run03_t4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Generated compare_run02_run03_t4.ipynb:',len(cells),'cells')
