"""Package user's existing artifacts and build an automatic Colab training launcher."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUNDLE = HERE/'run02_colab_inputs.zip'
ADAPTERS = Path('/Users/cls-lab/Downloads/drive-download-20261013T094957Z-1-001')
files = {}
for stage in ('stage1','stage2'):
    slug = 't4_20261013_run01_'+stage
    for p in sorted((ADAPTERS/slug).iterdir()):
        if p.is_file():
            files['adapters/'+slug+'/'+p.name] = p
for name in ('gsm8k_2000.jsonl','tulu3_persona_if_2000.jsonl'):
    files['data_cache/'+name] = ROOT/'data_cache'/name
for name in ('split_v1.json','heldout_problem_hashes.json'):
    files['dev_eval/'+name] = ROOT/'local_eval'/name
manifest = {'files': {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}}
with ZipFile(BUNDLE,'w',compression=ZIP_DEFLATED,compresslevel=1) as z:
    z.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    for name,path in files.items():
        z.write(path,name)
bundle_sha = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
manual = json.loads((HERE/'train_stage2.ipynb').read_text())
parts = ['# Generated from train_stage2.ipynb; readable worker, no notebook execution.\nimport sys\n']
for cell in manual['cells']:
    if cell['cell_type'] != 'code':
        continue
    s=''.join(cell['source'])
    if s.startswith('# [config]'):
        s=s.replace('PHASE = "setup"', 'PHASE = sys.argv[1]')
    if s.startswith('# [workspace]'):
        assert "    drive.mount('/content/drive')" in s
        s=s.replace("    from google.colab import drive\n    drive.mount('/content/drive')", "    if not Path('/content/drive/MyDrive').is_dir():\n        raise RuntimeError('先に親notebookでDriveをマウントしてください。')")
    s=s.replace("print('merge完了。セッションのみ再起動 → PHASE=\"train\" → すべて実行。')", "print('merge完了。親notebookが次の学習プロセスを起動します。')")
    parts.append(s)
worker='\n\n'.join(parts)
compile(worker,'run02_worker.py','exec')
(HERE/'run02_worker.py').write_text(worker)
launcher=(HERE/'run02_launcher.py').read_text()
cells=[]
def md(s):
    cells.append(dict(cell_type='markdown',metadata={},source=s.strip().splitlines(True)))
def code(s,collapsed=False):
    cells.append(dict(cell_type='code',metadata={'collapsed':collapsed},execution_count=None,outputs=[],source=s.strip().splitlines(True)))
md('''# run02 自動学習：Run Allは1回、PHASE編集・手動再起動なし

**新しいColab T4**で実行してください。使用中の学習・推論ランタイムとは別にします。
PRM 1,500＋GSM 500、全文、1 epochでStage 2を新規学習します。Stage 1はrun01を再利用。
0.22超えはまだ未検証。このnotebookは学習と保存までを行います。

## VS Codeで行う操作

1. このnotebookを開き、右上のカーネル選択 → Colab → New Colab Server → T4。
2. 同じフォルダの **run02_colab_inputs.zip** をExplorerで右クリック → **Upload to Colab**。
3. 「すべて実行」を1回押す。
4. Driveの認証画面で、今回の保存先にするアカウントを選んで認証する。

あとは、環境構築 → Stage 1復元 → 新しいStage 2学習 → Drive保存まで順に実行します。
各工程を別のPythonプロセスで動かすので、途中のカーネル再起動は不要です。
ZIPには元のStage 1/2、過去の学習cache、固定dev/finalが含まれます。Baseモデル約8GBは含みません。
ZIPは自動展開するので、別アカウントへのフォルダ移動・手動展開も不要です。
元のファイルと違う内容がDriveにある場合は上書きせず停止します。

Web版Colabなら、このnotebookを開き、T4に変更し、左側のファイル欄からZIPを `/content/` へアップロードして「すべて実行」。

実行中はこのnotebook/Colabサーバーを削除・切断しないでください。完了までの時間はまだ実測していません。
''')
code('''# [preflight] このセルはGPUの有無だけを別プロセスで確認します。
from pathlib import Path
import subprocess, sys, json, os
if not Path('/content').is_dir():
    raise RuntimeError('Colab T4のカーネルを選んでください。ローカルMacでは実行しません。')
WORKSPACE = '/content/drive/MyDrive/geniac_workspace'
RUN_ID = 't4_20261013_run02_prmfull'
BUNDLE = Path('/content/run02_colab_inputs.zip')
if not BUNDLE.is_file():
    raise RuntimeError('run02_colab_inputs.zipを右クリック→Upload to Colabで /content/ に置いてください。')
subprocess.run([sys.executable, '-c',
    "import torch; assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'T4 GPU required'; print(torch.cuda.get_device_name(0))"], check=True)
print('T4とZIPを確認しました。次にDriveの認証へ進みます。')
''')
code('# [launcher-helpers]\n'+launcher+'\nBUNDLE_SHA256 = '+repr(bundle_sha),True)
code('''# [drive-and-restore] ユーザーのGoogle認証が必要になるのはこのセルです。
from google.colab import drive
drive.mount('/content/drive')
print('入力ファイルを確認・配置します:', restore_bundle(BUNDLE, WORKSPACE, BUNDLE_SHA256))
ALREADY_COMPLETE = completed_training(WORKSPACE, RUN_ID)
if ALREADY_COMPLETE:
    print('このrunの学習は完了済みです。再学習・上書きをスキップします。')
''')
md('## 学習コード\n次のセルは読みやすいPythonスクリプトをColabへ保存します。ここではモデルは動きません。')
code('%%writefile /content/run02_worker.py\n'+worker,True)
code('''# [install] 親kernelは学習ライブラリをimportせず、導入後の子プロセスで読み込みます。
LOG_DIR = Path(WORKSPACE)/'experiments'/RUN_ID/'logs'
if not ALREADY_COMPLETE:
    print('学習用ライブラリを導入中。ログ:', LOG_DIR/'install.log')
    run_logged([sys.executable, '-m', 'pip', 'install', 'unsloth', 'trl', 'bitsandbytes',
                'datasets', 'peft', 'accelerate'], LOG_DIR/'install.log', echo=False)
    print('環境構築完了。手動再起動は不要です。')
''')
code('''# [automatic-merge-and-train]
if not ALREADY_COMPLETE:
    print('1/2 Stage 1を復元します。')
    run_logged([sys.executable, '-u', '/content/run02_worker.py', 'merge'], LOG_DIR/'merge.log')
    print('2/2 Stage 2の学習を開始します。')
    run_logged([sys.executable, '-u', '/content/run02_worker.py', 'train'], LOG_DIR/'train.log')
if not completed_training(WORKSPACE, RUN_ID):
    raise RuntimeError('学習完了を確認できませんでした。logsを確認してください。')
report_path = Path(WORKSPACE)/'experiments'/RUN_ID/'training_summary.json'
report = json.loads(report_path.read_text())
print('\\n=== RUN02_TRAINING_COMPLETE ===')
print('epoch:', report['epoch'], '/ optimizer steps:', report['global_step'])
print('Stage 2:', report['stage2_dir'])
print('共有する集計:', report_path)
''')
md('''## 完了・中断の見分け方

**`RUN02_TRAINING_COMPLETE` が表示されたら学習完了です。**
Driveの `geniac_workspace/experiments/t4_20261013_run02_prmfull/training_summary.json` が集計です。
新しい重みは `geniac_workspace/adapters/t4_20261013_run02_prmfull_stage2/` に保存されます。

- 途中でエラーが出たら、エラーが出たセルから先には進まないでください。
- セッションの中断後は、同じnotebookで「すべて実行」。同じ条件なら50stepごとの最新checkpointから再開します。
- サーバーが削除された場合は新しいT4に接続し、ZIPをもう一度アップロードします。
- 最初の50step以前にはcheckpointがないので、その場合は学習開始からになります。
- `logs/install.log`、`logs/merge.log`、`logs/train.log` は上記Driveのrunディレクトリにあります。
- 既に学習完了なら、完了時の重みハッシュを確認してスキップします。

ここでは公式問題の推論・提出・HFへの公開は行いません。学習lossと公式スコアは別物です。
完了後は `training_summary.json` を共有し、保存したモデルで開発評価または公式の提出推論へ進みます。
''')
for i,c in enumerate(cells):c['id']=f'auto-run02-{i:02d}'
nb={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'run02_auto_t4.ipynb'}}}
(HERE/'run02_auto_t4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Notebook:',HERE/'run02_auto_t4.ipynb')
print('Bundle MB:',round(BUNDLE.stat().st_size/1e6,1),'files:',len(files))
