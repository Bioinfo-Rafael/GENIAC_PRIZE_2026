"""Build the 6,000-example candidate from the tested run02 launcher; reuse Drive Stage 1."""
import copy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
nb=copy.deepcopy(json.loads((HERE/'run02_auto_t4.ipynb').read_text()))
for cell in nb['cells']:
 s=''.join(cell['source'])
 if cell['cell_type']=='code':
  cell['outputs']=[];cell['execution_count']=None
  s=s.replace('t4_20261013_run02_prmfull','t4_20261014_run03_prmfull6k')
  s=s.replace('/content/run02_worker.py','/content/run03_worker.py')
  s=s.replace('COUNTS = {"prm800k": 1500, "gsm8k": 500}', 'COUNTS = {"prm800k": 4500, "gsm8k": 1500}')
  s=s.replace('RUN02_TRAINING_COMPLETE','RUN03_TRAINING_COMPLETE')
  if s.startswith('# [preflight]'):
   s='''# [preflight]
from pathlib import Path
import subprocess, sys, json, os
if not Path('/content').is_dir():
    raise RuntimeError('新しいColab T4に接続してください。')
WORKSPACE = '/content/drive/MyDrive/geniac_workspace'
RUN_ID = 't4_20261014_run03_prmfull6k'
subprocess.run([sys.executable, '-c', "import torch; assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0), 'T4 required'; print(torch.cuda.get_device_name(0))"], check=True)
'''
  elif s.startswith('# [drive-and-restore]'):
   s='''# [drive-and-restore]
from google.colab import drive
drive.mount('/content/drive')
stage1 = Path(WORKSPACE)/'adapters/t4_20261013_run01_stage1'
if not (stage1/'adapter_model.safetensors').is_file():
    raise RuntimeError('run02と同じGoogleアカウントのDriveに接続してください。Stage 1が見つかりません。')
ALREADY_COMPLETE = completed_training(WORKSPACE, RUN_ID)
print('学習完了済み:', ALREADY_COMPLETE)
'''
  cell['source']=s.splitlines(True)
nb['cells']=[c for c in nb['cells'] if c['cell_type']=='code']
intro='''# run03：PRM 4,500＋GSM8K 1,500件、全文、1 epoch
run02の公式スコアはユーザー報告で0.174014。今回はStage 2の件数だけ2,000→6,000へ増やす候補です。
Stage 1はrun01を再利用し、新規Stage 2を学習します。run02のStage 2への継ぎ足しではありません。
学習率5e-5、LoRA r8/alpha16、最大2048トークン、実効batch16を維持。固定評価800問を除外。
学習の目安はrun02の実測から約70分＋環境構築・モデル取得・復元。ただし未実測です。

1. 新しいColab T4サーバーに接続。推論用環境と分けてください。
2. 「すべて実行」を1回押す。
3. run02と同じGoogleアカウントでDriveを認証。

ZIPのアップロードは不要。手動再起動も不要。完了表示は `RUN03_TRAINING_COMPLETE`。
保存先：`geniac_workspace/adapters/t4_20261014_run03_prmfull6k_stage2/`。
既存run01/run02は上書きしません。中断時は同じ条件で再実行するとcheckpointから再開します。
学習後は固定した開発問題でrun02と比較してから公式提出を判断します。0.22超えは未検証です。
'''
nb['cells'].insert(0,dict(cell_type='markdown',metadata={},source=intro.splitlines(True)))
for i,c in enumerate(nb['cells']):
 c['id']=f'run03-{i:02d}'
 if c['cell_type']=='code':
  source=''.join(c['source'])
  if source.startswith('%%writefile'):source=source.split('\n',1)[1]
  compile(source,c['id'],'exec')
nb['metadata']['colab']={'name':'run03_auto_t4.ipynb'}
(HERE/'run03_auto_t4.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('Created run03_auto_t4.ipynb; all Python cells compile.')
