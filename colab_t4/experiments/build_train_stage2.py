"""Build the self-contained run02 Colab notebook. Never edit executed run01 notebooks."""
import hashlib
import json
import pprint
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
reference = json.loads((HERE/'run02_reference.json').read_text())
recipe_code = (HERE/'stage2_recipe.py').read_text()
template = json.loads((ROOT/'inference_config.json').read_text())['chat_template']['jinja']
cells = []

def md(s):
    cells.append(dict(cell_type='markdown', metadata={}, source=s.strip().splitlines(True)))

def code(s, collapsed=False):
    cells.append(dict(cell_type='code', metadata={'collapsed': collapsed}, execution_count=None,
                      outputs=[], source=s.strip().splitlines(True)))

md('''# run02：Stage 2だけをPRM中心で再学習

**狙い：数学教材を広げる。0.22超えは未検証です。**
Stage 1はrun01のadapterを再利用。Stage 2は新規LoRAを作り、PRM800K 1,500件＋GSM8K 500件を解説全文で1周学習します。
run01のStage 2は読み込まず、上書きもしません。学習率5e-5、rank 8 / alpha 16、実効batch 16、系列長2048を維持します。
全文が収まらない例は機械的に除外し、解説の途中で切りません。lossは全非paddingトークンが対象です。

## 実行手順：新しいColab T4で、この1冊だけをアップロード

1. `PHASE="setup"` → **すべて実行** → セッション再起動。
2. `PHASE="merge"` → **すべて実行** → セッション再起動。ランタイムは削除しない。
3. `PHASE="train"` → **すべて実行**。終了すると新しいStage 2がDriveに保存されます。

Driveに `geniac_workspace/adapters/t4_20261013_run01_stage1/` が必要です。
別アカウントなら同じパスに元のStage 1フォルダを置いてください。配布教材はこのノートブックが取得します。
停止した学習はセッションのみ再起動し、同じ設定で `PHASE="train"` を再実行すると最新checkpointから再開します。
ランタイム削除後はsetupとmergeも再実行してください。最初の50step以前にはcheckpointがありません。

固定した評価用800問は、同梱した問題ハッシュで学習から除外します。問題・解答プレビューは表示しません。
''')
code('''# [config]
PHASE = "setup"                   # setup → 再起動 → merge → 再起動 → train
RUN_ID = "t4_20261013_run02_prmfull"
WORKSPACE = "/content/drive/MyDrive/geniac_workspace"
STAGE1_DIR = WORKSPACE + "/adapters/t4_20261013_run01_stage1"
STAGE2_DIR = WORKSPACE + "/adapters/" + RUN_ID + "_stage2"
MERGED_STAGE1 = "/content/run02_stage1_merged"
RUN_DIR = WORKSPACE + "/experiments/" + RUN_ID
COUNTS = {"prm800k": 1500, "gsm8k": 500}
MAX_LENGTH = 2048
EPOCHS = 1.0
LEARNING_RATE = 5e-5
LORA_R, LORA_ALPHA = 8, 16
GRAD_ACCUM = 16
SEED = 42
''')
code('''# [guard-and-install]
import sys, subprocess, json, os, hashlib, time, gc
from pathlib import Path
if not Path('/content').is_dir():
    raise RuntimeError('このnotebookはColab CUDA用です。T4へ接続してください。')
import google.colab
if PHASE not in ('setup', 'merge', 'train'):
    raise ValueError('PHASEを確認してください。')
if PHASE == 'setup':
    with open('/content/run02_install.log', 'w') as log:
        result = subprocess.run([sys.executable, '-m', 'pip', 'install',
                                 'unsloth', 'trl', 'bitsandbytes', 'datasets', 'peft', 'accelerate'],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('導入失敗。/content/run02_install.log を確認してください。')
    print('setup完了。セッション再起動 → PHASE="merge" → すべて実行。')
''')
md('## 固定情報と教材準備の関数\n評価問題の本文は含めず、除外用ハッシュ・許可教材のrevision・元adapterのハッシュだけを同梱しています。')
code('# [reference]\nREFERENCE = '+pprint.pformat(reference,width=110,sort_dicts=True)
     +'\nCHAT_TEMPLATE = '+repr(template)+'\nRECIPE_CODE_SHA256 = '+repr(hashlib.sha256(recipe_code.encode()).hexdigest()), True)
code('# [recipe-functions]\n'+recipe_code)
code('''# [io-functions]
def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\\n')
    os.replace(tmp, path)

def versions():
    from importlib.metadata import version
    return {k: version(k) for k in ['torch','transformers','trl','peft','unsloth','bitsandbytes','datasets']}

def saved_tokenizer(path):
    from transformers import AutoTokenizer
    from importlib.metadata import version
    cfg = read_json(Path(path)/'tokenizer_config.json')
    kwargs = {}
    if int(version('transformers').split('.')[0]) < 5 and isinstance(cfg.get('extra_special_tokens'), list):
        kwargs = {'extra_special_tokens': {}, 'additional_special_tokens': cfg['extra_special_tokens']}
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False, **kwargs)
    for token_id, token in cfg.get('added_tokens_decoder', {}).items():
        assert tok.convert_tokens_to_ids(token['content']) == int(token_id), 'token IDが変わっています。'
    return tok
''')
code('''# [workspace]
if PHASE != 'setup':
    # Unsloth must precede transformers/peft/trl imports in the training process.
    if PHASE == 'train':
        from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('ランタイムをT4 GPUへ変更してください。')
    from google.colab import drive
    drive.mount('/content/drive')
    if Path(STAGE2_DIR).exists():
        raise RuntimeError('同名のStage 2保存先が既にあります。上書きしません。完了済みか確認してください。')
    for filename, expected in REFERENCE['stage1_files'].items():
        path = Path(STAGE1_DIR)/filename
        if not path.is_file() or file_sha(path) != expected:
            raise RuntimeError('run01 Stage 1と異なる/不足するファイル: '+filename)
    MERGE_SIGNATURE = {'base': REFERENCE['base'], 'stage1': REFERENCE['stage1_files'],
                       'template_sha256': hashlib.sha256(CHAT_TEMPLATE.encode()).hexdigest()}
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
''')
md('## Stage 1を復元（mergeのみ）\n公式Baseに保存済みStage 1を統合します。これは学習ではありません。元の4bit学習直後に保存したモデルとの数値的同一性は仮定しません。')
code('''# [restore-stage1]
if PHASE == 'merge':
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    marker = Path(MERGED_STAGE1)/'run02_merge.json'
    if Path(MERGED_STAGE1).exists():
        if not marker.exists() or read_json(marker)['signature'] != MERGE_SIGNATURE:
            raise RuntimeError('未完了/別構成のMERGED_STAGE1です。空の別パスを指定してください。')
        print('同じStage 1の復元記録を再利用します。')
    else:
        model = AutoModelForCausalLM.from_pretrained(
            REFERENCE['base']['model'], revision=REFERENCE['base']['commit'],
            dtype=torch.float16, device_map='cuda', low_cpu_mem_usage=True)
        model = PeftModel.from_pretrained(model, STAGE1_DIR).merge_and_unload(safe_merge=True)
        tokenizer = saved_tokenizer(STAGE1_DIR)
        if tokenizer.chat_template != CHAT_TEMPLATE:
            raise RuntimeError('Stage 1 tokenizerと配布templateが一致しません。')
        model.save_pretrained(MERGED_STAGE1, safe_serialization=True, max_shard_size='2GB')
        tokenizer.save_pretrained(MERGED_STAGE1)
        files = {p.name: file_sha(p) for p in sorted(Path(MERGED_STAGE1).iterdir()) if p.is_file()}
        save_json(marker, {'signature': MERGE_SIGNATURE, 'files': files})
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    print('merge完了。セッションのみ再起動 → PHASE="train" → すべて実行。')
''')
md('## 教材の選択（trainのみ）\nPRM 1,500件＋GSM 500件を固定seedで機械的に選びます。解説全文とEOSを含めて2048以内の例のみ採用します。')
code('''# [prepare-training-data]
if PHASE == 'train':
    marker = Path(MERGED_STAGE1)/'run02_merge.json'
    if not marker.exists() or read_json(marker)['signature'] != MERGE_SIGNATURE:
        raise RuntimeError('PHASE="merge"を先に実行してください。')
    merged_record = read_json(marker)
    for name, expected in merged_record['files'].items():
        if not (Path(MERGED_STAGE1)/name).is_file() or file_sha(Path(MERGED_STAGE1)/name) != expected:
            raise RuntimeError('Stage 1統合ファイルが欠損/変更されています。')
    tokenizer = saved_tokenizer(MERGED_STAGE1)
    raw, tokenized, data_stats = build_recipe(fetch_allowed_records(REFERENCE['sources']), tokenizer,
        REFERENCE['hashes'], COUNTS, max_length=MAX_LENGTH, seed=SEED)
    signature = {'run_id': RUN_ID, 'merge': MERGE_SIGNATURE,
                 'merged_files': merged_record['files'], 'versions': versions(),
                 'recipe_code_sha256': RECIPE_CODE_SHA256, 'heldout_split': REFERENCE['split_sha256'],
                 'selected_ids': [r['problem_hash'] for r in raw],
                 'config': {'counts': COUNTS, 'max_length': MAX_LENGTH, 'epochs': EPOCHS,
                            'learning_rate': LEARNING_RATE, 'rank': LORA_R, 'alpha': LORA_ALPHA,
                            'grad_accum': GRAD_ACCUM, 'seed': SEED, 'loss': 'all_nonpadding',
                            'compression': 'none', 'explicit_eos': True}}
    signature_path = Path(RUN_DIR)/'training_signature.json'
    if signature_path.exists() and read_json(signature_path) != signature:
        raise RuntimeError('以前の学習と構成が異なります。別RUN_IDを設定してください。')
    save_json(signature_path, signature)
    cache = Path(WORKSPACE)/'data_cache'/(RUN_ID+'_'+str(sum(COUNTS.values()))+'.jsonl')
    cache.parent.mkdir(parents=True, exist_ok=True)
    data_text = ''.join(json.dumps(r, ensure_ascii=False)+'\\n' for r in raw)
    if cache.exists() and cache.read_text() != data_text:
        raise RuntimeError('同名の学習キャッシュが別内容です。')
    cache.write_text(data_text)
    save_json(Path(RUN_DIR)/'data_summary.json', data_stats)
    print('学習教材:', COUNTS, '/ 集計:', data_stats)
    print('学習cache:', str(cache))
''')
md('## 新しいStage 2を学習\nStage 1統合モデルを4bitで読み込み、新しいLoRAを追加します。回答のみlossなどの追加変更は今回は行いません。')
code('''# [new-stage2-model]
if PHASE == 'train':
    if 'model' in globals():
        raise RuntimeError('モデルが残っています。セッション再起動後、trainを再実行してください。')
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MERGED_STAGE1, max_seq_length=MAX_LENGTH,
        dtype=torch.float16, load_in_4bit=True, use_exact_model_name=True)
    assert tokenizer.chat_template == CHAT_TEMPLATE, '学習templateが変わっています。'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(model, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=0.0, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        bias='none', use_gradient_checkpointing='unsloth', random_state=SEED)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError('学習対象のパラメータがありません。')
    print('学習対象パラメータ:', trainable)
''')
code('''# [trainer-and-checks]
if PHASE == 'train':
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    class CheckFiniteLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            import math
            for key in ('loss', 'grad_norm'):
                value = (logs or {}).get(key)
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise RuntimeError('非有限の'+key+'を検出しました。学習を停止します。')
    cfg = SFTConfig(output_dir=RUN_DIR+'/checkpoints', num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE, per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM, warmup_ratio=0.03, lr_scheduler_type='cosine',
        max_length=MAX_LENGTH, packing=False, completion_only_loss=False,
        bf16=False, fp16=True, optim='paged_adamw_8bit', logging_steps=5, seed=SEED,
        save_strategy='steps', save_steps=50, save_total_limit=2, save_only_model=False,
        report_to='none')
    trainer = SFTTrainer(model=model, processing_class=tokenizer,
        train_dataset=Dataset.from_list(tokenized), args=cfg, callbacks=[CheckFiniteLoss()])
    # Check real collated labels rather than assuming a config flag worked.
    for i in (0, len(tokenized)-1):
        example = trainer.train_dataset[i]
        batch = trainer.data_collator([example])
        # A one-example padding-free batch can omit attention_mask.
        ids, labels = batch['input_ids'], batch['labels']
        if ids.ndim != 2 or ids.shape[0] != 1 or labels.shape != ids.shape:
            raise RuntimeError('検査用の単一example batchの形状が不正です。')
        if 'attention_mask' in batch:
            mask = batch['attention_mask'].bool()
        else:
            # Do not infer padding from token IDs: pad and EOS can share an ID.
            if ids.numel() != len(tokenized[i]['input_ids']):
                raise RuntimeError('attention_maskなしで系列長が変わっています。')
            mask = torch.ones_like(ids, dtype=torch.bool)
        actual_ids, actual_labels = ids[mask], labels[mask]
        if (actual_ids.tolist() != tokenized[i]['input_ids']
                or actual_ids[-1].item() != tokenizer.eos_token_id):
            raise RuntimeError('データの切詰め/EOS欠落を検出しました。')
        expected_labels = actual_ids.clone()
        # The first label has no preceding token and is never a causal target.
        # Padding-free collators may explicitly ignore that boundary label.
        if actual_labels[0].item() == -100:
            expected_labels[0] = -100
        if not torch.equal(expected_labels, actual_labels):
            raise RuntimeError('想定外のlossマスクを検出しました。')
        if not torch.all(labels[~mask] == -100):
            raise RuntimeError('paddingのlossマスクが不正です。')
    print('実際のlabelsを確認：非padding全体を学習、全文/EOSを保持。')
''')
code('''# [train-and-save]
if PHASE == 'train':
    from transformers.trainer_utils import get_last_checkpoint
    checkpoint_dir = Path(RUN_DIR)/'checkpoints'
    checkpoint = get_last_checkpoint(str(checkpoint_dir)) if checkpoint_dir.exists() else None
    print('再開checkpoint:', checkpoint or 'なし（新規学習）')
    t0 = time.monotonic()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    # Do not mark a partial/early-stopped run as a final adapter.
    if trainer.state.epoch is None or trainer.state.epoch < EPOCHS - 1e-3:
        raise RuntimeError('予定epoch未完了。最終adapterとして保存しません。')
    if Path(STAGE2_DIR).exists():
        raise RuntimeError('保存先が既にあります。上書きしません。')
    model.save_pretrained(STAGE2_DIR)
    tokenizer.save_pretrained(STAGE2_DIR)
    report = {'complete': True, 'run_id': RUN_ID, 'stage1_dir': STAGE1_DIR,
              'stage2_dir': STAGE2_DIR, 'global_step': trainer.state.global_step,
              'epoch': trainer.state.epoch, 'metrics': result.metrics,
              'this_session_seconds': time.monotonic()-t0, 'data_stats': data_stats,
              'signature_sha256': stable_hash(signature),
              'stage2_weights_sha256': file_sha(Path(STAGE2_DIR)/'adapter_model.safetensors')}
    save_json(Path(RUN_DIR)/'training_summary.json', report)
    print('Stage 2学習完了:', STAGE2_DIR)
    print('集計:', RUN_DIR+'/training_summary.json')
''')
md('''## 学習が終わったら

新しいT4推論ランタイムで、既存の `evaluate_dev.ipynb` または配布の推論notebookを使用します。
**Stage 1はrun01、Stage 2だけrun02** にします。

```python
STAGE1_SLUG = "t4_20261013_run01_stage1"
STAGE2_SLUG = "t4_20261013_run02_prmfull_stage2"
RUN_ID = "t4_20261013_run02_prmfull"
```

開発評価側では `STAGE2_ADAPTER_DIR` を新しい保存先へ変え、`EVAL_RUN_ID`も別名にします。
`TRAIN_CACHE_SPECS` に今回の `data_cache/t4_20261013_run02_prmfull_2000.jsonl`（2,000件）を追加してください。
固定splitは800問の除外ハッシュに対応する元のものを使います。ローカルの `local_eval/split_v1.json` と
`heldout_problem_hashes.json` をDriveの `geniac_workspace/dev_eval/` に置けば同じ評価問題を使えます。

公式提出は、配布推論コードで全問生成した未編集のJSONを使用します。このnotebookは提出しません。
先に開発比較する場合、run01とrun02を同じT4/vLLM条件で測ってください。学習lossの低下だけでは性能改善を判断できません。

初回の学習時間は未実測です。ローカル事前準備では平均約264トークン/件、合計約53万トークンでした。
元のGSM8Kより長いため、run01の18.5分をそのまま当てはめないでください。

参考：[TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)、
[Unsloth checkpoint再開](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint)。
''')
for i, cell in enumerate(cells):
    cell['id'] = f'run02-{i:02d}'
nb = {'nbformat':4, 'nbformat_minor':5, 'cells':cells,
      'metadata': {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'},
                   'language_info':{'name':'python'}, 'accelerator':'GPU',
                   'colab': {'name':'train_stage2.ipynb'}}}
(HERE/'train_stage2.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print('train_stage2.ipynb:',len(cells),'cells')
