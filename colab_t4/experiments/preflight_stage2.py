"""Local mechanical data/tokenizer preparation only. No Base download or training."""
from pathlib import Path
import json
from transformers import AutoTokenizer
from stage2_recipe import fetch_allowed_records, build_recipe, stable_hash
ROOT = Path(__file__).resolve().parents[2]
split = json.loads((ROOT/'local_eval/split_v1.json').read_text())
heldout = json.loads((ROOT/'local_eval/heldout_problem_hashes.json').read_text())
path = Path('/Users/cls-lab/Downloads/drive-download-20261013T094957Z-1-001/t4_20261013_run01_stage1')
cfg = json.loads((path/'tokenizer_config.json').read_text())
kwargs = ({'extra_special_tokens': {}, 'additional_special_tokens': cfg['extra_special_tokens']}
          if isinstance(cfg.get('extra_special_tokens'), list) else {})
tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
template = json.loads((ROOT/'inference_config.json').read_text())['chat_template']['jinja']
assert tokenizer.chat_template == template
raw, tokens, stats = build_recipe(fetch_allowed_records(split['sources']), tokenizer,
                                  heldout['hashes'], {'prm800k': 1500, 'gsm8k': 500})
out = ROOT/'local_eval/run02_preflight'
out.mkdir(exist_ok=True)
with (out/'train_private.jsonl').open('w') as f:
    for row in raw:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
report = {'stats': stats, 'selected_problem_hashes_sha256': stable_hash([r['problem_hash'] for r in raw]),
          'heldout_split_sha256': heldout['split_sha256'], 'mean_tokens': stats['tokens']/stats['selected']}
(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
