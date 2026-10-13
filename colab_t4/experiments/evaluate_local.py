"""Local Mac MPS development evaluation. No official evaluation dataset access.
Run with the project .venv-local Python. Answers are saved privately, never printed.
"""
import argparse
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['GENIAC_EVAL_RUNTIME'] = 'local'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
import dev_eval as d

ROOT = Path(__file__).resolve().parents[2]


class TransformersBackend:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    def chat(self, conversations, params, use_tqdm=False):
        import torch
        outputs = []
        for messages in conversations:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors='pt', return_dict=True).to('mps')
            n = inputs['input_ids'].shape[-1]
            if n + params.max_tokens > 4096:
                raise RuntimeError('Input plus generation exceeds fixed 4096-token context.')
            with torch.inference_mode():
                result = self.model.generate(**inputs, max_new_tokens=params.max_tokens,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
            torch.mps.synchronize()
            tokens = result[0, n:].tolist()
            eos = self.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, list) else [eos]
            finish = 'stop' if tokens and tokens[-1] in eos else 'length'
            outputs.append(SimpleNamespace(prompt_token_ids=inputs['input_ids'][0].tolist(), outputs=[
                SimpleNamespace(text=self.tokenizer.decode(tokens, skip_special_tokens=True),
                                token_ids=tokens, finish_reason=finish)]))
            del inputs, result
        return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapters', type=Path,
        default=Path('/Users/cls-lab/Downloads/drive-download-20261013T094957Z-1-001'))
    parser.add_argument('--size', choices=['quick', 'dev'], default='quick')
    parser.add_argument('--run-id', default='run01_quick_mps_v1')
    args = parser.parse_args()
    if not __import__('re').fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
        raise ValueError('Invalid run ID')
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError('Apple GPU / MPS unavailable; CPU fallback is not automatic.')
    print('[device] Apple MPS, float16, transformers; NOT the official vLLM backend', flush=True)
    for name in ('math_verify', 'math_verify.parser', 'math_verify.grader', 'latex2sympy2_extended'):
        logging.getLogger(name).disabled = True
    workspace = ROOT / 'local_eval'
    specs = [dict(name=name, path=str(ROOT/'data_cache'/filename), expected_rows=2000)
             for name, filename in [('run01_stage1', 'tulu3_persona_if_2000.jsonl'),
                                    ('run01_stage2', 'gsm8k_2000.jsonl')]]
    print('[data] Preparing frozen holdout; excluding both training caches', flush=True)
    split, inventory, training = d.prepare_split(workspace, specs)
    rows = split['split']['dev']
    if args.size == 'quick':
        ids = set(split['split']['quick_ids'])
        rows = [r for r in rows if r['id'] in ids]
    gold, failures = d.prepare_gold(rows)
    print('[data] selected:', len(rows), '/ gold parse failures:', failures, flush=True)
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    stage1 = args.adapters / 't4_20261013_run01_stage1'
    stage2 = args.adapters / 't4_20261013_run01_stage2'
    template = d.read_json(ROOT/'colab_t4/inference_config.json')['chat_template']['jinja']
    model_lock = workspace/'base_revision.json'
    if model_lock.exists():
        revision = d.read_json(model_lock)['commit']
    else:
        revision = HfApi().model_info('Qwen/Qwen3-4B-Base').sha
        d.atomic_json(model_lock, {'model': 'Qwen/Qwen3-4B-Base', 'commit': revision})
    signature = {'backend': 'transformers_mps', 'dtype': 'float16', 'base_commit': revision,
        'stage1': d.adapter_fingerprint(stage1), 'stage2': d.adapter_fingerprint(stage2),
        'split_sha256': split['split_sha256'], 'eval_size': args.size,
        'selected_ids_sha256': d.digest([r['id'] for r in rows]),
        'training_inventory': inventory, 'versions': d.package_versions(),
        'helper_sha256': d.file_hash(Path(d.__file__)), 'runner_sha256': d.file_hash(Path(__file__)),
        'template_sha256': d.digest(template),
        'generation': {'max_new_tokens': 2048, 'max_model_len': 4096, 'do_sample': False, 'seed': 42,
                       'attention': 'sdpa', 'chunk_size': 1}}
    output = workspace/'runs'/args.run_id
    d.resume_state(output/'predictions_private.json', signature, [r['id'] for r in rows])
    d.atomic_json(output/'run_manifest.json', signature)
    tok_config = d.read_json(stage2/'tokenizer_config.json')
    # Saved by newer Transformers: v4 expects a mapping; preserve special token IDs.
    tok_kwargs = {}
    if isinstance(tok_config.get('extra_special_tokens'), list):
        tok_kwargs = {'extra_special_tokens': {},
                      'additional_special_tokens': tok_config['extra_special_tokens']}
    tokenizer = AutoTokenizer.from_pretrained(stage2, trust_remote_code=False, **tok_kwargs)
    for token_id, token in tok_config.get('added_tokens_decoder', {}).items():
        if tokenizer.convert_tokens_to_ids(token['content']) != int(token_id):
            raise RuntimeError('Saved tokenizer token IDs changed during compatibility loading.')
    diagnostics = d.training_diagnostics(tokenizer, training['run01_stage2'])
    diagnostics['saved_adapter_template_matches_inference'] = tokenizer.chat_template == template
    tokenizer.chat_template = template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    d.atomic_json(output/'training_diagnostics.json', diagnostics)
    print('[diagnostics]', diagnostics, flush=True)
    for row in rows:
        n = len(tokenizer.apply_chat_template([{'role': 'user', 'content': row['problem']}],
                                              tokenize=True, add_generation_prompt=True))
        if n + 2048 > 4096:
            raise RuntimeError('Fixed evaluation contains prompt too long for configured context.')
    print('[model] Download/load official Base, then merge Stage 1 and Stage 2 on CPU', flush=True)
    t0 = time.monotonic()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-4B-Base', revision=revision,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='sdpa',
        trust_remote_code=False)
    for name, adapter in [('Stage 1', stage1), ('Stage 2', stage2)]:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload(safe_merge=True)
        print('[merge]', name, 'complete', flush=True)
    model = model.to(device='mps', dtype=torch.float16).eval()
    print('[model] Ready in %.1f seconds' % (time.monotonic()-t0), flush=True)
    summary = d.evaluate_rows(TransformersBackend(model, tokenizer), rows, gold, signature,
        output, max_tokens=2048, chunk_size=1, sampling_params_factory=SimpleNamespace)
    summary.update(run_id=args.run_id, training_diagnostics=diagnostics, gold_parse_failures=failures,
                   backend_note='Mac MPS/Transformers FP16. Compare within this backend; not official vLLM score.')
    d.atomic_json(output/'summary.json', summary)
    print('[complete]', summary['by_dataset'], flush=True)
    print('[summary]', output/'summary.json', flush=True)


if __name__ == '__main__':
    main()
