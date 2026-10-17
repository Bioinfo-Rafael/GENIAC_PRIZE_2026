"""Isolated GPU phases; use run.py instead of invoking this directly."""
import argparse
import dataclasses
import importlib.metadata
import math
import os
import random
import shutil
import signal
import sys
import time
from pathlib import Path

from core import (read, write, sha, fingerprint, tokenize_answer, update_groups,
                  digest, pairs_from_outputs, comparison)
import grading

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def tokenizer_for(path):
    from transformers import AutoTokenizer
    cfg = read(Path(path) / 'tokenizer_config.json')
    extra = cfg.get('extra_special_tokens')
    kw = {'extra_special_tokens': {}, 'additional_special_tokens': extra} if isinstance(extra, list) else {}
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=False, **kw)
    template = (Path(__file__).parent / 'chat_template.jinja').read_text()
    if tok.chat_template != template:
        raise ValueError('Saved tokenizer differs from the fixed inference template')
    for tid, token in cfg.get('added_tokens_decoder', {}).items():
        if tok.convert_tokens_to_ids(token['content']) != int(tid):
            raise ValueError('Special token ID changed')
    if tok.pad_token_id is None or tok.pad_token_id == tok.eos_token_id:
        raise ValueError('A distinct existing PAD token is required')
    return tok


def compatible_adapter(source, target):
    """Drop only null/default future metadata; never silently change LoRA semantics."""
    from peft import LoraConfig
    target.mkdir(parents=True, exist_ok=True)
    fields = {f.name for f in dataclasses.fields(LoraConfig)}
    cfg = read(source / 'adapter_config.json')
    for key in list(cfg):
        if key not in fields:
            if cfg[key] not in (None, False, {}, []) and key != 'peft_version':
                raise ValueError('Unsupported non-default adapter field: ' + key)
            del cfg[key]
    if cfg['r'] != 8 or cfg['lora_alpha'] != 16:
        raise ValueError('Expected run03 LoRA rank8/alpha16')
    write(target / 'adapter_config.json', cfg)
    weight = target / 'adapter_model.safetensors'
    if not weight.exists():
        weight.symlink_to((source / weight.name).resolve())
    return target


def save_model(model, tok, destination):
    tmp = destination.with_name(destination.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp, safe_serialization=True, max_shard_size='2GB')
    tok.save_pretrained(tmp)
    write(tmp / 'artifacts.json', fingerprint(tmp))
    tmp.rename(destination)


def check_model(destination):
    manifest = destination / 'artifacts.json'
    if not manifest.exists():
        return False
    for name, expected in read(manifest).items():
        if sha(destination / name) != expected:
            raise ValueError('Model artifact hash mismatch: ' + name)
    return True


def merge(ctx, candidate):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    cfg, root, inputs = ctx['config'], Path(ctx['run']), Path(ctx['inputs'])
    tok = tokenizer_for(inputs / 'stage2')
    out = root / 'models' / candidate
    if check_model(out):
        return
    parent = root / 'models' / 'stage1'
    if not check_model(parent):
        model = AutoModelForCausalLM.from_pretrained(cfg['base_model'], revision=cfg['base_revision'],
                  torch_dtype=torch.float16, device_map='cuda', attn_implementation='sdpa')
        source = compatible_adapter(inputs / 'stage1', root / 'compatible' / 'stage1')
        model = PeftModel.from_pretrained(model, source).merge_and_unload(safe_merge=True)
        save_model(model, tok, parent)
    else:
        model = AutoModelForCausalLM.from_pretrained(parent, torch_dtype=torch.float16,
                                                    device_map='cuda', attn_implementation='sdpa')
    if candidate == 'b0':
        adapter = compatible_adapter(inputs / 'stage2', root / 'compatible' / 'stage2')
    else:
        adapter = root / 'training' / candidate / 'final'
        if not check_model(adapter):
            raise ValueError('Candidate training is incomplete')
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
    save_model(model, tok, out)


def generate(ctx, candidate, purpose, count):
    from vllm import LLM, SamplingParams
    cfg, root = ctx['config'], Path(ctx['run'])
    rows = read(root / {'train':'train_rows.json', 'eval':'eval_rows.json',
                        'final':'final_rows.json'}[purpose])[:count]
    dest = root / 'generations' / f'{candidate}_{purpose}.json'
    saved = read(dest) if dest.exists() else {}
    pending = [r for r in rows if r['id'] not in saved]
    if not pending:
        return
    model = root / 'models' / candidate
    if not check_model(model):
        raise ValueError('Missing merged model')
    llm = LLM(model=str(model), dtype='half', max_model_len=cfg['context'],
              gpu_memory_utilization=cfg['vllm_memory'], max_num_seqs=cfg['eval_chunk'] * cfg['generations'],
              seed=cfg['seed'], trust_remote_code=False, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    if tok.chat_template != (Path(__file__).parent / 'chat_template.jinja').read_text():
        raise ValueError('Generation template mismatch')
    train = purpose == 'train'
    write(root / 'generation_environment.json', {
        'vllm':importlib.metadata.version('vllm'),
        'transformers':importlib.metadata.version('transformers'),
        'eos_token_id':tok.eos_token_id, 'template_sha256':sha(Path(__file__).parent/'chat_template.jinja'),
        'generation_config':read(model/'generation_config.json') if (model/'generation_config.json').exists() else None,
        'config':cfg})
    timing_path = root / 'timings' / f'{candidate}_{purpose}.json'
    timings = read(timing_path) if timing_path.exists() else []
    for start in range(0, len(pending), cfg['eval_chunk']):
        if STOP:
            break
        chunk = pending[start:start+cfg['eval_chunk']]
        ids = [tok.apply_chat_template([{'role': 'user', 'content': r['problem']}],
                                       tokenize=True, add_generation_prompt=True) for r in chunk]
        if any(len(x) + cfg['max_tokens'] > cfg['context'] for x in ids):
            raise ValueError('Prompt exceeds reserved context; refusing silent truncation')
        # Per-question seeds make resume/chunk sizes independent of sampling order.
        params = [SamplingParams(n=cfg['generations'] if train else 1,
                   temperature=cfg['temperature'] if train else 0,
                   top_p=cfg['top_p'] if train else 1, max_tokens=cfg['max_tokens'],
                   seed=(cfg['seed'] + int(r['id'][:8], 16)) % (2**31),
                   skip_special_tokens=True) for r in chunk]
        t0 = time.monotonic()
        outputs = llm.generate([{'prompt_token_ids': x} for x in ids], params, use_tqdm=False)
        for row, output in zip(chunk, outputs):
            try:
                gold = grading.prepare_gold([row])[0][row['id']]
            except Exception:
                gold = None
            candidates = []
            for o in output.outputs:
                candidates.append({'index': o.index, 'text': o.text, 'token_ids': list(o.token_ids),
                     'finish_reason': o.finish_reason, 'stop_reason': o.stop_reason,
                     'grade': grading.grade(row, o.text, gold)})
            saved[row['id']] = candidates
        timings.append({'questions': len(chunk), 'seconds': time.monotonic()-t0})
        write(timing_path, timings)
        write(dest, saved)
        print(f'{candidate}/{purpose}: {len(saved)}/{len(rows)}', flush=True)
    write(timing_path, timings)
    if STOP:
        raise SystemExit(75)


def token_pairs(ctx):
    root, cfg = Path(ctx['run']), ctx['config']
    rows = read(root / 'train_rows.json')
    generated = read(root / 'generations' / 'b0_train.json')
    pairs, stats = pairs_from_outputs([r for r in rows if r['id'] in generated], generated,
                                     cfg['seed'], cfg.get('data_mode', 'mixed'))
    tok = tokenizer_for(Path(ctx['inputs']) / 'stage2')
    accepted = []
    for pair in pairs:
        chosen = tokenize_answer(tok, pair['problem'], pair['chosen'], cfg['context'])
        rejected = (tokenize_answer(tok, pair['problem'], pair['rejected'], cfg['context'])
                    if 'rejected' in pair else None)
        if chosen is not None and ('rejected' not in pair or rejected is not None):
            item = dict(pair, chosen_tokens=chosen)
            if rejected is not None:
                item['rejected_tokens'] = rejected
            accepted.append(item)
    stats.update(pairs=len(accepted), overlength_pairs=len(pairs)-len(accepted))
    if (root/'pairs.json').exists() and read(root/'pairs.json') != accepted:
        raise ValueError('Pair set changed on resume; use a new run')
    write(root / 'pairs.json', accepted)
    write(root / 'pair_summary.json', stats)


def load_train_model(ctx, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training
    cfg, root = ctx['config'], Path(ctx['run'])
    dtype = getattr(torch, cfg['dtype'])
    base = AutoModelForCausalLM.from_pretrained(root / 'models' / 'stage1', device_map='cuda',
        torch_dtype=dtype, attn_implementation='sdpa', quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype))
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter is None:
        adapter = compatible_adapter(Path(ctx['inputs']) / 'stage2', root / 'compatible' / 'stage2')
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    model.config.use_cache = False
    return model


def logps(model, tok, rows, dtype, dft=False):
    import torch
    from losses import collate, sequence_logps
    batch = collate(rows, tok.pad_token_id, 'cuda')
    labels = batch.pop('labels')
    with torch.autocast('cuda', dtype=dtype):
        out = model(**batch)
        return sequence_logps(out.logits, labels, dft=dft)


def reference(ctx):
    import torch
    root, cfg = Path(ctx['run']), ctx['config']
    model = load_train_model(ctx)
    model.eval()
    tok = tokenizer_for(Path(ctx['inputs']) / 'stage2')
    pairs = read(root / 'pairs.json')
    dest = root / 'reference.json'
    signature = digest({'pairs': sha(root / 'pairs.json'), 'stage2': cfg['stage2_sha256'],
                        'dtype': cfg['dtype'], 'code': ctx['code']})
    saved = read(dest) if dest.exists() else {'signature': signature, 'values': {}}
    if saved['signature'] != signature:
        raise ValueError('Reference cache signature mismatch')
    with torch.no_grad():
        for pair in pairs:
            if pair['id'] in saved['values']:
                continue
            if STOP:
                raise SystemExit(75)
            values, _ = logps(model, tok, [pair['chosen_tokens'], pair['rejected_tokens']],
                              getattr(torch, cfg['dtype']))
            saved['values'][pair['id']] = values.cpu().tolist()
            write(dest, saved)


def train(ctx, branch, benchmark=False):
    import torch
    from losses import preference_objective
    from transformers import get_cosine_schedule_with_warmup
    root, cfg = Path(ctx['run']), ctx['config']
    dest = root / 'training' / branch
    dest.mkdir(parents=True, exist_ok=True)
    if not benchmark and check_model(dest / 'final'):
        return
    pairs = read(root / 'pairs.json')
    objective = cfg.get('objective', branch)
    if objective not in ('rft', 'dft', 'dpo', 'dpo_nll'):
        raise ValueError('Unknown objective: ' + objective)
    preference = objective in ('dpo', 'dpo_nll')
    refs = read(root / 'reference.json')['values'] if preference else {}
    training_signature = digest([ctx['signature'],sha(root/'pairs.json'),
                                 sha(root/'reference.json') if preference else None])
    train_seed = cfg.get('train_seed', cfg['seed'])
    groups = update_groups(len(pairs), cfg['epochs'], cfg['effective_batch'], train_seed)
    if not groups:
        raise ValueError('No training groups')
    total_updates = len(groups)
    if benchmark and cfg.get('suite_benchmark'):
        # Probe the longest examples so a faster setting also fits later training batches.
        order = sorted(range(len(pairs)), key=lambda i:max(
            len(pairs[i][key]['input_ids']) for key in ('chosen_tokens','rejected_tokens')
            if key in pairs[i]), reverse=True)
        groups = [order[i:i+cfg['effective_batch']] for i in
                  range(0,min(len(order),2*cfg['effective_batch']),cfg['effective_batch'])]
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    state_path = dest / 'latest.json'
    state_dir = dest / read(state_path)['checkpoint'] if state_path.exists() and not benchmark else None
    model = load_train_model(ctx, state_dir)
    tok = tokenizer_for(Path(ctx['inputs']) / 'stage2')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['learning_rate'], weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, round(len(groups)*.05)), len(groups))
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['dtype']=='float16')
    step = 0
    if state_dir:
        state = torch.load(state_dir / 'state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != training_signature:
            raise ValueError('Training resume signature mismatch')
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        step = state['step']
    dtype = getattr(torch, cfg['dtype'])
    model.train()
    elapsed = []

    def checkpoint_save():
        checkpoint = dest / f'checkpoint-{step:05d}'
        if checkpoint.exists():
            return
        tmp = dest / f'checkpoint-{step:05d}.partial'
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        torch.save({'step': step, 'signature': training_signature, 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'python_rng': random.getstate()}, tmp / 'state.pt')
        tmp.rename(checkpoint)
        write(state_path, {'checkpoint': checkpoint.name, 'step': step})
        for old in sorted(dest.glob('checkpoint-*'))[:-2]:
            if old.is_dir() and not old.name.endswith('.partial'):
                shutil.rmtree(old)

    for group in groups[step:]:
        if STOP:
            if not benchmark:
                checkpoint_save()
            raise SystemExit(75)
        t0 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        token_total = sum(sum(t != -100 for t in pairs[i]['chosen_tokens']['labels'][1:]) for i in group)
        loss_value = 0.
        diagnostics = {'chosen_score_per_token':0., 'rejected_logp_per_token':0.,
                       'chosen_reference_delta':0., 'rejected_reference_delta':0.,
                       'preference_margin':0., 'preference_accuracy':0.}
        for offset in range(0, len(group), cfg['microbatch']):
            sub = [pairs[i] for i in group[offset:offset+cfg['microbatch']]]
            if not preference:
                values, counts = logps(model, tok, [p['chosen_tokens'] for p in sub],
                                       dtype, dft=objective=='dft')
                loss = -values.sum() / token_total
            else:
                values, counts = logps(model, tok, [p['chosen_tokens'] for p in sub] +
                                  [p['rejected_tokens'] for p in sub], dtype)
                ref = torch.tensor([refs[p['id']] for p in sub], device='cuda')
                loss = preference_objective(values[:len(sub)],values[len(sub):],ref[:,0],ref[:,1],
                    cfg['beta'],len(group),token_total,
                    cfg.get('nll_weight',.1) if objective=='dpo_nll' else 0.)
                with torch.no_grad():
                    dc, dr = values[:len(sub)]-ref[:,0], values[len(sub):]-ref[:,1]
                    margin = cfg['beta']*(dc-dr)
                    diagnostics['chosen_reference_delta'] += dc.sum().item()/len(group)
                    diagnostics['rejected_reference_delta'] += dr.sum().item()/len(group)
                    diagnostics['preference_margin'] += margin.sum().item()/len(group)
                    diagnostics['preference_accuracy'] += (margin>0).sum().item()/len(group)
                    diagnostics['rejected_logp_per_token'] += (values[len(sub):]/counts[len(sub):]).sum().item()/len(group)
            diagnostics['chosen_score_per_token'] += (values[:len(sub)].detach()/counts[:len(sub)]).sum().item()/len(group)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            loss_value += loss.detach().item()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient; aborting without advancing scheduler')
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        torch.cuda.synchronize()
        elapsed.append(time.monotonic()-t0)
        print(f'{branch} step={step}/{len(groups)} loss={loss_value:.6f} seconds={elapsed[-1]:.2f}', flush=True)
        with (dest / ('benchmark.jsonl' if benchmark else 'train.jsonl')).open('a') as f:
            import json
            f.write(json.dumps({'step':step, 'loss':loss_value, 'seconds':elapsed[-1],
                                'max_vram':torch.cuda.max_memory_allocated(),
                                'objective':objective, 'gradient_norm':norm.item(),
                                'learning_rate':optimizer.param_groups[0]['lr'], **diagnostics})+'\n')
        if benchmark and step >= min(2, len(groups)):
            write(root / 'timings' / f'bench_{branch}.json', {'seconds_per_update': max(elapsed),
                        'updates':total_updates, 'microbatch':cfg['microbatch']})
            return
        if cfg.get('save_milestones') and step in {max(1, len(groups)//2), len(groups)}:
            milestone = dest / 'milestones' / f'step-{step:05d}'
            if not check_model(milestone):
                milestone.parent.mkdir(parents=True, exist_ok=True)
                save_model(model, tok, milestone)
        if step % cfg['checkpoint_every'] == 0:
            checkpoint_save()
    checkpoint_save()
    save_model(model, tok, dest / 'final')
    write(dest / 'complete.json', {'complete':True, 'updates':step, 'pairs':len(pairs),
                                 'signature':training_signature})


def audit(ctx):
    """Check actual terminal labels and save an unmodified adapter for roundtrip evaluation."""
    import torch
    from losses import collate
    root, cfg = Path(ctx['run']), ctx['config']
    tok = tokenizer_for(Path(ctx['inputs'])/'stage2')
    model_config = read(root/'models/b0/config.json')
    gen_path = root/'models/b0/generation_config.json'
    gen_config = read(gen_path) if gen_path.exists() else {}
    eos = gen_config.get('eos_token_id') or model_config.get('eos_token_id') or tok.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if not eos or tok.eos_token_id not in eos:
        raise ValueError('Training tokenizer EOS is not a generation stop ID')
    data = read(root/'pairs.json')[:32]
    if not data:
        raise ValueError('No valid examples available for the terminal-token audit')
    examples = []
    for row in data:
        tokens = row['chosen_tokens']
        labels = tokens['labels']
        stops = [i for i, label in enumerate(labels) if label in eos]
        if not stops or labels[-1] != tok.eos_token_id:
            raise ValueError('Completion does not supervise a generation stop token')
        examples.append({'id':row['id'], 'tail_ids':tokens['input_ids'][-12:],
                         'tail_tokens':tok.convert_ids_to_tokens(tokens['input_ids'][-12:]),
                         'tokens_after_first_stop':len(labels)-stops[0]-1})
    model = load_train_model(ctx)
    model.eval()
    batch = collate([data[0]['chosen_tokens']], tok.pad_token_id, 'cuda')
    labels = batch.pop('labels')
    with torch.no_grad(), torch.autocast('cuda', dtype=getattr(torch,cfg['dtype'])):
        out = model(**batch).logits
        stop_index = next(i for i,x in enumerate(labels[0].tolist()) if x in eos)
        logits_before = out[0,stop_index-1].float().clone()
        stop_probability = logits_before.softmax(-1)[eos].sum().item()
    dest = root/'training/rft/final'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not check_model(dest):
        save_model(model, tok, dest)
    # Loading the saved adapter on the same quantized base must preserve logits.
    model.load_adapter(dest, adapter_name='roundtrip', is_trainable=False)
    model.set_adapter('roundtrip')
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=getattr(torch,cfg['dtype'])):
        after = model(**batch).logits[0,stop_index-1].float()
    maximum = (after-logits_before).abs().max().item()
    if not torch.allclose(after, logits_before, atol=1e-3, rtol=1e-3):
        raise ValueError('Adapter save/reload changed logits: '+str(maximum))
    write(root/'audit.json', {'eos_ids':eos, 'tokenizer_eos':tok.eos_token_id,
        'examples':examples, 'roundtrip_max_abs_logit_difference':maximum,
        'teacher_forced_stop_probability_first_example':stop_probability,
        'terminal_policy':'legacy template preserved; trailing tokens explicitly reported'})


def report(ctx):
    root = Path(ctx['run'])
    result = {'status': 'pilot_only', 'candidates': {}, 'comparisons': {}, 'official_score': None}
    outputs = {}
    for name in ('b0', 'rft', 'dpo'):
        p = root / 'generations' / f'{name}_eval.json'
        if not p.exists():
            continue
        outputs[name] = {k:v[0] for k,v in read(p).items()}
        vals = list(outputs[name].values())
        n = len(vals)
        result['candidates'][name] = {'n':n, 'accuracy':sum(v['grade']['correct'] for v in vals)/n,
              'mean_output_tokens':sum(len(v['token_ids']) for v in vals)/n,
              'non_stop':sum(v['finish_reason']!='stop' for v in vals),
              'unknown':sum(any(v['grade'][k] for k in ('extraction_failed','grader_error','gold_unparseable')) for v in vals)}
        from collections import Counter
        result['candidates'][name]['repeated_10word_4times'] = sum(
            any(count >= 4 for count in Counter(tuple(v['text'].split()[i:i+10])
                for i in range(max(0,len(v['text'].split())-9))).values()) for v in vals)
        datasets = {r['id']:r['dataset'] for r in read(root/'eval_rows.json')}
        result['candidates'][name]['by_dataset'] = {}
        for dataset in ('prm800k','gsm8k'):
            subset = [v for key,v in outputs[name].items() if datasets[key]==dataset]
            result['candidates'][name]['by_dataset'][dataset] = {
                'n':len(subset), 'accuracy':sum(v['grade']['correct'] for v in subset)/len(subset) if subset else None}
    for name in ('rft', 'dpo'):
        if name in outputs and 'b0' in outputs:
            common = outputs[name].keys() & outputs['b0'].keys()
            if common:
                result['comparisons'][name] = comparison({k:outputs['b0'][k] for k in common},
                                                          {k:outputs[name][k] for k in common})
    if all(name in outputs and len(outputs[name])>=128 for name in ('b0','rft','dpo')):
        result['status'] = 'quick_comparison_complete' if min(map(len,outputs.values()))<400 else 'dev_comparison_complete'
    write(root / 'report.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['merge','generate','pairs','reference','bench','train','report','preflight','audit','audit_weights'])
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--candidate', default='b0', choices=['b0','rft','dpo'])
    parser.add_argument('--purpose', default='eval', choices=['train','eval','final'])
    parser.add_argument('--count', type=int, default=128)
    args = parser.parse_args()
    ctx = read(args.context)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if args.phase == 'merge': merge(ctx, args.candidate)
    elif args.phase == 'generate': generate(ctx, args.candidate, args.purpose, args.count)
    elif args.phase == 'pairs': token_pairs(ctx)
    elif args.phase == 'reference': reference(ctx)
    elif args.phase in ('train','bench'): train(ctx, args.candidate, args.phase=='bench')
    elif args.phase == 'report': report(ctx)
    elif args.phase == 'audit': audit(ctx)
    elif args.phase == 'audit_weights':
        from diagnose_roundtrip import verify_model_equivalence
        root = Path(ctx['run'])
        result = verify_model_equivalence(root/'models/b0',root/'models/rft')
        write(root/'integrity.json',result)
        if not result['identical']:
            raise ValueError('No-update model weights or inference configuration differ; see audit/integrity.json')
    else:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
        if torch.cuda.get_device_properties(0).total_memory < 40*2**30: raise RuntimeError('Expected a 48GB-class GPU')
        write(Path(ctx['run']) / 'gpu.json', {'name':torch.cuda.get_device_name(),
              'torch':torch.__version__, 'cuda':torch.version.cuda,
              'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes']}})


if __name__ == '__main__':
    main()
