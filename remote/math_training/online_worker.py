"""On-policy group-normalized REINFORCE, one update per fresh rollout batch.

No offline DPO, no reference KL, no clipped multi-epoch GRPO claims. Sampling uses
the same QLoRA policy and untempered full softmax used by the gradient. Exact
sampled token IDs are supervised, including EOS only when actually sampled.
"""
import argparse
import json
import random
import shutil
import signal
from pathlib import Path
import time
from core import read, write, sha, digest
from accuracy_utils import advantages, online_groups, rollout_tokens
import worker


def policy_objective(sequence_logps, token_counts, group_advantages, divisor, fixed_length=None):
    """Legacy per-response mean, or fixed-length Dr.-inspired normalization.

    E2/E3 opts in; other suites retain reward-std and response-length scaling.
    This remains one on-policy REINFORCE update, without PPO clipping or KL.
    """
    import torch
    adv = torch.as_tensor(group_advantages, device=sequence_logps.device,
                          dtype=sequence_logps.dtype)
    denominator = fixed_length if fixed_length is not None else token_counts.clamp_min(1)
    return -(sequence_logps/denominator*adv).sum()/divisor


def train(ctx, until):
    import torch
    import grading
    from transformers import GenerationConfig
    from losses import scaled_optimizer_step
    cfg, root = ctx['config'], Path(ctx['run'])
    dest = root/'training/rft'
    dest.mkdir(parents=True, exist_ok=True)
    rows = read(root/'online_rows.json')
    groups = online_groups(rows, cfg['online_rounds'], cfg['online_prompts'], cfg['seed'])
    signature = digest([ctx['signature'], sha(root/'online_rows.json'), groups])
    tok = worker.tokenizer_for(Path(ctx['inputs'])/'stage2')
    gold, failures = grading.prepare_gold(rows)
    # Gold parsing is independent of model answers; unavailable graders are excluded mechanically.
    usable = {k for k, v in gold.items() if v is not None and v != []}
    write(root/'reward_audit.json', {'gold_parse_failures': failures, 'usable': len(usable),
          'reward': 'final-answer correctness; no length/stop/format bonus',
          'policy': 'full-softmax temperature1 top_p1 top_k0; no dropout; same quantized policy',
          'objective': ('mean-centered REINFORCE, fixed max_tokens normalization; Dr. GRPO-inspired, not a full reproduction'
                        if cfg.get('dr_normalization') else
                        'group-normalized REINFORCE, one update per rollout batch; beta=0')})
    if not usable:
        raise ValueError('No parseable reward targets')
    seed = cfg['seed']
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    latest = dest/'latest.json'
    checkpoint = dest/read(latest)['checkpoint'] if latest.exists() else None
    model = worker.load_train_model(ctx, checkpoint)
    # Keep rollout and gradient policies identical even for adapters with LoRA dropout.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['online_lr'], weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['dtype'] == 'float16')
    step, updates = 0, 0
    sample_batch = cfg['online_generations']
    if checkpoint:
        state = torch.load(checkpoint/'state.pt', map_location='cpu', weights_only=False)
        if state['signature'] != signature:
            raise ValueError('Online resume identity mismatch')
        optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step, updates = state['step'], state['updates']
        sample_batch = state.get('sample_batch', sample_batch)
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
    until = min(until, len(groups))
    generation = GenerationConfig(do_sample=True, temperature=1., top_p=1., top_k=0,
        max_new_tokens=cfg['max_tokens'], eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, use_cache=True, repetition_penalty=1.)

    def save():
        target = dest/f'checkpoint-{step:05d}'
        if not target.exists():
            tmp = target.with_name(target.name+'.partial')
            if tmp.exists(): shutil.rmtree(tmp)
            model.save_pretrained(tmp, safe_serialization=True); tok.save_pretrained(tmp)
            torch.save({'signature': signature, 'step': step, 'updates': updates,
                        'sample_batch': sample_batch,
                        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()},
                       tmp/'state.pt')
            tmp.rename(target)
        write(latest, {'checkpoint': target.name, 'step': step})
        old = sorted(p for p in dest.glob('checkpoint-*') if not p.name.endswith('.partial'))
        for path in old[:-2]: shutil.rmtree(path)

    while step < until:
        if worker.STOP: raise SystemExit(75)
        started = time.monotonic()
        records, all_rewards, variable_groups, raw = [], [], 0, {}
        model.eval()
        model.gradient_checkpointing_disable()
        for index in groups[step]:
            row = rows[index]
            if row['id'] not in usable:
                continue
            prefix = tok.apply_chat_template([{'role': 'user', 'content': row['problem']}],
                                            tokenize=True, add_generation_prompt=True)
            if len(prefix)+cfg['max_tokens'] > cfg['context']:
                raise ValueError('Online prompt exceeds context')
            prompt = torch.tensor([prefix], device='cuda')
            samples, rewards = [], []
            while len(samples) < cfg['online_generations']:
                if worker.STOP: raise SystemExit(75)
                sample = len(samples)
                count = min(sample_batch, cfg['online_generations']-sample)
                torch.manual_seed((seed + step*100003 + int(row['id'][:8], 16) + sample) % 2**31)
                try:
                    with torch.no_grad():
                        output = model.generate(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                            generation_config=generation, num_return_sequences=count)
                    generated = output[:, len(prefix):].tolist()
                    del output
                except torch.cuda.OutOfMemoryError:
                    if count == 1: raise
                    import gc
                    gc.collect(); torch.cuda.empty_cache()
                    sample_batch = max(1, count//2)
                    print('ROLLOUT_OOM_RETRY sample_batch=', sample_batch, flush=True)
                    continue
                for ids in generated:
                    tokens = rollout_tokens(prefix, ids, tok.eos_token_id, cfg['context'])
                    text = tok.decode(tokens['input_ids'][len(prefix):], skip_special_tokens=True)
                    result = grading.grade(row, text, gold[row['id']])
                    if result['grader_error']:
                        raise ValueError('Online verifier errored; refusing to train on an invalid reward')
                    reward = float(result['correct'])
                    rewards.append(reward)
                    samples.append({'tokens': tokens, 'text': text, 'grade': result,
                                    'finish_reason': 'stop' if tok.eos_token_id in ids else 'length'})
            adv = ([r-sum(rewards)/len(rewards) for r in rewards]
                   if cfg.get('dr_normalization') else advantages(rewards))
            all_rewards.extend(rewards)
            raw[row['id']] = samples
            if any(a != 0 for a in adv):
                variable_groups += 1
                records.extend((s['tokens'], a) for s, a in zip(samples, adv))
        # Store every rollout before updating. Resume repeats a whole uncommitted round.
        write(root/f'rollouts/round-{step+1:05d}.json', raw)
        model.train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
        loss_value = 0.
        if records:
            for retry in range(8):
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.
                for tokens, adv in records:
                    values, counts = worker.logps(model, tok, [tokens], getattr(torch, cfg['dtype']))
                    # Include zero-advantage groups in the denominator.
                    divisor = len(raw)*cfg['online_generations']
                    loss = policy_objective(values, counts, [adv], divisor,
                        fixed_length=cfg['max_tokens'] if cfg.get('dr_normalization') else None)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite online loss')
                    scaler.scale(loss).backward()
                    loss_value += loss.detach().item()
                updated, norm = scaled_optimizer_step(optimizer, scaler, model.parameters())
                if updated:
                    updates += 1
                    break
            else:
                raise FloatingPointError('Online gradients still nonfinite after 8 retries')
        step += 1
        # A completed round is the atomic resume boundary, including optimizer state.
        save()
        record = {'round': step, 'optimizer_updates': updates, 'variable_groups': variable_groups,
                  'reward_mean': sum(all_rewards)/len(all_rewards) if all_rewards else None,
                  'sample_count': len(all_rewards), 'loss': loss_value,
                  'rewards': all_rewards,
                  'generation_lengths': [sum(x != -100 for x in s['tokens']['labels'])
                                         for samples in raw.values() for s in samples],
                  'optimizer_updates_this_round': int(bool(records)),
                  'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                  'sample_batch': sample_batch,
                  'seconds': time.monotonic()-started}
        write(dest/f'round_metrics/{step:05d}.json', record)
        print('ONLINE', json.dumps(record), flush=True)
    milestone = dest/f'milestones/step-{step:05d}'
    if not worker.check_model(milestone):
        milestone.parent.mkdir(parents=True, exist_ok=True)
        worker.save_model(model, tok, milestone)
    if step == len(groups):
        if not worker.check_model(dest/'final'):
            worker.save_model(model, tok, dest/'final')
        write(dest/'complete.json', {'rounds': step, 'optimizer_updates': updates, 'signature': signature})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', type=Path, required=True)
    p.add_argument('--until-round', type=int, required=True)
    a = p.parse_args()
    signal.signal(signal.SIGINT, worker.request_stop)
    signal.signal(signal.SIGTERM, worker.request_stop)
    train(read(a.context), a.until_round)
