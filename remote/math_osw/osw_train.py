"""Current-policy measurements, then one weighted teacher update per unique batch."""
import json
import os
import random
import time
from helpers import *


def train(root):
    import gpu
    import worker
    original = worker.train
    worker.train = lambda ctx, branch, **kwargs: train_context(root,ctx)
    try:
        gpu.train(root/'chain','stage2',recipe=read(root/'chain/recipe.json'))
    finally:
        worker.train = original


def train_context(root, ctx):
    import torch
    import worker
    import grading
    from evaluate import strict_grade
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
    from losses import scaled_optimizer_step

    work = root/'chain/stage2'
    dest = work/'training/rft'
    dest.mkdir(parents=True,exist_ok=True)
    rows, plan = read(work/'pairs.json'), read(work/'schedule.json')
    budget = read(root/'budget.json')
    cutoff = budget['train_deadline_epoch']
    signature = digest([ctx['signature'],sha(work/'pairs.json'),plan,read(root/'request.json')])
    latest = dest/'latest.json'
    saved = dest/read(latest)['checkpoint'] if latest.exists() else None
    cfg = ctx['config']
    random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    model = worker.load_train_model(ctx,saved)
    tok = worker.tokenizer_for(Path(ctx['inputs'])/'stage2')
    gold_rows = read(root/'data/gold.json')
    gold, failures = grading.prepare_gold(list(gold_rows.values()))
    if failures or any(v is None or v == [] for v in gold.values()):
        raise ValueError('Unparseable pinned training gold; no reward fabrication')
    for row in rows:
        if row['chosen_tokens']['labels'][-1] != tok.eos_token_id:
            raise ValueError('Missing actual supervised EOS')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=.01)
    # Preserve the successful 3k->10k schedule, even for a short pilot prefix.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_factor(step,438))
    scaler = torch.amp.GradScaler('cuda',enabled=True)
    step = updates = nonzero_lr_updates = 0
    seconds = []
    if saved:
        state = torch.load(saved/'state.pt',map_location='cpu',weights_only=False)
        if state['signature'] != signature: raise ValueError('OSW resume identity changed')
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        step,updates = state['step'],state['updates']
        nonzero_lr_updates,seconds = state['nonzero_lr_updates'],state['seconds']

    class Deadline(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            return worker.STOP or time.time() >= cutoff-30

    generation = GenerationConfig(do_sample=True,temperature=1.,top_p=1.,top_k=0,
        max_new_tokens=2048,eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,
        use_cache=True,repetition_penalty=1.)

    def save():
        checkpoint = dest/f'checkpoint-{step:05d}'
        if not checkpoint.exists():
            partial = checkpoint.with_suffix('.partial')
            if partial.exists():
                import shutil
                shutil.rmtree(partial)
            worker.save_model(model,tok,partial)
            torch.save(dict(signature=signature,step=step,updates=updates,
                nonzero_lr_updates=nonzero_lr_updates,seconds=seconds,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate()),partial/'state.pt')
            partial.rename(checkpoint)
        write(latest,dict(checkpoint=checkpoint.name,step=step))
        link_readonly(checkpoint,dest/'milestones'/f'step-{step:05d}')

    while step < len(plan['groups']):
        if worker.STOP or not can_start_batch(time.time(),cutoff,seconds): break
        started = time.time()
        group = [rows[i] for i in plan['groups'][step]]
        weights, records = [], []
        interrupted = False
        model.eval(); model.gradient_checkpointing_disable()
        for row in group:
            if worker.STOP or time.time() >= cutoff-30:
                interrupted = True; break
            prefix = prefix_ids(row['chosen_tokens'])
            prompt = torch.tensor([prefix],device='cuda')
            # Two draws from the same current NF4+LoRA policy, no teacher/API.
            with torch.inference_mode():
                output = model.generate(input_ids=prompt,attention_mask=torch.ones_like(prompt),
                    generation_config=generation,num_return_sequences=2,
                    stopping_criteria=StoppingCriteriaList([Deadline()]))
            if worker.STOP or time.time() >= cutoff-30:
                interrupted = True; break  # Never grade deadline-truncated rollouts.
            rewards, samples = [], []
            for ids in output[:,len(prefix):].tolist():
                if tok.eos_token_id in ids: ids = ids[:ids.index(tok.eos_token_id)+1]
                text = tok.decode(ids,skip_special_tokens=True)
                grade = strict_grade(gold_rows[row['id']],text,gold[row['id']])
                if grade['grader_error']: raise ValueError('Training verifier error')
                rewards.append(bool(grade['correct']))
                samples.append(dict(token_ids=ids,grade=grade))
            weights.append(weight(rewards))
            records.append(dict(id=row['id'],dataset=row['dataset'],weight=weights[-1],samples=samples))
            del output,prompt
        model.train(); model.gradient_checkpointing_enable()
        if interrupted:
            write(dest/'discarded_batch.json',dict(step=step+1,reason='time_or_signal',
                completed_rollout_questions=len(records),teacher_update_applied=False))
            break
        token_total = sum(sum(v!=-100 for v in row['chosen_tokens']['labels'][1:]) for row in group)
        lr_used = optimizer.param_groups[0]['lr']
        loss_value, norm_value = 0.,0.
        if any(weights):
            for attempt in range(8):
                if time.time() >= cutoff or worker.STOP:
                    interrupted = True; break
                optimizer.zero_grad(set_to_none=True); loss_value = 0.
                for row,w in zip(group,weights):
                    if not w: continue
                    values,_ = worker.logps(model,tok,[row['chosen_tokens']],torch.float16)
                    loss = -values.sum()*w/token_total
                    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite OSW loss')
                    scaler.scale(loss).backward(); loss_value += loss.detach().item()
                updated,norm = scaled_optimizer_step(optimizer,scaler,params)
                if updated:
                    norm_value = norm.item(); updates += 1
                    nonzero_lr_updates += int(lr_used > 0)
                    break
            else:
                raise FloatingPointError('Nonfinite gradients after 8 scaler retries')
        if interrupted:
            optimizer.zero_grad(set_to_none=True); break
        # A zero-weight batch consumes data/schedule, but never applies weight decay.
        if any(weights): scheduler.step()
        else:
            # Match the predeclared batch-position LR without optimizer.step().
            scheduler.last_epoch += 1
            scheduler._last_lr = [base*lr_factor(step+1,438) for base in scheduler.base_lrs]
            for pg,lr in zip(optimizer.param_groups,scheduler._last_lr): pg['lr'] = lr
        step += 1
        torch.cuda.synchronize(); seconds.append(time.time()-started)
        record = dict(step=step,questions=step*16,optimizer_updates=updates,
            nonzero_lr_updates=nonzero_lr_updates,loss=loss_value,gradient_norm=norm_value,
            learning_rate=lr_used,seconds=seconds[-1],weights=weights,
            teacher_tokens=token_total,weighted_teacher_tokens=sum(w*sum(v!=-100 for v in r['chosen_tokens']['labels'][1:]) for r,w in zip(group,weights)),
            peak_vram_bytes=torch.cuda.max_memory_allocated())
        immutable_json(dest/f'rollouts_{step:05d}.json',records)
        with (dest/'train.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        save()
        print('OSW_BATCH',json.dumps(record),flush=True)

    if not step or not nonzero_lr_updates:
        raise RuntimeError('No nonzero-LR update completed; refusing to label the ancestor as a new trained model')
    save()
    name = candidate_name(step*16)
    exp = root/name
    link_readonly(dest/f'checkpoint-{step:05d}',exp/'checkpoint')
    immutable_json(exp/'training_complete.json',dict(complete=True,pilot=True,
        planned_questions=len(rows),consumed_questions=step*16,optimizer_updates=updates,
        nonzero_lr_updates=nonzero_lr_updates,step=step,signature=signature,
        trained_this_chain=counts(rows[:step*16]),realized=read(exp/'experiment.json')['realized'],
        stop_reason='pilot_maximum' if step==len(plan['groups']) else 'submission_time_reserve',
        checkpoint_files=file_receipt(exp,[exp/'checkpoint']),
        algorithm='OSW-inspired LoRA; fixed token denominator; strict boxed verifier; 2 current-policy samples',
        recipe=read(root/'chain/recipe.json')))
    write(root/'selected.json',dict(name=name,consumed_questions=step*16,
        selection='last completed unique batch, time/size only; no evaluation or official score selection'))
    print('OSW_TRAINING_READY',name,flush=True)
