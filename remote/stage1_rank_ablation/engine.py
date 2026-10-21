"""Isolated GPU worker. B/C/D start with a NEW math adapter. A is inference only."""
import argparse
import gc
import math
import os
import random
import shutil
import signal
import time
from pathlib import Path
from common import *

STOP = False

def stop(*_):
    global STOP
    STOP = True


def collate(rows, pad, layout, device):
    import torch
    if layout == 'packed':
        ids, labels, positions = [], [], []
        for row in rows:
            seq = row['input_ids']
            ids.extend(seq); labels.extend([-100]+seq[1:]); positions.extend(range(len(seq)))
        return {k:torch.tensor([v],device=device) for k,v in
                [('input_ids',ids),('labels',labels),('position_ids',positions)]}
    length = max(len(r['input_ids']) for r in rows)
    ids, labels, masks = [], [], []
    for r in rows:
        seq = r['input_ids']; n = length-len(seq)
        ids.append(seq+[pad]*n); labels.append([-100]+seq[1:]+[-100]*n)
        masks.append([1]*len(seq)+[0]*n)
    return {k:torch.tensor(v,device=device) for k,v in
            [('input_ids',ids),('labels',labels),('attention_mask',masks)]}


def target_count(rows): return sum(len(r['input_ids'])-1 for r in rows)


def fresh_model(root, condition, cfg):
    # Imports must follow Unsloth in that backend's process.
    if cfg['backend'] == 'unsloth':
        os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
        from unsloth import FastLanguageModel
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from worker import tokenizer_for
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED)
    parent = root/'parents'/('stage1' if condition['stage1'] else 'base')
    if not verify_model(parent): raise ValueError('Parent missing or incomplete')
    tok = tokenizer_for(root/'inputs/stage1')
    if cfg['backend'] == 'unsloth':
        model, loaded_tok = FastLanguageModel.from_pretrained(model_name=str(parent),
            max_seq_length=2048,dtype=torch.float16,load_in_4bit=cfg['weights']=='nf4',
            use_exact_model_name=True)
        if loaded_tok.get_vocab() != tok.get_vocab(): raise ValueError('Unsloth changed vocabulary')
        model = FastLanguageModel.get_peft_model(model,r=condition['rank'],lora_alpha=condition['alpha'],
            target_modules=MODULES,lora_dropout=0.,bias='none',random_state=SEED,
            use_gradient_checkpointing='unsloth' if cfg['checkpointing'] else False)
    else:
        kwargs = dict(torch_dtype=torch.float16,device_map='cuda',attn_implementation=cfg['attention'])
        if cfg['weights']=='nf4':
            kwargs['quantization_config']=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(parent,**kwargs)
        if cfg['weights']=='nf4':
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=cfg['checkpointing'],
                gradient_checkpointing_kwargs={'use_reentrant':False})
        else:
            model.requires_grad_(False)
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        model=get_peft_model(model,LoraConfig(r=condition['rank'],lora_alpha=condition['alpha'],
            target_modules=MODULES,lora_dropout=0.,bias='none',task_type='CAUSAL_LM',init_lora_weights=True))
        if cfg['checkpointing']:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data=parameter.data.float()
    quant=model.config.quantization_config
    quant=quant.to_dict() if hasattr(quant,'to_dict') else quant
    if not quant.get('load_in_4bit',quant.get('_load_in_4bit',False)) or quant.get('bnb_4bit_quant_type')!='nf4' or not quant.get('bnb_4bit_use_double_quant'):
        raise ValueError('Backend changed the NF4 double-quantization contract')
    if str(quant.get('bnb_4bit_compute_dtype')).replace('torch.','')!='float16':
        raise ValueError('Backend changed FP16 computation')
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable!=16515072*condition['rank']//8:
        raise ValueError(f'Unexpected trainable parameter count: {trainable}')
    model.config.use_cache=False
    config=model.peft_config['default']
    if config.r!=condition['rank'] or config.lora_alpha!=condition['alpha']:
        raise ValueError('LoRA configuration mismatch')
    # No learned math adapter has been loaded; default LoRA B must be zero.
    for name,p in model.named_parameters():
        if 'lora_B' in name and torch.count_nonzero(p).item(): raise ValueError('Math adapter is not fresh')
    return model,tok


def native_loss(model,tok,rows,cfg):
    import torch
    batch=collate(rows,tok.pad_token_id,cfg['layout'],'cuda')
    with torch.autocast('cuda',dtype=torch.float16):
        result=model(**batch,use_cache=False)
    if result.loss is None or not torch.isfinite(result.loss): raise FloatingPointError('Nonfinite or missing loss')
    return result.loss


def audit_loss(model,tok,rows,cfg):
    """Native reduction + packed attention must reproduce independent causal CE and gradients."""
    import torch
    import torch.nn.functional as F
    # Audit two real examples (short prefixes only for numerical checks; actual training is unmodified).
    tiny=[dict(input_ids=r['input_ids'][:min(48,len(r['input_ids']))]) for r in rows[:2]]
    model.eval(); model.zero_grad(set_to_none=True)
    total=target_count(tiny)
    expected=0.
    for row in tiny:
        batch=collate([row],tok.pad_token_id,'padded','cuda');labels=batch.pop('labels')
        with torch.autocast('cuda',dtype=torch.float16): out=model(**batch,use_cache=False)
        logits=out.logits
        if not isinstance(logits,torch.Tensor): raise ValueError('Backend does not expose logits for audit')
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),
                             labels[:,1:].reshape(-1),ignore_index=-100,reduction='sum')/total
        loss.backward();expected+=loss.detach().item()
    reference={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    actual=native_loss(model,tok,tiny,cfg);actual.backward()
    num=den=0.
    for n,p in model.named_parameters():
        if n in reference:
            if p.grad is None: raise ValueError('Missing native gradient')
            num+=(p.grad.float()-reference[n].float()).square().sum().item()
            den+=reference[n].float().square().sum().item()
    relative=(num/max(den,1e-20))**.5
    if abs(actual.item()-expected)>max(.02,abs(expected)*.005) or relative>.06:
        raise ValueError(f'Loss/attention audit failed: native={actual.item()} reference={expected} grad={relative}')
    result=dict(native_loss=actual.item(),independent_loss=expected,relative_gradient_error=relative,
                packing=cfg['layout'],target_count=total)
    del reference,actual,out,logits,loss
    model.zero_grad(set_to_none=True);model.train();gc.collect();torch.cuda.empty_cache()
    return result


def optimizer_for(model):
    import bitsandbytes as bnb
    return bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],lr=5e-5,
                                  betas=(.9,.999),eps=1e-8,weight_decay=0.)


def update(model,tok,rows,cfg,optimizer,scaler):
    import torch
    # Retry the same effective batch after AMP overflow; never advance the scheduler on a skipped update.
    for attempt in range(12):
        optimizer.zero_grad(set_to_none=True)
        total=target_count(rows);value=0.
        for start in range(0,len(rows),cfg['microbatch']):
            part=rows[start:start+cfg['microbatch']]
            loss=native_loss(model,tok,part,cfg)*target_count(part)/total
            scaler.scale(loss).backward();value+=loss.detach().item()
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            old=scaler.get_scale();scaler.update(new_scale=old/2)
            print(f'AMP overflow retry={attempt+1} scale={old/2}',flush=True)
            continue
        scaler.step(optimizer);scaler.update()
        return value,norm.item()
    raise FloatingPointError('Nonfinite gradients persist; update not committed')


def benchmark(root,condition,cfg,target,smoke=False):
    import torch
    start=time.monotonic();model,tok=fresh_model(root,condition,cfg)
    rows=read(root/'data/tokens.json');groups=read(root/'data/groups.json')
    audit=audit_loss(model,tok,rows,cfg)
    optimizer=optimizer_for(model)
    scaler=torch.amp.GradScaler("cuda",init_scale=256.)
    torch.cuda.reset_peak_memory_stats()
    # All candidates must fit 16 longest examples, even if they do not co-occur in training.
    stress=sorted(rows,key=lambda r:len(r['input_ids']),reverse=True)[:16]
    update(model,tok,stress,cfg,optimizer,scaler);torch.cuda.synchronize()
    cold=time.monotonic()-start
    timings=[];tokens=0
    ranked=sorted(range(len(groups)),key=lambda idx:sum(len(rows[i]['input_ids']) for i in groups[idx]))
    representatives=[ranked[int((len(ranked)-1)*q)] for q in (.17,.5,.83)]
    for idx in ([ranked[len(ranked)//2]] if smoke else representatives):
        part=[rows[i] for i in groups[idx]]
        torch.cuda.synchronize();t=time.monotonic()
        update(model,tok,part,cfg,optimizer,scaler);torch.cuda.synchronize()
        timings.append(time.monotonic()-t);tokens+=target_count(part)
    estimate=sum(timings)/len(timings)*TOTAL_STEPS
    print(f'BENCH tokens/sec={tokens/sum(timings):.1f} peak_VRAM_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}',flush=True)
    print(f'BENCH rank={condition["rank"]} seconds/update={estimate/TOTAL_STEPS:.3f} estimated_375_updates_minutes={estimate/60:.1f} (approximate, training only)',flush=True)
    write(target,dict(estimated_training_seconds=estimate,approximate=True,updates=TOTAL_STEPS,excluded='initialization, warmup, audit, saving, evaluation',status='ok',config=cfg,condition=condition,audit=audit,
        cold_seconds=cold,seconds_per_update=sum(timings)/len(timings),samples=timings,representative_groups=representatives,
        estimated_training_seconds_range=[min(timings)*TOTAL_STEPS,max(timings)*TOTAL_STEPS],
        supervised_tokens_per_second=tokens/sum(timings),peak_vram_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def train(root,name,cfg):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    if name=='A': raise ValueError('A is an existing run03 model; training is forbidden')
    destination=root/'conditions'/name
    if (destination/'complete.json').exists() and verify_model(destination/'final'): return
    destination.mkdir(parents=True,exist_ok=True)
    signature=digest([read(root/'identity.json'),read(root/'data/report.json'),cfg,CONDITIONS[name]])
    immutable_json(destination/'contract.json',dict(signature=signature,condition=CONDITIONS[name],config=cfg))
    model,tok=fresh_model(root,CONDITIONS[name],cfg)
    optimizer=optimizer_for(model)
    scaler=torch.amp.GradScaler("cuda",init_scale=256.)
    scheduler=get_cosine_schedule_with_warmup(optimizer,math.ceil(TOTAL_STEPS*.03),TOTAL_STEPS)
    rows=read(root/'data/tokens.json');groups=read(root/'data/groups.json')
    step=0; seconds=0.; peak=0; wall=0.
    latest=destination/'latest.json'
    if latest.exists():
        checkpoint=destination/read(latest)['checkpoint']
        if not verify_model(checkpoint): raise ValueError('Invalid resume checkpoint')
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        if state['signature']!=signature: raise ValueError('Resume signature mismatch')
        loaded=set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
        if loaded.unexpected_keys: raise ValueError('Unexpected adapter keys during resume')
        scaler.load_state_dict(state['scaler'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng'])
        step,seconds,peak,wall=state['step'],state['seconds'],state['peak'],state['wall']
    model.train();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    def save_checkpoint():
        target=destination/f'checkpoint-{step:04d}'
        if target.exists(): return
        tmp=target.with_name(target.name+'.partial')
        if tmp.exists():shutil.rmtree(tmp)
        model.save_pretrained(tmp,safe_serialization=True);tok.save_pretrained(tmp)
        torch.save(dict(signature=signature,step=step,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
            seconds=seconds,peak=peak,wall=wall+time.monotonic()-started),tmp/'state.pt')
        write(tmp/'artifacts.json',dict(fingerprint(tmp),**{'state.pt':sha(tmp/'state.pt')}))
        tmp.rename(target);write(latest,dict(checkpoint=target.name,step=step))
        for old in sorted(p for p in destination.glob('checkpoint-*') if not p.name.endswith('.partial'))[:-2]:
            shutil.rmtree(old)
    while step<TOTAL_STEPS:
        if STOP:save_checkpoint();raise SystemExit(75)
        part=[rows[i] for i in groups[step]]
        torch.cuda.synchronize();t=time.monotonic()
        loss,norm=update(model,tok,part,cfg,optimizer,scaler);scheduler.step();torch.cuda.synchronize()
        elapsed=time.monotonic()-t;seconds+=elapsed;step+=1
        peak=max(peak,torch.cuda.max_memory_allocated())
        write(destination/f'updates/{step:04d}.json',dict(step=step,loss=loss,gradient_norm=norm,
            seconds=elapsed,remaining_training_seconds=seconds/step*(TOTAL_STEPS-step),supervised_tokens=target_count(part),learning_rate=optimizer.param_groups[0]['lr']))
        print(f'{name} step={step}/{TOTAL_STEPS} loss={loss:.5f} seconds={elapsed:.2f} ETA_training_minutes={seconds/step*(TOTAL_STEPS-step)/60:.1f} (approx)',flush=True)
        if step%25==0: save_checkpoint()
    save_checkpoint();save_model(model,tok,destination/'final')
    total=read(root/'data/report.json')['supervised_tokens']
    write(destination/'complete.json',dict(signature=signature,complete=True,steps=step,epoch=1,
        train_seconds=seconds,wall_seconds=wall+time.monotonic()-started,peak_vram_bytes=peak,
        supervised_tokens=total,tokens_per_second=total/seconds,condition=CONDITIONS[name],backend=cfg))


def parents(root):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from worker import compatible_adapter,tokenizer_for
    tok=tokenizer_for(root/'inputs/stage1')
    base=root/'parents/base';stage1=root/'parents/stage1'
    if verify_model(stage1) and verify_model(base): return
    model=AutoModelForCausalLM.from_pretrained(BASE,revision=REVISION,torch_dtype=torch.float16,
                                              device_map='cuda',attn_implementation='sdpa')
    save_model(model,tok,base)
    if not verify_model(stage1):
        source=compatible_adapter(root/'inputs/stage1',root/'compatible/stage1')
        model=PeftModel.from_pretrained(model,source).merge_and_unload(safe_merge=True)
        save_model(model,tok,stage1)


def merge(root,name):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from worker import tokenizer_for
    out=root/'merged'/name
    if verify_model(out):return
    adapter=root/'conditions'/name/'final'
    if name=='A':
        from worker import compatible_adapter
        validate_inputs(root/'inputs')
        adapter=compatible_adapter(root/'inputs/stage2',root/'compatible/stage2')
    elif not verify_model(adapter):raise ValueError('Training not complete')
    parent=root/'parents'/('stage1' if CONDITIONS[name]['stage1'] else 'base')
    model=AutoModelForCausalLM.from_pretrained(parent,torch_dtype=torch.float16,device_map='cuda',attn_implementation='sdpa')
    model=PeftModel.from_pretrained(model,adapter).merge_and_unload(safe_merge=True)
    save_model(model,tokenizer_for(root/'inputs/stage1'),out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['parents','bench','smoke','train','merge'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--condition',choices=list(CONDITIONS),default='D')
    p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    cfg=read(a.config) if a.config else None
    if cfg and cfg['backend']=='unsloth':
        os.environ['UNSLOTH_RETURN_LOGITS']='1'
        import unsloth
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    if torch.cuda.get_device_properties(0).total_memory<40*2**30:raise RuntimeError('48GB-class GPU required')
    if a.phase=='parents':parents(a.root)
    elif a.phase=='merge':merge(a.root,a.condition)
    elif a.phase=='train':train(a.root,a.condition,cfg)
    else:
        try:benchmark(a.root,CONDITIONS[a.condition],cfg,a.output,a.phase=='smoke')
        except Exception as exc:
            write(a.output,dict(status='failed',error=str(exc),config=cfg));raise
