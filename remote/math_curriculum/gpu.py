"""Reuse training/checkpoint engine; supply fresh Unsloth models and fused NLL."""
import argparse
import os
import signal
from common import *


def train(root, stage, until=None):
    # Must precede transformers/peft imports. Do not request full 8k vocabulary logits.
    os.environ['UNSLOTH_RETURN_LOGITS']='0'
    from unsloth import FastLanguageModel
    import torch
    import worker
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from losses import collate
    work=root/stage
    cfg=dict(CONFIG, dtype='float16', objective='rft', learning_rate=1e-4 if stage=='stage1' else 5e-5,
        epochs=1, effective_batch=16, microbatch=2 if stage=='stage1' else 1,
        context=2048 if stage=='stage1' else 8192, published_schedule=True,
        save_milestones=True, checkpoint_at_milestones=True, checkpoint_every=10,
        auto_oom_microbatch=True)
    parent=BASE if stage=='stage1' else str(root/'models/stage1')
    identity=dict(config=cfg,parent_revision=REVISION if stage=='stage1' else
                  sha(root/'models/stage1/artifacts.json'),code=read(root/'identity.json'))
    immutable_json(work/'training_contract.json',identity)
    ctx=dict(run=str(work),inputs=str(root/'tokenizer'),config=cfg,signature=digest(identity))
    def loader(context, adapter=None):
        kwargs={'revision':REVISION} if stage=='stage1' else {}
        model,tok=FastLanguageModel.from_pretrained(model_name=parent,max_seq_length=cfg['context'],
            dtype=torch.float16,load_in_4bit=True,use_exact_model_name=True,**kwargs)
        expected=worker.tokenizer_for(root/'tokenizer/stage2')
        if tok.get_vocab()!=expected.get_vocab(): raise ValueError('Tokenizer vocabulary changed')
        model=FastLanguageModel.get_peft_model(model,r=8,lora_alpha=16,lora_dropout=0,bias='none',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
            random_state=SEED,use_gradient_checkpointing='unsloth')
        if adapter:
            state=set_peft_model_state_dict(model,load_file(str(adapter/'adapter_model.safetensors')))
            if state.unexpected_keys: raise ValueError('Resume adapter keys differ')
        else:
            if any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n):
                raise ValueError('Expected a newly initialized adapter')
        for parameter in model.parameters():
            if parameter.requires_grad: parameter.data=parameter.data.float()
        if sum(p.numel() for p in model.parameters() if p.requires_grad)!=16515072:
            raise ValueError('Unexpected rank8 trainable parameter count')
        model.config.pad_token_id=expected.pad_token_id
        model.config.eos_token_id=expected.eos_token_id
        model.config.use_cache=False
        return model
    def native_logps(model,tok,rows,dtype,dft=False):
        if dft: raise ValueError('NLL only')
        batch=collate(rows,tok.pad_token_id,'cuda')
        count=(batch['labels'][:,1:]!=-100).sum().reshape(1)
        with torch.autocast('cuda',dtype=dtype):
            loss=model(**batch,use_cache=False).loss
        # The shared trainer needs the SUM of response log probabilities. Native
        # fused CE returns the mean over shifted nonmasked tokens in this microbatch.
        return -loss.reshape(1)*count,count
    worker.load_train_model=loader
    worker.logps=native_logps
    signal.signal(signal.SIGTERM,worker.request_stop)
    signal.signal(signal.SIGINT,worker.request_stop)
    torch.cuda.reset_peak_memory_stats()
    worker.train(ctx,'rft',until_step=until)


def merge(root, stage, milestone=None):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    from worker import save_model,check_model,tokenizer_for
    destination=root/'models'/(stage if stage=='stage1' else milestone)
    if stage=='stage1':
        adapter=root/'stage1/training/rft/final';parent=BASE;kwargs=dict(revision=REVISION)
    else:
        step=read(root/'stage2/schedule.json')['milestone_steps'][milestone]
        adapter=root/f'stage2/training/rft/milestones/step-{step:05d}'
        parent=root/'models/stage1';kwargs={}
        if not check_model(parent): raise ValueError('GSM parent not complete')
    if not check_model(adapter): raise ValueError('Adapter missing or changed')
    if not check_model(destination):
        model=AutoModelForCausalLM.from_pretrained(parent,torch_dtype=torch.float16,
            device_map='cuda',attn_implementation='sdpa',**kwargs)
        model=PeftModel.from_pretrained(model,adapter).merge_and_unload(safe_merge=True)
        tok=tokenizer_for(root/'tokenizer/stage2')
        model.config.pad_token_id=tok.pad_token_id
        model.generation_config.pad_token_id=tok.pad_token_id
        save_model(model,tok,destination)
    # Include tokenizer merges.txt as well as the shared helper's JSON/weights.
    artifacts=read(destination/'artifacts.json')
    for file in destination.iterdir():
        if file.is_file() and file.name not in ('artifacts.json','lineage.json'):
            artifacts[file.name]=sha(file)
    write(destination/'artifacts.json',artifacts)
    immutable_json(destination/'lineage.json',dict(base=BASE,revision=REVISION,
        stage1_sha256=sha(root/'stage1/training/rft/final/adapter_model.safetensors'),
        stage2_sha256=sha(adapter/'adapter_model.safetensors') if stage=='stage2' else None))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['train','merge'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--milestone');p.add_argument('--until',type=int);a=p.parse_args()
    if a.phase=='train': train(a.root,a.stage,a.until)
    else: merge(a.root,a.stage,a.milestone)
