import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from peft import LoraConfig,PeftModel,get_peft_model
from losses import sequence_logps,dpo_loss


def test_saved_adapter_reloads_trainable_and_dpo_updates(tmp_path):
    torch.manual_seed(1)
    config=Qwen3Config(vocab_size=50,hidden_size=32,intermediate_size=64,
                      num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=16)
    base=Qwen3ForCausalLM(config)
    base_state={k:v.clone() for k,v in base.state_dict().items()}
    model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.save_pretrained(tmp_path/'adapter')
    new_base=Qwen3ForCausalLM(config);new_base.load_state_dict(base_state)
    model=PeftModel.from_pretrained(new_base,tmp_path/'adapter',is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.config.use_cache=False
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    ids=torch.randint(0,50,(2,7));labels=ids.clone();labels[:,:3]=-100
    with torch.no_grad():ref=sequence_logps(model(ids).logits,labels)[0]
    values,_=sequence_logps(model(ids).logits,labels)
    loss=dpo_loss(values[:1],values[1:],ref[:1],ref[1:],.1).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    optimizer.step()
    with torch.no_grad():after=sequence_logps(model(ids).logits,labels)[0]
    assert (after[0]-after[1]) > (ref[0]-ref[1])
