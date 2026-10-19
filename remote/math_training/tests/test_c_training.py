"""Real CPU autograd/Adam/scheduler checks for pause/resume and whole-update OOM retry."""
import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import worker
from core import write,read
from c_utils import make_plan


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([.4]))
    def save_pretrained(self,path,**kwargs):
        Path(path).mkdir(parents=True,exist_ok=True)
        torch.save(self.state_dict(),Path(path)/'weights.pt')
        write(Path(path)/'config.json',{'synthetic':True})


class Tokenizer:
    def save_pretrained(self,path): write(Path(path)/'tokenizer_config.json',{'synthetic':True})


def setup(tmp_path,monkeypatch,microbatch=2,oom=False):
    root=tmp_path; rows=[{'id':str(i),'dataset':'gsm8k','chosen_tokens':
          {'input_ids':[1,i+2,3,99],'labels':[-100,i+2,3,99]}} for i in range(32)]
    plan=make_plan(rows,[list(range(16)),list(range(16,32)),list(range(16)),list(range(16,32))],
                   milestones={'first':1,'half':2,'full':4})
    write(root/'pairs.json',rows);write(root/'schedule.json',plan)
    calls=[]; raised=[False]
    def load(ctx,adapter=None):
        model=Model()
        if adapter: model.load_state_dict(torch.load(Path(adapter)/'weights.pt',weights_only=True))
        return model
    def logps(model,tok,sub,dtype,dft=False):
        calls.append(len(sub))
        if oom and len(sub)==4 and len(calls)==2 and not raised[0]:
            raised[0]=True
            raise torch.cuda.OutOfMemoryError('synthetic OOM after partial gradients')
        targets=torch.tensor([r['input_ids'][1]/100 for r in sub])
        weights=torch.rand(len(sub))+.5
        return -3*weights*(model.weight-targets).square(),torch.full((len(sub),),3)
    monkeypatch.setattr(worker,'load_train_model',load)
    monkeypatch.setattr(worker,'tokenizer_for',lambda _:Tokenizer())
    monkeypatch.setattr(worker,'logps',logps)
    monkeypatch.setattr(worker,'STOP',False)
    for name in ('manual_seed_all','synchronize','set_rng_state_all','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    ctx={'run':str(root),'inputs':str(root/'inputs'),'signature':'fixed',
         'config':{'objective':'rft','published_schedule':True,'published_mixed':False,
                   'epochs':1,'effective_batch':16,'seed':42,'learning_rate':.01,
                   'dtype':'bfloat16','microbatch':microbatch,'checkpoint_every':1,
                   'save_milestones':True,'auto_oom_microbatch':True}}
    return ctx,calls


def test_paused_training_keeps_full_scheduler_and_exact_optimizer_state(tmp_path,monkeypatch):
    full,_=setup(tmp_path/'full',monkeypatch)
    worker.train(full,'rft')
    paused,_=setup(tmp_path/'paused',monkeypatch)
    worker.train(paused,'rft',until_step=1)
    assert not (tmp_path/'paused/training/rft/final').exists()
    state=torch.load(tmp_path/'paused/training/rft/checkpoint-00001/state.pt',weights_only=False)
    assert state['step']==1 and state['scheduler']['last_epoch']==1
    worker.train(paused,'rft',until_step=4)
    left=torch.load(tmp_path/'full/training/rft/final/weights.pt',weights_only=True)
    right=torch.load(tmp_path/'paused/training/rft/final/weights.pt',weights_only=True)
    assert torch.equal(left['weight'],right['weight'])
    assert read(tmp_path/'paused/training/rft/complete.json')['updates']==4


def test_oom_discards_partial_gradients_rewinds_rng_and_retries_whole_update(tmp_path,monkeypatch):
    clean,_=setup(tmp_path/'clean',monkeypatch,2)
    worker.train(clean,'rft',until_step=1)
    retry,calls=setup(tmp_path/'retry',monkeypatch,4,oom=True)
    worker.train(retry,'rft',until_step=1)
    left=torch.load(tmp_path/'clean/training/rft/checkpoint-00001/weights.pt',weights_only=True)
    right=torch.load(tmp_path/'retry/training/rft/checkpoint-00001/weights.pt',weights_only=True)
    torch.testing.assert_close(left['weight'],right['weight'],rtol=0,atol=1e-7)
    state=torch.load(tmp_path/'retry/training/rft/checkpoint-00001/state.pt',weights_only=False)
    assert state['runtime_microbatch']==2 and state['scheduler']['last_epoch']==1
    assert calls[:2]==[4,4] and calls[2:]==[2]*8


def test_schedule_tamper_rejected_before_model_load(tmp_path,monkeypatch):
    ctx,_=setup(tmp_path,monkeypatch)
    plan=read(tmp_path/'schedule.json');plan['groups'][0][0]=1;write(tmp_path/'schedule.json',plan)
    with pytest.raises(ValueError,match='schedule'):
        worker.train(ctx,'rft')


def test_representative_benchmark_does_not_save_training_checkpoints(tmp_path,monkeypatch):
    ctx,_=setup(tmp_path,monkeypatch)
    ctx['config'].update(benchmark_mode='representative',benchmark_steps=3)
    worker.train(ctx,'rft',benchmark=True)
    assert not (tmp_path/'training/rft/latest.json').exists()
    assert not (tmp_path/'training/rft/milestones').exists()
    timing=read(tmp_path/'timings/bench_rft.json')
    assert len(timing['sample_update_seconds'])==3
    assert timing['updates']==4
