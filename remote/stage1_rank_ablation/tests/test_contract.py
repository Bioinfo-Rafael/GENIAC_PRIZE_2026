import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import *
from engine import collate,target_count,train


def test_no_A_training(tmp_path):
    with pytest.raises(ValueError,match='forbidden'): train(tmp_path,'A',{})


def test_only_quantized_candidates():
    assert all(w=='nf4' for b,w,a in backend_candidates(True,True))
    assert CONDITIONS['C']['rank']==32 and CONDITIONS['D']['alpha']==128


def test_sampler_is_one_epoch():
    order=order_indices(6000)
    assert order==order_indices(6000)
    assert len(groups_for(order))==375
    with pytest.raises(ValueError):groups_for(order[:-1])


def test_microbatch_loss_and_gradients():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(42)
    model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.))
    rows=[{'input_ids':[1,2,3,4,5]},{'input_ids':[1,3,4]}]
    model.eval()
    loss=model(**collate(rows,0,'padded','cpu')).loss;loss.backward()
    reference={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad();total=0.
    for r in rows:
        value=model(**collate([r],0,'padded','cpu')).loss*target_count([r])/target_count(rows)
        value.backward();total+=value.item()
    assert total==pytest.approx(loss.item(),abs=1e-6)
    for n,p in model.named_parameters():
        if n in reference:torch.testing.assert_close(p.grad,reference[n],atol=1e-6,rtol=1e-5)


def test_paired_statistics():
    from core import comparison
    a={str(i):{'grade':{'correct':i%2==0}} for i in range(10)}
    b={str(i):{'grade':{'correct':True}} for i in range(10)}
    assert comparison(a,a)['paired_bootstrap_95']==[0,0]
    assert comparison(a,b)['wrong_to_right']==5


def test_reference_reuse_requires_exact_contract(tmp_path):
    from evaluate import reusable_outputs
    rows=[{'id':'one','dataset':'gsm8k'}]
    output={'one':[dict(text='answer',token_ids=[1],finish_reason='stop',grade=dict(correct=True,
        extraction_failed=False,grader_error=False,gold_unparseable=False))]}
    c={'model_sha':'old','config':{'context':4096}}
    write(tmp_path/'contract.json',c);write(tmp_path/'generations/b0_eval.json',output)
    assert reusable_outputs(tmp_path,c,rows)==output
    assert reusable_outputs(tmp_path,dict(c,model_sha='different'),rows) is None
    assert reusable_outputs(tmp_path,dict(c,config={'context':8192}),rows) is None


def test_immutable_resume_contract(tmp_path):
    file=tmp_path/'contract.json'
    immutable_json(file,{'rank':32});immutable_json(file,{'rank':32})
    with pytest.raises(ValueError):immutable_json(file,{'rank':64})


def test_report_exports_all_pairs(tmp_path):
    from report import report
    from suite_utils import summarize
    rows=[{'id':'one','dataset':'gsm8k'}]
    output={'one':[dict(text='answer',token_ids=[1],finish_reason='stop',grade=dict(correct=True,
        extraction_failed=False,grader_error=False,gold_unparseable=False))]}
    for name in CONDITIONS:
        write(tmp_path/'evaluation'/name/'generations/b0_eval.json',output)
        write(tmp_path/'evaluation'/name/'metrics.json',summarize(output,rows))
        if name!='A':write(tmp_path/'conditions'/name/'complete.json',dict(train_seconds=10,
            wall_seconds=11,peak_vram_bytes=1024,tokens_per_second=100))
    report(tmp_path)
    r=read(tmp_path/'report.json')
    assert len(r['comparisons'])==6
    assert r['conditions']['A']['train_seconds'] is None
    assert (tmp_path/'comparison.png').stat().st_size>0
    assert 'D_correct' in (tmp_path/'question_changes.csv').read_text()
