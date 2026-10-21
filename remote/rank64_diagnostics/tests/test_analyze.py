import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('rank_diagnose',Path(__file__).resolve().parents[1]/'analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

class Tok:
    eos_token_id=ord('~')
    def decode(self,ids,**kw):return ''.join(map(chr,ids))
    def __call__(self,text,**kw):return {'input_ids':list(map(ord,text)), 'offset_mapping':[(i,i+1) for i in range(len(text))]}


def test_exact_region_accounting():
    text='<|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\nwork \\boxed{\\frac{1}{2}} tail<|im_end|>\n~'
    counts,flags=a.split_tokens(Tok(),list(map(ord,text)))
    assert sum(counts.values())==len(text)-1
    assert counts['problem']==8 and counts['explanation']==5
    assert counts['final_box']==len('\\boxed{\\frac{1}{2}}')
    assert counts['after_final_box']==5 and counts['eos']==1
    assert flags==dict(has_final_box=True,eos_count=1,tokens_after_first_eos=0)


def test_final_box_is_last_balanced_one():
    text='\\boxed{a} \\boxed{b{c}} \\boxed{unfinished'
    l,r=a.last_box_span(text,0,len(text));assert text[l:r]=='\\boxed{b{c}}'


def test_token_mismatch_refused():
    class Bad(Tok):
        def __call__(self,text,**kw):return dict(input_ids=[],offset_mapping=[])
    with pytest.raises(ValueError,match='round-trip'):a.split_tokens(Bad(),[65])


def test_transition_counts_not_inferred_from_net_change():
    rows=[dict(id=str(i),dataset='gsm8k' if i<2 else 'prm800k',problem='synthetic') for i in range(4)]
    def outputs(correct):return {str(i):dict(grade={'correct':v},text='short',token_ids=[1],finish_reason='stop') for i,v in enumerate(correct)}
    records,s=a.transitions(rows,outputs([True,False,True,False]),outputs([False,True,False,False]))
    c=s['all']['comparison']
    assert c['wrong_to_right']==1 and c['right_to_wrong']==2 and c['delta_accuracy']==-.25
    assert len(records)==4 and s['dataset:gsm8k']['comparison']['delta_accuracy']==0


def test_retention_inventory_no_weight_loading(tmp_path):
    for n in (725,750):
        p=tmp_path/f'conditions/R12/checkpoint-{n:04d}';p.mkdir(parents=True)
        (p/'adapter_model.safetensors').write_bytes(b'not a real model')
        a.write(p/'artifacts.json',{})
    r=a.checkpoint_inventory(tmp_path)
    assert [v['step'] for v in r['available']]==[725,750]
    assert all(v['status']=='missing_no_retraining' for v in r['requested'][:4])


def test_update_weights_and_log_alignment(tmp_path):
    text='<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\nwork \\boxed{1}<|im_end|>\n~'
    rows=[dict(id=str(i),dataset='gsm8k',input_ids=list(map(ord,text))) for i in range(16)]
    a.write(tmp_path/'data/tokens.json',rows);a.write(tmp_path/'data/groups.json',[list(range(16))])
    n=16*(len(text)-1)
    a.write(tmp_path/'conditions/D/updates/0001.json',dict(step=1,supervised_tokens=n,loss=1.25,seconds=1))
    s,_,u,w=a.teacher_audit(tmp_path,'D',Tok(),{str(i) for i in range(16)})
    assert s['all']['targets']==n and sum(s['update_weight_mean'].values())==pytest.approx(1)
    assert w[0]['mean_loss']==1.25 and u[0]['original_examples']==16
    a.write(tmp_path/'conditions/D/updates/0001.json',dict(step=1,supervised_tokens=1))
    with pytest.raises(ValueError,match='mismatch'):a.teacher_audit(tmp_path,'D',Tok(),set())
