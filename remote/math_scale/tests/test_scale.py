import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from shared import count_plan,lr_factor,counts,extend_plan,read,write,sha,verify_files,file_receipt,NAMES
from c_utils import validate_plan
spec=importlib.util.spec_from_file_location('scale_runner',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def rows(n):
    return [dict(id=str(i),dataset='limo' if i%5==0 else 'prm800k',chosen_tokens=dict(input_ids=[0,3],labels=[-100,3])) for i in range(n)]


def test_count_boundaries_and_identical_order():
    pool=rows(11000);used=list(range(3000))
    more,_,report=extend_plan(pool,used,10000)
    full=pool[:3000]+more
    r8=count_plan(more,[('R8_6k',3000),('R8_10k',7000)])
    r32=count_plan(full,[('warmup1040',1040),('history3000',3000),('R32_6k',6000),('R32_10k',10000)])
    validate_plan(more,r8);validate_plan(full,r32)
    assert r8['milestone_steps']=={'R8_6k':188,'R8_10k':438}
    assert r32['milestone_steps']['R32_6k']==376 and r32['milestone_steps']['R32_10k']==626
    assert r32['milestone_steps']['warmup1040']==65
    for plan,rs in [(r8,more),(r32,full)]:
        flat=[i for g in plan['groups'] for i in g]
        assert flat==list(range(len(rs))) and len(set(flat))==len(rs)
    assert full[3000:]==more
    assert len([i for g in r8['groups'][:188] for i in g])==3000
    assert len([i for g in r32['groups'][:376] for i in g])==6000


def test_insufficient_pool_not_repeated():
    more,_,report=extend_plan(rows(9300),list(range(3000)),10000)
    assert len(more)==6300 and report['shortfall']==700
    plan=count_plan(more,[('R8_6k',3000),('R8_10k',6300)])
    assert plan['examples']==6300
    with pytest.raises(ValueError):count_plan(rows(5),[('six',6)])


def test_resume_scheduler_matches_uninterrupted():
    import torch
    for total,switch,lr,boundary in [(438,None,5e-6,188),(626,65,5e-5,376)]:
        def new():
            p=torch.nn.Parameter(torch.tensor([1.]));o=torch.optim.AdamW([p],lr=lr)
            s=torch.optim.lr_scheduler.LambdaLR(o,lambda i:lr_factor(i,total,switch))
            return p,o,s
        p,o,s=new();before=[]
        for i in range(boundary):
            p.grad=torch.ones_like(p);o.step();s.step();before.append(o.param_groups[0]['lr'])
        os,ss=o.state_dict(),s.state_dict()
        q,other,resume=new();other.load_state_dict(os);resume.load_state_dict(ss)
        assert other.param_groups[0]['lr']==o.param_groups[0]['lr']
        for i in range(boundary,total):
            o.step();s.step();other.step();resume.step()
            assert other.param_groups[0]['lr']==o.param_groups[0]['lr']
        assert s.last_epoch==total and resume.last_epoch==total
        assert max(before)<=lr
        if switch:
            assert all(lr*lr_factor(i,total,switch)<=5e-6+1e-15 for i in range(switch,total+1))
        assert lr_factor(total,total,switch)==0


def test_receipt_detects_changes(tmp_path):
    write(tmp_path/'x.json',{'one':1});files=file_receipt(tmp_path,[tmp_path/'x.json'])
    verify_files(tmp_path,files)
    write(tmp_path/'x.json',{'one':2})
    with pytest.raises(ValueError,match='changed'):verify_files(tmp_path,files)


def test_budget_never_renews_on_resume(tmp_path,monkeypatch):
    monkeypatch.setenv('MATH_SCALE_STARTED','1000')
    args=SimpleNamespace(root=tmp_path,source=tmp_path.parent/'source')
    r=runner.Runner(args)
    monkeypatch.setenv('MATH_SCALE_STARTED','9000')
    r2=runner.Runner(args)
    assert r2.deadline==r.deadline==37000
    assert runner.jsonl(tmp_path/'missing.jsonl')==[]


def test_sequential_submission_gate(tmp_path,monkeypatch):
    import scale_prepare
    monkeypatch.setattr(scale_prepare,'prepare',lambda *a:None)
    args=SimpleNamespace(root=tmp_path/'suite',source=tmp_path/'source',eval_python='eval',data_python='data')
    r=runner.Runner(args)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    def output(cmd,**kw):
        if '--query-gpu=name,memory.total' in cmd:return 'NVIDIA RTX 6000 Ada Generation, 49140'
        if '--query-compute-apps=pid' in cmd:return ''
        return 'package==1'
    monkeypatch.setattr(runner.subprocess,'check_output',output)
    monkeypatch.setattr(runner,'estimate',lambda *a:dict(required_seconds=1))
    for name in NAMES:
        write(r.root/name/'experiment.json',{})
    calls=[]
    def phase(name,phase,*args):
        calls.append((name,phase))
        if phase=='submit':raise RuntimeError('synthetic submission failure')
    monkeypatch.setattr(r,'phase',phase)
    with pytest.raises(RuntimeError,match='synthetic'):r.run()
    assert calls==[('suite','prepare'),('R8_6k','train'),('R8_6k','merge'),('R8_6k','fetch'),('R8_6k','submit')]
    assert read(r.root/'suite_report.json')['status']=='stopped_resumable'


def test_low_budget_skips_unstarted_training(tmp_path,monkeypatch):
    import scale_prepare
    monkeypatch.setattr(scale_prepare,'prepare',lambda *a:None)
    args=SimpleNamespace(root=tmp_path/'suite',source=tmp_path/'source',eval_python='eval',data_python='data')
    r=runner.Runner(args)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(runner.subprocess,'check_output',lambda cmd,**kw:'NVIDIA RTX 6000 Ada Generation, 49140' if '--query-gpu=name,memory.total' in cmd else '')
    monkeypatch.setattr(runner,'estimate',lambda *a:dict(required_seconds=99999))
    for name in NAMES:write(r.root/name/'experiment.json',{})
    calls=[];monkeypatch.setattr(r,'phase',lambda name,phase,*a:calls.append((name,phase)))
    r.run()
    assert calls==[('suite','prepare')]
    assert all(x['status']=='skipped_budget' for x in r.report['experiments'].values())


def test_source_reconstruction_and_readonly_parent(tmp_path):
    from shared import digest,make_plan,REVISION,CONFIG
    from scale_prepare import prepare
    source=tmp_path/'source';original=tmp_path/'original';root=tmp_path/'suite'
    pool=rows(10919);first=pool[:1040];second=pool[1040:3000]
    def save_model(path):
        path.mkdir(parents=True,exist_ok=True)
        (path/'adapter_model.safetensors').write_bytes(b'fixture')
        write(path/'adapter_config.json',dict(r=8,lora_alpha=16))
        write(path/'artifacts.json',{n:sha(path/n) for n in ('adapter_config.json','adapter_model.safetensors')})
    evalrows=[dict(id='eval'+str(i),dataset='prm800k' if i<200 else 'limo') for i in range(400)]
    for run,rs,plan in [(original,pool,make_plan(pool,[list(range(i,i+16)) for i in range(0,1040,16)],42,{'300000':65})),
                        (source,second,count_plan(second,[('cumulative3000',1960)]))]:
        write(run/'stage2/pairs.json',rs);write(run/'stage2/schedule.json',plan)
        identity={'fixture':True};write(run/'stage2/training_contract.json',identity)
        write(run/'stage2/training/rft/complete.json',dict(complete=True,updates=len(plan['groups']),supervised_tokens=plan['actual_tokens'],signature=digest([digest(identity),sha(run/'stage2/pairs.json'),None,digest(plan)])))
        write(run/'data/eval.json',evalrows)
        write(run/'data/complete.json',file_receipt(run,[run/n for n in ('stage2/pairs.json','stage2/schedule.json','data/eval.json')]))
    write(source/'continuation.json',dict(source=str(original),files={}))
    write(source/'data/used_stage2_ids.json',[r['id'] for r in first])
    write(source/'data/additional_stage2_ids.json',[r['id'] for r in second])
    write(original/'tokenizer/stage2/tokenizer_config.json',dict(eos_token='<eos>',added_tokens_decoder={'3':{'content':'<eos>'}}))
    write(original/'stage1/training_contract.json',{})
    for p in [source/'stage2/training/rft/final',source/'stage2/training/rft/milestones/step-00123',original/'models/stage1',source/'models/cumulative3000']:save_model(p)
    lineage=dict(revision=REVISION,stage1_sha256='parent',stage2_sha256=sha(source/'stage2/training/rft/final/adapter_model.safetensors'))
    write(original/'models/stage1/lineage.json',lineage);write(source/'models/cumulative3000/lineage.json',lineage)
    write(root/'identity.json',dict(code='fixture',config=CONFIG))
    original_files=file_receipt(tmp_path,[source,original])
    prepare(root,source)
    prepare(root,source)
    verify_files(tmp_path,original_files)
    order=read(root/'data/full_order.json')
    assert len(order)==len(set(order))==10000 and order[:3000]==[r['id'] for r in first+second]
    assert read(root/'R8_6k/experiment.json')['realized']['questions']==6000
    assert (root/'chains/R8/models/stage1').resolve()==original/'models/stage1'
    assert read(root/'chains/R8/continuation.json')['initial_adapter']==str(source/'stage2/training/rft/final')
    assert not (root/'chains/R32/continuation.json').exists()
    (source/'stage2/training/rft/final/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):prepare(root,source)
