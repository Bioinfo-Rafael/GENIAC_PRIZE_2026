from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from shared import read,write,sha,digest,file_receipt,verify_files,link_readonly
from followup_plan import build_plan,schedule_for
import followup
import followup_phases


def examples(n):
    return [dict(id=str(i),dataset='limo' if i%5==0 else 'prm800k',chosen_tokens=dict(input_ids=[0,3],labels=[-100,3])) for i in range(n)]


def test_fixed_queue_distinct_branches_and_no_repeated_questions():
    plan=build_plan();full=examples(10000)
    assert len(plan['execution_order'])==len(set(plan['execution_order']))==30
    assert plan['chains'][0]['learning_rate']==2e-6
    assert plan==build_plan()
    for chain in plan['chains']:
        rows,schedule=schedule_for(full,chain)
        flat=[i for g in schedule['groups'] for i in g]
        assert flat==list(range(len(rows)))
        assert [r['id'] for r in rows]==[r['id'] for r in full[chain['start_questions']:chain['targets'][-1]]]
        for e in chain['experiments']:
            step=schedule['milestone_steps'][e['name']]
            assert sum(map(len,schedule['groups'][:step]))+chain['start_questions']==e['target']
        if chain['switch_step'] is not None:assert schedule['milestone_steps']['warmup1040']==65


def instance(tmp_path):
    pred=tmp_path/'predecessor';root=tmp_path/'followup'
    write(pred/'budget.json',dict(started_epoch=followup.time.time(),deadline_epoch=followup.time.time()+36000,max_seconds=36000))
    write(pred/'suite_report.json',dict(status='running'))
    return followup.Followup(SimpleNamespace(root=root,source=pred,eval_python='eval',data_python='data'))


def test_budget_inherited_and_waiting_does_not_launch_gpu(tmp_path,monkeypatch):
    r=instance(tmp_path)
    assert read(r.root/'budget.json')==read(r.predecessor/'budget.json')
    assert r.report['status']=='waiting_predecessor' and len(r.report['experiments'])==30
    monkeypatch.setattr(followup.time,'sleep',lambda _:(_ for _ in ()).throw(InterruptedError('test wait')))
    with pytest.raises(InterruptedError):r.wait_predecessor()
    assert not r.locks
    r2=followup.Followup(r.a)
    assert r2.deadline==r.deadline
    write(r.predecessor/'suite_report.json',dict(status='stopped_resumable'))
    with pytest.raises(RuntimeError,match='Preceding suite stopped'):r.wait_predecessor()


def test_each_submission_gates_evaluation_and_later_training(tmp_path,monkeypatch):
    r=instance(tmp_path);name=r.plan['execution_order'][0];calls=[]
    def phase(n,p,*a):
        calls.append(p)
        if p=='submit':raise RuntimeError('submission failed')
    monkeypatch.setattr(r,'phase',phase)
    with pytest.raises(RuntimeError,match='submission failed'):r.execute_one(name)
    assert calls==['train','merge','fetch','submit']
    calls=[]
    def succeed(n,p,*a):
        calls.append(p)
        if p=='submit':write(r.root/n/'submission/submission.json',{})
    monkeypatch.setattr(r,'phase',succeed)
    r.execute_one(name)
    assert calls==['train','merge','fetch','submit','evaluate','cleanup']
    assert r.report['experiments'][name]['status']=='complete'


def test_only_verified_followup_merge_is_released(tmp_path):
    root=tmp_path/'new';name='candidate';chain=root/'chains/branch';exp=root/name
    model=chain/'models'/name
    write(model/'weight.json',{'weights':'temporary'})
    write(tmp_path/'original_model/keep.json',{'immutable':True})
    write(exp/'checkpoint/state.json',{'optimizer':'saved'})
    write(exp/'submission/submission.json',{'ready':True})
    write(exp/'report.json',{'evaluation':'complete'})
    write(exp/'evaluations/metrics.json',{'correct':1})
    link_readonly(model,exp/'evaluations/models/b0')
    link_readonly(chain/'models',exp/'models')
    write(exp/'experiment.json',dict(chain=str(chain)))
    write(exp/'model_manifest/artifacts.json',{'weight.json':sha(model/'weight.json')})
    for phase,path in [('train',exp/'checkpoint'),('submit',exp/'submission'),('evaluate',exp/'evaluations')]:
        write(root/'receipts'/f'{name}_{phase}.json',dict(files=file_receipt(root,[path])))
    followup_phases.cleanup(root,name)
    assert not model.exists() and (exp/'checkpoint/state.json').exists()
    assert read(tmp_path/'original_model/keep.json')=={'immutable':True}
    verify_files(root,read(root/'receipts'/f'{name}_evaluate.json')['files'])
    followup_phases.cleanup(root,name)  # crash after deletion is resumable


def test_cleanup_refuses_missing_submission(tmp_path):
    root=tmp_path/'new';name='candidate'
    write(root/name/'models'/name/'weight.json',{})
    with pytest.raises(FileNotFoundError):followup_phases.cleanup(root,name)
    assert (root/name/'models'/name/'weight.json').exists()


@pytest.mark.parametrize('lr',[2e-6,1e-6,5e-7])
def test_continuation_uses_frozen_lower_learning_rate(tmp_path,monkeypatch,lr):
    import torch,worker,gpu
    monkeypatch.setitem(sys.modules,'unsloth',SimpleNamespace(FastLanguageModel=SimpleNamespace()))
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace(set_peft_model_state_dict=None,get_peft_model_state_dict=None))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(gpu.signal,'signal',lambda *a:None)
    monkeypatch.setattr(worker,'load_train_model',worker.load_train_model)
    monkeypatch.setattr(worker,'logps',worker.logps)
    write(tmp_path/'continuation.json',dict(initial_adapter='existing',initial_adapter_sha256='frozen'))
    write(tmp_path/'identity.json',{});write(tmp_path/'models/stage1/artifacts.json',{})
    captured=[];monkeypatch.setattr(worker,'train',lambda ctx,*a,**kw:captured.append(ctx))
    gpu.train(tmp_path,'stage2',recipe=dict(learning_rate=lr,lora_rank=8,lora_alpha=16,lr_policy={'fixed':True}))
    assert captured[0]['config']['learning_rate']==lr
    assert captured[0]['config']['initial_adapter']=='existing'


def test_prepare_preserves_full_order_baseline_and_initial_weights(tmp_path):
    r=instance(tmp_path);root=r.root;pred=r.predecessor
    source=tmp_path/'three_thousand';parent=tmp_path/'original'
    write(pred/'suite_report.json',dict(status='complete'))
    full=examples(10000)
    write(pred/'chains/R32/stage2/pairs.json',full)
    write(pred/'data/full_order.json',[x['id'] for x in full])
    write(pred/'data/report.json',{'frozen':True})
    write(pred/'data/complete.json',file_receipt(pred,[pred/n for n in ('chains/R32/stage2/pairs.json','data/full_order.json','data/report.json')]))
    for base in (source,parent):
        adapter=base/'stage2/training/rft/final';adapter.mkdir(parents=True)
        (adapter/'adapter_model.safetensors').write_bytes(b'baseline')
    for n in ('models/stage1','tokenizer','stage1'):(parent/n).mkdir(parents=True)
    write(source/'data/eval.json',[dict(id='e'+str(i),dataset='prm800k' if i<200 else 'limo') for i in range(400)])
    write(pred/'source.json',dict(source=str(source),original=str(parent),absolute_files={}))
    write(root/'identity.json',dict(test=True))
    frozen=file_receipt(tmp_path,[source,parent,pred])
    followup_phases.prepare(root,pred);followup_phases.prepare(root,pred)
    verify_files(tmp_path,frozen)
    for chain in r.plan['chains']:
        work=root/'chains'/chain['key']
        actual=read(work/'stage2/pairs.json')
        assert actual==full[chain['start_questions']:chain['targets'][-1]]
        if chain['start_questions']:
            expected=source if chain['start_questions']==3000 else parent
            assert read(work/'continuation.json')['initial_adapter']==str(expected/'stage2/training/rft/final')
        else:assert not (work/'continuation.json').exists()
