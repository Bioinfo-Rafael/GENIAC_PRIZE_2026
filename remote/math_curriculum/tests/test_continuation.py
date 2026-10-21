"""Continuation accounting, immutable parent, adapter loading and orchestration."""
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import continuation as c
from common import read,write,sha,digest,CONFIG
from c_utils import make_plan,validate_plan


def pair(i,source='prm800k',n=10):
    return dict(id=str(i),dataset=source,chosen_tokens=dict(input_ids=[7]+[3]*n,labels=[-100]+[3]*n))


def test_exact_cumulative_count_and_tail_no_repetition():
    rows=[pair(i,'limo' if i%5==0 else 'prm800k',30 if i%5==0 else 10) for i in range(4000)]
    used=list(range(1040))
    added,plan,report=c.extend_plan(rows,used)
    validate_plan(added,plan)
    assert len(added)==1960 and report['cumulative']['questions']==3000
    assert len(plan['groups'])==123 and len(plan['groups'][-1])==8
    assert all(len(g)==16 for g in plan['groups'][:-1])
    assert len({r['id'] for r in added})==1960
    assert not {r['id'] for r in added}.intersection(rows[i]['id'] for i in used)
    assert abs(report['cumulative']['teacher_token_ratio']['limo']-.2)<.005
    assert c.extend_plan(rows,used)==(added,plan,report)


def test_exhausted_limo_uses_prm_and_shortfall_never_repeats():
    rows=[pair(i,'limo' if i==0 else 'prm800k') for i in range(39)]
    added,plan,report=c.extend_plan(rows,list(range(16)),target=40)
    assert len(added)==23 and report['shortfall']==1
    assert report['limo_exhausted'] and report['additional']['questions_by_source']=={'prm800k':23}
    assert len(plan['groups'][-1])==7
    assert report['cumulative']['teacher_tokens']==390
    with pytest.raises(ValueError,match='already'): c.extend_plan(rows,list(range(16)),16)
    with pytest.raises(ValueError,match='Duplicate'): c.extend_plan(rows+[rows[0]],[0])
    with pytest.raises(ValueError,match='PRM/LIMO'): c.extend_plan(rows+[pair(100,'gsm8k')],[0])


def source_fixture(tmp_path):
    source=tmp_path/'source';root=tmp_path/'new'
    rows=[pair(i,'limo' if i%5==0 else 'prm800k') for i in range(80)]
    plan=make_plan(rows,[list(range(16)),list(range(16,32))],42,{'100000':1,'300000':2})
    identity=dict(config={'learning_rate':5e-5},code='original')
    write(source/'stage2/pairs.json',rows);write(source/'stage2/schedule.json',plan)
    write(source/'data/eval.json',[dict(id='held-out',dataset='prm800k')])
    write(source/'stage2/training_contract.json',identity)
    write(source/'stage1/training_contract.json',{})
    write(source/'tokenizer/stage2/tokenizer.json',{})
    write(source/'stage2/training/rft/complete.json',dict(complete=True,updates=2,
        signature=digest([digest(identity),sha(source/'stage2/pairs.json'),None,digest(plan)]),
        supervised_tokens=plan['actual_tokens']))
    for name in ['stage2/training/rft/milestones/step-00002','stage2/training/rft/final','models/stage1',
                 'models/300000','stage1/training/rft/final']:
        dest=source/name;dest.mkdir(parents=True,exist_ok=True)
        (dest/'adapter_model.safetensors').write_bytes(b'fake weights')
        write(dest/'adapter_config.json',dict(r=8,lora_alpha=16))
        write(dest/'artifacts.json',{n:sha(dest/n) for n in ('adapter_model.safetensors','adapter_config.json')})
    lineage=dict(revision=c.REVISION,stage1_sha256=sha(source/'stage1/training/rft/final/adapter_model.safetensors'),
                 stage2_sha256=sha(source/'stage2/training/rft/final/adapter_model.safetensors'))
    write(source/'models/300000/lineage.json',lineage);write(source/'models/stage1/lineage.json',lineage)
    write(source/'data/complete.json',{n:sha(source/n) for n in ('stage2/pairs.json','stage2/schedule.json','data/eval.json')})
    write(root/'continuation_request.json',dict(source=str(source),target=40))
    return source,root


def fingerprint(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}


def test_prepare_uses_exact_parent_and_never_writes_source(tmp_path):
    source,root=source_fixture(tmp_path);before=fingerprint(source)
    c.prepare(root);c.prepare(root)
    assert fingerprint(source)==before
    assert read(root/'data/report.json')['existing']['questions']==32
    assert read(root/'data/report.json')['additional']['questions']==8
    assert (root/'models/stage1').resolve()==source/'models/stage1'
    assert (root/'models/300000').resolve()==source/'models/300000'
    assert read(root/'continuation.json')['initial_adapter'].endswith('step-00002')
    assert not (root/'stage2/training').exists()
    (source/'models/stage1/adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):c.prepare(root)


@pytest.mark.parametrize('kind',['incomplete','wrong_signature','wrong_adapter','overlap'])
def test_prepare_rejects_wrong_source(tmp_path,kind):
    source,root=source_fixture(tmp_path)
    if kind in ('incomplete','wrong_signature'):
        path=source/'stage2/training/rft/complete.json';x=read(path)
        x['updates' if kind=='incomplete' else 'signature']=1 if kind=='incomplete' else 'wrong';write(path,x)
    elif kind=='wrong_adapter':
        (source/'stage2/training/rft/milestones/step-00002/adapter_model.safetensors').write_bytes(b'wrong')
    else:
        write(source/'data/eval.json',[dict(id='0',dataset='prm800k')])
        x=read(source/'data/complete.json');x['data/eval.json']=sha(source/'data/eval.json');write(source/'data/complete.json',x)
    with pytest.raises(ValueError):c.prepare(root)


@pytest.mark.parametrize('resume',[False,True])
def test_gpu_loads_existing_lora_and_new_learning_rate(tmp_path,monkeypatch,resume):
    import torch
    import worker
    import gpu
    from safetensors.torch import save_file
    source=tmp_path/'adapter';source.mkdir()
    save_file({'lora.weight':torch.tensor([2.,3.])},str(source/'adapter_model.safetensors'))
    resumed=tmp_path/'resumed';resumed.mkdir()
    save_file({'lora.weight':torch.tensor([4.,5.])},str(resumed/'adapter_model.safetensors'))
    write(tmp_path/'continuation.json',dict(initial_adapter=str(source),
          initial_adapter_sha256=sha(source/'adapter_model.safetensors')))
    write(tmp_path/'identity.json',{});write(tmp_path/'models/stage1/artifacts.json',{})
    parameter=SimpleNamespace(requires_grad=True,data=torch.tensor([1.]),numel=lambda:16515072)
    model=SimpleNamespace(parameters=lambda:[parameter],config=SimpleNamespace())
    tok=SimpleNamespace(get_vocab=lambda:{'fixed':1},pad_token_id=0,eos_token_id=1)
    class Fast:
        @staticmethod
        def from_pretrained(**kw):
            assert kw['model_name']==str(tmp_path/'models/stage1');return model,tok
        @staticmethod
        def get_peft_model(m,**kw):return m
    loaded=[]
    monkeypatch.setitem(sys.modules,'unsloth',SimpleNamespace(FastLanguageModel=Fast))
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace(
        get_peft_model_state_dict=lambda m:{'lora.weight':torch.zeros(2)},
        set_peft_model_state_dict=lambda m,w:(loaded.append(w['lora.weight'].tolist()) or SimpleNamespace(unexpected_keys=[]))))
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(worker,'tokenizer_for',lambda p:tok)
    monkeypatch.setattr(worker,'load_train_model',worker.load_train_model)
    monkeypatch.setattr(worker,'logps',worker.logps)
    monkeypatch.setattr(gpu.signal,'signal',lambda *a:None)
    def train(ctx,branch,**kw):
        assert ctx['config']['learning_rate']==5e-6 and ctx['config']['effective_batch']==16
        assert not (tmp_path/'stage2/training/rft/latest.json').exists()
        worker.load_train_model(ctx,resumed if resume else None)
    monkeypatch.setattr(worker,'train',train)
    gpu.train(tmp_path,'stage2')
    assert loaded==([[4.,5.]] if resume else [[2.,3.]])
    with pytest.raises(ValueError,match='never retrain'):gpu.train(tmp_path,'stage1')


@pytest.mark.parametrize('compatible',[True,False])
def test_baseline_reuse_requires_identical_contract(tmp_path,monkeypatch,compatible):
    import evaluate
    source=tmp_path/'source';root=tmp_path/'new';old=source/'evaluations/300000'
    write(root/'continuation.json',dict(source=str(source)))
    rows=[dict(id='q',dataset='prm800k')];write(root/'data/eval.json',rows)
    write(root/'models/300000/artifacts.json',{})
    monkeypatch.setattr(c.importlib.metadata,'version',lambda _: 'test')
    expected=dict(rows_sha256=sha(root/'data/eval.json'),model_sha256=sha(root/'models/300000/artifacts.json'),
        config=CONFIG,grader_sha256=sha(c.LEGACY/'grading.py'),extraction_sha256=sha(HERE/'evaluate.py'),
        packages={p:'test' for p in ('vllm','transformers','math-verify')})
    contract=dict(expected)
    if not compatible:contract['extraction_sha256']='other'
    write(old/'contract.json',contract);write(old/'generations/b0_eval.json',{'q':[{}]})
    write(old/'metrics.json',{'n':1});write(old/'eval_rows.json',rows)
    monkeypatch.setattr(evaluate,'metrics',lambda *a:{'n':1})
    calls=[]
    def ev(r,n):
        calls.append(n)
        assert (r/'evaluations/300000/generations/b0_eval.json').exists()==compatible
    monkeypatch.setattr(evaluate,'evaluate',ev)
    before=fingerprint(source);c.evaluate_baseline(root)
    assert calls==['300000'] and fingerprint(source)==before


def test_continuation_pipeline_failure_resume_compare_and_source_unchanged(tmp_path,monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('continuation_runner',HERE/'run.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    source,root=source_fixture(tmp_path)
    # The CLI has a fixed target of 3000; this small pool exercises shortfall reporting.
    write(root/'continuation_request.json',dict(source=str(source),target=3000))
    before=fingerprint(source);calls=[];fail={'value':True}
    monkeypatch.setattr(runner.subprocess,'check_output',lambda cmd,**kw:
        ('49140\n' if 'memory.total' in cmd[1] else '') if cmd[0]=='nvidia-smi' else 'stable-package==1\n')
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    monkeypatch.setattr(runner.signal,'signal',lambda *a:None)
    class Process:
        def __init__(self,cmd,**kw):
            self.returncode=0;script=Path(cmd[2]).name
            milestone=cmd[cmd.index('--milestone')+1] if '--milestone' in cmd else None
            calls.append((script,tuple(cmd[5:])))
            assert '--stage' not in cmd or cmd[cmd.index('--stage')+1]=='stage2'
            if script=='continuation.py' and 'prepare' in cmd:c.prepare(root)
            elif script=='gpu.py' and 'train' in cmd:
                plan=read(root/'stage2/schedule.json')
                write(root/'stage2/training/rft/final/artifacts.json',{})
                write(root/'stage2/training/rft/complete.json',dict(complete=True,updates=len(plan['groups']),supervised_tokens=plan['actual_tokens']))
                (root/'stage2/training/rft/train.jsonl').write_text('{"step":1,"seconds":1.2,"max_vram":100}\n')
            elif script=='gpu.py':write(root/f'models/{milestone}/artifacts.json',{})
            elif (script=='continuation.py' and 'baseline' in cmd) or script=='evaluate.py':
                name=milestone or '300000'
                if name==c.NAME and fail['value']:self.returncode=1;return
                dest=root/'evaluations'/name
                output={'held-out':[dict(text=r'\boxed{1}',token_ids=[1],finish_reason='stop',grade=dict(correct=True,answer_method='last_boxed',
                    extraction_failed=False,grader_error=False,gold_unparseable=False))]}
                import evaluate
                write(dest/'generations/b0_eval.json',output)
                write(dest/'metrics.json',evaluate.metrics(output,read(root/'data/eval.json')))
                write(dest/'contract.json',dict(config=CONFIG,rows_sha256=sha(root/'data/eval.json'),model_sha256=name))
            elif script=='continuation.py':c.report(root)
            elif script=='submit.py':
                for name in ['question_identity.json','submission.json','format_check.json','hashes.json','manifest.json','complete.json']:
                    write(root/'submission'/name,{})
                write(root/'submission/inference_times.json',[])
        def poll(self):return self.returncode
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    a=SimpleNamespace(root=root,continue_from=source,eval_python='eval',data_python='data')
    with pytest.raises(RuntimeError,match='eval_cumulative3000'):runner.run(a)
    assert not (root/'selection.json').exists() and not (root/'submission').exists()
    mark=len(calls);fail['value']=False;runner.run(a)
    assert all(s!='gpu.py' for s,args in calls[mark:])
    assert not any('baseline' in args for s,args in calls[mark:])
    assert read(root/'selection.json')['candidate']==c.NAME
    assert read(root/'report.json')['realized_counts']['questions']==80
    assert read(root/'status.json')['status']=='complete'
    assert list(read(root/'runtime_summary.json')['training_summary'])==['stage2']
    assert (root/'README.md').exists() and fingerprint(source)==before
    mark=len(calls);runner.run(a);assert len(calls)==mark
