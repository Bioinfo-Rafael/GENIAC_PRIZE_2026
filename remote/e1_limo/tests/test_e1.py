import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e1_common import *
import pytest
from c_utils import validate_plan
from pipeline import E1


def rows():
    return [dict(id=f'{s}-{i}',dataset=s,chosen_tokens=dict(input_ids=[1]*102,
        labels=[-100]*2+[2]*(10+i%11))) for s in WEIGHTS for i in range(100)]


def test_whole_response_mixture_and_only_two_milestones():
    data=rows();plan=plan_for(data,(1000,6000))
    validate_plan(data,plan)
    assert set(plan['milestone_steps'])=={'E1-1k','E1-6k'}
    for total,step in zip((1000,6000),plan['milestone_steps'].values()):
        assert total<=plan['cumulative_tokens'][step-1]<total+20
    for source,weight in WEIGHTS.items():
        assert abs(plan['tokens_by_source'][source]/plan['actual_tokens']-weight)<.015
    assert plan==plan_for(data,(1000,6000))
    assert max(map(int,plan['exposures']))<=3


def test_exhausted_source_does_not_silently_change_mix():
    with pytest.raises(ValueError,match='Insufficient'):
        plan_for(rows(),(1000,600000))


def test_selection_uses_accuracy_only_and_freezes_tie():
    trials={n:dict(status='complete',metrics=dict(correct=250,repetition_rate=r))
            for n,r in [('E1-200k',1),('E1-600k',0)]}
    assert choose(trials)=='E1-200k'
    assert not adopt(dict(correct=250),trials['E1-200k'],dict(comparison=dict(paired_bootstrap_95=[.01,.1])))
    trials['E1-600k']['metrics']['correct']=251
    assert choose(trials)=='E1-600k'
    assert not adopt(dict(correct=250),trials['E1-600k'],dict(comparison=dict(paired_bootstrap_95=[0,.1])))
    assert adopt(dict(correct=250),trials['E1-600k'],dict(comparison=dict(paired_bootstrap_95=[.001,.1])))
    del trials['E1-200k']
    with pytest.raises(ValueError):choose(trials)


def test_reject_alternate_independent_set():
    with pytest.raises(ValueError,match='already frozen'):validate_independent([])


def test_prompt_mask_and_whole_solution():
    from published_data import canonical_tokens
    class Tok:
        all_special_ids=[99];eos_token_id=99
        def apply_chat_template(self,*a,**k):return [1,2,3]
        def encode(self,text,**k):return [5]*len(text)
    tokens,why=canonical_tokens(Tok(),'p','long solution',100)
    assert tokens['labels'][:3]==[-100]*3
    assert tokens['labels'][3:]==[5]*13+[99]
    assert canonical_tokens(Tok(),'p','long solution',10)==(None,'overlength')


def test_pipeline_freezes_before_independent_and_restarts_without_training(tmp_path,monkeypatch):
    # Exercise real run() ordering without GPU/data/network access.
    import pipeline as run
    obj=object.__new__(E1);obj.root=tmp_path;obj.inputs=tmp_path/'inputs'
    obj.summary={'trials':{}};obj.args=type('Args',(),dict(data_python='python',eval_python='python'))()
    obj.save=lambda:write(tmp_path/'report.json',obj.summary)
    obj.dev=[];obj.final=[];events=[]
    obj.initialize=lambda:obj.summary.update(baseline=dict(correct=250))
    def train():
        events.append('train')
        obj.summary['trials']={n:dict(status='complete',adapter=str(tmp_path/n),path=str(tmp_path/n),
            metrics=dict(correct=c),recipe={}) for n,c in [('E1-200k',260),('E1-600k',255)]}
    obj.training=train
    def baseline(split,rows):
        assert read(tmp_path/'selection.json')['candidate']=='E1-200k'
        events.append('independent_A');return {}
    obj.baseline=baseline
    def evaluate(name,*args):
        assert name=='E1-200k';events.append('independent_winner')
        return dict(comparison=dict(paired_bootstrap_95=[-.01,.05]))
    obj.evaluate=evaluate
    monkeypatch.setattr(run,'summarize',lambda *a:dict(correct=250))
    obj.command=lambda cmd,name:events.append(name)
    obj.phase=lambda *a,**k:pytest.fail('Fallback A needs no new merge')
    obj.reports=lambda:None
    obj.run()
    assert events==['train','independent_A','independent_winner','submission_fetch','submission_generate']
    assert read(tmp_path/'adoption.json')['model']=='A'
    # Resume after selection: no second train/search; downstream caches handle generation.
    obj.summary['status']='interrupted';events.clear();obj.run()
    assert 'train' not in events
    assert read(tmp_path/'selection.json')['candidate']=='E1-200k'


def test_evaluation_cache_requires_exact_identity(tmp_path,monkeypatch):
    from suite import Suite
    obj=object.__new__(E1);obj.workspace=tmp_path;obj.code={'worker.py':'abc'}
    obj.packages={'eval':{'vllm':'fixed'}}
    current=tmp_path/'suites/new/baselines/dev'
    prior=tmp_path/'suites/prior/baselines/dev'
    for root in (current,prior):
        write(root/'models/b0/artifacts.json',{'weights':'123'})
    rows=[dict(id='q',dataset='gsm8k')]
    cfg={'seed':42}
    write(current/'eval_rows.json',rows)
    write(current/'context.json',dict(run=str(current),config=cfg))
    identity=dict(config=cfg,rows_sha256=digest(rows),
        model_sha256=sha(current/'models/b0/artifacts.json'),code=obj.code,packages=obj.packages['eval'])
    write(prior/'evaluation_identity.json',identity)
    values={'q':[dict(text='answer',token_ids=[1],finish_reason='stop',
                     grade=dict(correct=True,extraction_failed=False,grader_error=False,gold_unparseable=False))]}
    write(prior/'generations/b0_eval.json',values)
    monkeypatch.setattr(Suite,'phase',lambda *a,**k:'generated')
    assert obj.phase(current/'context.json','generate')==0
    assert read(current/'generations/b0_eval.json')==values
    (current/'generations/b0_eval.json').unlink()
    # Changed model identity must never reuse previous answers.
    (current/'evaluation_identity.json').unlink()
    write(current/'models/b0/artifacts.json',{'weights':'456'})
    assert obj.phase(current/'context.json','generate')=='generated'
