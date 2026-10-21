import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import write
from short_diagnostic import reward_summary, sft_summary


def sample(correct):
    return {'text':'1','token_ids':[1],'finish_reason':'stop',
            'grade':dict(correct=correct,extraction_failed=False,grader_error=False,gold_unparseable=False)}


def test_reward_efficiency_uses_committed_rounds(tmp_path):
    write(tmp_path/'training/rft/round_metrics/00001.json',
          dict(round=1,optimizer_updates=1,seconds=10))
    write(tmp_path/'training/rft/round_metrics/00002.json',
          dict(round=2,optimizer_updates=1,seconds=12))
    write(tmp_path/'rollouts/round-00001.json',{'a':[sample(True),sample(False)],
          'b':[sample(True),sample(True)]})
    write(tmp_path/'rollouts/round-00002.json',{'c':[sample(False),sample(False)]})
    x=reward_summary(tmp_path,2)
    assert x['mixed_group_fraction']==1/3
    assert x['optimizer_updates']==1 and x['optimizer_updates_per_round']==.5
    assert x['all_correct']==x['all_incorrect']==x['mixed']==1
    assert reward_summary(tmp_path,3)['optimizer_updates'] is None
    assert reward_summary(tmp_path,3)['missing_metric_rounds']==[3]


def test_sft_comparison_matches_initialization_lr_and_budget(tmp_path):
    trials={}
    for name,n,lr,correct in [('small',6000,1e-5,False),('large',12000,1e-5,True),
                              ('other_lr',12000,5e-6,True)]:
        root=tmp_path/name
        write(root/'generations/rft_eval.json',{'a':[sample(correct)]})
        trials[name]=dict(status='complete',path=str(root),metrics=dict(accuracy=float(correct),correct=int(correct)),
            recipe=dict(kind='sft',examples=n,form='F11',initialization='continue',learning_rate=lr,
                        supervised_tokens=50000))
    r=sft_summary({'trials':trials},[{'id':'a','dataset':'gsm8k'}])
    assert len(r['size_comparisons'])==1
    assert r['size_comparisons'][0]['delta_accuracy']==1
    assert r['size_comparisons'][0]['large']=='large'


def test_stop_refuses_unverified_lock_owner(tmp_path, monkeypatch):
    import pytest
    import short_diagnostic as m
    def locked(*args): raise BlockingIOError
    monkeypatch.setattr(m.fcntl,'flock',locked)
    monkeypatch.setattr(m,'read',lambda _: {'pid':999999999})
    sent=[]
    monkeypatch.setattr(m.os,'kill',lambda *args:sent.append(args))
    with pytest.raises(RuntimeError,match='unverified'):
        m.stop_owner(tmp_path,None)
    assert sent==[]
