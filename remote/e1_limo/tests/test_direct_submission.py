import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from e1_common import read,write,sha
from direct_submit import validate_training


def completed(tmp_path,monkeypatch):
    import worker
    monkeypatch.setattr(worker,'check_model',lambda p: True)
    source=tmp_path/'suites/e1';training=source/'training_trials/E1'
    cfg=dict(suite_kind='e1_limo_v1')
    write(source/'suite_context.json',dict(config=cfg,code={}))
    write(training/'context.json',dict(config=cfg))
    write(training/'schedule.json',dict(groups=[[0],[1]],actual_tokens=600123))
    write(training/'training/rft/complete.json',dict(complete=True,updates=2,supervised_tokens=600123))
    return source,training


def test_direct_requires_completed_600k(tmp_path,monkeypatch):
    source,training=completed(tmp_path,monkeypatch)
    assert validate_training(source)[2]['supervised_tokens']==600123
    write(training/'training/rft/complete.json',dict(complete=True,updates=1,supervised_tokens=200001))
    with pytest.raises(ValueError,match='not complete'):validate_training(source)


def test_direct_rejects_changed_source_snapshot(tmp_path,monkeypatch):
    source,training=completed(tmp_path,monkeypatch)
    write(source/'code/example.json',dict(value=1))
    ctx=read(source/'suite_context.json');ctx['code']={'example.json':sha(source/'code/example.json')}
    write(source/'suite_context.json',ctx)
    write(source/'code/example.json',dict(value=2))
    with pytest.raises(ValueError,match='Source code changed'):validate_training(source)


def test_dev_does_not_access_unused_confirmation(tmp_path,monkeypatch):
    import accuracy_suite as module
    obj=object.__new__(module.AccuracySuite)
    obj.summary={};obj.root=tmp_path;obj.cfg={};obj.dev=[dict(id='q')]
    obj.save=lambda:None;obj.share_models=lambda root:None
    obj.context=lambda root,cfg:root/'context.json'
    obj.baseline=lambda *a:{}
    obj.phase=lambda *a,**k:write(tmp_path/'evaluations/dev/E1-200k/generations/rft_eval.json',{})
    monkeypatch.setattr(module,'link_dir',lambda *a:None)
    monkeypatch.setattr(module,'candidate_report',lambda *a:dict(metrics=dict(accuracy=.5)))
    result=obj.evaluate('E1-200k',tmp_path/'adapter',{},'dev')
    assert result['status']=='complete'
    assert not hasattr(obj,'confirmation')
