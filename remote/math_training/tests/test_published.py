"""No real dataset bodies: synthetic unit cases and a simulated GPU suite."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import question_id,read,write
from published_data import canonical_tokens,partition_limo,response_tokens,schedule
from published_suite import PublishedSuite
from suite import PhaseFailure
from suite_utils import summarize


class Tokenizer:
    eos_token_id=99
    all_special_ids=[98,99]
    def apply_chat_template(self,*args,**kwargs): return [98,3,4]
    def encode(self,text,**kwargs): return [ord(x) for x in text]


def sample(i,source='gsm8k',length=100):
    return {'id':question_id(str(i)),'dataset':source,
            'chosen_tokens':{'input_ids':[1,2]+[3]*(length-1)+[99],
                             'labels':[-100,-100]+[3]*(length-1)+[99]}}


def test_canonical_preserves_response_and_masks_only_prompt():
    tok=Tokenizer()
    value,reason=canonical_tokens(tok,'question','123',7)
    assert reason is None
    assert value=={'input_ids':[98,3,4,49,50,51,99],'labels':[-100,-100,-100,49,50,51,99]}
    assert canonical_tokens(tok,'question','123',6)==(None,'overlength')
    assert canonical_tokens(tok,'question','c',8)==(None,'embedded_special_or_empty')


def test_token_budget_mixing_exposures_and_milestones():
    rows=[sample(i,length=7) for i in range(100)] + [sample(i+100,'limo',83) for i in range(10)]
    plan=schedule(rows,1800,mixed=True)
    flat=[i for group in plan['groups'] for i in group]
    assert max(Counter(flat).values())<=3
    assert 1800 <= plan['actual_tokens'] < 1800+83
    assert abs(plan['tokens_by_source']['gsm8k']-plan['tokens_by_source']['limo'])<=83
    assert sum(response_tokens(rows[i]) for i in flat)==plan['actual_tokens']
    assert all(len(group)<=16 for group in plan['groups'])
    for label,fraction in [('quarter',.25),('half',.5),('full',1.)]:
        step=plan['milestone_steps'][label]
        assert plan['cumulative_tokens'][step-1]>=1800*fraction
        if step>1: assert plan['cumulative_tokens'][step-2]<1800*fraction
    assert schedule(rows,1800,mixed=True)==plan
    assert schedule(rows,1800,seed=43,mixed=True)['groups']!=plan['groups']
    with pytest.raises(ValueError): schedule(rows,10**7,mixed=True)


def test_limo_split_excludes_prior_training_before_length_filter():
    rows=[{'question':f'Problem {i}','solution':'long','answer':'1'} for i in range(101)]
    rows += [rows[1]]
    split,counts=partition_limo(rows,{question_id('Problem 0')})
    assert counts=={'duplicates':1,'known_overlap':1,'unique':100}
    assert {k:len(v) for k,v in split.items()}=={'dev':5,'test':5,'train':90}
    all_ids=[r['id'] for group in split.values() for r in group]
    assert len(set(all_ids))==100
    assert question_id('Problem 0') not in all_ids
    reversed_split,_=partition_limo(list(reversed(rows)),{question_id('Problem 0')})
    assert split==reversed_split


def output(correct):
    return [{'text':'1','token_ids':[1,99],'finish_reason':'stop',
             'grade':{'correct':correct,'extraction_failed':False,'grader_error':False,'gold_unparseable':False}}]


def fake_suite(tmp_path,monkeypatch,improve=True,fail_training=None):
    args=SimpleNamespace(workspace=tmp_path/'work',run_id='published_test',source_run=None,
        started_at=None,budget_minutes=480,train_python=sys.executable,eval_python=sys.executable,
        dtype='bfloat16',resume=False)
    suite=PublishedSuite(args)
    events=[]
    def initialize():
        suite.dev=[{'id':question_id(f'dev {i}'),'dataset':'gsm8k' if i<100 else 'prm800k',
                    'problem':'test','answer':'1'} for i in range(400)]
        suite.final=[dict(r,id=question_id(f'final {i}')) for i,r in enumerate(suite.dev)]
        suite.gsm=[sample(i,length=100) for i in range(100)]
        suite.limo=[sample(i+100,'limo',600) for i in range(100)]
        suite.summary['data']={'mixed_available':True}
        for kind in ('dev','test'): write(suite.data/f'limo_{kind}.json',[])
        for name in ('stage1','b0'):
            write(suite.shared/f'models/{name}/artifacts.json',{})
        suite.context(suite.shared,suite.cfg)
        write(suite.shared/'eval_rows.json',suite.dev)
        write(suite.shared/'final_rows.json',suite.final)
        suite.base_outputs={r['id']:output(i<240) for i,r in enumerate(suite.dev)}
        suite.summary['baseline']=summarize(suite.base_outputs,suite.dev)
    def phase(context,phase,branch='b0',purpose='eval',count=128,tag=None):
        ctx=read(context);root=Path(ctx['run'])
        events.append((phase,str(root),branch,purpose))
        if phase=='bench':
            write(root/'timings/bench_rft.json',{'seconds_per_update':.01,
                  'updates':len(read(root/'schedule.json')['groups']), 'microbatch':ctx['config']['microbatch']})
        elif phase=='train':
            if fail_training and fail_training in root.name:
                raise PhaseFailure('simulated training failure')
            plan=read(root/'schedule.json')
            for step in plan['milestone_steps'].values():
                write(root/f'training/rft/milestones/step-{step:05d}/artifacts.json',{})
            write(root/'training/rft/final/artifacts.json',{})
        elif phase=='merge':
            write(root/f'models/{branch}/artifacts.json',{})
        elif phase=='generate':
            if purpose=='final':
                assert (suite.root/'selection.json').exists()
            rows=read(root/('final_rows.json' if purpose=='final' else 'eval_rows.json'))[:count]
            correct=280 if branch=='rft' and improve else 240
            write(root/f'generations/{branch}_{purpose}.json',
                  {r['id']:output(i<correct) for i,r in enumerate(rows)})
        return .01
    monkeypatch.setattr(suite,'initialize',initialize)
    monkeypatch.setattr(suite,'phase',phase)
    return suite,events


@pytest.mark.parametrize('improve',[True,False])
def test_full_suite_selection_replication_and_frozen_final(tmp_path,monkeypatch,improve):
    suite,events=fake_suite(tmp_path,monkeypatch,improve)
    suite.run()
    selected=read(suite.root/'selection.json')['candidate']
    assert bool(selected)==improve
    assert len([x for x in events if x[0]=='train'])==4+int(improve)
    assert len([r for r in suite.summary['trials'].values() if r['status']=='complete'])==12+int(improve)
    if improve:
        assert suite.summary['replication']['seed43_dev_gate']
        assert suite.summary['repeat_dev']['passed']
        assert suite.summary['recommendation']=='candidate_improvement_supported'
    else:
        assert suite.summary['recommendation']=='keep_b0'
    # Resume uses frozen selection; never rerun model selection or training after final.
    events.clear()
    suite.run()
    assert not any(e[0]=='train' for e in events)


def test_one_failed_condition_does_not_abort_other_conditions(tmp_path,monkeypatch):
    suite,_=fake_suite(tmp_path,monkeypatch,False,fail_training='A_gsm_nll')
    suite.run()
    assert suite.summary['training']['A_gsm_nll']['status']=='failed'
    assert len([r for r in suite.summary['training'].values() if r['status']=='complete'])==3
    assert suite.summary['recommendation']=='keep_b0'
