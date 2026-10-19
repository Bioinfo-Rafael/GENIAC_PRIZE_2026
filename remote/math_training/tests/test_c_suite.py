"""Synthetic public data only; real interpolation and simulated GPU orchestration."""
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import pytest
import torch
from safetensors.torch import save_file,load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import read,write,sha,digest
from c_utils import make_plan,reshuffle_plan,validate_plan,category,select
from c_worker import interpolate,gsm_legacy_target
from c_suite import CSuite,PRIOR_CODE
from suite import PhaseFailure,PhaseTimeout
from suite_utils import summarize
from test_published import sample,output


def test_interpolation_exact_endpoints_rank_preserved_and_input_unchanged(tmp_path):
    cfg={'r':8,'lora_alpha':16,'target_modules':['q_proj'],'bias':'none'}
    tensors={'layer.lora_A.weight':torch.arange(32,dtype=torch.float32).reshape(8,4),
             'layer.lora_B.weight':torch.arange(32,dtype=torch.float32).reshape(4,8)}
    for name,offset in [('left',0),('right',8)]:
        write(tmp_path/name/'adapter_config.json',cfg)
        save_file({k:v+offset for k,v in tensors.items()},str(tmp_path/name/'adapter_model.safetensors'))
    before=sha(tmp_path/'right/adapter_model.safetensors')
    for alpha in (0.,.25,1.):
        dst=tmp_path/str(alpha)
        interpolate(tmp_path/'left',tmp_path/'right',dst,alpha)
        for key,value in load_file(str(dst/'adapter_model.safetensors')).items():
            assert torch.equal(value,tensors[key]+8*alpha)
        assert read(dst/'adapter_config.json')['r']==8
        interpolate(tmp_path/'left',tmp_path/'right',dst,alpha)
    assert sha(tmp_path/'right/adapter_model.safetensors')==before
    with pytest.raises(ValueError,match='source changed'):
        interpolate(tmp_path/'left',tmp_path/'right',tmp_path/'0.25',.5)
    write(tmp_path/'right/adapter_config.json',dict(cfg,r=16))
    with pytest.raises(ValueError,match='configuration'):
        interpolate(tmp_path/'left',tmp_path/'right',tmp_path/'bad',.5)


def test_fixed_schedule_seed_changes_order_not_multiset_or_horizon():
    rows=[sample(i,length=20+i) for i in range(32)]
    plan=make_plan(rows,[list(range(16)),list(range(16,32))],milestones={'half':1,'full':2})
    other=reshuffle_plan(rows,plan,43)
    validate_plan(rows,other)
    assert other['groups']!=plan['groups']
    assert Counter(i for g in other['groups'] for i in g)==Counter(i for g in plan['groups'] for i in g)
    assert other['actual_tokens']==plan['actual_tokens']
    assert other['milestone_steps']==plan['milestone_steps']
    with pytest.raises(ValueError,match='Exposure'):
        make_plan(rows,[[0]]*4)


def test_mechanical_gsm_formatting_keeps_derivation_and_structured_final():
    assert gsm_legacy_target('2+3=<<2+3=5>>5.\n#### 5')=='2+3=5.\n\nTherefore, $\\boxed{5}$.'
    with pytest.raises(ValueError): gsm_legacy_target('No structured answer field')


def test_observed_c_quarter_is_tradeoff_half_is_ineligible():
    base=dict(n=400,correct=254,accuracy=.635,repeated=126,repetition_rate=.315,
        non_stop_rate=129/400,correct_and_stop=189,unknown=25,mean_output_tokens=780,
        by_dataset={'gsm8k':{'correct':84}})
    quarter=dict(base,correct=244,accuracy=.61,repeated=62,repetition_rate=.155,
        non_stop_rate=.16,correct_and_stop=221,unknown=34,by_dataset={'gsm8k':{'correct':78}})
    half=dict(quarter,correct=239,accuracy=239/400,repeated=38,repetition_rate=.095,
        non_stop_rate=39/400,correct_and_stop=230,unknown=41,by_dataset={'gsm8k':{'correct':72}})
    assert category(base,quarter)=='termination_tradeoff'
    assert category(base,half) is None
    accurate=dict(quarter,correct=254,accuracy=.635)
    trials={k:{'status':'complete','metrics':v} for k,v in [('quarter',quarter),('half',half),('accurate',accurate)]}
    assert select(base,trials)=='accurate'
    trials['accurate']['selectable']=False
    assert select(base,trials)=='quarter'


def fake_suite(tmp_path,monkeypatch,improve=True,failed=None):
    args=SimpleNamespace(workspace=tmp_path/'work',run_id='c_test',source_run=tmp_path/'pilot',
        source_suite=tmp_path/'prior',started_at=None,budget_minutes=960,
        train_python=sys.executable,eval_python=sys.executable,dtype='bfloat16',resume=False)
    suite=CSuite(args); events=[]
    def results(rows,good):
        values={r['id']:output(i<(280 if good else 240)) for i,r in enumerate(rows)}
        if not good:
            for i,r in enumerate(rows):
                if i<100:
                    values[r['id']][0].update(text=' '.join(['word']*50),finish_reason='length')
        return values
    def initialize():
        suite.dev=[{'id':f'dev{i}','dataset':'gsm8k' if i<100 else 'prm800k'} for i in range(400)]
        suite.final=[dict(r,id=f'final{i}') for i,r in enumerate(suite.dev)]
        rows=[sample(i,length=3000) for i in range(100)]+[sample(i+100,'limo',3000) for i in range(100)]
        plan=make_plan(rows,[list(range(16)),list(range(16,32))],milestones={'quarter':1,'half':1,'full':2})
        suite.old_plan=plan
        suite.old_recipe_root=suite.root/'training_trials/old_C_recipe'
        suite.training_setup(suite.old_recipe_root,rows,plan)
        for form in ('F00','F10','F01','F11'):
            write(suite.form_data/form/'pairs.json',rows);write(suite.form_data/form/'schedule.json',plan)
        for kind in ('dev','test'): write(suite.data/f'limo_{kind}.json',[])
        for name in ('stage1','b0'): write(suite.shared/f'models/{name}/artifacts.json',{})
        suite.context(suite.shared,suite.cfg)
        write(suite.shared/'eval_rows.json',suite.dev);write(suite.shared/'final_rows.json',suite.final)
        suite.base_outputs=results(suite.dev,False)
        suite.summary['baseline']=summarize(suite.base_outputs,suite.dev)
    def phase(context,phase,branch='b0',purpose='eval',count=128,tag=None):
        root=Path(read(context)['run']); events.append((phase,root.name,purpose))
        if phase=='bench':
            cfg=read(context)['config']
            write(root/'timings/bench_rft.json',{'seconds_per_update':.01,'microbatch':cfg['microbatch']})
        elif phase=='merge': write(root/f'models/{branch}/artifacts.json',{})
        elif phase=='audit_weights': write(root/'integrity.json',{'identical':True})
        elif phase=='generate':
            if purpose=='final': assert (suite.root/'selection.json').exists()
            rows=read(root/('final_rows.json' if purpose=='final' else 'eval_rows.json'))[:count]
            write(root/f'generations/{branch}_{purpose}.json',results(rows,improve and branch=='rft'))
        return .01
    def train(root,step,tag):
        events.append(('train',root.name,step))
        if root.name==failed: raise PhaseFailure('synthetic train failure')
        write(suite.adapter(root,step)/'artifacts.json',{})
        return .01
    def interpolation(name,right,alpha):
        p=suite.root/'interpolations'/name;write(p/'artifacts.json',{});return p
    monkeypatch.setattr(suite,'initialize',initialize)
    monkeypatch.setattr(suite,'phase',phase)
    monkeypatch.setattr(suite,'train_until',train)
    monkeypatch.setattr(suite,'interpolate',interpolation)
    return suite,events


@pytest.mark.parametrize('improve',[True,False])
def test_all_stages_confirm_one_frozen_candidate_then_final_and_resume(tmp_path,monkeypatch,improve):
    suite,events=fake_suite(tmp_path,monkeypatch,improve)
    suite.run()
    selected=read(suite.root/'selection.json')['candidate']
    assert bool(selected)==improve
    assert set(suite.summary['training'])=={'F00','F10','F01','F11','C_standard','C_gentle'}
    assert suite.summary['main_budget']['target_tokens']==900_000  # synthetic GSM exposure capacity
    assert suite.summary['final']['fixed_C_half_reference']['status']=='complete'
    if improve:
        assert suite.summary['replication']['seed43_passed']
        assert suite.summary['repeat_dev']['passed']
        assert suite.summary['recommendation']=='accuracy_improvement_supported'
    else: assert suite.summary['recommendation']=='keep_b0'
    events.clear(); suite.run()
    assert not any(e[0]=='train' for e in events)
    assert read(suite.root/'selection.json')['candidate']==selected


def test_failed_main_branch_keeps_other_results(tmp_path,monkeypatch):
    suite,_=fake_suite(tmp_path,monkeypatch,False,failed='C_standard')
    suite.run()
    assert suite.summary['incomplete_training']['C_standard']=='failed'
    assert suite.summary['training']['C_gentle']['status']=='complete'


def test_timed_operation_restores_cutoff_and_charges_failed_work(tmp_path,monkeypatch):
    suite,_=fake_suite(tmp_path,monkeypatch)
    old=suite.cutoff
    def fail(): raise PhaseFailure('synthetic')
    with pytest.raises(PhaseFailure): suite.main_timed('training',18000,fail)
    assert suite.cutoff==old and suite.summary['main_wall_seconds']['training']>=0
    suite.summary['main_wall_seconds']['training']=18000
    with pytest.raises(PhaseTimeout): suite.main_timed('training',18000,fail)


def test_prior_verification_rejects_unknown_builder_before_training(tmp_path,monkeypatch):
    suite,_=fake_suite(tmp_path,monkeypatch)
    write(suite.prior/'suite_context.json',{'code':{},'config':{}})
    with pytest.raises(ValueError,match='Unsupported prior'):
        suite.verify_prior()


def test_prior_identity_checks_completed_training_and_adapter_hashes(tmp_path,monkeypatch):
    import c_suite
    suite,_=fake_suite(tmp_path,monkeypatch)
    code={}
    for name in set(PRIOR_CODE)|{'input_manifest.json','published_sources.json','chat_template.jinja','grading.py'}:
        path=suite.prior/'code'/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('synthetic code contract '+name);code[name]=sha(path)
    monkeypatch.setattr(c_suite,'PRIOR_CODE',{k:code[k] for k in PRIOR_CODE})
    monkeypatch.setattr(suite,'code',code)
    write(suite.prior/'suite_context.json',{'code':code,'config':suite.cfg})
    root=suite.prior/'training_trials/C_gsm_limo_nll'
    write(root/'context.json',{'signature':'synthetic'})
    write(root/'pairs.json',[sample(0)])
    plan={'milestone_steps':{'quarter':1,'half':2}}
    write(root/'schedule.json',plan)
    write(root/'training/rft/complete.json',{'signature':digest(['synthetic',sha(root/'pairs.json'),None,digest(plan)])})
    for step in (1,2):
        adapter=root/f'training/rft/milestones/step-{step:05d}'
        adapter.mkdir(parents=True)
        save_file({'tensor':torch.tensor([step],dtype=torch.float32)},str(adapter/'adapter_model.safetensors'))
        write(adapter/'artifacts.json',{'adapter_model.safetensors':sha(adapter/'adapter_model.safetensors')})
    suite.verify_prior()
    suite.verify_prior()  # Stable immutable manifest on resume.
    write(root/'pairs.json',[sample(1)])
    with pytest.raises(ValueError,match='identity changed'): suite.verify_prior()


def test_format_preparation_matches_ids_order_and_mechanical_targets(tmp_path,monkeypatch):
    import json
    import c_worker,worker
    from core import question_id
    from published_data import canonical_tokens
    class Tok:
        eos_token_id=10000
        all_special_ids=[10000,10001]
        def encode(self,text,**kw): return [ord(c) for c in text]
        def apply_chat_template(self,messages,**kw):
            prefix=[10001,1,2]
            return prefix if len(messages)==1 else prefix+self.encode(messages[1]['content'])+[10001,10]
    tok=Tok()
    raw_gsm=[{'question':'synthetic GSM','answer':'2+3=<<2+3=5>>5.\n#### 5'}]
    raw_limo=[{'question':'synthetic LIMO','solution':'Synthetic derivation ends here.','answer':'5'}]
    fake_pq=SimpleNamespace(read_table=lambda _:SimpleNamespace(to_pylist=lambda:raw_gsm))
    monkeypatch.setitem(sys.modules,'pyarrow',SimpleNamespace(parquet=fake_pq))
    monkeypatch.setitem(sys.modules,'pyarrow.parquet',fake_pq)
    (tmp_path/'limo.jsonl').write_text(json.dumps(raw_limo[0])+'\n')
    monkeypatch.setattr(c_worker,'fetch_public',lambda _:{'gsm8k':'synthetic','limo':tmp_path/'limo.jsonl'})
    monkeypatch.setattr(worker,'tokenizer_for',lambda _:tok)
    rows=[]
    for dataset,raw in [('gsm8k',raw_gsm[0]),('limo',raw_limo[0])]:
        response=raw['answer'] if dataset=='gsm8k' else raw['solution']
        tokens,_=canonical_tokens(tok,raw['question'],response,8192)
        rows.append({'id':question_id(raw['question']),'dataset':dataset,'chosen_tokens':tokens})
        write(tmp_path/f'public/{dataset}_pairs.json',[rows[-1]])
    write(tmp_path/'prior/training_trials/C_gsm_limo_nll/pairs.json',rows)
    write(tmp_path/'prior/training_trials/C_gsm_limo_nll/schedule.json',
          {'groups':[[0],[1]],'milestone_steps':{'half':1,'full':2}})
    c_worker.prepare({'run':str(tmp_path/'formats'),'inputs':str(tmp_path/'inputs'),
        'config':{'c_source_suite':str(tmp_path/'prior'),'c_public_data':str(tmp_path/'public'),'context':8192}})
    plans=[]
    for form in ('F00','F10','F01','F11'):
        processed=read(tmp_path/f'formats/{form}/pairs.json')
        plan=read(tmp_path/f'formats/{form}/schedule.json'); validate_plan(processed,plan)
        assert [r['id'] for r in processed]==[r['id'] for r in rows]
        assert processed[0]['chosen_tokens']['labels'][-1]==tok.eos_token_id
        plans.append(plan['groups'])
    assert plans==[[[0],[1]]]*4
    assert read(tmp_path/'formats/F00/pairs.json')==rows
    assert read(tmp_path/'formats/F10/pairs.json')[1]==rows[1]  # LIMO solution untouched
    assert read(tmp_path/'formats/F01/pairs.json')[0]['chosen_tokens']['labels'][-3:]==[10001,10,10000]
