import importlib.util
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from common import bounded_plan,choose,read,write,sha,question_id
from prepare import split_rows,families
from evaluate import strict_grade,metrics
spec=importlib.util.spec_from_file_location('curriculum_run',HERE/'run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
verify_receipt,phase_order=runner.verify_receipt,runner.phase_order
from c_utils import validate_plan
from published_data import response_tokens,canonical_tokens


def pair(i,source,n):
    return dict(id=str(i),dataset=source,chosen_tokens=dict(input_ids=[7]+[3]*n,labels=[-100]+[3]*n))


def test_budget_full_batches_and_token_ratio():
    rows=[pair(i,'prm800k' if i<1000 else 'limo',73+(i%31)) for i in range(1400)]
    plan=bounded_plan(rows,targets=(10000,30000))
    validate_plan(rows,plan)
    flat=[i for g in plan['groups'] for i in g]
    assert len(flat)==len(set(flat))
    assert all(len(g)==16 for g in plan['groups'])
    for target,step in plan['milestone_steps'].items():
        counts={s:sum(response_tokens(rows[i]) for g in plan['groups'][:step] for i in g if rows[i]['dataset']==s)
                for s in ('prm800k','limo')}
        assert .95*int(target)<=sum(counts.values())<=int(target)
        assert counts['prm800k']<=.8*int(target)
        assert counts['limo']<=.2*int(target)
        assert abs(counts['limo']/sum(counts.values())-.2)<.025
    assert plan==bounded_plan(rows,targets=(10000,30000))


def test_insufficient_whole_solutions_fails():
    with pytest.raises(ValueError,match='Insufficient'):
        bounded_plan([pair(0,'prm800k',80000),pair(1,'limo',20000)])


def row(text,source='prm800k',sid=None):
    return dict(id=question_id(text),dataset=source,problem=text,answer='7',solution='published solution',source_id=sid)


def test_transitive_family_and_parameterized_variant():
    rows=[row('Compute the sum of 12 and 31',sid='a'),row('Compute the sum of 13 and 32',sid='b'),
          row('A differently worded derivative',sid='b'),row('Independent theorem')]
    group=families(rows)
    assert group[0]==group[1]==group[2]
    assert group[3]!=group[0]


def test_split_freezes_limo_first_and_removes_all_train_overlap():
    rows=[row('Unique alpha','limo'),row('Unique beta','limo'),row('Unique alpha','prm800k'),
          row('Unique gamma'),row('Unique delta'),row('GSM omega','gsm8k')]
    evaluation,train,stats=split_rows(rows,count=2)
    assert [r['dataset'] for r in evaluation]==['limo','limo','prm800k','prm800k']
    assert {r['id'] for r in evaluation}.isdisjoint(r['id'] for r in train)
    assert train==[rows[-1]]
    assert split_rows(rows,count=2)==(evaluation,train,stats)
    assert all(r['answer']=='7' for r in evaluation)


def test_short_limo_and_prior_dev_exclusion():
    rows=[row('Limo alpha','limo'),row('PRM beta'),row('PRM gamma'),row('Old eval'),row('GSM phi','gsm8k')]
    evaluation,train,stats=split_rows(rows,count=2,excluded=[row('OLD  EVAL')])
    assert stats['evaluation']['limo']['actual']==1
    assert all(r['problem']!='Old eval' for r in evaluation+train)


def test_last_box_only_and_bad_gold_in_denominator():
    import grading
    r=row('question');gold=grading.parse_math('7')
    assert strict_grade(r,r'\boxed{1} then \boxed{7}',gold)['correct']
    assert not strict_grade(r,r'\boxed{7} then \boxed{1}',gold)['correct']
    assert strict_grade(r,'answer: 7',gold)['extraction_failed']
    assert strict_grade(r,r'\boxed{7} then \boxed{',gold)['extraction_failed']
    result=strict_grade(r,r'\boxed{7}',None)
    assert result['gold_unparseable'] and not result['correct']
    output={r['id']:[dict(text=r'\boxed{7}',token_ids=[1]*2048,finish_reason='length',grade=result)]}
    m=metrics(output,[r])
    assert m['n']==1 and m['accuracy']==0 and m['ungradable_count']==1
    assert m['max_tokens_reached_rate']==1
    assert m['by_dataset']['limo']['n']==0


def test_choose_accuracy_only_and_require_both():
    def outputs(values): return {str(i):[dict(grade={'correct':v})] for i,v in enumerate(values)}
    assert choose({'100000':outputs([True,False]),'300000':outputs([True,True])})=='300000'
    assert choose({'100000':outputs([True]),'300000':outputs([True])})=='100000'
    with pytest.raises(ValueError): choose({'100000':outputs([True])})
    with pytest.raises(ValueError): choose({'100000':outputs([True]),'300000':outputs([True,True])})


def test_completed_phase_hash_tamper_detected(tmp_path):
    write(tmp_path/'data.json',{'a':1});receipt={'files':{'data.json':sha(tmp_path/'data.json')}}
    verify_receipt(tmp_path,receipt)
    write(tmp_path/'data.json',{'a':2})
    with pytest.raises(ValueError): verify_receipt(tmp_path,receipt)


def test_training_continues_after_first_evaluation_before_selection():
    order=phase_order()
    assert order.index('eval_100000')<order.index('stage2_300000')<order.index('eval_300000')<order.index('select')
    assert order.index('select')<order.index('submission_fetch')


def test_full_limo_never_truncated_and_prompt_masked():
    class Tok:
        all_special_ids=[0,1];eos_token_id=1
        def apply_chat_template(self,*a,**kw): return [9,9,9]
        def encode(self,*a,**kw): return [5]*10
    tok=Tok()
    value,reason=canonical_tokens(tok,'question','complete solution',14)
    assert reason is None and value['labels']==[-100]*3+[5]*10+[1]
    assert canonical_tokens(tok,'question','complete solution',13)==(None,'overlength')


def test_native_mean_to_sum_matches_masked_nll_gradient():
    import torch
    from losses import sequence_logps
    logits=torch.randn(2,6,13,requires_grad=True)
    labels=torch.tensor([[-100,-100,3,4,5,-100],[-100,-100,-100,6,7,8]])
    native=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,13),labels[:,1:].reshape(-1),ignore_index=-100)
    count=(labels[:,1:]!=-100).sum()
    values,_=sequence_logps(logits,labels)
    assert torch.allclose(-native*count,values.sum(),atol=1e-5)
    a=torch.autograd.grad(native,logits,retain_graph=True)[0]
    b=torch.autograd.grad(-values.sum()/count,logits)[0]
    assert torch.allclose(a,b,atol=1e-6)
    assert torch.count_nonzero(a[:,0])==0


def test_pipeline_failure_resume_and_no_retraining(tmp_path,monkeypatch):
    """Execute real orchestration with fake GPU children; failure cannot select early."""
    import json
    from types import SimpleNamespace
    root=tmp_path/'run';calls=[];fail={'enabled':True}
    def check_output(cmd,**kw):
        if cmd[0]=='nvidia-smi': return '49140\n' if 'memory.total' in cmd[1] else ''
        return 'stable-package==1\n'
    monkeypatch.setattr(runner.subprocess,'check_output',check_output)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*2**30))
    class Process:
        def __init__(self,cmd,stdout,**kw):
            script=Path(cmd[2]).name;self.returncode=0
            stage=cmd[cmd.index('--stage')+1] if '--stage' in cmd else None
            milestone=cmd[cmd.index('--milestone')+1] if '--milestone' in cmd else None
            calls.append((script,stage,milestone,tuple(cmd[5:])))
            def put(name,value={}): write(root/name,value)
            if script=='entry.py':
                for n in ['data/complete.json','data/eval.json','data/report.json','stage1/pairs.json',
                          'stage1/schedule.json','stage2/pairs.json','tokenizer/stage2/tokenizer.json']:put(n)
                put('stage2/schedule.json',{'milestone_steps':{'100000':1,'300000':2}})
            elif script=='gpu.py' and 'train' in cmd:
                put(f'{stage}/training/rft/final/artifacts.json')
                put(f'{stage}/training/rft/milestones/step-00001/artifacts.json')
                put(f'{stage}/training/rft/milestones/step-00002/artifacts.json')
                (root/stage/'training/rft/train.jsonl').write_text('{"step":1}\n')
            elif script=='gpu.py':put('models/'+(milestone or stage)+'/artifacts.json')
            elif script=='evaluate.py' and milestone:
                if fail['enabled'] and milestone=='100000': self.returncode=1;return
                put(f'evaluations/{milestone}/metrics.json');put(f'evaluations/{milestone}/generations/b0_eval.json')
            elif script=='evaluate.py':
                for n in ['selection.json','question_changes.json','report.json']:put(n)
            elif script=='submit.py':
                for n in ['question_identity.json','submission.json','format_check.json','hashes.json','manifest.json','complete.json']:
                    put('submission/'+n)
                put('submission/inference_times.json',[])
        def poll(self): return self.returncode
    monkeypatch.setattr(runner.subprocess,'Popen',Process)
    a=SimpleNamespace(root=root,eval_python='eval-python',data_python='data-python')
    with pytest.raises(RuntimeError,match='eval_100000'):runner.run(a)
    assert not (root/'selection.json').exists()
    before=len(calls);fail['enabled']=False;runner.run(a)
    assert read(root/'status.json')['status']=='complete'
    assert not any(s=='gpu.py' and stage=='stage1' for s,stage,m,args in calls[before:])
    assert sum('train' in args and stage=='stage2' for s,stage,m,args in calls)==2
    before=len(calls);runner.run(a);assert len(calls)==before


def test_grading_timeout_is_recorded_not_dropped(monkeypatch):
    import grading
    from math_verify.errors import TimeoutException
    def timeout(*a,**kw): raise TimeoutException('test')
    monkeypatch.setattr(grading,'parse_math',timeout)
    result=strict_grade(row('q'),r'\boxed{1}',[1])
    assert result['grader_error'] and not result['correct']


def test_curriculum_submission_uses_selected_model_and_raw_outputs(tmp_path,monkeypatch):
    """Full shared generator + curriculum wrapper, with synthetic questions/engine."""
    import torch
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('curriculum_submit',HERE/'submit.py')
    submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)
    import submission
    from common import CONFIG
    root=tmp_path/'run';dest=root/'submission'
    with pytest.raises(FileNotFoundError):submit.run(root,'fetch')
    write(root/'selection.json',{'candidate':'100000'})
    for name in ['models/100000/artifacts.json','models/100000/lineage.json','identity.json','data/complete.json','stage1/training_contract.json','stage2/training_contract.json']:
        write(root/name,{})
    questions=[dict(question_id='0',task='math_comp',subtask='synthetic',turns=['synthetic question'])]
    write(dest/'questions.json',questions);write(dest/'question_identity.json',{'sha256':sha(dest/'questions.json')})
    monkeypatch.setattr(submission.checker,'QUESTION_IDS',frozenset(['0']))
    monkeypatch.setattr(submission.importlib.metadata,'version',lambda _: 'synthetic')
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'synthetic GPU')
    template=(HERE.parent/'math_training/chat_template.jinja').read_text();calls=[]
    class Tok:
        chat_template=template
        def apply_chat_template(self,messages,tokenize,**kw):
            assert messages==[{'role':'user','content':'synthetic question'}]
            return [1,2] if tokenize else '<|im_start|>user\nsynthetic question<|im_end|>\n<|im_start|>assistant\n'
    class LLM:
        def __init__(self,**kw): assert kw['model']==str(root/'models/100000')
        def get_tokenizer(self): return Tok()
        def chat(self,conversations,params,**kw):
            assert params[0]['n']==1 and params[0]['temperature']==0
            calls.append(conversations)
            return [SimpleNamespace(prompt_token_ids=[1,2],outputs=[SimpleNamespace(
                text='  raw synthetic response\n',token_ids=[3,4],finish_reason='stop',stop_reason=None)])]
    original=LLM.chat
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:kw))
    submit.run(root,'generate');submit.run(root,'generate')
    assert len(calls)==1 and LLM.chat is original
    output=read(dest/'submission.json')
    assert output['predictions'][0]['answer_text']=='  raw synthetic response\n'
    assert output['submission_meta']['stage1_slug']=='gsm8k_2000_rank8'
    assert not read(dest/'format_check.json')['errors']
    assert len(read(dest/'inference_times.json'))==1
    assert read(dest/'hashes.json')['files']['submission.json']==sha(dest/'submission.json')
