"""Synthetic examples only; never load competition questions."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from shared import groups_for_12k,RECIPE,read,write,sha
from run04_recipe import build_recipe,question_hash


def test_recipe_selection_filtering_and_order():
    class Tok:
        eos_token_id=99
        def apply_chat_template(self,messages,**kw):
            return list(range(10)) if messages[0]['content']=='long' else [1,2]
    records=[dict(problem=p,solution='work',answer='2',dataset=d) for d,p in
             [('prm800k','blocked'),('prm800k','long'),('prm800k','p1'),('prm800k','p1'),
              ('prm800k','p2'),('gsm8k','g1'),('gsm8k','g2')]]
    a,t,stats=build_recipe(records,Tok(),[question_hash('blocked')],{'prm800k':2,'gsm8k':2},max_length=4)
    b,_,_=build_recipe(list(reversed(records)),Tok(),[question_hash('blocked')],{'prm800k':2,'gsm8k':2},max_length=4)
    assert a==b and len(a)==4
    assert all(row['input_ids']==[1,2,99] for row in t)
    assert stats['prm800k_over_length_excluded']==1 and stats['prm800k_duplicate_excluded']==1


def test_750_updates_and_no_duplicate_examples():
    groups=groups_for_12k(list(range(12000)))
    assert len(groups)==750 and all(len(g)==16 for g in groups)
    with pytest.raises(ValueError):groups_for_12k([0]*12000)
    assert RECIPE['warmup_updates']==23 and RECIPE['initialization']=='fresh math LoRA'


def test_original_message_builder_unchanged():
    import official_prompt
    official_prompt._prompt_spec={'system_prompt':None,'turns':[{'role':'user','content':'{QUESTION}'}]}
    official_prompt._placeholder='{QUESTION}'
    assert official_prompt._build_messages('synthetic')==[{'role':'user','content':'synthetic'}]
    source=(HERE/'official_prompt.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef))
    import hashlib
    assert hashlib.sha256(ast.get_source_segment(source,node).encode()).hexdigest()==read(HERE/'official_reference.json')['builder_sha256']


def test_submission_custom_model_and_raw_outputs(tmp_path,monkeypatch):
    import submission,torch
    from types import SimpleNamespace
    rows=[dict(question_id=str(i),task='math_comp',subtask='synthetic',turns=['synthetic '+str(i)]) for i in range(2)]
    monkeypatch.setattr(submission.checker,'QUESTION_IDS',frozenset(['0','1']))
    write(tmp_path/'questions.json',rows);write(tmp_path/'question_identity.json',{'sha256':sha(tmp_path/'questions.json')})
    write(tmp_path/'manifest.json',dict(candidate='R12',config=dict(base_model='Qwen/Qwen3-4B-Base',context=8192,max_tokens=2048,seed=42,eval_chunk=1,vllm_memory=.8)))
    model=tmp_path/'custom_model';write(model/'artifacts.json',{})
    calls=[]
    class Tok:
        chat_template=(submission.HERE/'chat_template.jinja').read_text()
        def apply_chat_template(self,messages,tokenize,**kw):
            text=messages[0]['content']
            return [1,2] if tokenize else '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'
    class LLM:
        def __init__(self,**kw):assert kw['model']==str(model);calls.append('init')
        def get_tokenizer(self):return Tok()
        def chat(self,messages,params,**kw):
            assert all(p['n']==1 and p['temperature']==0 for p in params)
            calls.append('chat')
            return [SimpleNamespace(prompt_token_ids=[1,2],outputs=[SimpleNamespace(text='  raw\n',token_ids=[3],finish_reason='stop',stop_reason=None)]) for _ in messages]
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:kw))
    monkeypatch.setattr(submission.importlib.metadata,'version',lambda _: 'synthetic')
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'synthetic')
    args=dict(model_path=model,recipe_detail='fresh rank64 PRM9000 GSM3000',stage1_slug='run01_stage1',run_id='rank12_synthetic',build_messages=lambda q:[{'role':'user','content':q}])
    submission.generate(tmp_path,**args)
    result=read(tmp_path/'submission.json')
    assert result['submission_meta']['run_id']=='rank12_synthetic'
    assert result['submission_meta']['recipe_detail']==args['recipe_detail']
    assert all(p['answer_text']=='  raw\n' for p in result['predictions'])
    assert not submission.checker.check(str(tmp_path/'submission.json')).errors
    n=len(calls);submission.generate(tmp_path,**args);assert len(calls)==n


def load_local(name):
    spec=importlib.util.spec_from_file_location('rank12_test_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_accuracy_report_is_not_repetition_gated(tmp_path):
    report=load_local('report')
    from suite_utils import summarize
    rows=[dict(id=str(i),dataset='gsm8k' if i%2 else 'prm800k') for i in range(4)]
    write(tmp_path/'source.json',{'path':str(tmp_path/'source')})
    for name in ['A','D','R12']:
        out={r['id']:[dict(text=' '.join(['word']*60) if name=='R12' else 'short',
              token_ids=[1,2],finish_reason='length' if name=='R12' else 'stop',
              grade=dict(correct=name=='R12' or r['id']=='0',extraction_failed=False,
                         grader_error=False,gold_unparseable=False))] for r in rows}
        write(tmp_path/f'evaluation/{name}/generations/b0_eval.json',out)
        write(tmp_path/f'evaluation/{name}/metrics.json',summarize(out,rows))
    write(tmp_path/'conditions/R12/complete.json',dict(train_seconds=30,tokens_per_second=100,peak_vram_bytes=1024))
    report.report(tmp_path)
    r=read(tmp_path/'report.json')
    assert r['interpretation']=='higher observed accuracy'
    assert r['comparisons']['D_vs_R12']['wrong_to_right']==3
    assert r['metrics']['R12']['repetition_rate']==1
    assert (tmp_path/'README.md').exists() and len(read(tmp_path/'question_changes.json'))==8


def test_common_training_engine_is_750_updates(monkeypatch):
    import engine
    from shared import configure_engine,NAME
    monkeypatch.setattr(engine,'TOTAL_STEPS',375)
    monkeypatch.setattr(engine,'CONDITIONS',dict(engine.CONDITIONS))
    e=configure_engine()
    assert e is engine and e.TOTAL_STEPS==750
    assert e.CONDITIONS[NAME]==dict(stage1=True,rank=64,alpha=128)
