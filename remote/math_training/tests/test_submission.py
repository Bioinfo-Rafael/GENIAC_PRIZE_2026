"""Synthetic inference only: no competition questions or answers loaded."""
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import submission
from core import read,write,sha


def test_official_assets_intact_and_question_ids_complete():
    submission.verify_assets()
    assert len(submission.checker.QUESTION_IDS)==232


def test_prediction_preserves_raw_text_and_rejects_multiple_outputs():
    q={'question_id':'a','task':'math_comp','subtask':'synthetic'}
    out=SimpleNamespace(prompt_token_ids=[1,2],outputs=[SimpleNamespace(
        text='  answer\n\n',token_ids=[3,4],finish_reason='length',stop_reason=None)])
    p,r=submission.prediction(q,out)
    assert p['answer_text']=='  answer\n\n' and p['answer_text']==r['text']
    partial={'signature':'fixed','predictions':[p],'raw_outputs':[r]}
    submission.validate_partial(partial,'fixed',[q])
    p['answer_text']='edited'
    with pytest.raises(ValueError,match='recorded model'):
        submission.validate_partial(partial,'fixed',[q])
    with pytest.raises(ValueError,match='signature'):
        submission.validate_partial(partial,'changed',[q])
    out.outputs*=2
    with pytest.raises(ValueError,match='exactly one'): submission.prediction(q,out)


def test_generation_resume_checks_ids_and_exports_official_schema(tmp_path,monkeypatch):
    import torch
    import worker
    rows=[{'question_id':str(i),'task':'math_comp','subtask':'synthetic','turns':['synthetic '+str(i)]} for i in range(5)]
    monkeypatch.setattr(submission.checker,'QUESTION_IDS',frozenset(str(i) for i in range(5)))
    cfg={'base_model':'Qwen/Qwen3-4B-Base','context':8192,'max_tokens':2048,'seed':42,'eval_chunk':2,'vllm_memory':.8}
    write(tmp_path/'manifest.json',{'candidate':'test','config':cfg})
    write(tmp_path/'questions.json',rows)
    write(tmp_path/'question_identity.json',{'sha256':sha(tmp_path/'questions.json')})
    write(tmp_path/'models/rft/artifacts.json',{})
    template=(submission.HERE/'chat_template.jinja').read_text()
    class Tok:
        chat_template=template
        def apply_chat_template(self,messages,tokenize,**kwargs):
            text=messages[0]['content']
            return [1,int(text[-1])+10,2] if tokenize else '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'
    calls=[];fail=[True]
    class LLM:
        def __init__(self,**kw): pass
        def get_tokenizer(self): return Tok()
        def generate(self,prompts,params,**kw):
            calls.append([p['prompt_token_ids'][1]-10 for p in prompts])
            if len(calls)==2 and fail[0]: raise RuntimeError('synthetic interruption')
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(
                text='  raw answer '+str(p['prompt_token_ids'][1])+'\n',token_ids=[3,4,5],
                finish_reason='stop',stop_reason=None)]) for p in prompts]
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(LLM=LLM,SamplingParams=lambda **kw:kw))
    monkeypatch.setattr(submission.importlib.metadata,'version',lambda _: 'synthetic')
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _: 'synthetic GPU')
    with pytest.raises(RuntimeError,match='interruption'): submission.generate(tmp_path)
    assert len(read(tmp_path/'partial.json')['predictions'])==2
    fail[0]=False; calls.clear(); submission.generate(tmp_path)
    assert calls==[[2,3],[4]]
    result=read(tmp_path/'submission.json')
    assert len(result['predictions'])==5
    assert all(p['answer_text'].startswith('  raw') and p['answer_text'].endswith('\n') for p in result['predictions'])
    assert not submission.checker.check(str(tmp_path/'submission.json')).errors
    assert read(tmp_path/'complete.json')['sha256']==sha(tmp_path/'submission.json')
    assert 'correct' not in result['predictions'][0]
    rows[1]['question_id']='0'
    with pytest.raises(ValueError,match='IDs/count'): submission.validate_questions(rows)
