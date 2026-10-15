import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from core import (digest, question_id, prepare, pairs_from_outputs, classify,
                  tokenize_answer, update_groups, comparison)


def fixtures():
    train = [{'problem':f'train {i}', 'dataset':'prm800k' if i<4500 else 'gsm8k',
              'answer':'1'} for i in range(6000)]
    heldout = [{'problem':f'heldout {i}'} for i in range(800)]
    split = {'dev':heldout[:400], 'final':heldout[400:]}
    return train, {'split':split, 'split_sha256':digest(split)}


def test_sampling_nested_and_ratio():
    train, split = fixtures()
    a, b = prepare(train,split,128,42), prepare(list(reversed(train)),split,512,42)
    assert a == b[:128]
    assert sum(r['dataset']=='gsm8k' for r in b)==128


def test_overlap_and_changed_split_rejected():
    train, split = fixtures()
    train[0]['problem']='heldout 1'
    with pytest.raises(ValueError,match='overlap'): prepare(train,split,128,42)
    split['split']['dev'][0]['problem']='changed'
    with pytest.raises(ValueError,match='hash'): prepare(train,split,128,42)


def output(index, correct=False, finish='stop', unknown=False):
    return {'index':index,'text':str(index),'finish_reason':finish,
            'grade':{'correct':correct,'extraction_failed':unknown,
                     'grader_error':False,'gold_unparseable':False}}


def test_only_verifiable_complete_pairs():
    rows=[{'id':'a','problem':'q','dataset':'gsm8k'}]
    gen={'a':[output(0,True),output(1,finish='length'),output(2,unknown=True)]}
    assert pairs_from_outputs(rows,gen,42)[0]==[]
    gen['a'].append(output(3))
    pairs,stats=pairs_from_outputs(rows,gen,42)
    assert pairs[0]['chosen']=='0' and pairs[0]['rejected']=='3'
    assert stats['outputs_unknown']==1 and stats['outputs_truncated']==1


class Tokenizer:
    eos_token_id=9
    def apply_chat_template(self,messages,tokenize=True,add_generation_prompt=False):
        return [1,2,3] if len(messages)==1 else [1,2,3,4,5,6]


def test_prompt_mask_and_terminal_tokens():
    row=tokenize_answer(Tokenizer(),'q','a',10)
    assert row['labels']==[-100,-100,-100,4,5,6,9]
    assert tokenize_answer(Tokenizer(),'q','a',6) is None


def test_epoch_order_and_comparison():
    groups=update_groups(19,2,16,42)
    assert sorted(groups[0]+groups[1])==list(range(19))
    assert groups==update_groups(19,2,16,42)
    a={'a':output(0),'b':output(1,True)}
    b={'a':output(0,True),'b':output(1,True)}
    assert comparison(a,b,draws=100)['wrong_to_right']==1
    with pytest.raises(ValueError):comparison(a,{'a':b['a']})
