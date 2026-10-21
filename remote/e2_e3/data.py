"""Only pinned GSM8K train is needed; PRM/LIMO are permitted but unused here."""
import argparse
from collections import Counter
import re
from common import *


def prepare(ctx,independent):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import grading
    sys.path.insert(0,str(HERE.parent/'public_eval_diagnostics'))
    from data import DuplicateIndex
    root,inputs=Path(ctx['run']),Path(ctx['inputs'])
    ind=read(independent)
    if digest(ind)!=INDEPENDENT_SHA:raise ValueError('Independent400 changed')
    split=read(inputs/'split.json')
    if digest(split['split'])!=split['split_sha256']:raise ValueError('Split changed')
    excluded=DuplicateIndex(split['split']['dev']+split['split']['final']+ind)
    source=read(LEGACY/'published_sources.json')['gsm8k']
    path=Path(hf_hub_download(repo_id=source['repo_id'],repo_type='dataset',
        revision=source['revision'],filename=source['filename']))
    if sha(path)!=source['sha256']:raise ValueError('GSM8K checksum')
    raw=pq.read_table(path).to_pylist();raw.sort(key=lambda r:digest([42,r['question']]))
    online=[];items=[];filters=Counter();seen=DuplicateIndex([])
    for r in raw:
        row=dict(id=question_id(r['question']),dataset='gsm8k',problem=r['question'],
                 answer=r['answer'].split('####')[-1].strip())
        reason=excluded.reason(row) or seen.reason(row)
        if reason:filters['duplicate_'+reason]+=1;continue
        if grading.numeric_answer(row['answer']) is None:filters['invalid_gold']+=1;continue
        seen.add(row);online.append(row)
        # One equation per original question, selected mechanically in source order.
        for lhs,rhs in re.findall(r'<<([^<>==]+)=([^<>==]+)>>',r['answer']):
            try:
                if arithmetic(lhs)!=arithmetic(rhs):filters['source_equation_wrong']+=1;continue
            except (ValueError,SyntaxError,ZeroDivisionError):filters['unsupported_expression']+=1;continue
            form=('json','equation','boxed')[len(items)%3]
            instruction={'json':'Represent this calculation as JSON with exactly two string fields: expression (copy the expression exactly) and value (its numerical result).',
                'equation':'Complete the calculation as a single equation: copy the expression, then =, then the numerical result.',
                'boxed':'Complete the missing right-hand side of this calculation. Return only the numerical result inside \\boxed{...}.'}[form]
            problem=instruction+'\n'+lhs.strip()+(' = ?' if form=='boxed' else '')
            items.append(dict(id=digest([row['id'],form,lhs]),source_id=row['id'],dataset='gsm8k',
                lhs=lhs.strip(),rhs=rhs.strip(),form=form,problem=problem,source_solution_sha256=digest(r['answer'])))
            break
    # Precommit pool/order; never inspect model quality manually.
    immutable_json(root/'online_rows.json',online[:512])
    immutable_json(root/'synthesis_pool.json',items[:900])
    immutable_json(root/'sources.json',dict(gsm8k=source,filters=dict(filters),eligible=len(items),
        online_count=min(512,len(online)),heldout_sha256=digest(split['split']['dev']+split['split']['final']+ind)))


def synthesize(ctx,deadline):
    import time
    from vllm import LLM,SamplingParams
    from published_data import canonical_tokens
    root=Path(ctx['run']);cfg=ctx['config']
    if (root/'synthesis_complete.json').exists():return
    state_path=root/'synthesis_state.json'
    state=read(state_path) if state_path.exists() else dict(cursor=0,accepted=[],rejected={},generated=0)
    pool=read(root/'synthesis_pool.json')
    model=Path(ctx['baseline_model'])
    from worker import check_model
    if not check_model(model):raise ValueError('Invalid run03 generator')
    llm=LLM(model=str(model),dtype='half',max_model_len=4096,gpu_memory_utilization=.8,
            max_num_seqs=16,seed=42,trust_remote_code=False)
    tok=llm.get_tokenizer();timings=[]
    while state['cursor']<len(pool) and len(state['accepted'])<300:
        # The first small batch is the speed probe, included in the useful pool.
        estimate=max(timings,default=15)*1.5
        if time.time()+estimate+20>=deadline:break
        batch=pool[state['cursor']:state['cursor']+min(16,300-len(state['accepted']))]
        prompts=[tok.apply_chat_template([{'role':'user','content':r['problem']}],
                    tokenize=True,add_generation_prompt=True) for r in batch]
        started=time.monotonic()
        outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],
                             SamplingParams(n=1,temperature=0,max_tokens=192,seed=42),use_tqdm=False)
        for item,out in zip(batch,outputs):
            text=out.outputs[0].text;reason=validate_generated(item,text)
            tokens=None
            if reason is None:tokens,reason=canonical_tokens(tok,item['problem'],text,cfg['context'])
            if reason:state['rejected'][reason]=state['rejected'].get(reason,0)+1
            else:state['accepted'].append(dict(item,chosen_tokens=tokens,generated_response=text))
            state['generated']+=1;state['cursor']+=1
        timings.append(time.monotonic()-started)
        write(state_path,state)
        write(root/'synthesis_timing.json',dict(batch_seconds=timings,remaining_seconds=max(0,deadline-time.time())))
        print('SYNTHESIS',state['generated'],len(state['accepted']),flush=True)
    immutable_json(root/'pairs.json',state['accepted'])
    immutable_json(root/'synthesis_complete.json',dict(generated=state['generated'],accepted=len(state['accepted']),
        rejected=state['rejected'],generator='run03',model_sha256=sha(model/'artifacts.json'),
        target=300,minimum=32,verification='exact rational arithmetic; strict output schema; source equation identity'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','synthesize'])
    p.add_argument('--context',type=Path,required=True);p.add_argument('--independent',type=Path)
    p.add_argument('--deadline',type=float);a=p.parse_args()
    if a.phase=='prepare':prepare(read(a.context),a.independent)
    else:synthesize(read(a.context),a.deadline)
