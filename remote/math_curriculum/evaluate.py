"""Same frozen questions, greedy one answer, fixed existing last-box extraction."""
import argparse
from common import *


def strict_grade(row,text,gold):
    import grading
    from math_verify import verify
    from math_verify.errors import TimeoutException
    result=dict(correct=False,extraction_failed=True,answer_method='last_boxed',
                grader_error=False,gold_unparseable=gold is None or gold==[])
    candidate=grading.last_boxed(text)
    if not candidate: return result
    try:
        pred=grading.parse_math(candidate)
        result['extraction_failed']=not bool(pred)
        result['correct']=bool(pred) and not result['gold_unparseable'] and bool(
            verify(gold,pred,strict=True,timeout_seconds=3,raise_on_error=True))
    except (Exception,TimeoutException):
        result['grader_error']=True
    return result


def metrics(outputs,rows):
    import grading
    from suite_utils import summarize
    result=summarize(outputs,rows)
    flat=[dict(outputs[r['id']][0]['grade'],finish_reason=outputs[r['id']][0]['finish_reason'],
               completion_tokens=len(outputs[r['id']][0]['token_ids']),dataset=r['dataset']) for r in rows]
    result.update(grading.metrics(flat))
    result['extraction_failed_count']=sum(r['extraction_failed'] for r in flat)
    result['ungradable_count']=sum(r['gold_unparseable'] or r['grader_error'] for r in flat)
    result['max_tokens_reached_rate']=sum(r['completion_tokens']>=2048 for r in flat)/len(flat)
    result['by_dataset']={s:grading.metrics([r for r in flat if r['dataset']==s]) for s in ('prm800k','limo')}
    return result


def evaluate(root,name):
    import importlib.metadata
    import worker
    import grading
    rows=read(root/'data/eval.json');dest=root/'evaluations'/name
    contract=dict(rows_sha256=sha(root/'data/eval.json'),model_sha256=sha(root/'models'/name/'artifacts.json'),
        config=CONFIG,grader_sha256=sha(LEGACY/'grading.py'),extraction_sha256=sha(HERE/'evaluate.py'),
        packages={p:importlib.metadata.version(p) for p in ('vllm','transformers','math-verify')})
    immutable_json(dest/'contract.json',contract)
    immutable_json(dest/'eval_rows.json',rows)
    link=dest/'models/b0';link.parent.mkdir(parents=True,exist_ok=True)
    if not link.exists(): link.symlink_to(root/'models'/name,target_is_directory=True)
    elif link.resolve()!=(root/'models'/name).resolve(): raise ValueError('Evaluation model changed')
    # Local process only: retain authoritative extraction/parsing; forbid marker fallbacks.
    grading.grade=strict_grade
    worker.generate(dict(run=str(dest),config=CONFIG),'b0','eval',len(rows))
    outputs=read(dest/'generations/b0_eval.json')
    immutable_json(dest/'metrics.json',metrics(outputs,rows))


def report(root):
    rows=read(root/'data/eval.json')
    outputs={n:read(root/f'evaluations/{n}/generations/b0_eval.json') for n in ('100000','300000')}
    # Validation must run before selection, including on a resumed run.
    scores={n:metrics(o,rows) for n,o in outputs.items()}
    chosen=choose(outputs)
    changes=[dict(id=r['id'],dataset=r['dataset'],correct_100k=outputs['100000'][r['id']][0]['grade']['correct'],
                  correct_300k=outputs['300000'][r['id']][0]['grade']['correct']) for r in rows]
    result=dict(metrics=scores,comparison=comparison(*[{k:v[0] for k,v in outputs[n].items()}
        for n in ('100000','300000')]),selected=chosen,data=read(root/'data/report.json'),
        limitations='Selection-set accuracy; bootstrap covers question sampling, not training variance or selection bias.')
    immutable_json(root/'selection.json',dict(candidate=chosen,rule='maximum correct; ties prefer 100k',
        evaluation_sha256=sha(root/'data/eval.json'),scores={n:s['correct'] for n,s in scores.items()}))
    immutable_json(root/'question_changes.json',changes)
    # Timings include resumed attempts; final phase metrics are written by the runner.
    write(root/'report.json',result)
    print('SELECTED',chosen,'CORRECT',{n:s['correct'] for n,s in scores.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--milestone');a=p.parse_args()
    if a.milestone: evaluate(a.root,a.milestone)
    else: report(a.root)
