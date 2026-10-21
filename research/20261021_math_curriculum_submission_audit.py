"""Read-only audit of submitted outputs; never loads official questions/gold.
Usage: python this_script.py --downloads ~/Downloads --output metrics.json
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'remote/math_training'))
from grading import last_boxed
from submission_assets import check_submission

FILES={'run03':('t4_20261014_run03_prmfull6k.json',.18758),
       'D':('submission_rank64_D.json',.16999),
       'R12':('submission_rank64_12k.json',.12394),
       'E1':('submission_e1_limo_600k.json',.18418),
       'math_curriculum':('submission_math_curriculum.json',.19317)}

def repeated(text):
    words=text.split()
    return any(n>=4 for n in Counter(tuple(words[i:i+10]) for i in range(max(0,len(words)-9))).values())

def features(row,meta):
    config=meta['inference_config'];text=row['answer_text']
    cap=min(config['max_tokens_by_task'].get(row['task'],2048),config['max_model_len']-row['prompt_tokens'])
    return dict(tokens=row['completion_tokens'],cap=cap,cap_reached=row['completion_tokens']>=cap,
        repeated=repeated(text),last_box_missing=last_boxed(text) is None,
        box_openings=len(re.findall(r'\\(?:boxed|fbox)\s*\{',text)),
        answer_sha256=hashlib.sha256(text.encode()).hexdigest())

def stats(rows,meta):
    f=[features(r,meta) for r in rows];lengths=[r['tokens'] for r in f];n=len(rows)
    return dict(n=n,mean_tokens=statistics.mean(lengths),median_tokens=statistics.median(lengths),
        total_tokens=sum(lengths),min_tokens=min(lengths),max_tokens=max(lengths),
        cap_reached=sum(r['cap_reached'] for r in f),repeated=sum(r['repeated'] for r in f),
        cap_and_repeated=sum(r['cap_reached'] and r['repeated'] for r in f),
        last_box_missing=sum(r['last_box_missing'] for r in f),
        cap_and_last_box_missing=sum(r['cap_reached'] and r['last_box_missing'] for r in f),
        multiple_box_openings=sum(r['box_openings']>1 for r in f))

def audit(downloads):
    result=dict(score_source='user-reported competition scores, not recomputed accuracy',
        scope='Submitted output consistency only; no official question/gold access; no per-question correctness labels',
        files={},metrics={},comparisons={},by_question={});data={}
    for name,(filename,score) in FILES.items():
        path=downloads/filename;a=json.loads(path.read_text());data[name]=a
        checked=check_submission.check(str(path))
        result['files'][name]=dict(filename=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reported_score=score,run_id=a['submission_meta']['run_id'],
            inference_config=a['submission_meta']['inference_config'],
            packages=a['submission_meta'].get('packages'),
            format_check=dict(errors=checked.errors,violations=checked.violations,warnings=checked.warnings))
        rows=a['predictions'];meta=a['submission_meta']
        assert len({r['question_id'] for r in rows})==len(rows)
        result['metrics'][name]=dict(overall=stats(rows,meta),by_task={t:stats([r for r in rows if r['task']==t],meta)
            for t in sorted({r['task'] for r in rows})})
        result['by_question'][name]={r['question_id']:dict(index=i,task=r['task'],**features(r,meta)) for i,r in enumerate(rows)}
    new={r['question_id']:r for r in data['math_curriculum']['predictions']}
    for name,a in data.items():
        if name=='math_curriculum':continue
        old={r['question_id']:r for r in a['predictions']};assert set(old)==set(new)
        result['comparisons'][name]=dict(
            reported_score_delta=FILES['math_curriculum'][1]-FILES[name][1],
            prompt_token_changes=sum(old[k]['prompt_tokens']!=new[k]['prompt_tokens'] for k in old),
            task_changes=sum(old[k]['task']!=new[k]['task'] for k in old),
            same_text=sum(old[k]['answer_text']==new[k]['answer_text'] for k in old),
            same_last_box_text=sum(last_boxed(old[k]['answer_text']) is not None and
                last_boxed(old[k]['answer_text'])==last_boxed(new[k]['answer_text']) for k in old),
            newly_repeated=sum(not repeated(old[k]['answer_text']) and repeated(new[k]['answer_text']) for k in old),
            resolved_repetition=sum(repeated(old[k]['answer_text']) and not repeated(new[k]['answer_text']) for k in old))
    # Verify selected operations transcribed from answers, not official gold.
    from fractions import Fraction
    import math
    import sympy as sp
    x=sp.Symbol('x')
    result['answer_internal_checks']={
        '76_gcd':math.gcd(6175,5187,-7163),
        '77_gcd':math.gcd(196,225,256),
        '83_gcd':math.gcd(22176,17556),
        '158_exact_quotient':str(Fraction(43)/(Fraction('1.075')*Fraction('0.8'))),
        '195_factorization_residual':str(sp.expand((5*x-40)*(x+26)-(5*x*x+90*x-1040))),
        '202_derivative_residual':str(sp.trigsimp(sp.diff(sp.tan(5*x+1)-sp.cos(7-8*x),x)
            -(5/sp.cos(5*x+1)**2-8*sp.sin(7-8*x)),method='fu')),
        '207_last_box_minus_previous_line':str(sp.expand(4*x**3*(5*sp.exp(4*x**5-4)-2*sp.sin(2*x**4+8))
            -(20*x**4*sp.exp(4*x**5-4)-8*x**3*sp.sin(2*x**4+8)))),
        '213_derivative_of_output_minus_integrand':str(sp.simplify(sp.diff(-sp.cos(1-7*x),x)-7*sp.sin(1-7*x))),
        '226_determinant':int(sp.Matrix([[-5,-5,5,1],[-3,-3,-4,-1],[-4,1,1,4],[2,2,4,4]]).det()),
        '226_penultimate_arithmetic':-5*10+5*10-5*10+10,
        '227_fraction':str(-Fraction(13,50)-Fraction(1443,100)),
        '228_fraction':str(Fraction(34,15)*Fraction(53,15)-Fraction(2,5)*Fraction(47,15)),
    }
    result['curriculum_provenance']=data['math_curriculum']['submission_meta']['provenance']
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.downloads)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    current=result['by_question']['math_curriculum']
    with a.output.with_suffix('.csv').open('w',newline='') as stream:
        fields=['question_id','index','task','tokens','cap_reached','repeated','last_box_missing',
                'run03_tokens','run03_repeated','run03_cap_reached']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
        for key,row in current.items():
            old=result['by_question']['run03'][key]
            writer.writerow(dict(question_id=key,**{f:row[f] for f in fields[1:7]},
                run03_tokens=old['tokens'],run03_repeated=old['repeated'],run03_cap_reached=old['cap_reached']))
    print(json.dumps({k:result[k] for k in ('metrics','comparisons')},ensure_ascii=False,indent=2))
