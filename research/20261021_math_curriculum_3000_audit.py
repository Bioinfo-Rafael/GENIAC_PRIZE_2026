"""Read-only submission audit; reuse prior definitions, never read official questions/gold."""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_audit',HERE/'20261021_math_curriculum_submission_audit.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def audit(downloads):
    result={'score_source':'User-reported competition score; not recomputed accuracy','models':{}}
    data={}
    for name,filename,score in [('300k','submission_math_curriculum.json',.19317),
                                ('3000','submission_math_curriculum_3000.json',.20384)]:
        path=downloads/filename;obj=json.loads(path.read_text());data[name]=obj
        rows=obj['predictions'];meta=obj['submission_meta'];check=previous.check_submission.check(str(path))
        assert len({r['question_id'] for r in rows})==len(rows)==232
        result['models'][name]=dict(filename=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            reported_score=score,run_id=meta['run_id'],inference_config=meta['inference_config'],packages=meta['packages'],
            generated_at=meta['generated_at'],chat_template=meta['chat_template'],
            format_check=dict(errors=check.errors,violations=check.violations,warnings=check.warnings),
            overall=previous.stats(rows,meta),by_task={t:previous.stats([r for r in rows if r['task']==t],meta)
                for t in sorted({r['task'] for r in rows})})
    old={r['question_id']:r for r in data['300k']['predictions']}
    new={r['question_id']:r for r in data['3000']['predictions']}
    assert set(old)==set(new)
    result['comparison']=dict(reported_score_delta=.20384-.19317,
        same_inference_config=result['models']['300k']['inference_config']==result['models']['3000']['inference_config'],
        same_packages=result['models']['300k']['packages']==result['models']['3000']['packages'],
        same_chat_template=result['models']['300k']['chat_template']==result['models']['3000']['chat_template'],
        prompt_token_changes=sum(old[k]['prompt_tokens']!=new[k]['prompt_tokens'] for k in old),
        task_changes=sum(old[k]['task']!=new[k]['task'] for k in old),
        same_text=sum(old[k]['answer_text']==new[k]['answer_text'] for k in old),
        newly_repeated=sum(not previous.repeated(old[k]['answer_text']) and previous.repeated(new[k]['answer_text']) for k in old),
        resolved_repetition=sum(previous.repeated(old[k]['answer_text']) and not previous.repeated(new[k]['answer_text']) for k in old))
    result['by_question']=[dict(question_id=k,task=old[k]['task'],
        old=previous.features(old[k],data['300k']['submission_meta']),
        new=previous.features(new[k],data['3000']['submission_meta'])) for k in old]
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,default=Path.home()/'Downloads')
    p.add_argument('--output',type=Path,default=HERE/'20261021_math_curriculum_3000_submission_metrics.json')
    a=p.parse_args();result=audit(a.downloads);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='by_question'},ensure_ascii=False,indent=2))
