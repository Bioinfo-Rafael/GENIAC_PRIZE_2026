"""Only the frozen winner accesses official inputs; outputs are never edited."""
import argparse
import time
from common import *
from submission import verify_assets, fetch, generate
sys.path.insert(0,str(HERE.parent/'rank64_12k'))
import official_prompt


def run(root,phase):
    verify_assets();decision=read(root/'selection.json');name=decision['candidate']
    if name not in ('100000','300000'): raise ValueError('Unknown selected checkpoint')
    dest=root/'submission';dest.mkdir(exist_ok=True)
    if phase=='fetch': fetch(dest);return
    from worker import check_model
    model=root/'models'/name
    if not check_model(model): raise ValueError('Selected model changed')
    manifest=dict(candidate=name,config=CONFIG,selection_sha256=sha(root/'selection.json'),
        model_artifacts=read(model/'artifacts.json'),lineage=read(model/'lineage.json'),
        data_sha256=sha(root/'data/complete.json'),code=read(root/'identity.json'),
        training_config_sha256={s:sha(root/s/'training_contract.json') for s in ('stage1','stage2')},
        inference_config_sha256=digest(CONFIG),
        official_assets=read(LEGACY/'submission_assets/manifest.json'))
    immutable_json(dest/'manifest.json',manifest)
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt'];official_prompt._placeholder=official['prompt']['placeholder']
    # Instrument only duration/length; the shared official generator owns all outputs.
    from vllm import LLM
    original_chat=LLM.chat
    timing_path=dest/'inference_times.json'
    timings=read(timing_path) if timing_path.exists() else []
    def timed_chat(self,*args,**kwargs):
        started=time.monotonic()
        result=original_chat(self,*args,**kwargs)
        timings.append(dict(seconds=time.monotonic()-started,questions=len(result),
            completion_tokens=sum(len(o.token_ids) for r in result for o in r.outputs)))
        write(timing_path,timings)
        return result
    LLM.chat=timed_chat
    try:
        generate(dest,model_path=model,stage1_slug='gsm8k_2000_rank8',run_id=root.name,
            recipe_detail='Qwen3-4B-Base pinned revision; new GSM8K Stage1 2000 examples, rank8 alpha16; '
                'merged then fresh rank8 alpha16 PRM/LIMO response NLL, teacher-token target 80/20; '+name,
            build_messages=official_prompt._build_messages)
    finally:
        LLM.chat=original_chat
    from submission_assets import check_submission as checker
    checked=checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=checked.errors,violations=checked.violations,warnings=checked.warnings))
    if checked.errors or checked.violations: raise ValueError('Official format check failed')
    write(dest/'hashes.json',dict(files={n:sha(dest/n) for n in ('submission.json','format_check.json','manifest.json')},
        model_tokenizer_config=read(model/'artifacts.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch','generate'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root,a.phase)
