"""Generate an unedited official submission from completed condition D only."""
import argparse
import os
from common import *

RECIPE_DETAIL = ('run01 Tulu Stage1 reused; fresh math Stage2 rank64 alpha128; '
    'PRM800K 4500 + GSM8K 1500; run03 selection; 1 epoch / 375 updates; '
    'lr5e-5; effective batch16; maxlen2048; NF4 FP16; all-nonpadding loss; seed42')


def prepare(source, dest):
    completed=read(source/'conditions/D/complete.json')
    if completed.get('steps') != 375 or not completed.get('complete'):
        raise ValueError('Condition D must have completed 375 updates')
    adapter=source/'conditions/D/final'
    cfg=read(adapter/'adapter_config.json')
    if cfg.get('r') != 64 or cfg.get('lora_alpha') != 128:
        raise ValueError('Expected condition D rank64 alpha128')
    for model in (adapter, source/'parents/stage1', source/'merged/D'):
        if not verify_model(model):
            raise ValueError('Missing verified D artifact: '+str(model))
    manifest=dict(candidate='D',source_run=str(source),training_recipe=RECIPE_DETAIL,
        adapter_sha256=sha(adapter/'adapter_model.safetensors'),
        adapter_artifacts_sha256=sha(adapter/'artifacts.json'),
        stage1_artifacts_sha256=sha(source/'parents/stage1/artifacts.json'),
        merged_artifacts_sha256=sha(source/'merged/D/artifacts.json'),
        completed_training=completed,backend=read(source/'selected_backend.json'),
        config=dict(base_model=BASE,base_revision=REVISION,context=8192,max_tokens=2048,
                    seed=42,eval_chunk=32,vllm_memory=.8),
        official_assets=read(LEGACY/'submission_assets/manifest.json'))
    immutable_json(dest/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','fetch','generate'])
    p.add_argument('--source-rank',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();source=a.source_rank.resolve();dest=a.root.resolve()
    os.umask(0o077);dest.mkdir(parents=True,exist_ok=True)
    from submission import verify_assets,fetch,generate
    verify_assets()
    if a.phase=='prepare':prepare(source,dest);return
    if read(dest/'manifest.json')['source_run'] != str(source):
        raise ValueError('Source differs from frozen manifest')
    if a.phase=='fetch':fetch(dest);return
    sys.path.insert(0,str(HERE.parent/'rank64_12k'))
    import official_prompt
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt']
    official_prompt._placeholder=official['prompt']['placeholder']
    generate(dest,model_path=source/'merged/D',recipe_detail=RECIPE_DETAIL,
             stage1_slug='run01_stage1',run_id=dest.name,
             build_messages=official_prompt._build_messages)
    from submission_assets import check_submission as checker
    result=checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=result.errors,warnings=result.warnings,violations=result.violations))
    if result.errors or result.violations:raise ValueError('Official format check failed')
    write(dest/'hashes.json',{name:sha(dest/name) for name in
          ['submission.json','manifest.json','generation_profile.json','format_check.json']})


if __name__=='__main__':main()
