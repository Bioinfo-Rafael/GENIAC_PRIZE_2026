"""Run only on the authorized remote. Questions and raw answers stay local."""
import argparse
import os
from shared import *


def run(root,phase):
    from submission import verify_assets,fetch,generate
    verify_assets()
    dest=root/'submission';dest.mkdir(parents=True,exist_ok=True)
    if phase=='fetch':fetch(dest);return
    import official_prompt
    from submission_assets import check_submission as checker
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt']
    official_prompt._placeholder=official['prompt']['placeholder']
    cfg=dict(base_model=BASE,base_revision=REVISION,context=8192,max_tokens=2048,seed=42,eval_chunk=32,vllm_memory=.8)
    manifest=dict(candidate=NAME,source_run=str(root),training_recipe=RECIPE,
        adapter_sha256=sha(root/f'conditions/{NAME}/final/adapter_model.safetensors'),
        stage1_artifacts_sha256=sha(root/'parents/stage1/artifacts.json'),
        merged_artifacts_sha256=sha(root/f'merged/{NAME}/artifacts.json'),
        data_report_sha256=sha(root/'data/report.json'),backend=read(root/'selected_backend.json'),
        experiment_identity=read(root/'identity.json'),config=cfg,
        official_reference=read(HERE/'official_reference.json'),
        official_assets=read(LEGACY/'submission_assets/manifest.json'))
    immutable_json(dest/'manifest.json',manifest)
    if (dest/'complete.json').exists():
        if sha(dest/'submission.json')!=read(dest/'complete.json')['sha256']:raise ValueError('Submission hash changed')
    else:
        generate(dest,model_path=root/f'merged/{NAME}',stage1_slug='run01_stage1',run_id=root.name,
            recipe_detail='run01 Tulu Stage1 reused; fresh math Stage2 rank64 alpha128; PRM800K 9000 + GSM8K 3000; run04 selection; 1 epoch / 750 updates; lr5e-5; effective batch16; maxlen2048; NF4 FP16; all-nonpadding loss; seed42',
            build_messages=official_prompt._build_messages)
    # Official checker, unchanged; never print its messages or prediction contents.
    check=checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=check.errors,violations=check.violations,warnings=check.warnings))
    if check.errors or check.violations:raise ValueError('Official format check failed')
    write(dest/'hashes.json',{str(p.relative_to(dest)):sha(p) for p in [dest/'submission.json',dest/'manifest.json',dest/'format_check.json']})

if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch','generate'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root,a.phase)
