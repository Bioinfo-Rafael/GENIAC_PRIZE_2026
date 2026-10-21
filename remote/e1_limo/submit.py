"""Reuse the distributed prompt builder and unchanged official format checker."""
import argparse
from e1_common import *
from submission import verify_assets,fetch,generate
sys.path.insert(0,str(HERE.parent/'rank64_12k'))
import official_prompt


def run(root,phase):
    verify_assets();dest=root/'submission';dest.mkdir(exist_ok=True)
    decision=read(root/'adoption.json')  # Required before accessing competition inputs.
    if phase=='fetch':fetch(dest);return
    cfg=read(root/'suite_context.json')['config'];name=decision['model']
    report=read(root/'report.json')
    model=root/'shared/models/b0' if name=='A' else Path(report['trials'][name]['path'])/'models/rft'
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt'];official_prompt._placeholder=official['prompt']['placeholder']
    manifest=dict(candidate=name,source_suite=str(root),config=cfg,config_sha256=digest(cfg),
        adapter_sha256=sha(Path(decision['adapter'])/'adapter_model.safetensors'),
        model_artifacts_sha256=sha(model/'artifacts.json'),
        stage1_artifacts_sha256=sha(root/'shared/models/stage1/artifacts.json'),
        adoption_sha256=sha(root/'adoption.json'),selection_sha256=sha(root/'selection.json'),
        data_identity=read(root/'prepared_identity.json'),code=read(root/'suite_context.json')['code'],
        e1_code=cfg['e1_code'],official_assets=read(LEGACY/'submission_assets/manifest.json'),
        official_prompt_sha256=sha(HERE.parent/'rank64_12k/official_prompt.py'))
    immutable_json(dest/'manifest.json',manifest)
    generate(dest,model_path=model,stage1_slug='run01_stage1',run_id=root.name,
        recipe_detail=('Original run03 Stage1 + rank8 math SFT; unchanged baseline' if name=='A' else
            'run03 rank8 continued SFT; unedited PRM/GSM/LIMO solutions; teacher tokens 60/20/20; '
            'NLL response only; initial lr5e-6 cosine; BF16 NF4; maxlen8192; seed42; '+name),
        build_messages=official_prompt._build_messages)
    from submission_assets import check_submission as checker
    result=checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=result.errors,violations=result.violations,warnings=result.warnings))
    if result.errors or result.violations:raise ValueError('Official format check failed')
    write(dest/'hashes.json',{n:sha(dest/n) for n in ('submission.json','manifest.json','format_check.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch','generate'])
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root,a.phase)
