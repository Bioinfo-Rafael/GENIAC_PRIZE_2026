"""Official greedy path. No competition text is printed or used by experiments."""
import argparse
import shutil
from common import *
from submission import verify_assets,fetch,generate
from submission_assets import check_submission as checker
sys.path.insert(0,str(HERE.parent/'rank64_12k'))
import official_prompt


def check(path):
    result=checker.check(str(path))
    report=dict(errors=result.errors,violations=result.violations,warnings=result.warnings)
    if result.errors or result.violations:raise ValueError('Official submission check failed')
    return report


def reuse(source,dest,model,adapter_sha):
    """Require model identity, completed-file hash, and official checker success."""
    path=source/'submission.json';done=source/'complete.json';manifest=source/'manifest.json'
    if not all(p.exists() for p in (path,done,manifest)):return False
    m=read(manifest)
    if m.get('adapter_sha256')!=adapter_sha:return False
    if m.get('model_artifacts_sha256')!=sha(model/'artifacts.json'):return False
    if read(done).get('sha256')!=sha(path):return False
    report=check(path)
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(path,dest/'submission.json')
    write(dest/'format_check.json',report)
    write(dest/'manifest.json',dict(reused_from=str(source),source_manifest_sha256=sha(manifest),
        adapter_sha256=adapter_sha,model_artifacts_sha256=sha(model/'artifacts.json'),original_manifest=m))
    write(dest/'complete.json',dict(status='submission_ready',sha256=sha(dest/'submission.json')))
    write(dest/'hashes.json',{n:sha(dest/n) for n in ('submission.json','format_check.json','manifest.json')})
    return True


def run(root,phase,name):
    verify_assets();dest=root/('fallback_submission' if name=='A' else 'candidate_submission')
    dest.mkdir(exist_ok=True)
    if phase=='fetch':fetch(dest);return
    cfg=dict(read(root/'suite_context.json')['config'],context=4096,eval_chunk=58,vllm_memory=.85)
    report=read(root/'suite_report.json')
    model=root/'shared/models/b0' if name=='A' else Path(report['evaluations']['dev'][name]['path'])/'models/rft'
    adapter=Path(report['inputs'])/'stage2' if name=='A' else Path(report['evaluations']['dev'][name]['adapter'])
    official=read(LEGACY/'submission_assets/inference_config.json')
    official_prompt._prompt_spec=official['prompt'];official_prompt._placeholder=official['prompt']['placeholder']
    immutable_json(dest/'manifest.json',dict(candidate=name,config=cfg,config_sha256=digest(cfg),adapter_sha256=sha(adapter/'adapter_model.safetensors'),
        model_artifacts_sha256=sha(model/'artifacts.json'),stage1_artifacts_sha256=sha(root/'shared/models/stage1/artifacts.json'),
        selection_sha256=sha(root/'selection.json') if (root/'selection.json').exists() else None,
        code=read(root/'suite_context.json')['code'],e2_e3_code=cfg['experiment_code'],
        official_assets=read(LEGACY/'submission_assets/manifest.json')))
    generate(dest,model_path=model,stage1_slug='run01_stage1',run_id=root.name,
        recipe_detail='run03 unchanged' if name=='A' else 'run03 independent '+name+'; '+str(report['recipes'][name]),
        build_messages=official_prompt._build_messages)
    write(dest/'format_check.json',check(dest/'submission.json'))
    write(dest/'hashes.json',{n:sha(dest/n) for n in ('submission.json','manifest.json','format_check.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fetch','generate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--name',required=True);a=p.parse_args();run(a.root,a.phase,a.name)
