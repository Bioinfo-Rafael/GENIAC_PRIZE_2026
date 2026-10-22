"""Isolated training, merge and official-inference processes."""
import argparse
import shutil
import sys
from helpers import *


def submit(root,name):
    from submission import verify_assets,generate
    from submission_assets import check_submission as checker
    sys.path.insert(0,str(HERE.parent/'rank64_12k'))
    import official_prompt
    verify_assets()
    exp = root/name
    dest = exp/'submission'; dest.mkdir(exist_ok=True)
    # Questions were preflighted in the isolated fetch process, not read by train.
    for file in ('questions.json','question_identity.json'):
        target = dest/file
        if not target.exists(): shutil.copyfile(root/'official_inputs'/file,target)
    model = exp/'models'/name
    manifest = dict(candidate=name,config=CONFIG,experiment=read(exp/'experiment.json'),
        training=read(exp/'training_complete.json'),model_artifacts=read(model/'artifacts.json'),
        lineage=read(model/'lineage.json'),code=read(root/'identity.json'),
        source_sha256=sha(root/'source.json'),data_sha256=sha(root/'data/complete.json'))
    immutable_json(dest/'manifest.json',manifest)
    official = read(HERE.parent/'math_training/submission_assets/inference_config.json')
    official_prompt._prompt_spec = official['prompt']
    official_prompt._placeholder = official['prompt']['placeholder']
    generate(dest,model_path=model,stage1_slug='gsm8k_2000_rank8',run_id=root.name+'_'+name,
        recipe_detail='OSW pilot: existing GSM Stage1 and cumulative3k rank8; weighted original PRM/LIMO response+EOS NLL; two training-only current-policy samples; '+name,
        build_messages=official_prompt._build_messages)
    result = checker.check(str(dest/'submission.json'))
    write(dest/'format_check.json',dict(errors=result.errors,violations=result.violations,warnings=result.warnings))
    ids = [r['question_id'] for r in read(dest/'submission.json')['predictions']]
    if result.errors or result.violations or len(ids)!=232 or set(ids)!=checker.QUESTION_IDS:
        raise ValueError('Official submission validation failed')
    write(dest/'hashes.json',file_receipt(dest,[dest/f for f in (
        'submission.json','format_check.json','manifest.json','generation_profile.json')]))
    print('SUBMISSION_READY',dest/'submission.json',flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase',choices=['prepare','fetch','train','merge','submit','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    a = p.parse_args(); root = a.root
    request = read(root/'request.json')
    if a.phase=='prepare':
        from osw_prepare import prepare
        prepare(root,Path(request['predecessor']),request['maximum_questions'])
    elif a.phase=='fetch':
        from submission import fetch,verify_assets
        verify_assets()
        dest = root/'official_inputs'; dest.mkdir(exist_ok=True)
        source = Path(read(root/'source.json')['source'])/'submission'
        for file in ('questions.json','question_identity.json'):
            if (source/file).exists() and not (dest/file).exists():
                shutil.copyfile(source/file,dest/file)
        fetch(dest)
    elif a.phase=='train':
        from osw_train import train
        train(root)
    else:
        name = read(root/'selected.json')['name']
        if a.phase=='merge':
            import gpu
            gpu.merge(root/'chain','stage2',name)
        elif a.phase=='submit': submit(root,name)
        else:
            import evaluate
            evaluate.evaluate(root/name,name)


if __name__=='__main__': main()
