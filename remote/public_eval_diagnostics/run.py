"""CPU-only saved public dev diagnosis + frozen independent public test set."""
import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'math_training'))
from core import comparison, digest, read, sha, write
from suite_utils import immutable_json
from grading import final_candidate, grade, prepare_gold
from data import SOURCES, build_independent, fetch, identity
from extraction import METHODS, select, summarize


def rows_from(path):
    if path.suffix=='.jsonl':return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    data=read(path)
    if isinstance(data,list):return data
    if isinstance(data,dict):
        # Existing split.json explicitly contains public dev/final; never search arbitrary JSON trees.
        if 'split' in data:
            if digest(data['split'])!=data['split_sha256']:raise ValueError('Split checksum mismatch')
            data=data['split']
        if not any(k in data for k in ('dev','final')):raise ValueError('Not a public split manifest')
        return [r for key in ('dev','final') for r in data.get(key,[])]
    raise ValueError('Expected public row list or split.json')


def csv_write(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def code_identity():
    paths=list(HERE.glob('*.py'))+[HERE.parent/'math_training'/n for n in ('core.py','grading.py','suite_utils.py')]
    return {p.parent.name+'/'+p.name:sha(p) for p in paths}


def load_saved(root):
    dev=read(root/'data/dev.json');ids={r['id'] for r in dev}
    if len(dev)!=400 or len(ids)!=400 or any(identity(r['problem'])!=r['id'] for r in dev):
        raise ValueError('Expected normalized-ID public dev400')
    contracts={};outputs={};paths=[root/'data/dev.json']
    for name in ('A','D','R12'):
        folder=root/'evaluation'/name
        paths.extend([folder/'contract.json',folder/'generations/b0_eval.json'])
        contracts[name]=read(folder/'contract.json');values=read(folder/'generations/b0_eval.json')
        if set(values)!=ids or any(not isinstance(v,list) or len(v)!=1 for v in values.values()):
            raise ValueError('Expected one saved response per dev question: '+name)
        outputs[name]={k:v[0] for k,v in values.items()}
        if contracts[name]['dev_sha']!=digest(dev):raise ValueError('Different saved dev labels: '+name)
        if contracts[name].get('grader_sha')!=sha(HERE.parent/'math_training/grading.py'):
            raise ValueError('Grader code changed; use matching snapshot rather than silently changing verifier')
        packages=contracts[name].get('packages',{})
        if packages.get('math-verify')!=version('math-verify'):
            raise ValueError('Install saved math-verify version in a dedicated CPU environment: '+str(packages))
    controls=[{k:v for k,v in c.items() if k!='model_sha'} for c in contracts.values()]
    if any(c!=controls[0] for c in controls[1:]):raise ValueError('A/D/R12 evaluation contracts differ; no automatic re-inference')
    return dev,outputs,paths


def analyze(dev,outputs,sources,out):
    metadata={r['id']:r for r in sources['prm_train']+sources['prm_test']}
    if any(r['dataset']=='prm800k' and r['id'] not in metadata for r in dev):
        raise ValueError('Cannot join PRM official difficulty metadata')
    gold,failures=prepare_gold(dev);reports={};all_records={};flat=[]
    for name,values in outputs.items():
        checkpoint=out/f'extraction_{name}_rows.json'
        records=read(checkpoint) if checkpoint.exists() else []
        done={r['id'] for r in records}
        for row in dev:
            key=row['id']
            if key in done:continue
            output=values[key];text=output['text'];methods={};cache={}
            for method in METHODS:
                candidate=final_candidate(text)[0] if method=='current' else select(text,method)
                if method=='current':result=grade(row,text,gold[key])
                elif candidate in cache:result=dict(cache[candidate])
                else:
                    result=grade(row,'\\boxed{'+candidate+'}' if candidate else '',gold[key])
                cache[candidate]=result
                methods[method]=dict(result,candidate=candidate)
            meta=metadata.get(key,{})
            records.append(dict(id=key,dataset=row['dataset'],subject=meta.get('subject','gsm8k'),
                level=meta.get('level','not_provided'),tokens=len(output['token_ids']),
                finish_reason=output['finish_reason'],saved_correct=bool(output['grade']['correct']),methods=methods))
            if len(records)%25==0:
                write(checkpoint,records);print('EXTRACTION',name,len(records),'/ 400',flush=True)
        write(checkpoint,records);all_records[name]=records
        reports[name]=dict(groups=summarize(records),
            saved_current_correctness_disagreements=sum(r['saved_correct']!=r['methods']['current']['correct'] for r in records))
        for group,summary in reports[name]['groups'].items():
            for method in METHODS:flat.append(dict(model=name,group=group,method=method,**summary[method]))
    comparisons={}
    for method in METHODS:
        scored={name:{r['id']:{'grade':r['methods'][method]} for r in rs} for name,rs in all_records.items()}
        comparisons[method]={a+'_vs_'+b:comparison(scored[a],scored[b]) for a,b in [('A','D'),('A','R12'),('D','R12')]}
    result=dict(models=reports,comparisons=comparisons,gold_parse_failures=failures,
        note='Diagnostic extraction policies, not submission scores. Missing/unparseable selections count as incorrect; no oracle selection.')
    write(out/'extraction_report.json',result);csv_write(out/'extraction_summary.csv',flat)
    questions=[]
    for name,rs in all_records.items():
        for r in rs:
            for method in METHODS:
                questions.append(dict(model=name,id=r['id'],dataset=r['dataset'],subject=r['subject'],
                    level=r['level'],tokens=r['tokens'],finish_reason=r['finish_reason'],method=method,**r['methods'][method]))
    csv_write(out/'extraction_questions.csv',questions)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r12',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--exclude-public-rows',type=Path,action='append',default=[])
    args=parser.parse_args();started=time.monotonic()
    root=args.r12.resolve();source=Path(read(root/'source.json')['path']).resolve();out=args.output.resolve()
    if any(out==p or p in out.parents for p in (root,source)):
        raise ValueError('Diagnostic artifacts must be outside existing experiment directories')
    out.mkdir(parents=True,exist_ok=True)
    dev,outputs,paths=load_saved(root)
    exclusion_paths=[source/'inputs/split.json']+args.exclude_public_rows
    exclusions=list(dev)
    for p in exclusion_paths:
        for r in rows_from(p):
            if not isinstance(r,dict) or 'problem' not in r:raise ValueError('Only public rows containing problem are supported')
            exclusions.append(dict(r,id=identity(r['problem'])))
    # Actual training IDs must be covered by the pinned public train corpus.
    paths.extend([root/'source.json',source/'data/tokens.json',root/'data/tokens.json',*exclusion_paths])
    run_identity=dict(files={str(p):sha(p) for p in paths},code=code_identity(),sources=SOURCES,
        packages={k:version(k) for k in ('math-verify','sympy','pyarrow')},seed=20261021)
    # JSON roundtrip normalizes source tuples for resumable identity comparisons.
    immutable_json(out/'identity.json',json.loads(json.dumps(run_identity)))
    sources=fetch(args.cache)
    train_ids={r['id'] for k in ('prm_train','gsm_train') for r in sources[k]}
    for p in (source/'data/tokens.json',root/'data/tokens.json'):
        if not {r['id'] for r in read(p)}<=train_ids:raise ValueError('Actual training IDs absent from exclusion corpus')
    print('PHASE independent_public_set',flush=True)
    if (out/'independent_report.json').exists():
        independent=read(out/'independent_report.json')
        if digest(read(out/'independent_dev400.json'))!=independent['rows_sha256']:raise ValueError('Frozen public set changed')
    else:
        rows,independent=build_independent(sources,exclusions)
        independent['rows_sha256']=digest(rows)
        independent['source_hashes']={k:v[1] for k,v in SOURCES.items()}
        independent['gold_parse_failures']=prepare_gold(rows)[1]
        immutable_json(out/'independent_dev400.json',rows);write(out/'independent_report.json',independent)
    print('INDEPENDENT',json.dumps(independent,ensure_ascii=False),flush=True)
    print('PHASE saved_dev_extraction',flush=True)
    report=analyze(dev,outputs,sources,out)
    lines=['# 公開dev回答抽出診断・独立公開評価セット','',
        '保存済み回答だけを再採点したCPU診断。学習・生成・提出用データへのアクセスは行っていない。','',
        '| Model | Current | First box | Last box | Explicit final | First correct → last wrong |',
        '|---|---:|---:|---:|---:|---:|']
    warnings=[]
    for name,r in report['models'].items():
        a=r['groups']['all']
        lines.append('| '+name+' | '+' | '.join(f"{a[m]['accuracy']:.2%}" for m in METHODS)+f" | {a['transitions']['first_correct_last_wrong']} |")
        if r['saved_current_correctness_disagreements']:
            warnings.append(f"\n注意: {name} は保存済み判定と {r['saved_current_correctness_disagreements']} 問不一致。grader timeout等を確認し、従来のスコアと同一視しない。\n")
    lines.extend(warnings)
    lines.extend(['','全手法とも分母は400。抽出失敗を除いた正答率ではない。正解に応じた抽出方法の選択は行わない。',
        '抽出失敗は「選択した規則で解析不能」を意味し、本文に正しい答えがあると証明するものではない。',
        '詳細・失敗数・分野/難易度/生成長別: extraction_summary.csv / extraction_report.json。問題別: extraction_questions.csv。',
        'PRMは公式level、GSMは難易度未提供。生成長の層はモデルごとに構成が異なり、因果効果や同一問題群の比較ではない。','',
        '## 独立した公開評価セット','',
        f"400問 (PRM300 + GSM100)。SHA256: `{independent['rows_sha256']}`。",
        'independent_dev400.json に固定。公開train全体・既存dev/final・指定した追加公開除外リストとの重複を機械的に除外。',
        '対象は数学SFTに対する既知の重複。Stage 1やベースモデル事前学習との独立性は保証していない。',
        '新セットでの推論はまだ行っていない。今後の性能確認まで学習やモデル選択に使用しない。','',
        '選択の内訳・重複除外理由・採点不能ラベル数: independent_report.json。',''])
    (out/'README.md').write_text('\n'.join(lines))
    write(out/'complete.json',dict(status='diagnostics_complete',seconds=time.monotonic()-started,
        inference_performed=False,training_performed=False))
    print('STATUS diagnostics_complete ARTIFACTS',out,flush=True)


if __name__=='__main__':main()
