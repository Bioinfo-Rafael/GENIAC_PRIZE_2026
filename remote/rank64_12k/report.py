"""Accuracy is primary; secondary generation metrics never veto an accuracy gain."""
import argparse
import csv
from shared import *
from core import comparison


def report(root):
    metrics={};outputs={}
    for name in ['A','D',NAME]:
        p=root/'evaluation'/name
        if not (p/'metrics.json').exists():continue
        metrics[name]=read(p/'metrics.json')
        outputs[name]={k:v[0] for k,v in read(p/'generations/b0_eval.json').items()}
    train=root/f'conditions/{NAME}/complete.json'
    result=dict(recipe=RECIPE,metrics=metrics,comparisons={},primary_metric='accuracy',
        uncertainty='Paired bootstrap over questions only; not multiple inference/training executions. Dev results are exploratory.',
        training=read(train) if train.exists() else None,official_score=None)
    for name in ['A','D']:
        if name in outputs and NAME in outputs:
            result['comparisons'][name+'_vs_'+NAME]=comparison(outputs[name],outputs[NAME])
    if NAME in metrics and 'D' in metrics:
        result['accuracy_delta_vs_D']=metrics[NAME]['accuracy']-metrics['D']['accuracy']
        result['interpretation']='higher observed accuracy' if result['accuracy_delta_vs_D']>0 else ('same observed accuracy' if result['accuracy_delta_vs_D']==0 else 'lower observed accuracy')
    write(root/'report.json',result)
    columns=['condition','accuracy','correct','gsm8k_accuracy','prm800k_accuracy','repetition_rate',
             'normal_stop_rate','mean_output_tokens','train_seconds','tokens_per_second','peak_vram_bytes']
    lines=['# Rank64: 12,000 vs 6,000 examples','',
           'Primary metric: accuracy. Secondary metrics do not gate selection. Not an official competition score.','',
           '| Condition | Correct / 400 | Accuracy | Repetition | Normal stop | Mean tokens |',
           '|---|---:|---:|---:|---:|---:|']
    source=Path(read(root/'source.json')['path'])
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader()
        for name,m in metrics.items():
            tp=root/f'conditions/{name}/complete.json' if name==NAME else source/f'conditions/{name}/complete.json'
            t=read(tp) if tp.exists() else {}
            writer.writerow(dict(condition=name,accuracy=m['accuracy'],correct=m['correct'],
                gsm8k_accuracy=m['by_dataset']['gsm8k']['correct']/m['by_dataset']['gsm8k']['n'],
                prm800k_accuracy=m['by_dataset']['prm800k']['correct']/m['by_dataset']['prm800k']['n'],
                repetition_rate=m['repetition_rate'],normal_stop_rate=1-m['non_stop_rate'],mean_output_tokens=m['mean_output_tokens'],
                **{k:t.get(k) for k in ['train_seconds','tokens_per_second','peak_vram_bytes']}))
            lines.append(f"| {name} | {m['correct']}/{m['n']} | {m['accuracy']:.2%} | {m['repetition_rate']:.2%} | {1-m['non_stop_rate']:.2%} | {m['mean_output_tokens']:.2f} |")
    changes=[]
    for base,c in result['comparisons'].items():
        lines+=['',f"{base}: delta {c['delta_accuracy']:+.2%}, paired 95% CI {c['paired_bootstrap_95']}; wrong→right {c['wrong_to_right']}, right→wrong {c['right_to_wrong']}."]
        name=base.split('_vs_')[0]
        for key in sorted(outputs[NAME]):
            before=bool(outputs[name][key]['grade']['correct']);after=bool(outputs[NAME][key]['grade']['correct'])
            changes.append(dict(reference=name,id=key,before_correct=before,after_correct=after,delta=int(after)-int(before)))
    write(root/'question_changes.json',changes)
    with (root/'question_changes.csv').open('w') as f:
        w=csv.DictWriter(f,['reference','id','before_correct','after_correct','delta']);w.writeheader();w.writerows(changes)
    if result['training']:
        t=result['training'];lines+=['',f"R12 training: {t['train_seconds']:.2f} s; {t['tokens_per_second']:.2f} supervised tokens/s; peak {t['peak_vram_bytes']/2**30:.2f} GiB.",
            'Training time excludes preparation, merging and evaluation.']
    lines+=['','Submission: `submission/submission.json` (only valid after `submission/complete.json` and successful format check).',
        'No upload to Omnicampus. Submission questions/answers must stay on the authorized remote; do not paste private logs.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
