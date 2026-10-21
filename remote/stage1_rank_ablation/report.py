"""Export metrics, paired question changes, confidence intervals and figures."""
import argparse
import csv
import itertools
from common import *
from core import comparison


def report(root):
    result={'conditions':{},'comparisons':{},'notes':[
        'A is existing run03; its training time and VRAM are not measured on this GPU.',
        'Bootstrap resamples questions, not executions; historical A is not a bitwise training control.']}
    outputs={}
    for name in CONDITIONS:
        file=root/'evaluation'/name/'metrics.json'
        if not file.exists():continue
        record=dict(condition=name,**CONDITIONS[name],**read(file))
        train=root/'conditions'/name/'complete.json'
        for key in ['train_seconds','wall_seconds','peak_vram_bytes','tokens_per_second']:
            record[key]=read(train).get(key) if train.exists() else None
        result['conditions'][name]=record
        outputs[name]={k:v[0] for k,v in read(root/'evaluation'/name/'generations/b0_eval.json').items()}
    for a,b in itertools.combinations(outputs,2):
        result['comparisons'][a+'_vs_'+b]=comparison(outputs[a],outputs[b])
    write(root/'report.json',result)
    columns=['condition','stage1','rank','alpha','accuracy','mean_output_tokens','repetition_rate',
             'non_stop_rate','train_seconds','wall_seconds','peak_vram_bytes','tokens_per_second']
    with (root/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,columns,extrasaction='ignore');writer.writeheader();writer.writerows(result['conditions'].values())
    if outputs:
        with (root/'question_changes.csv').open('w') as f:
            names=list(outputs);w=csv.writer(f);w.writerow(['id']+[n+'_correct' for n in names])
            for key in sorted(next(iter(outputs.values()))):w.writerow([key]+[outputs[n][key]['grade']['correct'] for n in names])
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,7));names=list(result['conditions'])
        for ax,key in zip(axes.flat,['accuracy','repetition_rate','mean_output_tokens','train_seconds','peak_vram_bytes','tokens_per_second']):
            values=[result['conditions'][n][key] for n in names]
            ax.bar([n for n,v in zip(names,values) if v is not None],[v for v in values if v is not None]);ax.set_title(key)
        fig.suptitle('Final dev400 / A training timing unavailable');fig.tight_layout()
        fig.savefig(root/'comparison.png',dpi=160);fig.savefig(root/'comparison.svg');plt.close(fig)
    print('REPORT',root/'report.json',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);report(p.parse_args().root)
