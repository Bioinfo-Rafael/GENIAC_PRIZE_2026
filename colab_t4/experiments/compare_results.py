"""Aggregate-only paired comparison; rejects incomplete or mismatched evaluation conditions."""
import json
from pathlib import Path

def compare_results(run02_dir, run03_dir):
    def read(directory, name):
        return json.loads((Path(directory)/name).read_text())
    a,b=[read(p,'summary.json') for p in (run02_dir,run03_dir)]
    if not a.get('complete') or not b.get('complete'):
        raise ValueError('Both evaluations must be complete.')
    fields=('code_version','helpers_sha256','base_commit','split_sha256','eval_size',
            'selected_ids_sha256','training_inventory','versions','gpu','config')
    for key in fields:
        if a['signature'][key] != b['signature'][key]:
            raise ValueError('Evaluation conditions differ: '+key)
    states=[read(p,'predictions_private.json') for p in (run02_dir,run03_dir)]
    for summary,state in zip((a,b),states):
        if state['signature'] != summary['signature']:
            raise ValueError('Prediction signature differs from summary.')
    x,y=[s['rows'] for s in states]
    if not x or [r['id'] for r in x] != [r['id'] for r in y] or len({r['id'] for r in x}) != len(x):
        raise ValueError('Prediction IDs differ or are duplicated.')
    for summary,rows in zip((a,b),(x,y)):
        if summary['overall']['n'] != len(rows) or summary['overall']['correct'] != sum(r['correct'] for r in rows):
            raise ValueError('Summary counts do not match predictions.')
    results={}
    for dataset in ('overall','prm800k','gsm8k'):
        pairs=[(l,r) for l,r in zip(x,y) if dataset=='overall' or l['dataset']==dataset]
        if any(l['dataset']!=r['dataset'] for l,r in pairs):raise ValueError('Dataset mismatch')
        n=len(pairs)
        if not n:raise ValueError('Empty dataset')
        old=sum(l['correct'] for l,r in pairs);new=sum(r['correct'] for l,r in pairs)
        results[dataset]={'n':n,'run02_correct':old,'run03_correct':new,'run02_accuracy':old/n,
            'run03_accuracy':new/n,'accuracy_difference':(new-old)/n,
            'run03_only_correct':sum(not l['correct'] and r['correct'] for l,r in pairs),
            'run02_only_correct':sum(l['correct'] and not r['correct'] for l,r in pairs)}
    return {'complete':True,'scope':'Fixed development holdout; not official competition score',
            'split_sha256':a['signature']['split_sha256'],'eval_size':a['signature']['eval_size'],
            'paired':results,'run02_diagnostics':a['by_dataset'],'run03_diagnostics':b['by_dataset']}
