"""CPU-only checks and selection rules for the eight-hour experiment suite."""
from collections import Counter
from pathlib import Path
from core import read, write, sha, comparison


GENERATION_KEYS = ('base_model', 'base_revision', 'stage2_sha256', 'seed',
                   'generations', 'temperature', 'top_p', 'max_tokens', 'context',
                   'eval_chunk', 'vllm_memory')
# The completed a3b82b4 pilot is the only legacy generation implementation we import.
LEGACY_CODE = {
    'worker.py':'1b9b03143e5f58e2854c5db227b3ba8184d2bbb7764a491a50ed3f001c880552',
    'core.py':'5b596388fd1fcf3a957d9496356decefe17bb100e2960742a4dbf85cffd0c2e2',
    'grading.py':'b0b37cb6114f78341fbb0459840ab5088566e432cbab2c1166c6ea2a2f640d14',
    'chat_template.jinja':'64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326',
}


def validate_source(source, config, current_code, packages, rows, dev):
    source = Path(source)
    ctx = read(source/'context.json')
    for key in GENERATION_KEYS:
        if ctx['config'].get(key) != config.get(key):
            raise ValueError('Source generation config differs: '+key)
    for name, expected in LEGACY_CODE.items():
        actual = sha(source/'code'/name)
        if actual != ctx['code'].get(name) or actual != expected:
            raise ValueError('Unrecognized source generation implementation: '+name)
    for name in ('grading.py','chat_template.jinja','input_manifest.json'):
        if sha(source/'code'/name) != current_code[name]:
            raise ValueError('Source inputs/grading/template differs: '+name)
    for label in ('train','eval'):
        if read(source/f'packages_{label}.json') != packages[label]:
            raise ValueError('Source package environment differs: '+label)
    old_rows = read(source/'train_rows.json')
    if old_rows != rows[:len(old_rows)] or read(source/'eval_rows.json') != dev:
        raise ValueError('Source question order/content differs')
    for purpose, expected_rows in [('train',old_rows), ('eval',dev)]:
        generated = read(source/f'generations/b0_{purpose}.json')
        ids = {r['id'] for r in expected_rows}
        if set(generated) != ids:
            raise ValueError('Source generation incomplete or contains unexpected questions')
        for outputs in generated.values():
            expected = config['generations'] if purpose == 'train' else 1
            if len(outputs) != expected or len({o['index'] for o in outputs}) != expected:
                raise ValueError('Source response count/index mismatch')
    return {name:sha(source/name) for name in
            ('context.json','train_rows.json','eval_rows.json',
             'generations/b0_train.json','generations/b0_eval.json')}


def summarize(outputs, rows):
    ids = [r['id'] for r in rows]
    if set(outputs) != set(ids) or not ids:
        raise ValueError('Evaluation must contain exactly the requested split')
    values = [outputs[k][0] for k in ids]
    n = len(values)
    repeated = sum(any(v >= 4 for v in Counter(
        tuple(words[i:i+10]) for i in range(max(0,len(words)-9))).values())
        for words in (v['text'].split() for v in values))
    return {'n':n, 'correct':sum(v['grade']['correct'] for v in values),
        'accuracy':sum(v['grade']['correct'] for v in values)/n,
        'repeated':repeated, 'repetition_rate':repeated/n,
        'non_stop':sum(v['finish_reason']!='stop' for v in values),
        'non_stop_rate':sum(v['finish_reason']!='stop' for v in values)/n,
        'unknown':sum(any(v['grade'][k] for k in
            ('extraction_failed','grader_error','gold_unparseable')) for v in values),
        'mean_output_tokens':sum(len(v['token_ids']) for v in values)/n,
        'by_dataset':{dataset:{'n':sum(r['dataset']==dataset for r in rows),
            'correct':sum(outputs[r['id']][0]['grade']['correct'] for r in rows if r['dataset']==dataset)}
            for dataset in sorted({r['dataset'] for r in rows})}}


def candidate_report(baseline, output, rows):
    return {'metrics':summarize(output, rows),
        'comparison':comparison({k:v[0] for k,v in baseline.items()},
                                {k:v[0] for k,v in output.items()})}


def eligible(base, candidate):
    """Practical dev gate, not a significance test. No final scores enter selection."""
    return (candidate['accuracy'] >= base['accuracy'] + .02 - 1e-9
        and candidate['repetition_rate'] <= base['repetition_rate'] + .03 + 1e-9
        and candidate['non_stop_rate'] <= base['non_stop_rate'] + .03 + 1e-9
        and candidate['by_dataset']['gsm8k']['correct'] > base['by_dataset']['gsm8k']['correct']-5)


def select_best(base, trials):
    ranked = [(name, result['metrics']) for name,result in trials.items()
              if result.get('status') == 'complete' and eligible(base,result['metrics'])]
    ranked.sort(key=lambda item:(-item[1]['correct'], item[1]['repeated'],
                                 item[1]['mean_output_tokens'], item[0]))
    return ranked[0][0] if ranked else None


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Immutable experiment artifact changed: '+str(path))
    else:
        write(path,value)
