"""Pinned official PRM/GSM solutions. Only mechanical processing; no text logging."""
import argparse
import urllib.request
from collections import Counter
from pathlib import Path
from core import read, write, jsonl, sha, digest, question_id, tokenize_answer
from published_data import canonical_tokens
from c_worker import gsm_legacy_target
from suite_utils import immutable_json
from accuracy_utils import balanced_subset, matched_plan
from published_data import response_tokens

PRM = {
    'repository': 'https://github.com/openai/prm800k',
    'revision': '7ecc794703b2877f63226f2477a49b34f9b25163',
    'path': 'prm800k/math_splits/train.jsonl',
    'sha256': '90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076'}
PRM_URL = ('https://media.githubusercontent.com/media/openai/prm800k/'
           + PRM['revision'] + '/' + PRM['path'])


def split_public(rows, old_train, excluded, n=400):
    """Two untouched 3:1 splits; exclude all known run03 training and old evaluation."""
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate public IDs')
    candidates = [r for r in rows if r['id'] not in old_train | excluded]
    confirmation = balanced_subset(candidates, n, 31415)
    ids = {r['id'] for r in confirmation}
    final = balanced_subset([r for r in candidates if r['id'] not in ids], n, 27182)
    heldout = ids | {r['id'] for r in final} | excluded
    return [r for r in rows if r['id'] not in heldout], confirmation, final


def prepare(ctx):
    import pyarrow.parquet as pq
    from worker import tokenizer_for
    root, inputs, cfg = Path(ctx['run']), Path(ctx['inputs']), ctx['config']
    root.mkdir(parents=True, exist_ok=True)
    path = root/'prm_official.jsonl'
    if not path.exists():
        tmp = path.with_suffix('.partial')
        with urllib.request.urlopen(PRM_URL, timeout=120) as src, tmp.open('wb') as out:
            import shutil
            shutil.copyfileobj(src, out)
        if sha(tmp) != PRM['sha256']:
            raise ValueError('Official PRM checksum mismatch')
        tmp.rename(path)
    if sha(path) != PRM['sha256']:
        raise ValueError('Official PRM checksum mismatch')
    from huggingface_hub import hf_hub_download
    gsm = read(Path(__file__).parent/'published_sources.json')['gsm8k']
    gsm_path = Path(hf_hub_download(repo_id=gsm['repo_id'], repo_type='dataset',
                                   revision=gsm['revision'], filename=gsm['filename']))
    if sha(gsm_path) != gsm['sha256']:
        raise ValueError('Official GSM checksum mismatch')
    immutable_json(root/'sources.json', {'prm800k': PRM, 'gsm8k': gsm})
    raw = [('prm800k', r['problem'], r['solution'], r['answer']) for r in jsonl(path)]
    raw += [('gsm8k', r['question'], r['answer'], r['answer'].split('####')[-1].strip())
            for r in pq.read_table(gsm_path).to_pylist()]
    rows, seen = [], set()
    for dataset, problem, solution, answer in raw:
        key = question_id(problem)
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(id=key, dataset=dataset, problem=problem, solution=solution, answer=answer))
    old = jsonl(inputs/'train.jsonl')
    by_id = {r['id']: r for r in rows}
    mapped = Counter()
    for r in old:
        key = question_id(r['problem'])
        if key not in by_id:
            raise ValueError('Inherited training question missing from official public sources')
        mapped[r['dataset']] += 1
    split = read(inputs/'split.json')
    if digest(split['split']) != split['split_sha256']:
        raise ValueError('Inherited split hash mismatch')
    excluded = {question_id(r['problem']) for kind in ('dev', 'final') for r in split['split'][kind]}
    train, confirmation, final = split_public(rows, {question_id(r['problem']) for r in old}, excluded)
    tok = tokenizer_for(inputs/'stage2')
    forms = {'F11': [], 'F10': []}
    online, rejected = [], Counter()
    for r in train:
        answer = gsm_legacy_target(r['solution']) if r['dataset'] == 'gsm8k' else r['solution']
        direct, reason = canonical_tokens(tok, r['problem'], answer, cfg['context'])
        legacy = tokenize_answer(tok, r['problem'], answer, cfg['context']) if direct else None
        prefix = tok.apply_chat_template([{'role': 'user', 'content': r['problem']}],
                                        tokenize=True, add_generation_prompt=True)
        if direct is None or legacy is None or len(prefix)+cfg['max_tokens'] > cfg['context']:
            rejected[reason or 'legacy_or_prompt_overlength'] += 1
            continue
        for form, tokens in [('F11', legacy), ('F10', direct)]:
            forms[form].append({'id': r['id'], 'dataset': r['dataset'], 'chosen_tokens': tokens})
        online.append({k: r[k] for k in ('id', 'dataset', 'problem', 'answer')})
    for kind, items in [('confirmation', confirmation), ('final', final)]:
        for r in items:
            prefix = tok.apply_chat_template([{'role': 'user', 'content': r['problem']}],
                                            tokenize=True, add_generation_prompt=True)
            if len(prefix)+cfg['max_tokens'] > cfg['context']:
                raise ValueError('Held-out prompt too long; no silent evaluation filtering')
        immutable_json(root/f'{kind}_rows.json',
                       [{k: r[k] for k in ('id', 'dataset', 'problem', 'answer')} for r in items])
    subsets = {n: balanced_subset(forms['F11'], n) for n in (6000, 12000)}
    target = min(cfg['sft_tokens'], *(3*sum(map(response_tokens, r)) for r in subsets.values()))
    for n, subset in subsets.items():
        immutable_json(root/f'pairs_{n}.json', subset)
        immutable_json(root/f'plan_{n}.json', matched_plan(subset, target))
    direct_index = {r['id']: r for r in forms['F10']}
    direct_rows = [direct_index[r['id']] for r in subsets[6000]]
    # The terminal control matches examples/update groups exactly, not token totals.
    from c_utils import make_plan
    legacy_plan = read(root/'plan_6000.json')
    immutable_json(root/'pairs_F10.json', direct_rows)
    immutable_json(root/'plan_F10.json', make_plan(direct_rows, legacy_plan['groups'],
                   milestones=legacy_plan['milestone_steps']))
    online_index = {r['id']: r for r in online}
    immutable_json(root/'online_rows.json', [online_index[r['id']] for r in subsets[12000]])
    immutable_json(root/'data_report.json', {
        'prm_source': PRM, 'inherited_questions_matched_to_official': dict(mapped),
        'mapping_scope': 'normalized question match; public solutions used unchanged, not cache answers',
        'source_note': 'Official math_splits/train.jsonl contains completed MATH solutions, not PRM step labels',
        'filters': dict(rejected), 'eligible_by_dataset': dict(Counter(r['dataset'] for r in forms['F11'])),
        'target_tokens': target, 'requested_tokens': cfg['sft_tokens'],
        'confirmation_n': len(confirmation), 'final_n': len(final),
        'splits': 'fresh hash splits exclude known run03 train and old dev/final; no competition data',
        'forms': 'F11 formatted GSM + legacy terminal; F10 same body + direct EOS',
        'limo': 'not used in this comparison; retain the run03 source mixture'})
    print('DATA_READY', read(root/'data_report.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--context', required=True, type=Path)
    prepare(read(p.parse_args().context))
