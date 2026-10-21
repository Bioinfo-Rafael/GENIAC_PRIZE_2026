"""Pinned original solutions, whole-example filtering; never competition data."""
import argparse
from collections import Counter
import shutil
import urllib.request
from e1_common import *
from accuracy_data import PRM, PRM_URL
from published_data import fetch_public, canonical_tokens
sys.path.insert(0, str(HERE.parent/'public_eval_diagnostics'))
from data import DuplicateIndex


def prepare(ctx, independent):
    import pyarrow.parquet as pq
    from worker import tokenizer_for
    root, inputs = Path(ctx['run']), Path(ctx['inputs'])
    rows_ind = read(independent); validate_independent(rows_ind)
    old_split = read(inputs/'split.json')
    if digest(old_split['split']) != old_split['split_sha256']:
        raise ValueError('Dev split hash mismatch')
    heldout = old_split['split']['dev'] + old_split['split']['final'] + rows_ind
    # Also protect previously frozen public confirmation/final splits, if present.
    workspace = inputs.parent
    for path in sorted((workspace/'suites').glob('*/data/*_rows.json')):
        if path.name in ('confirmation_rows.json','final_rows.json'):
            heldout += read(path)
    exclusion = DuplicateIndex(heldout)
    paths = fetch_public(root)
    prm = root/'prm_official.jsonl'
    if not prm.exists():
        cached = next((p for p in sorted((workspace/'suites').glob('*/data/prm_official.jsonl'))
                       if sha(p)==PRM['sha256']), None)
        if cached: shutil.copyfile(cached,prm)
        else:
            with urllib.request.urlopen(PRM_URL,timeout=120) as src, prm.with_suffix('.partial').open('wb') as out:
                shutil.copyfileobj(src,out)
            if sha(prm.with_suffix('.partial')) != PRM['sha256']: raise ValueError('PRM checksum')
            prm.with_suffix('.partial').replace(prm)
    if sha(prm)!=PRM['sha256']: raise ValueError('PRM checksum')
    raw = [('prm800k',r['problem'],r['solution']) for r in jsonl(prm)]
    raw += [('gsm8k',r['question'],r['answer']) for r in pq.read_table(paths['gsm8k']).to_pylist()]
    raw += [('limo',r['question'],r['solution']) for r in jsonl(paths['limo'])]
    # Give LIMO priority in cross-source dedup so PRM copies do not erase the LIMO pool.
    raw.sort(key=lambda r:({'limo':0,'prm800k':1,'gsm8k':2}[r[0]],question_id(r[1])))
    tok = tokenizer_for(inputs/'stage2'); pairs=[]; rejected=Counter(); seen=DuplicateIndex([])
    for source,problem,solution in raw:
        row = dict(id=question_id(problem),problem=problem)
        reason = exclusion.reason(row) or seen.reason(row)
        if reason: rejected[source+':duplicate_'+reason]+=1; continue
        tokens,reason = canonical_tokens(tok,problem,solution,8192)
        if reason: rejected[source+':'+reason]+=1; continue
        seen.add(row)
        pairs.append(dict(id=row['id'],dataset=source,chosen_tokens=tokens))
    plan=plan_for(pairs)
    immutable_json(root/'pairs.json',pairs); immutable_json(root/'schedule.json',plan)
    immutable_json(root/'data_report.json',dict(prm_source=PRM,public_sources=read(root/'sources.json'),
        policy='Unedited published solution + EOS; no truncation; prompt masked; dedup priority LIMO then PRM then GSM',
        requested_tokens=600000,actual_tokens=plan['actual_tokens'],
        tokens_by_source=plan['tokens_by_source'],filters=dict(rejected),
        eligible=dict(Counter(r['dataset'] for r in pairs)),heldout_digest=digest(heldout),
        independent_sha256=INDEPENDENT_SHA,exposures=plan['exposures'],
        milestones={name:plan['cumulative_tokens'][step-1] for name,step in plan['milestone_steps'].items()}))
    print('DATA_READY',read(root/'data_report.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--context',type=Path,required=True)
    p.add_argument('--independent',type=Path,required=True);a=p.parse_args()
    prepare(read(a.context),a.independent)
