"""Reuse exactly matching A/D evaluations; otherwise infer into this new run only."""
import argparse
from shared import *
import evaluate as rank_eval
from suite_utils import summarize


def run(root,source,check_only=False):
    rows=read(root/'data/dev.json')
    if rows!=read(source/'data/dev.json'):raise ValueError('Reference dev questions/order differ')
    needed=[]
    for name in ['A','D']:
        contract=rank_eval.contract(source,name)
        outputs=rank_eval.reusable_outputs(source/'evaluation'/name,contract,rows)
        if outputs is not None:
            dest=root/'evaluation'/name
            immutable_json(dest/'contract.json',contract)
            immutable_json(dest/'generations/b0_eval.json',outputs)
            immutable_json(dest/'metrics.json',summarize(outputs,rows))
            immutable_json(dest/'reuse.json',{'source':str(source/'evaluation'/name),
                'responses_sha256':sha(source/'evaluation'/name/'generations/b0_eval.json')})
            print('REFERENCE_REUSED',name,flush=True)
        elif check_only:
            needed.append(name)
            print('REFERENCE_REINFERENCE_REQUIRED',name,flush=True)
        else:
            # Only final trained adapters/parents are linked read-only; merge happens in our directory.
            if name=='D':
                link=root/'conditions/D';link.parent.mkdir(parents=True,exist_ok=True)
                if not link.exists():link.symlink_to((source/'conditions/D').resolve())
            if not verify_model(root/'merged'/name):raise ValueError('Reference merge needed: '+name)
            rank_eval.evaluate(root,name)
    if check_only:write(root/'reference_eval_plan.json',{'reinfer':needed})
    else:rank_eval.evaluate(root,NAME)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--check-only',action='store_true')
    a=p.parse_args();run(a.root,a.source,a.check_only)
