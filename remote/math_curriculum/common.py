"""One mathematical curriculum; CPU-only immutable contracts."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'math_training'
sys.path.insert(0, str(LEGACY))
from core import read, write, sha, digest, jsonl, question_id, comparison
from suite_utils import immutable_json
from published_data import response_tokens
from c_utils import make_plan
BASE = 'Qwen/Qwen3-4B-Base'
REVISION = '906bfd4b4dc7f14ee4320094d8b41684abff8539'
SEED = 42
CONFIG = dict(base_model=BASE, base_revision=REVISION, seed=SEED, generations=1,
              temperature=0, top_p=1, max_tokens=2048, context=8192,
              eval_chunk=32, vllm_memory=.8)


def bounded_plan(rows, targets=(100000, 300000), batch=16):
    """Complete responses/full effective batches, never cross either token ceiling.

    Token deficits determine the source. Only length and seeded order determine
    fitting examples. No semantic/answer/score selection and no repeated examples.
    """
    from collections import Counter
    pools = {s: sorted((i for i,r in enumerate(rows) if r['dataset']==s),
                      key=lambda i:digest([SEED,rows[i]['id']])) for s in ('prm800k','limo')}
    sizes = [response_tokens(r) for r in rows]
    if any(n <= 0 for n in sizes): raise ValueError('Empty response')
    used, selected, groups, milestones = Counter(), set(), [], {}
    for target in targets:
        caps = {'prm800k':target*4//5, 'limo':target//5}
        while True:
            group=[]; trial=used.copy()
            for _ in range(batch):
                choices=[]
                for source in pools:
                    fit=next((i for i in pools[source] if i not in selected and i not in group
                              and sizes[i] <= caps[source]-trial[source]),None)
                    if fit is not None: choices.append((trial[source]/caps[source],source,fit))
                if not choices: break
                _,source,i=min(choices); group.append(i);trial[source]+=sizes[i]
            if len(group)!=batch: break
            groups.append(group);selected.update(group);used=trial
        if sum(used.values()) < .95*target:
            raise ValueError(f'Insufficient whole examples near budget {target}: {dict(used)}')
        if abs(used['limo']/sum(used.values())-.2)>.01:
            raise ValueError('Whole-example mixture deviates by more than one percentage point')
        milestones[str(target)] = len(groups)
    if len(set(milestones.values())) != len(targets): raise ValueError('Distinct milestones required')
    return make_plan(rows,groups,SEED,milestones)


def choose(outputs):
    if set(outputs) != {'100000','300000'}: raise ValueError('Both checkpoints required')
    if set(outputs['100000']) != set(outputs['300000']): raise ValueError('Unequal evaluation IDs')
    return max(('100000','300000'),key=lambda n:sum(v[0]['grade']['correct'] for v in outputs[n].values()))
