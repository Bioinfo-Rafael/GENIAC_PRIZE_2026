"""Freeze new PRM/LIMO evaluation BEFORE selecting either stage's training data."""
import random
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from common import *
from published_data import fetch_public, canonical_tokens, response_tokens
from accuracy_data import PRM, PRM_URL
from c_worker import gsm_legacy_target
sys.path.insert(0,str(HERE.parent/'public_eval_diagnostics'))
from data import canonical, shingles


def families(rows):
    """Connected components: normalization, source IDs, near duplicates (transitive)."""
    parent=list(range(len(rows))); keys={}; inverse=defaultdict(list); sets=[]
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(i)]=find(j)
    for i,r in enumerate(rows):
        # Number-masked structural key catches mechanically parameterized variants.
        norm=canonical(r['problem'])
        structural=re.sub(r'\d+(?:\.\d+)?','#',norm)
        aliases=[('normalized',norm),('numeric_template',structural)]
        if r.get('source_id'): aliases.append((r['dataset'],r['source_id']))
        for key in aliases:
            if key in keys: union(i,keys[key])
            else: keys[key]=i
        s=shingles(r['problem']);counts=Counter(j for x in s for j in inverse[x])
        for j,n in counts.items():
            if n/(len(s)+len(sets[j])-n)>=.85: union(i,j)
        sets.append(s)
        for x in s: inverse[x].append(i)
    return [find(i) for i in range(len(rows))]


def split_rows(rows, count=200, excluded=()):
    all_groups=families(rows+list(excluded)); groups=all_groups[:len(rows)]
    rng=random.Random(SEED); taken=set(all_groups[len(rows):]); evaluation=[]; stats={}
    for source in ('limo','prm800k'):
        indices=[i for i,r in enumerate(rows) if r['dataset']==source];rng.shuffle(indices)
        rejected=Counter(); accepted=[]
        available=len({groups[i] for i in indices}-taken)
        for i in indices:
            if groups[i] in taken: rejected['same_or_related_problem']+=1;continue
            if len(accepted)==count: rejected['not_sampled_after_quota']+=1;continue
            taken.add(groups[i]);accepted.append(rows[i])
        evaluation+=accepted
        stats[source]=dict(requested=count,actual=len(accepted),pool=len(indices),independent_available=available,exclusions=dict(rejected))
        if source=='prm800k' and len(accepted)!=count: raise ValueError('Insufficient independent PRM evaluation')
    train=[];seen=set(taken);rejected=Counter()
    # Preserve LIMO membership when the same family occurs in PRM.
    for i in sorted(range(len(rows)),key=lambda i:({'limo':0,'prm800k':1,'gsm8k':2}[rows[i]['dataset']],i)):
        if groups[i] in seen:
            rejected[rows[i]['dataset']+(':evaluation_family' if groups[i] in taken else ':train_family')]+=1
        else: seen.add(groups[i]); train.append(rows[i])
    return evaluation,train,dict(evaluation=stats,training_exclusions=dict(rejected),families=len(set(groups)))


def prepare(root):
    if (root/'data/complete.json').exists():
        for name,h in read(root/'data/complete.json').items():
            if sha(root/name)!=h: raise ValueError('Prepared data changed: '+name)
        return
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    paths=fetch_public(root/'data')
    prm=root/'data/prm.jsonl'
    if not prm.exists():
        # Reuse pinned local cache when available; HF datasets use their own cache.
        cache=next((p for p in root.parent.parent.glob('suites/*/data/prm_official.jsonl') if sha(p)==PRM['sha256']),None)
        if cache: shutil.copyfile(cache,prm)
        else:
            with urllib.request.urlopen(PRM_URL,timeout=120) as src, prm.with_suffix('.partial').open('wb') as dest:
                shutil.copyfileobj(src,dest)
            if sha(prm.with_suffix('.partial'))!=PRM['sha256']: raise ValueError('PRM checksum')
            prm.with_suffix('.partial').replace(prm)
    if sha(prm)!=PRM['sha256']: raise ValueError('PRM checksum')
    raw=[]
    for source,rows,pfield,sfield,afield in (
        ('prm800k',jsonl(prm),'problem','solution','answer'),
        ('limo',jsonl(paths['limo']),'question','solution','answer'),
        ('gsm8k',pq.read_table(paths['gsm8k']).to_pylist(),'question','answer','answer')):
        for i,r in enumerate(rows):
            raw.append(dict(id=question_id(r[pfield]),dataset=source,problem=r[pfield],
                solution=r[sfield],answer=str(r[afield]),source_id=str(r.get('unique_id',i))))
    excluded=[]; prior_split=None
    for path in (root.parent.parent/'inputs/split.json',root.parent.parent/'inputs/dev_eval/split_v1.json',
                 root.parent/'split_v1.json'):
        if path.exists():
            prior=read(path)
            if digest(prior['split'])!=prior['split_sha256']: raise ValueError('Prior split checksum')
            excluded=prior['split']['dev']+prior['split']['final'];prior_split=dict(path=str(path),sha256=sha(path));break
    evaluation,training,stats=split_rows(raw,excluded=excluded)
    stats['previous_eval_exclusion']=prior_split
    immutable_json(root/'data/previous_eval_exclusions.json',excluded)
    immutable_json(root/'data/eval.json',evaluation)  # Freeze before length filtering / sampling train.
    tok=AutoTokenizer.from_pretrained(BASE,revision=REVISION)
    tok.chat_template=(LEGACY/'chat_template.jinja').read_text()
    # Use an existing reserved token; never resize the official vocabulary.
    tok.pad_token='<|fim_pad|>'
    if tok.pad_token_id is None or tok.pad_token_id==tok.eos_token_id: raise ValueError('Distinct padding required')
    tok.save_pretrained(root/'tokenizer/stage2')
    if any(len(tok.apply_chat_template([dict(role='user',content=r['problem'])],tokenize=True,
              add_generation_prompt=True))+2048>8192 for r in evaluation):
        raise ValueError('Frozen evaluation prompt too long; do not silently drop it')
    stage1=[];stage2=[];rejected=Counter()
    for r in sorted(training,key=lambda r:digest([SEED,r['id']])):
        gsm=r['dataset']=='gsm8k';limit=2048 if gsm else 8192
        solution=gsm_legacy_target(r['solution']) if gsm else r['solution']
        tokens,reason=canonical_tokens(tok,r['problem'],solution,limit)
        if reason: rejected[r['dataset']+':'+reason]+=1;continue
        if gsm:
            # run01-style full nonpadding loss in Stage 1; Stage 2 is response-only.
            tokens['labels']=[-100]+tokens['input_ids'][1:]
        (stage1 if gsm else stage2).append(dict(id=r['id'],dataset=r['dataset'],chosen_tokens=tokens))
    if len(stage1)<2000: raise ValueError('Fewer than 2000 independent GSM examples')
    stage1=stage1[:2000]; order=list(range(2000));random.Random(SEED).shuffle(order)
    plans={'stage1':make_plan(stage1,[order[i:i+16] for i in range(0,2000,16)],SEED,{'full':125}),
           'stage2':bounded_plan(stage2)}
    for name,rows in [('stage1',stage1),('stage2',stage2)]:
        immutable_json(root/name/'pairs.json',rows);immutable_json(root/name/'schedule.json',plans[name])
    stats.update(filters=dict(rejected),prm_source=PRM,stage1_examples=2000,
        stage2_tokens=plans['stage2']['tokens_by_source'],stage2_actual=plans['stage2']['actual_tokens'],
        milestones={k:dict(actual_tokens=plans['stage2']['cumulative_tokens'][v-1],step=v,
            tokens_by_source={source:sum(response_tokens(stage2[i]) for g in plans['stage2']['groups'][:v]
                for i in g if stage2[i]['dataset']==source) for source in ('prm800k','limo')})
            for k,v in plans['stage2']['milestone_steps'].items()},
        duplicate_policy='transitive normalized / source ID / numeric template / 5gram Jaccard>=.85; no semantic filtering')
    immutable_json(root/'data/report.json',stats)
    names=['data/previous_eval_exclusions.json','data/eval.json','data/report.json','stage1/pairs.json','stage1/schedule.json',
           'stage2/pairs.json','stage2/schedule.json']
    immutable_json(root/'data/complete.json',{n:sha(root/n) for n in names})
    print('DATA_READY',stats,flush=True)
