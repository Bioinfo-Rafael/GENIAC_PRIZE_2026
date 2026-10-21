"""Pinned public data only; never opens competition inputs or submissions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

PRM='https://media.githubusercontent.com/media/openai/prm800k/7ecc794703b2877f63226f2477a49b34f9b25163/prm800k/math_splits/'
GSM='https://huggingface.co/datasets/openai/gsm8k/resolve/740312add88f781978c0658806c59bc2815b9866/main/'
SOURCES={
 'prm_train':(PRM+'train.jsonl','90d96daeac3fe343ebb1e22ce93dd99690f75983e957f88de42f87cffe1e8076'),
 'prm_test':(PRM+'test.jsonl','35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132'),
 'gsm_train':(GSM+'train-00000-of-00001.parquet','ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d'),
 'gsm_test':(GSM+'test-00000-of-00001.parquet','ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59'),
}


def identity(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC',text).split()).encode()).hexdigest()


def canonical(text):
    # Preserve numbers/operators; remove only whitespace and case differences.
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text).casefold())


def shingles(text):
    words=re.findall(r'\w+|[^\w\s]',unicodedata.normalize('NFKC',text).casefold())
    return frozenset(tuple(words[i:i+5]) for i in range(len(words)-4)) if len(words)>=20 else frozenset()


def fetch(cache):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);result={}
    for name,(url,checksum) in SOURCES.items():
        path=cache/(name+('.jsonl' if name.startswith('prm') else '.parquet'))
        if not path.exists():
            print('FETCH_PUBLIC',name,flush=True)
            with urllib.request.urlopen(url,timeout=120) as response: blob=response.read()
            if hashlib.sha256(blob).hexdigest()!=checksum:raise ValueError('Public source hash mismatch: '+name)
            tmp=path.with_suffix('.partial');tmp.write_bytes(blob);tmp.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('Cached source hash mismatch: '+name)
        if name.startswith('prm'):
            rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            rows=[dict(id=identity(r['problem']),dataset='prm800k',problem=r['problem'],answer=str(r['answer']),
                       subject=r['subject'],level=r['level'],source_id=str(r['unique_id']),source=name) for r in rows]
        else:
            import pyarrow.parquet as pq
            rows=pq.read_table(path).to_pylist()
            if any('####' not in r['answer'] for r in rows):raise ValueError('Missing GSM gold delimiter')
            rows=[dict(id=identity(r['question']),dataset='gsm8k',problem=r['question'],
                       answer=r['answer'].rsplit('####',1)[1].strip(),subject='gsm8k',level='not_provided',
                       source_id=str(i),source=name) for i,r in enumerate(rows)]
        result[name]=rows
        print('PUBLIC_SOURCE',name,len(rows),checksum,flush=True)
    return result


class DuplicateIndex:
    def __init__(self, rows):
        self.exact=set();self.normalized=set();self.sets=[];self.inverted=defaultdict(list)
        for row in rows:self.add(row)

    def add(self,row):
        self.exact.add(row['id']);self.normalized.add(canonical(row['problem']))
        s=shingles(row['problem']);idx=len(self.sets);self.sets.append(s)
        for x in s:self.inverted[x].append(idx)

    def reason(self,row):
        if row['id'] in self.exact:return 'exact'
        if canonical(row['problem']) in self.normalized:return 'case_whitespace'
        s=shingles(row['problem'])
        if not s:return None
        counts=Counter(i for x in s for i in self.inverted.get(x,()))
        for i,intersection in counts.items():
            if intersection/(len(s)+len(self.sets[i])-intersection)>=.85:return 'near_5gram_jaccard_0.85'
        return None


def stratified(rows,n,seed):
    if len(rows)<n:raise ValueError(f'Only {len(rows)} eligible rows for requested {n}; do not silently shrink evaluation')
    groups=defaultdict(list)
    for r in rows:groups[f"{r['subject']}|{r['level']}"].append(r)
    quotas={k:n*len(v)//len(rows) for k,v in groups.items()}
    remainder=n-sum(quotas.values())
    order=sorted(groups,key=lambda k:(-(n*len(groups[k])%len(rows)),k))
    for k in order[:remainder]:quotas[k]+=1
    chosen=[]
    for k in sorted(groups):
        chosen.extend(sorted(groups[k],key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())[:quotas[k]])
    return chosen,quotas


def build_independent(sources, exclusions, seed=20261021):
    train=sources['prm_train']+sources['gsm_train'];index=DuplicateIndex(train+exclusions)
    counts=Counter();eligible=[]
    # Stable order makes dedup deterministic and independent of file ordering.
    for row in sorted(sources['prm_test']+sources['gsm_test'],key=lambda r:(r['id'],r['source'])):
        reason=index.reason(row)
        if reason:counts[row['dataset']+':'+reason]+=1
        else:eligible.append(row)
    # Candidate-to-candidate duplicate handling is separate from training overlap.
    unique=[];seen=DuplicateIndex([])
    for row in eligible:
        reason=seen.reason(row)
        if reason:counts[row['dataset']+':test_duplicate:'+reason]+=1
        else:unique.append(row);seen.add(row)
    selected=[];quotas={}
    for dataset,n in [('prm800k',300),('gsm8k',100)]:
        subset=[r for r in unique if r['dataset']==dataset]
        choice,q=stratified(subset,n,seed);selected.extend(choice);quotas[dataset]=q
    selected.sort(key=lambda r:hashlib.sha256(f"order:{seed}:{r['id']}".encode()).hexdigest())
    assert len({r['id'] for r in selected})==400
    assert not any(index.reason(r) for r in selected)
    return selected,dict(seed=seed,n=400,train_exclusion_rows=len(train),extra_exclusion_rows=len(exclusions),
        rejected=dict(counts),eligible=dict(Counter(r['dataset'] for r in unique)),quotas=quotas,
        overlap_after_filter=0,definition='Pinned public test; disjoint from known public math train/dev/final under documented matching. Pretraining and Stage 1 overlap not established.')
